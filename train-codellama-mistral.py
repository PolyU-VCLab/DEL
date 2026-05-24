 #    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os
import json
import math
import datasets

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import pathlib
import utils
import random

from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import (
    EvalPrediction,
)
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.utils import is_peft_available

if is_peft_available():
    from peft import PeftModel


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    padding_side: Optional[str] = field(default="right")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    template_variation: bool = field(
        default=True, metadata={"help": "whether to use template variation"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1500,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    flash_attn: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, template_variation: bool):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        if 'json' in data_path:
            with open(data_path) as f:
                list_data_dict = json.load(f)
        else:
            list_data_dict = datasets.load_dataset(data_path)["train"]

        logging.warning("Formatting inputs...")
        if template_variation:
            PROMPT_DICT = random.choice(utils.PROMPT_TEMPLATE)
        else:
            PROMPT_DICT = utils.PROMPT_TEMPLATE_SINGLE
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = []
        for example in list_data_dict:
            if example.get("input", "") != "":
                sources.append(prompt_input.format_map(example))
            else:
                sources.append(prompt_no_input.format_map(example))

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path,
                                      template_variation=data_args.template_variation)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class CustomTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_loss_func: Optional[Callable] = None,
    ):

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.compute_loss_func = compute_loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, self.tokenizer)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def custom_compute_loss(outputs, labels, tokenizer):

    def NTL_WAS(logits, labels, tokenizer):
        num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        num_ids = tokenizer.convert_tokens_to_ids(num)
        y=labels.clone()
        num_position = torch.tensor(num_ids, device=labels.device) # token ID for number 0~9
        number = torch.arange(10, device=labels.device) # digit 0~9

        mask = torch.isin(labels, num_position)
        logits = logits[:, num_position]
        logits = logits[mask]
        labels = labels[mask]

        Y_sorted, sorted_indices = torch.sort(num_position)
        pos = torch.searchsorted(Y_sorted, labels)
        original_indices = sorted_indices[pos]
        y[mask] = number[original_indices] # labels map to digits

        probs = F.softmax(logits, dim=-1)  #[N,10]
        seq = torch.where(mask)[0]
        abs_diff = torch.abs(y[seq].unsqueeze(-1) - number)
        loss_num = (abs_diff * probs).mean()
        if torch.any(torch.isnan(loss_num)):
            loss_num = 0
        return loss_num

    def generate_place(labels_matrix):

        non_number = (labels_matrix < 0) | (labels_matrix > 10) 
        number = (labels_matrix >= 0) & (labels_matrix <= 10)
        digit = (labels_matrix >= 0) & (labels_matrix <= 9) 
        dot = labels_matrix == 10
        place = labels_matrix.clone()
        place[non_number] = 1/1.02
        place[number] = 1
        place = place.prod(1) 
        m = labels_matrix == 10
        ind = torch.argmax(m.int(), dim=1)

        place_dot = (1/1.02) ** (labels_matrix.shape[1] - ind)
        place[ind>0] = place_dot[ind>0]
        
        exponents = torch.arange(labels_matrix.shape[1]-1, -1, -1, device=labels_matrix.device)
        r = 1.02 ** exponents # generate [100000, 10000, 1000, 100, 10, 1]
        place = r.unsqueeze(0).expand_as(labels_matrix)*place[:,None]

        find_decimal = place.clone()
        place[digit] = torch.log(place[digit]) / math.log(1.02) +1

        decimal = place < 0
        integer = place > 0
        place[integer] = place[integer] * 2
        place[decimal] = find_decimal[decimal]
        place[dot] = 0 
        place[non_number] = 0
        return place

    def generate_integer_place(labels_matrix):

        non_number = (labels_matrix < 0) | (labels_matrix > 9)
        digit = (labels_matrix >= 0) & (labels_matrix <= 9) 
        place = labels_matrix.clone()
        place[non_number] = 1/1.02
        place[digit] = 1
        place = place.prod(1)
        exponents = torch.arange(labels_matrix.shape[1]-1, -1, -1, device=labels_matrix.device)
        r = 1.02 ** exponents # generate [100000, 10000, 1000, 100, 10, 1]
        place = r.unsqueeze(0).expand_as(labels_matrix)*place[:,None]
        place[digit] = torch.log(place[digit]) / math.log(1.02) +1
        place[non_number] = 0

        return place

    def DIST2_loss(logits, labels, tokenizer):
        num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        num_ids = tokenizer.convert_tokens_to_ids(num)
        y=labels.clone()
        num_position = torch.tensor(num_ids, device=labels.device) # token ID for number 0~9
        number = torch.arange(10, device=labels.device) # digit 0~9, including '.'

        mask = torch.isin(labels, num_position)
        logits = logits[:, num_position]
        logits = logits[mask]
        labels = labels[mask]

        Y_sorted, sorted_indices = torch.sort(num_position)
        pos = torch.searchsorted(Y_sorted, labels)
        original_indices = sorted_indices[pos]
        y[mask] = number[original_indices] # labels map to digits

        seq = torch.where(mask)[0]
        
        # recognize floating-point numbers
        groups = []
        current_group = []
        for i, num in enumerate(seq):
            if i == 0:
                current_group.append(num)
            else:
                if num - seq[i-1] == 1:
                    current_group.append(num)
                else:
                    groups.append(current_group)
                    current_group = [num]
        groups.append(current_group)

        # groups: like [[101, 102, 103], [106, 107], [109], [111, 112, 113], [115, 116]]

        if seq.shape[0] > 0:
            tensors = [y[torch.tensor(g)] for g in groups]
            g = pad_sequence(tensors, batch_first=True, padding_value=-2)

            digit = (g >= 0) & (g <= 9)
            place = generate_integer_place(g.float())
            square_diff = (y[seq].unsqueeze(-1) - number) ** 2
            p_square_diff = F.softmax(-square_diff.float(), dim=-1)

            loss_fn = nn.KLDivLoss(reduction="none")
            loss_num = (loss_fn(logits.log_softmax(-1), p_square_diff)).mean(-1)
            loss_num = (loss_num * place[digit]).mean()

            if torch.any(torch.isnan(loss_num)):
                print('-'*50)
                print('place.shape', place.shape)
                print('place', place)
                print('place[digit]', place[digit])
                print('place[digit].shape', place[digit].shape)
        else:
            loss_num = 0

        return loss_num

    def DEL(logits, labels, tokenizer):
        num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
        num_ids = tokenizer.convert_tokens_to_ids(num)
        digit_ids = tokenizer.convert_tokens_to_ids(num[:10])
        y=labels.clone()
        num_position = torch.tensor(num_ids, device=labels.device) # token ID for number 0~9 and dot
        digit_position = torch.tensor(digit_ids, device=labels.device) # token ID for digit 0~9
        number = torch.arange(11, device=labels.device) # digit 0~9, including '.'

        mask = torch.isin(labels, num_position)
        mask_digit = torch.isin(labels, digit_position)
        logits = logits[:, digit_position[:10]]
        logits_digit = logits[mask_digit]
        labels = labels[mask]

        Y_sorted, sorted_indices = torch.sort(num_position)
        pos = torch.searchsorted(Y_sorted, labels)
        original_indices = sorted_indices[pos]
        y[mask] = number[original_indices] # labels map to digits

        probs = F.softmax(logits_digit, dim=-1)  #[N,10]
        seq = torch.where(mask)[0]
        seq_digit = torch.where(mask_digit)[0]
        
        # recognize floating-point numbers
        groups = []
        current_group = []
        for i, num in enumerate(seq):
            if i == 0:
                current_group.append(num)
            else:
                if num - seq[i-1] == 1:
                    current_group.append(num)
                else:
                    groups.append(current_group)
                    current_group = [num]
        groups.append(current_group) 

        # groups: like [[101, 102, 103], [106, 107], [109], [111, 112, 113], [115, 116]]

        if seq.shape[0] > 0:
            tensors = [y[torch.tensor(g)] for g in groups]
            g = pad_sequence(tensors, batch_first=True, padding_value=-2)

            digit = (g >= 0) & (g <= 10) # digits including decimal dots
            place = generate_place(g.float())

            bce_loss = F.binary_cross_entropy_with_logits(
                logits_digit,
                F.one_hot(y[seq_digit].long(), num_classes=10).float(),
                reduction='none'
            )

            # loss is the absolute difference weighted by the softmax probs

            digit1 = (g >= 0) & (g <= 9)
            criterion_entropy = bce_loss * probs_digit # criterion term

            if digit1.any():

                loss_num = 0.1 * (criterion_entropy * place[digit1][:, None]).mean()

                if torch.any(torch.isnan(loss_num)):
                    print('-'*50)
                    print('loss_num', loss_num)
                    print('place.shape', place.shape)
                    print('place', place)
                    print('place[digit]', place[digit])
                    print('place[digit].shape', place[digit].shape)
                    print('place[digit1]', place[digit1])
                    print('place[digit1].shape', place[digit1].shape)
            else:
                loss_num = 0

        else:
            loss_num = 0

        return loss_num

    def EMOloss(logits, labels, mask, vocab_size, tokenizer):

        labels_tmp = labels.clone()
        labels_tmp[labels_tmp==(-100)] = 0
        one_hot = F.one_hot(labels_tmp, num_classes=vocab_size).to(logits.dtype)
        stable_onehot = (one_hot+1e-15) / torch.linalg.vector_norm((one_hot+1e-15), ord=1, dim=-1, keepdim=True) # (bsz*seq_len, vocab_size)
        embedding_matrix = tokenizer.cost_embedding.to(dtype=logits.dtype) # (vocab_size, hidden_size)
        embedding_matrix = embedding_matrix / torch.linalg.vector_norm(embedding_matrix, ord=2, dim=1, keepdim=True)
        p_contextual_repr = stable_onehot @ embedding_matrix # (bsz*seq_len, hidden_size)
        q_grad = F.softmax(logits, dim=-1) # (bsz*seq_len, vocab_size)
        gt_q = (q_grad * one_hot).detach()
        q_final = q_grad - gt_q
        q_contextual_repr = q_final @ embedding_matrix # (bsz*seq_len, hidden_size)
        emo_loss = (1 - torch.sum(p_contextual_repr*q_contextual_repr, dim=-1)) # (bsz*seq_len,)
        emo_loss = emo_loss * mask
            
        return emo_loss
        
    logits = outputs.logits.float()
    vocab_size=logits.shape[-1]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        valid_mask = shift_labels != -100
        mle_loss = loss_fct(shift_logits, shift_labels)

        '''
        # MixCE
        mask = valid_mask.to(logits.dtype)

        with torch.no_grad():
            q = torch.exp(-mle_loss.detach())
        mle_loss = (
            0.5 * mle_loss
            + (1.0 - 0.5) * q * mle_loss
        )

        loss = (mle_loss * mask).sum()
        '''

        '''
        # EMO
        #emo_loss = EMOloss(shift_logits, shift_labels, mask, vocab_size, tokenizer)
        #loss = ((mle_loss / (emo_loss_optimized+1e-10)).detach()  * emo_loss_optimized + mle_loss) * 0.5
        #loss = (loss * mask).sum() / (1e-15 + mask.sum())
        '''
        
        loss_num = DEL(shift_logits[valid_mask], shift_labels[valid_mask], tokenizer)
        loss = mle_loss + loss_num

        return loss

def train():

    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('Start Loading Model')
    if training_args.flash_attn:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        ).to('cuda')
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        ).to('cuda')

    print('Start building tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    print("*"*50)
    print("Before adding, tokenizer length: ",len(tokenizer))
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("*"*50)
    print("After adding, tokenizer length: ",len(tokenizer))

    print('Start building data module')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    cost_embedding = copy.deepcopy(model.lm_head.weight.data)
    tokenizer.cost_embedding = cost_embedding
    print('--------------------------------cost_embedding setting complete--------------------------------')

    print('Start building the trainer module')

    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, compute_loss_func=custom_compute_loss, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
