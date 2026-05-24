# Running  evaluation on all the 7 benchmarks at a time

```
bash all.sh
```

# Running evaluation on a specific benchmark

run ```bash eval.sh```

### code generation

```
dataset='gsm8k'

###  ['gsm8k', 'svamp', 'math', 'simuleq']

python run_open.py \
 --model "your_model_path" \
 --shots 0 \
 --stem_flan_type "pot_prompt" \
 --dataset $dataset \
 --model_max_length 1500 \
 --cot_backup \
 --print \
```

### CoT mathematical reasoning

```
dataset='aqua'

###  ['aqua', 'sat', 'mmlu_mathematics']

python run_choice.py \
  --model "your_model_path" \
  --dataset $dataset \
  --cot_backup \
  --print
```
