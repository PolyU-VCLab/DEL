# Digit Entropy Loss for Numerical Learning of LLMs

```
    ██████╗  ███████╗ ██╗     
    ██╔══██╗ ██╔════╝ ██║     
    ██║  ██║ █████╗   ██║     
    ██║  ██║ ██╔══╝   ██║     
    ██████╔╝ ███████╗ ███████╗
    ╚═════╝  ╚══════╝ ╚══════╝
```

<a href="https://arxiv.org/abs/2605.20369"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=Arxiv"></a>

### DEL is made for accurate number generation in language models.

```
@article{zheng2026DEL,
  title={DEL: Digit Entropy Loss for Numerical Learning of Large Language Models},
  author={Zheng, Zhaohui and He, Chenhang and Wang, Shihao and Li, Yuxuan and Cheng, Ming-Ming and Zhang, Lei},
  journal={arXiv preprint arXiv:2605.20369},
  year={2026}
}
```

# Installation

 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/PolyU-VCLab/DEL.git
   cd DEL
   ```
   
 - Set up the environment for training Qwen and DeepSeek-Math,
  ```Shell
   conda create -n DEL-qwen python=3.10
   conda activate DEL-qwen
   pip install -r requirements-qwen-deepseek.txt
   ```

 - Set up the environment for training CodeLlama and Mistral,
   ```Shell
   conda create -n DEL-llama python=3.10
   conda activate DEL-llama
   pip install -r requirements-codellama-mistral.txt
   ```

# Training
   ```Shell
   bash train_qwen.sh  # train Qwen
   bash train_deepseek.sh  # train DeepSeek-Math-Instruct
   bash train_codellama.sh  # train CodeLlama
   bash train_mistral.sh  # train Mistral
   ```
When training is complete, evaluation will automatically process.

# Evaluation
   ```Shell
   cd eval
   unzip dataset.zip
   bash all.sh  # evaluate the seven mathematical reasoning benchmarks
   bash eval.sh  # evaluate one benchmark
   ```
You need to modify the model path in `all.sh` and `eval.sh`.

## Pretrained models

| Model | mACC |
|:----------:|:----:|
| [CodeLlama-7B](https://huggingface.co/Zzh-tju/CodeLlama-7B) | 49.0 |
| [Qwen2.5-1.5B](https://huggingface.co/Zzh-tju/qwen2.5-1.5B) | 55.4 |
| [Mistral-7B](https://huggingface.co/Zzh-tju/Mistral-7B) | 56.5 |
| [DeepSeek-math-7B-Instruct](https://huggingface.co/Zzh-tju/DeepSeek-Math-7B-instruct) | 66.1 |
| [Qwen2.5-7B](https://huggingface.co/Zzh-tju/qwen2.5-7B) | 70.6 |

## Loss ablation

The following results are evaluated on Qwen2.5-1.5B.

| Method | mACC | Venue | 
|:----------:|:----:|:----:|
| [MLE](https://huggingface.co/Zzh-tju/Qwen2.5-1.5B_CE) | 52.8 | - |
| [MixCE](https://huggingface.co/Zzh-tju/Qwen2.5-1.5B_MixCE) | 52.9 | [ACL 2023](https://aclanthology.org/2023.acl-long.502/) |
| [EMO](https://huggingface.co/Zzh-tju/Qwen2.5-1.5B_EMO) | 53.4 | [ICLR 2024](https://openreview.net/forum?id=4bLXfRd0CX&noteId=4bLXfRd0CX) |
| [NTL-WAS](https://huggingface.co/Zzh-tju/Qwen2.5-1.5B_NTL) | 53.8 | [ICML 2025](https://openreview.net/forum?id=V66xc5KxgY&noteId=V66xc5KxgY) |
| [DIST2Loss](https://huggingface.co/Zzh-tju/Qwen2.5-1.5B_DIST) | 53.1 | [ICLR 2026](https://openreview.net/forum?id=s0zLtkY7iu) |
| [DEL (Ours)](https://huggingface.co/Zzh-tju/qwen2.5-1.5B) | 55.4 | - |

# Acknowledgments

Thank you to Xiang Yue et al. for their fork of [MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH/), which is an exellent work for mathematical reasoning.
