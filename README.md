# DEL
Digit Entropy Loss for Numerical Learning of LLMs

```
    ██████╗  ███████╗ ██╗     
    ██╔══██╗ ██╔════╝ ██║     
    ██║  ██║ █████╗   ██║     
    ██║  ██║ ██╔══╝   ██║     
    ██████╔╝ ███████╗ ███████╗
    ╚═════╝  ╚══════╝ ╚══════╝
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

