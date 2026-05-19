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
 - Set up the environment
  ```Shell
   conda create -n DEL python=3.10
   conda activate DEL
   ```

 - For training Qwen and DeepSeek-Math, we use
   ```Shell
   pip install -r requirements-qwen-deepseek.txt
   ```
   For training CodeLlama and Mistral, we use
   ```Shell
   pip install -r requirements-codellama-mistral.txt
   ```
