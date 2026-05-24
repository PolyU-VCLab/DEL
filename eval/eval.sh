export VLLM_WORKER_MULTIPROC_METHOD=spawn
# dataset='gsm8k'
# python run_open.py \
#  --model "Zzh-tju/qwen2.5-1.5B" \
#  --shots 0 \
#  --stem_flan_type "pot_prompt" \
#  --dataset $dataset \
#  --model_max_length 1500 \
#  --cot_backup \
#  --print \

dataset='aqua'

python run_choice.py \
  --model "Zzh-tju/qwen2.5-1.5B" \
  --dataset $dataset \
  --cot_backup \
  --print