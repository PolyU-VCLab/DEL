#!/bin/bash
export MODEL_PATH="Qwen/Qwen2.5-1.5B"
export OUTPUT_PATH="checkpoints/DEL-qwen"
export MASTER_ADDR="localhost"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export NUM_GPUS=4

# Training
echo "=========================================="
echo "Start training..."
echo "=========================================="

torchrun --master_addr ${MASTER_ADDR} \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=6008 \
  train-qwen-deepseek.py \
  --model_name_or_path $MODEL_PATH \
  --data_path "TIGER-Lab/MathInstruct" \
  --bf16 True \
  --output_dir ${OUTPUT_PATH}\
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --save_strategy "steps" \
  --save_steps 2000\
  --save_total_limit 1 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
  --tf32 True

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training complete!"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed"
    echo "=========================================="
    exit 1
fi

# Evaluation
echo "=========================================="
echo "Start evaluation..."
echo "=========================================="

cd "./eval" || { echo "cannot enter ./eval, evaluation failed"; exit 1; }

export VLLM_WORKER_MULTIPROC_METHOD=spawn
model="$OUTPUT_PATH"
model_name=$(basename "$model")
eval_dir="./eval/${model_name}"
mkdir -p "$eval_dir"

datasets1=("aqua" "sat" "mmlu_mathematics")
datasets2=("svamp" "gsm8k" "simuleq" "math")

evaluate_dataset() {
    local dataset=$1
    local script_type=$2
    
    echo "Evaluating: $dataset (using: $script_type,: $(pwd))"

    if [ "$script_type" = "run_open.py" ]; then
        python run_open.py \
         --model ../$model \
         --shots 0 \
         --stem_flan_type "pot_prompt" \
         --dataset $dataset \
         --model_max_length 1500 \
         --cot_backup \
         --print \
         >"$eval_dir/${dataset}.log" 2>&1
    elif [ "$script_type" = "run_choice.py" ]; then
        python run_choice.py \
         --model ../$model \
         --shots 0 \
         --dataset $dataset \
         --cot_backup \
         --print \
         >"$eval_dir/${dataset}.log" 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        echo "$dataset evaluation complete"
        sleep 1
        
        if [ -f "$eval_dir/${dataset}.log" ]; then
            echo "=== $dataset accuracy ==="
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(final accuracy)" | tail -1
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(Log MAE)" | tail -1
            if [ $? -ne 0 ]; then
                tail -1 "$eval_dir/${dataset}.log"
            fi
            echo "=========================="
        else
            echo "warning: log file $eval_dir/${dataset}.log does not exist"
        fi
    else
        echo "$dataset evaluation failed"
        exit 1
    fi
}

for dataset in "${datasets1[@]}"; do
    evaluate_dataset "$dataset" "run_choice.py"
done

for dataset in "${datasets2[@]}"; do
    evaluate_dataset "$dataset" "run_open.py"
done

cd ..

echo "All evaluations complete!"