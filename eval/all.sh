export VLLM_WORKER_MULTIPROC_METHOD=spawn
model='your_model_path'

model_name=$(basename "$model")

eval_dir="./eval/${model_name}"
mkdir -p "$eval_dir"

datasets1=("aqua" "sat" "mmlu_mathematics")
datasets2=("svamp" "gsm8k" "simuleq" "math")

evaluate_dataset() {
    local dataset=$1
    local script_type=$2
    
    echo "evaluating: $dataset (using: $script_type)"
    
    if [ "$script_type" = "run_open.py" ]; then
        python run_open.py \
         --model $model \
         --shots 0 \
         --stem_flan_type "pot_prompt" \
         --dataset $dataset \
         --model_max_length 1500 \
         --cot_backup \
         --print \
         >"$eval_dir/${dataset}.log" 2>&1
    elif [ "$script_type" = "run_choice.py" ]; then
        python run_choice.py \
         --model $model \
         --shots 0 \
         --dataset $dataset \
         --cot_backup \
         --print \
         >"$eval_dir/${dataset}.log" 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset evaluation complete"
        sleep 1
        
        if [ -f "$eval_dir/${dataset}.log" ]; then
            echo "=== $dataset accuracy ==="
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(final accuracy)" | tail -1
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(Log MAE)" | tail -1
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(int_num)" | tail -1
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(Int Number accuracy)" | tail -1
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(frac_num)" | tail -1
            tail -30 "$eval_dir/${dataset}.log" | grep -iE "(Decimal Number accuracy)" | tail -1
            if [ $? -ne 0 ]; then
                tail -1 "$eval_dir/${dataset}.log"
            fi
            echo "=========================="
        else
            echo "warning: log file $eval_dir/${dataset}.log does not exist"
        fi
    else
        echo "✗ $dataset evaluation failed"
        exit 1
    fi
}

echo "evaluation dir: $eval_dir"

for dataset in "${datasets1[@]}"; do
    evaluate_dataset "$dataset" "run_choice.py"
done

for dataset in "${datasets2[@]}"; do
    evaluate_dataset "$dataset" "run_open.py"
done

echo "Evaluation complete！Log is saved at: $eval_dir"