#!/bin/bash

# Parallel evaluation using all available GPUs
# Usage: ./eval_multi.sh <config_name> <data_num> <checkpoint_num> [debug_mode] [task_name]

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <config_name> <data_num> <checkpoint_num> [debug_mode] [task_name]"
    exit 1
fi

CONFIG_NAME="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
DEBUG_MODE="${4:-0}"
TASK_NAME="${5:-$CONFIG_NAME}"

# Number of GPUs to use
NUM_GPUS=8
START_SEED=1000
END_SEED=1099
TOTAL_SEEDS=$((END_SEED - START_SEED + 1))

LOG_FILE="eval_results_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"
TEMP_DIR=$(mktemp -d)

echo "Evaluating task: $TASK_NAME"
echo "Using $NUM_GPUS GPUs for parallel evaluation"
echo "Seeds: $START_SEED to $END_SEED ($TOTAL_SEEDS total)"
echo "Evaluating task: $TASK_NAME" >> "$LOG_FILE"
echo "Parallel evaluation with $NUM_GPUS GPUs" >> "$LOG_FILE"

QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

# Function to run evaluation for a range of seeds on a specific GPU
run_gpu_eval() {
    local gpu_id=$1
    local start=$2
    local end=$3
    local result_file="$TEMP_DIR/gpu_${gpu_id}.txt"
    
    local success=0
    local total=0
    
    for ((seed=start; seed<=end; seed++)); do
        export PYOPENGL_PLATFORM=egl
        export MUJOCO_GL=egl
        export DISPLAY=""
        
        CUDA_VISIBLE_DEVICES=$gpu_id OUTPUT=$(python ./policy/Diffusion-Policy/eval_multi_dp.py \
            --config="$CONFIG_NAME" \
            --data-num=$DATA_NUM \
            --checkpoint-num=$CHECKPOINT_NUM \
            --render-mode="rgb_array" \
            -o="rgb" \
            -b="cpu" \
            -n 1 \
            -s $seed)
        
        LAST_LINE=$(echo "$OUTPUT" | tail -n 1)
        fine=0
        if [[ $LAST_LINE == *"success"* ]]; then
            fine=1
            ((success++))
        fi
        ((total++))
        echo "$seed,$fine" >> "$result_file"
    done
    
    echo "GPU $gpu_id: $success/$total successes"
}

# Calculate seeds per GPU
seeds_per_gpu=$(( (TOTAL_SEEDS + NUM_GPUS - 1) / NUM_GPUS ))

# Launch parallel jobs
echo "Launching parallel evaluation jobs..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    start=$((START_SEED + gpu * seeds_per_gpu))
    end=$((start + seeds_per_gpu - 1))
    if [ $end -gt $END_SEED ]; then
        end=$END_SEED
    fi
    if [ $start -le $END_SEED ]; then
        echo "  GPU $gpu: seeds $start to $end"
        run_gpu_eval $gpu $start $end &
    fi
done

# Wait for all jobs to complete
echo "Waiting for all GPU jobs to complete..."
wait
echo "All jobs completed!"

# Aggregate results
echo ""
echo "Aggregating results..."
SUCCESS=0
TOTAL=0

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    result_file="$TEMP_DIR/gpu_${gpu}.txt"
    if [ -f "$result_file" ]; then
        while IFS=',' read -r seed fine; do
            if [ "$fine" -eq 1 ]; then
                ((SUCCESS++))
            fi
            ((TOTAL++))
            SUCCESS_RATE=$(python3 -c "print(f'{$SUCCESS / $TOTAL * 100:.4f}')" 2>/dev/null || echo "0")
            echo "$seed, $fine, $SUCCESS_RATE%" >> "$LOG_FILE"
        done < "$result_file"
    fi
done

SUCCESS_RATE=$(python3 -c "print(f'{$SUCCESS / $TOTAL * 100:.4f}')" 2>/dev/null || echo "0")
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"
echo ""
echo "========================================"
echo "Final Results:"
echo "  Total Episodes: $TOTAL"
echo "  Successes: $SUCCESS"
echo "  Success Rate: $SUCCESS_RATE%"
echo "========================================"
echo "Results saved in $LOG_FILE"

# Cleanup
rm -rf "$TEMP_DIR"