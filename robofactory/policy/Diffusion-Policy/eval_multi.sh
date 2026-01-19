#!/bin/bash

# Parallel evaluation using all available GPUs
# Usage: ./eval_multi.sh <config_name> <data_num> <checkpoint_num> [debug_mode] [task_name] [max_steps] [num_eval]

# Change to the script's directory so relative paths work correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <config_name> <data_num> <checkpoint_num> [debug_mode] [task_name] [max_steps] [num_eval]"
    exit 1
fi

CONFIG_NAME="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
DEBUG_MODE="${4:-0}"
TASK_NAME="${5:-$CONFIG_NAME}"
MAX_STEPS="${6:-250}"
NUM_EVAL="${7:-100}"
START_SEED="${8:-1000}"

# Automatically detect number of available GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
else
    NUM_GPUS=1
    GPU_NAMES="Unknown"
fi

# Ensure at least 1 GPU
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

TOTAL_SEEDS=$NUM_EVAL

# Cap NUM_GPUS to TOTAL_SEEDS (no point using more GPUs than seeds)
if [ "$NUM_GPUS" -gt "$TOTAL_SEEDS" ]; then
    NUM_GPUS=$TOTAL_SEEDS
fi

LOG_FILE="eval_results_diffusion_policy_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"
TEMP_DIR=$(mktemp -d)

echo "Evaluating Diffusion Policy task: $TASK_NAME"
echo "Detected $NUM_GPUS GPUs ($GPU_NAMES)"
echo "Using $NUM_GPUS GPUs for parallel evaluation"
echo "Total episodes: $TOTAL_SEEDS (seeds $START_SEED to $((START_SEED + TOTAL_SEEDS - 1)))"
echo "Max steps per episode: $MAX_STEPS"
echo "Evaluating task: $TASK_NAME" >> "$LOG_FILE"
echo "Parallel evaluation with $NUM_GPUS GPUs" >> "$LOG_FILE"

QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

# Calculate episodes per GPU
episodes_per_gpu=$(( (TOTAL_SEEDS + NUM_GPUS - 1) / NUM_GPUS ))

# Function to run evaluation for a range of episodes on a specific GPU
run_gpu_eval() {
    local gpu_id=$1
    local start_seed=$2
    local num_episodes=$3
    local result_file="$TEMP_DIR/gpu_${gpu_id}.txt"
    
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_GL=egl
    export DISPLAY=""
    
    echo "GPU $gpu_id: Running $num_episodes episodes starting from seed $start_seed"
    
    # Run all episodes in a single Python process
    CUDA_VISIBLE_DEVICES=$gpu_id OUTPUT=$(cd ../.. && python ./policy/Diffusion-Policy/eval_multi_dp.py \
        --config="$CONFIG_NAME" \
        --data-num=$DATA_NUM \
        --checkpoint-num=$CHECKPOINT_NUM \
        --max-steps=$MAX_STEPS \
        --seed=$start_seed \
        --num-episodes=$num_episodes \
        --render-mode="rgb_array" \
        -o="rgb" \
        -b="cpu" \
        -n 1 \
        $QUIET_FLAG 2>&1)
    
    echo "$OUTPUT"
    
    # Parse results from output (format: RESULT:seed,success)
    echo "$OUTPUT" | grep "^RESULT:" | sed 's/RESULT://' >> "$result_file"
    
    local success=$(echo "$OUTPUT" | grep "^RESULT:" | grep ",1$" | wc -l)
    local total=$(echo "$OUTPUT" | grep "^RESULT:" | wc -l)
    echo "GPU $gpu_id: $success/$total successes"
}

# Launch parallel jobs
echo ""
echo "Launching parallel evaluation jobs..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    start=$((START_SEED + gpu * episodes_per_gpu))
    remaining=$((TOTAL_SEEDS - gpu * episodes_per_gpu))
    num_eps=$episodes_per_gpu
    if [ $num_eps -gt $remaining ]; then
        num_eps=$remaining
    fi
    if [ $num_eps -gt 0 ]; then
        echo "  GPU $gpu: seeds $start to $((start + num_eps - 1)) ($num_eps episodes)"
        run_gpu_eval $gpu $start $num_eps &
    fi
done

# Wait for all jobs to complete
echo ""
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
            if [ "$fine" -eq 1 ] 2>/dev/null; then
                ((SUCCESS++))
            fi
            ((TOTAL++))
            SUCCESS_RATE=$(python3 -c "print(f'{$SUCCESS / $TOTAL * 100:.4f}')" 2>/dev/null || echo "0")
            echo "$seed, $fine, $SUCCESS_RATE%" >> "$LOG_FILE"
        done < "$result_file"
    fi
done

if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$(python3 -c "print(f'{$SUCCESS / $TOTAL * 100:.4f}')" 2>/dev/null || echo "0")
else
    SUCCESS_RATE="0"
fi
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
