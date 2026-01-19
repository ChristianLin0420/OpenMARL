#!/bin/bash

# Parallel evaluation using all available GPUs (matching Diffusion Policy)
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

END_SEED=$((START_SEED + NUM_EVAL - 1))
TOTAL_SEEDS=$NUM_EVAL

# Cap NUM_GPUS to TOTAL_SEEDS (no point using more GPUs than seeds)
if [ "$NUM_GPUS" -gt "$TOTAL_SEEDS" ]; then
    NUM_GPUS=$TOTAL_SEEDS
fi

LOG_FILE="eval_results_openvla_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"
TEMP_DIR=$(mktemp -d)

echo "Evaluating OpenVLA task: $TASK_NAME"
echo "Detected $NUM_GPUS GPUs ($GPU_NAMES)"
echo "Using $NUM_GPUS GPUs for parallel evaluation (~$((TOTAL_SEEDS / NUM_GPUS)) seeds/GPU)"
echo "Seeds: $START_SEED to $END_SEED ($TOTAL_SEEDS total)"
echo "Max steps per episode: $MAX_STEPS"
echo "Evaluating OpenVLA task: $TASK_NAME" >> "$LOG_FILE"
echo "Parallel evaluation with $NUM_GPUS GPUs" >> "$LOG_FILE"

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
        
        # Run from robofactory directory (two levels up from script)
        # PYTHONPATH needs to be three levels up to reach /workspace (parent of robofactory)
        PYTHONPATH="$(pwd)/../../..:$PYTHONPATH" CUDA_VISIBLE_DEVICES=$gpu_id OUTPUT=$(cd ../.. && python ./policy/OpenVLA/eval_multi_openvla.py \
            --config="$CONFIG_NAME" \
            --data_num=$DATA_NUM \
            --checkpoint_num=$CHECKPOINT_NUM \
            --debug=0 \
            --seed=$seed \
            --max_steps=$MAX_STEPS \
            --render_mode="rgb_array" \
            --obs_mode="rgb" \
            --record_dir="./eval_video/openvla/{env_id}" 2>&1)
        
        fine=0
        # Check for "SUCCESS" in the output (matching Python's print format)
        if [[ $OUTPUT == *"SUCCESS"* ]]; then
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
