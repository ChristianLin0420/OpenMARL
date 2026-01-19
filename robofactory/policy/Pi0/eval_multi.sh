#!/bin/bash

# Parallel evaluation for Pi0/Pi0.5 using all available GPUs
# Usage: ./eval_multi.sh <config_name> <policy_type> <data_num> <checkpoint_step> [debug_mode] [task_name] [max_steps] [num_eval]
#
# Examples:
#   ./eval_multi.sh LiftBarrier pi0 150 5000
#   ./eval_multi.sh TwoRobotsStackCube pi05 150 10000 0 TwoRobotsStackCube-rf 1000 50

# Change to the script's directory so relative paths work correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: $0 <config_name> <policy_type> <data_num> <checkpoint_step> [debug_mode] [task_name] [max_steps] [num_eval]"
    echo ""
    echo "Arguments:"
    echo "  config_name      : Task config name (e.g., LiftBarrier)"
    echo "  policy_type      : 'pi0' or 'pi05'"
    echo "  data_num         : Number of training samples (e.g., 150)"
    echo "  checkpoint_step  : Training step of checkpoint (e.g., 5000)"
    echo "  debug_mode       : Debug mode 0/1 (default: 0)"
    echo "  task_name        : Full task name with suffix (default: <config_name>-rf)"
    echo "  max_steps        : Maximum steps per episode (default: 250)"
    echo "  num_eval         : Number of evaluation episodes (default: 100)"
    echo ""
    echo "Examples:"
    echo "  $0 LiftBarrier pi0 150 5000"
    echo "  $0 TwoRobotsStackCube pi05 150 10000 0 TwoRobotsStackCube-rf 1000 50"
    exit 1
fi

CONFIG_NAME="$1"
POLICY_TYPE="$2"
DATA_NUM="$3"
CHECKPOINT_STEP="$4"
DEBUG_MODE="${5:-0}"
TASK_NAME="${6:-${CONFIG_NAME}-rf}"
MAX_STEPS="${7:-250}"
NUM_EVAL="${8:-100}"
START_SEED="${9:-1000}"

# Validate policy type
if [ "$POLICY_TYPE" != "pi0" ] && [ "$POLICY_TYPE" != "pi05" ]; then
    echo "Error: policy_type must be 'pi0' or 'pi05'"
    exit 1
fi

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

LOG_FILE="eval_results_${POLICY_TYPE}_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_STEP}_$(date +"%Y%m%d_%H%M%S").log"
TEMP_DIR=$(mktemp -d)

echo "Evaluating Pi0/Pi0.5 task: $TASK_NAME"
echo "Policy Type: $POLICY_TYPE"
echo "Detected $NUM_GPUS GPUs ($GPU_NAMES)"
echo "Using $NUM_GPUS GPUs for parallel evaluation (~$((TOTAL_SEEDS / NUM_GPUS)) seeds/GPU)"
echo "Seeds: $START_SEED to $END_SEED ($TOTAL_SEEDS total)"
echo "Evaluating $POLICY_TYPE task: $TASK_NAME" >> "$LOG_FILE"
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
        
        # Construct config path
        CONFIG_PATH="../../configs/table/${CONFIG_NAME}.yaml"
        
        PYTHONPATH="$(pwd)/../..:$(pwd)/../../..:$PYTHONPATH" CUDA_VISIBLE_DEVICES=$gpu_id OUTPUT=$(python eval_multi_pi0.py \
            --config="$CONFIG_PATH" \
            --policy_type="$POLICY_TYPE" \
            --data_num=$DATA_NUM \
            --checkpoint_step=$CHECKPOINT_STEP \
            --debug=0 \
            --seed=$seed \
            --max_steps=$MAX_STEPS \
            --render_mode="rgb_array" \
            --obs_mode="rgb" \
            --num_eval_episodes=1 2>&1)
        
        fine=0
        # Check for "Success: True" in the output (matching Python's print format)
        if [[ $OUTPUT == *"Success: True"* ]]; then
            fine=1
            ((success++))
        fi
        ((total++))
        echo "$seed,$fine" >> "$result_file"
        
        if [ "$DEBUG_MODE" -eq 1 ]; then
            echo "Seed $seed: $([ $fine -eq 1 ] && echo 'SUCCESS' || echo 'FAIL')"
            echo "Output: $OUTPUT"
            echo "---"
        fi
        
        # Show first failure's full output for debugging
        if [ $fine -eq 0 ] && [ $total -eq 1 ]; then
            echo "First episode failed. Output:"
            echo "$OUTPUT" | tail -n 50
            echo "---"
        fi
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
echo "Final Results ($POLICY_TYPE):"
echo "  Task: $TASK_NAME"
echo "  Total Episodes: $TOTAL"
echo "  Successes: $SUCCESS"
echo "  Success Rate: $SUCCESS_RATE%"
echo "========================================"
echo "Results saved in $LOG_FILE"

# Cleanup
rm -rf "$TEMP_DIR"

