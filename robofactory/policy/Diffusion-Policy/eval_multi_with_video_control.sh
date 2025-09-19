#!/bin/bash

# Enhanced evaluation script with video recording control
# Usage: $0 <config_name> <data_num> <checkpoint_num> <debug_mode> <task_name> [video_interval]

if [ -z "$1" ]; then
    echo "Usage: $0 <config_name> <data_num> <checkpoint_num> <debug_mode> <task_name> [video_interval]"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <config_name> <data_num> <checkpoint_num> <debug_mode> <task_name> [video_interval]"
    exit 1
fi
if [ -z "$3" ]; then
    echo "Usage: $0 <config_name> <data_num> <checkpoint_num> <debug_mode> <task_name> [video_interval]"
    exit 1
fi
if [ -z "$4" ]; then
    echo "Usage: $0 <config_name> <data_num> <checkpoint_num> <debug_mode> <task_name> [video_interval]"
    exit 1
fi
if [ -z "$5" ]; then
    echo "Usage: $0 <config_name> <data_num> <checkpoint_num> <debug_mode> <task_name> [video_interval]"
    exit 1
fi

CONFIG_NAME="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
DEBUG_MODE="$4"
TASK_NAME="$5"
VIDEO_INTERVAL="${6:-1}"  # Default to 1 (record all videos)

# Generate log file with timestamp
LOG_FILE="eval_results_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"

echo "Evaluating task: $TASK_NAME"
echo "Video recording interval: $VIDEO_INTERVAL"
echo "Evaluating task: $TASK_NAME" >> "$LOG_FILE"
echo "Video recording interval: $VIDEO_INTERVAL" >> "$LOG_FILE"
TOTAL=0
SUCCESS=0
SUCCESS_RATE=0

# Set quiet flag based on debug mode
QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

for SEED in {1000..1099}
do
    echo "Running evaluation with seed $SEED for task $CONFIG_NAME..."
    
    # Determine if we should record video for this seed
    RECORD_VIDEO=""
    if [ "$VIDEO_INTERVAL" -gt 0 ]; then
        # Calculate if this seed should have video recording
        SEED_INDEX=$((SEED - 1000))
        if [ $((SEED_INDEX % VIDEO_INTERVAL)) -eq 0 ]; then
            RECORD_VIDEO="--record-dir ./eval_video/{env_id}"
            echo "  → Recording video for seed $SEED"
        else
            RECORD_VIDEO="--record-dir /tmp/no_video"
            echo "  → Skipping video for seed $SEED"
        fi
    else
        RECORD_VIDEO="--record-dir /tmp/no_video"
        echo "  → Video recording disabled"
    fi
    
    OUTPUT=""
    if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
        OUTPUT=$(python ./policy/Diffusion-Policy/eval_multi_dp.py \
                --config="$CONFIG_NAME" \
                --data-num=$DATA_NUM \
                --checkpoint-num=$CHECKPOINT_NUM \
                --render-mode="sensors" \
                -o="rgb" \
                -b="cpu" \
                -n 1 \
                -s $SEED \
                $RECORD_VIDEO \
                $QUIET_FLAG)
    else
        OUTPUT=$(python ./policy/Diffusion-Policy/eval_multi_dp.py \
                --config="$CONFIG_NAME" \
                --data-num=$DATA_NUM \
                --checkpoint-num=$CHECKPOINT_NUM \
                --render-mode="sensors" \
                -o="rgb" \
                -b="cpu" \
                -n 1 \
                -s $SEED \
                $RECORD_VIDEO \
                $QUIET_FLAG)
    fi
    echo "$OUTPUT"
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)
    FINE=0
    # Check if output contains "success"
    if [[ $LAST_LINE == *"success"* ]]; then
        FINE=1
        SUCCESS=$((SUCCESS + 1))
    fi
    TOTAL=$((TOTAL + 1))
    SUCCESS_RATE=$(echo "scale=4; $SUCCESS / $TOTAL * 100" | bc)
    echo "$SEED, $FINE, $SUCCESS_RATE%" >> "$LOG_FILE"
    echo "Seed $SEED done. Success Rate: $SUCCESS_RATE%"
done
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"

echo "Evaluation completed. Results saved in $LOG_FILE."
