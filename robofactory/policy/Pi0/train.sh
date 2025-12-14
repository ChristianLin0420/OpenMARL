#!/bin/bash

# Pi0/Pi0.5 Training Script
# Usage: ./train.sh <policy_type> <task_name> <agent_id> [num_gpus] [additional_args]
#
# Examples:
#   ./train.sh pi0 LiftBarrier-rf 0              # Single GPU, Pi0
#   ./train.sh pi0 LiftBarrier-rf 0 4            # 4 GPUs, Pi0
#   ./train.sh pi05 TwoRobotsStackCube-rf 1 2    # 2 GPUs, Pi0.5

set -e  # Exit on error

# Parse arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <policy_type> <task_name> <agent_id> [num_gpus] [additional_args]"
    echo ""
    echo "Arguments:"
    echo "  policy_type   : 'pi0' or 'pi05'"
    echo "  task_name     : Task name (e.g., LiftBarrier-rf)"
    echo "  agent_id      : Agent ID (0, 1, 2, ...)"
    echo "  num_gpus      : Number of GPUs for training (default: 1)"
    echo "  additional_args: Additional Hydra arguments (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 pi0 LiftBarrier-rf 0"
    echo "  $0 pi0 LiftBarrier-rf 0 4"
    echo "  $0 pi05 TwoRobotsStackCube-rf 1 2 training.num_epochs=500"
    exit 1
fi

POLICY_TYPE="$1"
TASK_NAME="$2"
AGENT_ID="$3"
NUM_GPUS="${4:-1}"
shift 4 || shift $#  # Remove first 4 args (or all if less than 4)
ADDITIONAL_ARGS="$@"

# Validate policy type
if [ "$POLICY_TYPE" != "pi0" ] && [ "$POLICY_TYPE" != "pi05" ]; then
    echo "Error: policy_type must be 'pi0' or 'pi05'"
    exit 1
fi

# Determine config name
if [ "$POLICY_TYPE" == "pi0" ]; then
    CONFIG_NAME="robot_pi0"
else
    CONFIG_NAME="robot_pi05"
fi

echo "========================================"
echo "Pi0/Pi0.5 Training"
echo "========================================"
echo "Policy Type:      $POLICY_TYPE"
echo "Task Name:        $TASK_NAME"
echo "Agent ID:         $AGENT_ID"
echo "Number of GPUs:   $NUM_GPUS"
echo "Config:           $CONFIG_NAME"
echo "Additional Args:  $ADDITIONAL_ARGS"
echo "========================================"
echo ""

# Set environment variables
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export DISPLAY=""

# Change to Pi0 policy directory
cd "$(dirname "$0")"
POLICY_DIR=$(pwd)

# Set PYTHONPATH
export PYTHONPATH="$POLICY_DIR:$POLICY_DIR/../..:$PYTHONPATH"

# Determine training command
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Starting single GPU training..."
    python train.py \
        --config-name "$CONFIG_NAME" \
        task.name="$TASK_NAME" \
        agent_id=$AGENT_ID \
        $ADDITIONAL_ARGS
else
    # Multi-GPU training with torchrun
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    
    # Set visible GPUs (0,1,2,...,NUM_GPUS-1)
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train.py \
        --config-name "$CONFIG_NAME" \
        task.name="$TASK_NAME" \
        agent_id=$AGENT_ID \
        $ADDITIONAL_ARGS
fi

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"

