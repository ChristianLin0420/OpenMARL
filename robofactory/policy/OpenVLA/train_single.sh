#!/bin/bash
# Single GPU training script for OpenVLA
#
# Usage:
#   bash train_single.sh <task_name> <load_num> <agent_id> <seed>
#
# Example:
#   bash train_single.sh LiftBarrier-rf 150 0 100

task_name=${1:-"LiftBarrier-rf"}
load_num=${2:-150}
agent_id=${3:-0}
seed=${4:-100}

echo "Starting single GPU training..."
bash policy/OpenVLA/train.sh ${task_name} ${load_num} ${agent_id} ${seed} 0

