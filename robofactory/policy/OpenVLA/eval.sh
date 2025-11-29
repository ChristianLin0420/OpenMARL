#!/bin/bash
# Simple evaluation script for OpenVLA
#
# Usage:
#   bash eval.sh <task_name> <agent_id> <data_num> <checkpoint_num>
#
# Example:
#   bash eval.sh LiftBarrier-rf 0 150 300

TASK_NAME=${1}
AGENT_ID=${2}
DATA_NUM=${3}
CHECKPOINT_NUM=${4}

# Determine config file
if [[ "$TASK_NAME" == *"robocasa"* ]]; then
    config_path="configs/robocasa/$(echo $TASK_NAME | sed 's/-rf//' | tr '[:upper:]' '[:lower:]').yaml"
else
    config_path="configs/table/$(echo $TASK_NAME | sed 's/-rf//' | tr '[:upper:]' '[:lower:]').yaml"
fi

echo "Using config: ${config_path}"

python policy/OpenVLA/eval_multi_openvla.py \
    --config ${config_path} \
    --data_num ${DATA_NUM} \
    --checkpoint_num ${CHECKPOINT_NUM} \
    --debug 1 \
    --seed 10000

