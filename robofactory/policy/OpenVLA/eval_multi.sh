#!/bin/bash
# Multi-agent evaluation script for OpenVLA
#
# Usage:
#   bash eval_multi.sh <config> <data_num> <checkpoint_num> <debug> <task_name>
#
# Example:
#   bash eval_multi.sh configs/table/lift_barrier.yaml 150 300 1 LiftBarrier-rf

config_name=${1}
DATA_NUM=${2}
CHECKPOINT_NUM=${3}
DEBUG_MODE=${4}
TASK_NAME=${5}

echo -e "\033[33m===========================================\033[0m"
echo -e "\033[33mEvaluating OpenVLA Multi-Agent Policies\033[0m"
echo -e "\033[33m===========================================\033[0m"
echo -e "\033[33mTask: ${TASK_NAME}\033[0m"
echo -e "\033[33mConfig: ${config_name}\033[0m"
echo -e "\033[33mData samples: ${DATA_NUM}\033[0m"
echo -e "\033[33mCheckpoint: ${CHECKPOINT_NUM}\033[0m"
echo -e "\033[33mDebug mode: ${DEBUG_MODE}\033[0m"
echo -e "\033[33m===========================================\033[0m"

python policy/OpenVLA/eval_multi_openvla.py \
    --config ${config_name} \
    --data_num ${DATA_NUM} \
    --checkpoint_num ${CHECKPOINT_NUM} \
    --debug ${DEBUG_MODE} \
    --seed 10000 \
    --max_steps 250

