#!/bin/bash
# Training script for OpenVLA on RoboFactory tasks
#
# Usage:
#   bash train.sh <task_name> <load_num> <agent_id> <seed> <gpu_id>
#
# Example:
#   bash train.sh LiftBarrier-rf 150 0 100 0

task_name=${1}
load_num=${2}
agent_id=${3}
seed=${4}
gpu_id=${5}

DEBUG=False

alg_name=robot_openvla
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-openvla-${addition_info}

echo -e "\033[33m===========================================\033[0m"
echo -e "\033[33mTraining OpenVLA Policy\033[0m"
echo -e "\033[33m===========================================\033[0m"
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mAgent ID: ${agent_id}\033[0m"
echo -e "\033[33mData samples: ${load_num}\033[0m"
echo -e "\033[33mSeed: ${seed}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33m===========================================\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Check if RLDS data exists, if not suggest conversion
rlds_path="robofactory/data/rlds_data/${task_name}_Agent${agent_id}"
if [ ! -d "$rlds_path" ]; then
    echo -e "\033[31mError: RLDS dataset not found at ${rlds_path}\033[0m"
    echo -e "\033[33mPlease convert ZARR data to RLDS format first:\033[0m"
    echo -e "\033[33m  python policy/OpenVLA/openvla_policy/utils/data_conversion.py \\\033[0m"
    echo -e "\033[33m    --zarr_path data/zarr_data \\\033[0m"
    echo -e "\033[33m    --output_dir data/rlds_data \\\033[0m"
    echo -e "\033[33m    --batch\033[0m"
    exit 1
fi

python ./robofactory/policy/OpenVLA/train.py --config-name=${config_name}.yaml \
                                 task.name=${task_name} \
                                 task.dataset.rlds_path="${rlds_path}" \
                                 training.debug=$DEBUG \
                                 training.seed=${seed} \
                                 training.device="cuda:0" \
                                 exp_name=${exp_name} \
                                 logging.mode=${wandb_mode}

