#!/bin/bash
#==============================================================================
# OpenMARL Unified Training Script
#==============================================================================
# Supports multiple policies with single/multi-GPU training
#
# Usage:
#   bash train.sh --policy <policy> --task <task> --agent_id <id> [options]
#
# Examples:
#   # Train Diffusion Policy (single GPU)
#   bash train.sh --policy dp --task LiftBarrier-rf --agent_id 0 --seed 100
#
#   # Train OpenVLA (8 GPUs)
#   bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0 --gpus 8
#
#   # Train all agents for a task
#   bash train.sh --policy dp --task LiftBarrier-rf --all_agents
#==============================================================================

set -e

# Default values
POLICY="dp"
TASK=""
AGENT_ID=""
SEED=100
GPUS=1
BATCH_SIZE=""
DEBUG=false
ALL_AGENTS=false
WANDB_MODE="online"
DATA_NUM=160

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Help function
show_help() {
    echo "OpenMARL Unified Training Script"
    echo ""
    echo "Usage: bash train.sh --policy <policy> --task <task> --agent_id <id> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  --policy        Policy type: 'dp' (Diffusion Policy) or 'openvla'"
    echo "  --task          Task name (e.g., LiftBarrier-rf, CameraAlignment-rf)"
    echo "  --agent_id      Agent ID (0, 1, 2, ...) or use --all_agents"
    echo ""
    echo "Optional Arguments:"
    echo "  --seed          Random seed (default: 100)"
    echo "  --gpus          Number of GPUs (default: 1, use >1 for multi-GPU)"
    echo "  --batch_size    Batch size (default: policy-specific)"
    echo "  --data_num      Number of data episodes (default: 150)"
    echo "  --debug         Enable debug mode (default: false)"
    echo "  --wandb         Wandb mode: 'online', 'offline', 'disabled' (default: online)"
    echo "  --all_agents    Train all agents sequentially"
    echo "  --help          Show this help message"
    echo ""
    echo "Supported Policies:"
    echo "  dp       - Diffusion Policy (CNN-based, ZARR data)"
    echo "  openvla  - OpenVLA (Vision-Language-Action, RLDS data)"
    echo ""
    echo "Examples:"
    echo "  # Single GPU Diffusion Policy"
    echo "  bash train.sh --policy dp --task LiftBarrier-rf --agent_id 0"
    echo ""
    echo "  # Multi-GPU OpenVLA"
    echo "  bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0 --gpus 8"
    echo ""
    echo "  # Train all agents"
    echo "  bash train.sh --policy dp --task LiftBarrier-rf --all_agents"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)     POLICY="$2";     shift 2 ;;
        --task)       TASK="$2";       shift 2 ;;
        --agent_id)   AGENT_ID="$2";   shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --gpus)       GPUS="$2";       shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --data_num)   DATA_NUM="$2";   shift 2 ;;
        --debug)      DEBUG=true;      shift ;;
        --wandb)      WANDB_MODE="$2"; shift 2 ;;
        --all_agents) ALL_AGENTS=true; shift ;;
        --help|-h)    show_help ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TASK" ]]; then
    echo -e "${RED}Error: --task is required${NC}"
    show_help
fi

if [[ "$ALL_AGENTS" == false && -z "$AGENT_ID" ]]; then
    echo -e "${RED}Error: --agent_id is required (or use --all_agents)${NC}"
    show_help
fi

# Validate policy
if [[ "$POLICY" != "dp" && "$POLICY" != "openvla" ]]; then
    echo -e "${RED}Error: Invalid policy '$POLICY'. Use 'dp' or 'openvla'${NC}"
    exit 1
fi

# Get script directory and change to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set debug mode
if [[ "$DEBUG" == true ]]; then
    WANDB_MODE="offline"
fi

export HYDRA_FULL_ERROR=1

#==============================================================================
# Policy-specific training functions
#==============================================================================

train_diffusion_policy() {
    local agent_id="$1"
    
    echo -e "${BLUE}Training Diffusion Policy${NC}"
    echo -e "  Task: ${TASK}, Agent: ${agent_id}, Seed: ${SEED}, GPUs: ${GPUS}"
    
    local zarr_path="robofactory/data/zarr_data/${TASK}_Agent${agent_id}_${DATA_NUM}.zarr"
    
    if [[ ! -d "$zarr_path" ]]; then
        echo -e "${RED}Error: ZARR dataset not found at ${zarr_path}${NC}"
        echo -e "${YELLOW}Run data preparation first: cd robofactory && bash prepare_all_data.sh${NC}"
        return 1
    fi
    
    local exp_name="diffusion_policy"
    local batch_size=${BATCH_SIZE:-64}
    
    cd robofactory
    
    if [[ "$GPUS" -gt 1 ]]; then
        # Multi-GPU training with torchrun (DDP)
        echo -e "${GREEN}Using multi-GPU training with ${GPUS} GPUs (DistributedDataParallel)${NC}"
        torchrun --standalone --nnodes=1 --nproc-per-node=${GPUS} \
            policy/Diffusion-Policy/train.py \
            --config-name=robot_dp.yaml \
            task.name=${TASK} \
            task.dataset.zarr_path="data/zarr_data/${TASK}_Agent${agent_id}_${DATA_NUM}.zarr" \
            training.debug=${DEBUG} \
            training.seed=${SEED} \
            training.device="cuda:0" \
            dataloader.batch_size=${batch_size} \
            agent_id=${agent_id} \
            exp_name=${exp_name} \
            logging.mode=${WANDB_MODE}
    else
        # Single GPU training
        export CUDA_VISIBLE_DEVICES=0
        python policy/Diffusion-Policy/train.py \
            --config-name=robot_dp.yaml \
            task.name=${TASK} \
            task.dataset.zarr_path="data/zarr_data/${TASK}_Agent${agent_id}_${DATA_NUM}.zarr" \
            training.debug=${DEBUG} \
            training.seed=${SEED} \
            training.device="cuda:0" \
            dataloader.batch_size=${batch_size} \
            agent_id=${agent_id} \
            exp_name=${exp_name} \
            logging.mode=${WANDB_MODE}
    fi
    
    cd ..
}

train_openvla() {
    local agent_id="$1"
    
    echo -e "${BLUE}Training OpenVLA${NC}"
    echo -e "  Task: ${TASK}, Agent: ${agent_id}, Seed: ${SEED}, GPUs: ${GPUS}"
    
    local rlds_path="robofactory/data/rlds_data/${TASK}_Agent${agent_id}"
    
    if [[ ! -d "$rlds_path" ]]; then
        echo -e "${RED}Error: RLDS dataset not found at ${rlds_path}${NC}"
        echo -e "${YELLOW}Convert ZARR to RLDS first:${NC}"
        echo -e "${YELLOW}  python robofactory/policy/OpenVLA/openvla_policy/utils/data_conversion.py \\${NC}"
        echo -e "${YELLOW}    --zarr_path robofactory/data/zarr_data --output_dir robofactory/data/rlds_data --batch${NC}"
        return 1
    fi
    
    local batch_size=${BATCH_SIZE:-32}
    
    if [[ "$GPUS" -gt 1 ]]; then
        # Multi-GPU training with torchrun
        echo -e "${GREEN}Using multi-GPU training with ${GPUS} GPUs${NC}"
        torchrun --standalone --nnodes=1 --nproc-per-node=${GPUS} \
            robofactory/policy/OpenVLA/train.py \
            --config-name=robot_openvla.yaml \
            task.name=${TASK} \
            task.dataset.rlds_path=${rlds_path} \
            agent_id=${agent_id} \
            training.seed=${SEED} \
            training.debug=${DEBUG} \
            dataloader.batch_size=${batch_size} \
            logging.mode=${WANDB_MODE}
    else
        # Single GPU training
        export CUDA_VISIBLE_DEVICES=0
        python robofactory/policy/OpenVLA/train.py \
            --config-name=robot_openvla.yaml \
            task.name=${TASK} \
            task.dataset.rlds_path=${rlds_path} \
            agent_id=${agent_id} \
            training.seed=${SEED} \
            training.debug=${DEBUG} \
            dataloader.batch_size=${batch_size} \
            logging.mode=${WANDB_MODE}
    fi
}

#==============================================================================
# Get number of agents for a task
#==============================================================================
get_agent_count() {
    local task_name="$1"
    local base_name=$(echo "$task_name" | sed 's/-rf$//' | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
    
    for config_dir in "robofactory/configs/table" "robofactory/configs/robocasa"; do
        local config_file="${config_dir}/${base_name}.yaml"
        if [[ -f "$config_file" ]]; then
            local count=$(grep -c "robot_uid:" "$config_file" 2>/dev/null || echo "0")
            echo "$count"
            return 0
        fi
    done
    echo "0"
}

#==============================================================================
# Main execution
#==============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}OpenMARL Training${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Policy: ${POLICY}"
echo -e "Task: ${TASK}"
echo -e "Seed: ${SEED}"
echo -e "GPUs: ${GPUS}"
echo -e "Wandb: ${WANDB_MODE}"
echo -e "${BLUE}================================================${NC}"

# Train function dispatcher
train_agent() {
    local agent_id="$1"
    case "$POLICY" in
        dp)      train_diffusion_policy "$agent_id" ;;
        openvla) train_openvla "$agent_id" ;;
    esac
}

if [[ "$ALL_AGENTS" == true ]]; then
    # Train all agents
    agent_count=$(get_agent_count "$TASK")
    if [[ "$agent_count" -eq 0 ]]; then
        echo -e "${RED}Error: Could not determine agent count for task ${TASK}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Training all ${agent_count} agents for ${TASK}${NC}"
    
    for ((i=0; i<agent_count; i++)); do
        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "${BLUE}Training Agent ${i}/${agent_count}${NC}"
        echo -e "${BLUE}----------------------------------------${NC}"
        train_agent "$i"
    done
    
    echo -e "${GREEN}All agents trained successfully!${NC}"
else
    # Train single agent
    train_agent "$AGENT_ID"
fi

echo -e "${GREEN}Training complete!${NC}"
