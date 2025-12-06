#!/bin/bash
#==============================================================================
# OpenMARL Unified Evaluation Script
#==============================================================================
# Supports multiple policies with consistent evaluation interface
#
# Usage:
#   bash eval.sh --policy <policy> --task <task> --config <config> [options]
#   bash eval.sh --policy <policy> --all_tasks [options]
#
# Examples:
#   # Evaluate single task
#   bash eval.sh --policy dp --task LiftBarrier-rf --config configs/table/lift_barrier.yaml
#
#   # Evaluate all trained tasks
#   bash eval.sh --policy dp --all_tasks
#==============================================================================

set -e

# Default values
POLICY="dp"
TASK=""
CONFIG=""
DATA_NUM=150
CHECKPOINT_NUM=300
DEBUG_MODE=0
SEED_START=1000
NUM_EVAL=100
MAX_STEPS=250
ALL_TASKS=false
RESULTS_DIR="evaluation_results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Help function
show_help() {
    echo "OpenMARL Unified Evaluation Script"
    echo ""
    echo "Usage:"
    echo "  bash eval.sh --policy <policy> --task <task> --config <config> [options]"
    echo "  bash eval.sh --policy <policy> --all_tasks [options]"
    echo ""
    echo "Required Arguments (for single task):"
    echo "  --policy        Policy type: 'dp' (Diffusion Policy) or 'openvla'"
    echo "  --task          Task name (e.g., LiftBarrier-rf)"
    echo "  --config        Config file path (e.g., configs/table/lift_barrier.yaml)"
    echo ""
    echo "Batch Evaluation:"
    echo "  --all_tasks     Evaluate all trained tasks automatically"
    echo ""
    echo "Optional Arguments:"
    echo "  --data_num      Number of data episodes used for training (default: 150)"
    echo "  --checkpoint    Checkpoint epoch to evaluate (default: 300)"
    echo "  --debug         Debug mode: 0=quiet, 1=verbose (default: 0)"
    echo "  --num_eval      Number of evaluation episodes (default: 100)"
    echo "  --seed          Starting seed for evaluation (default: 1000)"
    echo "  --max_steps     Maximum steps per episode (default: 250)"
    echo "  --results_dir   Results directory (default: evaluation_results)"
    echo "  --help          Show this help message"
    echo ""
    echo "Supported Policies:"
    echo "  dp       - Diffusion Policy (CNN-based)"
    echo "  openvla  - OpenVLA (Vision-Language-Action)"
    echo ""
    echo "Examples:"
    echo "  # Evaluate single task with Diffusion Policy"
    echo "  bash eval.sh --policy dp --task LiftBarrier-rf --config configs/table/lift_barrier.yaml"
    echo ""
    echo "  # Evaluate single task with OpenVLA"
    echo "  bash eval.sh --policy openvla --task LiftBarrier-rf --config configs/table/lift_barrier.yaml"
    echo ""
    echo "  # Evaluate all trained Diffusion Policy models"
    echo "  bash eval.sh --policy dp --all_tasks"
    echo ""
    echo "  # Evaluate all trained OpenVLA models"
    echo "  bash eval.sh --policy openvla --all_tasks"
    echo ""
    echo "  # Quick debug evaluation"
    echo "  bash eval.sh --policy dp --task LiftBarrier-rf --config configs/table/lift_barrier.yaml --debug 1 --num_eval 10"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)      POLICY="$2";         shift 2 ;;
        --task)        TASK="$2";           shift 2 ;;
        --config)      CONFIG="$2";         shift 2 ;;
        --data_num)    DATA_NUM="$2";       shift 2 ;;
        --checkpoint)  CHECKPOINT_NUM="$2"; shift 2 ;;
        --debug)       DEBUG_MODE="$2";     shift 2 ;;
        --num_eval)    NUM_EVAL="$2";       shift 2 ;;
        --seed)        SEED_START="$2";     shift 2 ;;
        --max_steps)   MAX_STEPS="$2";      shift 2 ;;
        --results_dir) RESULTS_DIR="$2";    shift 2 ;;
        --all_tasks)   ALL_TASKS=true;      shift ;;
        --help|-h)     show_help ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate policy
if [[ "$POLICY" != "dp" && "$POLICY" != "openvla" ]]; then
    echo -e "${RED}Error: Invalid policy '$POLICY'. Use 'dp' or 'openvla'${NC}"
    exit 1
fi

# Validate arguments based on mode
if [[ "$ALL_TASKS" == false ]]; then
    if [[ -z "$TASK" ]]; then
        echo -e "${RED}Error: --task is required (or use --all_tasks)${NC}"
        show_help
    fi
    if [[ -z "$CONFIG" ]]; then
        echo -e "${RED}Error: --config is required (or use --all_tasks)${NC}"
        show_help
    fi
fi

# Get script directory and change to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log file for batch evaluation
LOG_FILE="${RESULTS_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    if [[ "$ALL_TASKS" == true ]]; then
        echo -e "$1" | tee -a "$LOG_FILE"
    else
        echo -e "$1"
    fi
}

#==============================================================================
# Helper functions for batch evaluation
#==============================================================================

# Convert task name to config file path
get_config_file() {
    local task_name="$1"
    
    # Convert PascalCase-rf to snake_case
    local base_name=$(echo "$task_name" | sed 's/-rf$//' | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
    
    # Try table config first
    local table_config="robofactory/configs/table/${base_name}.yaml"
    if [[ -f "$table_config" ]]; then
        echo "$table_config"
        return 0
    fi
    
    # Try robocasa config
    local robocasa_config="robofactory/configs/robocasa/${base_name}.yaml"
    if [[ -f "$robocasa_config" ]]; then
        echo "$robocasa_config"
        return 0
    fi
    
    echo ""
    return 1
}

# Check if task has trained checkpoints
is_task_trained() {
    local task_name="$1"
    local checkpoint_dir="robofactory/checkpoints/${task_name}_Agent0_${DATA_NUM}"
    
    if [[ -d "$checkpoint_dir" ]]; then
        # Check if checkpoint file exists
        if [[ -f "${checkpoint_dir}/${CHECKPOINT_NUM}.ckpt" ]] || [[ -d "${checkpoint_dir}/epoch_${CHECKPOINT_NUM}" ]]; then
            return 0
        fi
    fi
    return 1
}

# Get all trained tasks
get_trained_tasks() {
    local tasks=()
    
    for checkpoint_dir in robofactory/checkpoints/*_Agent0_${DATA_NUM}; do
        if [[ -d "$checkpoint_dir" ]]; then
            local task_name=$(basename "$checkpoint_dir" | sed "s/_Agent0_${DATA_NUM}$//")
            if is_task_trained "$task_name"; then
                tasks+=("$task_name")
            fi
        fi
    done
    
    echo "${tasks[@]}"
}

# Get number of agents for a task
get_agent_count() {
    local task_name="$1"
    local count=0
    
    for agent_dir in robofactory/checkpoints/${task_name}_Agent*_${DATA_NUM}; do
        if [[ -d "$agent_dir" ]]; then
            count=$((count + 1))
        fi
    done
    
    echo "$count"
}

#==============================================================================
# Policy-specific evaluation functions
#==============================================================================

eval_diffusion_policy() {
    local task="$1"
    local config="$2"
    
    log "${BLUE}Evaluating Diffusion Policy${NC}"
    log "  Task: ${task}"
    log "  Config: ${config}"
    log "  Checkpoint: ${CHECKPOINT_NUM}"
    log "  Episodes: ${NUM_EVAL}"
    
    cd robofactory
    
    # Check if checkpoint exists
    local checkpoint_dir="checkpoints/${task}_Agent0_${DATA_NUM}"
    if [[ ! -d "$checkpoint_dir" ]]; then
        log "${RED}Error: Checkpoint directory not found: ${checkpoint_dir}${NC}"
        log "${YELLOW}Train the policy first: bash train.sh --policy dp --task ${task} --agent_id 0${NC}"
        cd ..
        return 1
    fi
    
    # Use config path relative to robofactory
    local relative_config="${config#robofactory/}"
    
    bash policy/Diffusion-Policy/eval_multi.sh \
        "$relative_config" \
        "$DATA_NUM" \
        "$CHECKPOINT_NUM" \
        "$DEBUG_MODE" \
        "$task"
    
    cd ..
    return 0
}

eval_openvla() {
    local task="$1"
    local config="$2"
    
    log "${BLUE}Evaluating OpenVLA${NC}"
    log "  Task: ${task}"
    log "  Config: ${config}"
    log "  Checkpoint: ${CHECKPOINT_NUM}"
    log "  Max Steps: ${MAX_STEPS}"
    
    cd robofactory
    
    # Check if checkpoint exists
    local checkpoint_dir="checkpoints/${task}_Agent0_${DATA_NUM}"
    if [[ ! -d "$checkpoint_dir" ]]; then
        log "${YELLOW}Warning: Checkpoint directory not found: ${checkpoint_dir}${NC}"
        log "${YELLOW}Make sure the policy is trained before evaluation${NC}"
        cd ..
        return 1
    fi
    
    # Use config path relative to robofactory
    local relative_config="${config#robofactory/}"
    
    python policy/OpenVLA/eval_multi_openvla.py \
        --config "$relative_config" \
        --data_num "$DATA_NUM" \
        --checkpoint_num "$CHECKPOINT_NUM" \
        --debug "$DEBUG_MODE" \
        --seed "$SEED_START" \
        --max_steps "$MAX_STEPS"
    
    cd ..
    return 0
}

# Dispatcher function
evaluate_task() {
    local task="$1"
    local config="$2"
    
    case "$POLICY" in
        dp)      eval_diffusion_policy "$task" "$config" ;;
        openvla) eval_openvla "$task" "$config" ;;
    esac
}

#==============================================================================
# Batch evaluation of all tasks
#==============================================================================

evaluate_all_tasks() {
    log "${BLUE}================================================${NC}"
    log "${BLUE}OpenMARL Batch Evaluation${NC}"
    log "${BLUE}================================================${NC}"
    log "Policy: ${POLICY}"
    log "Checkpoint: ${CHECKPOINT_NUM}"
    log "Data episodes: ${DATA_NUM}"
    log "Results directory: ${RESULTS_DIR}"
    log "${BLUE}================================================${NC}"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Get all trained tasks
    local trained_tasks=($(get_trained_tasks))
    
    if [[ ${#trained_tasks[@]} -eq 0 ]]; then
        log "${RED}No trained tasks found!${NC}"
        log "${YELLOW}Train some tasks first: bash train.sh --policy ${POLICY} --task <task> --agent_id 0${NC}"
        exit 1
    fi
    
    log ""
    log "${GREEN}Found ${#trained_tasks[@]} trained tasks:${NC}"
    for task in "${trained_tasks[@]}"; do
        local agent_count=$(get_agent_count "$task")
        log "  - $task (${agent_count} agents)"
    done
    log ""
    
    local total_tasks=${#trained_tasks[@]}
    local completed_tasks=0
    local failed_tasks=0
    local task_results=()
    
    # Evaluate each task
    for task_name in "${trained_tasks[@]}"; do
        log "${BLUE}----------------------------------------${NC}"
        log "${BLUE}Evaluating: $task_name${NC}"
        log "${BLUE}----------------------------------------${NC}"
        
        # Get config file
        local config_file=$(get_config_file "$task_name")
        if [[ -z "$config_file" ]]; then
            log "${RED}✗ Config file not found for $task_name, skipping...${NC}"
            failed_tasks=$((failed_tasks + 1))
            task_results+=("$task_name: SKIPPED (no config)")
            continue
        fi
        
        log "Config: $config_file"
        
        # Create task results directory
        mkdir -p "$RESULTS_DIR/$task_name"
        
        # Evaluate
        if evaluate_task "$task_name" "$config_file"; then
            completed_tasks=$((completed_tasks + 1))
            task_results+=("$task_name: SUCCESS")
            log "${GREEN}✓ Completed: $task_name${NC}"
        else
            failed_tasks=$((failed_tasks + 1))
            task_results+=("$task_name: FAILED")
            log "${RED}✗ Failed: $task_name${NC}"
        fi
        
        log ""
    done
    
    # Summary
    log "${BLUE}================================================${NC}"
    log "${BLUE}Evaluation Summary${NC}"
    log "${BLUE}================================================${NC}"
    log "Total tasks: $total_tasks"
    log "${GREEN}Completed: $completed_tasks${NC}"
    log "${RED}Failed: $failed_tasks${NC}"
    log ""
    log "${BLUE}Results:${NC}"
    for result in "${task_results[@]}"; do
        if [[ "$result" == *"SUCCESS"* ]]; then
            log "${GREEN}  ✓ $result${NC}"
        else
            log "${RED}  ✗ $result${NC}"
        fi
    done
    log ""
    log "${BLUE}Results saved in: ${RESULTS_DIR}/${NC}"
    log "${BLUE}Log file: ${LOG_FILE}${NC}"
    
    if [[ $failed_tasks -eq 0 ]]; then
        log "${GREEN}🎉 All evaluations completed successfully!${NC}"
        return 0
    else
        log "${YELLOW}⚠️  Some evaluations failed. Check the log for details.${NC}"
        return 1
    fi
}

#==============================================================================
# Main execution
#==============================================================================

if [[ "$ALL_TASKS" == true ]]; then
    # Batch evaluation mode
    evaluate_all_tasks
else
    # Single task evaluation mode
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}OpenMARL Evaluation${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Policy: ${POLICY}"
    echo -e "Task: ${TASK}"
    echo -e "Config: ${CONFIG}"
    echo -e "Checkpoint: ${CHECKPOINT_NUM}"
    echo -e "Debug Mode: ${DEBUG_MODE}"
    echo -e "${BLUE}================================================${NC}"
    
    evaluate_task "$TASK" "$CONFIG"
    
    echo -e "${GREEN}Evaluation complete!${NC}"
fi
