#!/bin/bash

# Comprehensive Training Script for RoboFactory
# This script trains all available tasks with prepared datasets

set -e  # Exit on any error

# Configuration
NUM_EPISODES=150
EPOCHS=300
SEED=100
GPU_ID=0
LOG_FILE="training_all_tasks_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Check if task is already trained
is_task_trained() {
    local task_name="$1"
    local agent_id="$2"
    
    # Check if checkpoint exists for the specific agent
    if [ -f "checkpoints/${task_name}_Agent${agent_id}_${NUM_EPISODES}/${EPOCHS}.ckpt" ]; then
        return 0  # Agent is trained
    else
        return 1  # Agent is not trained
    fi
}

# Check if dataset exists for training
is_dataset_available() {
    local task_name="$1"
    
    # Get the number of agents for this task
    local agent_count=$(get_agent_count "$task_name")
    
    if [ "$agent_count" -eq 0 ]; then
        return 1  # No agents found
    fi
    
    # Check if zarr data exists for all agents
    for ((i=0; i<agent_count; i++)); do
        local zarr_path="data/zarr_data/${task_name}_Agent${i}_${NUM_EPISODES}.zarr"
        if [ ! -d "$zarr_path" ]; then
            return 1  # Missing dataset for agent i
        fi
    done
    
    return 0  # All datasets are available
}

# Get the number of agents for a task from config file
get_agent_count() {
    local task_name="$1"
    
    # Convert task name to config file name
    local base_name=$(echo "$task_name" | sed 's/-rf$//' | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
    
    # Try table config first
    local table_config="configs/table/${base_name}.yaml"
    if [ -f "$table_config" ]; then
        local agent_count=$(grep -c "robot_uid:" "$table_config" 2>/dev/null || echo "0")
        echo "$agent_count"
        return 0
    fi
    
    # Try robocasa config
    local robocasa_config="configs/robocasa/${base_name}.yaml"
    if [ -f "$robocasa_config" ]; then
        local agent_count=$(grep -c "robot_uid:" "$robocasa_config" 2>/dev/null || echo "0")
        echo "$agent_count"
        return 0
    fi
    
    # If not found, return 0
    echo "0"
    return 1
}

# Train a single task
train_task() {
    local task_name="$1"
    
    log "${BLUE}Training task: $task_name${NC}"
    
    # Get the number of agents for this task
    local agent_count=$(get_agent_count "$task_name")
    
    if [ "$agent_count" -eq 0 ]; then
        log "${RED}✗ No agents found for $task_name, skipping...${NC}"
        return 1
    fi
    
    log "  Found $agent_count agents for this task"
    
    # Check if already trained (check all agents)
    local all_trained=true
    for ((i=0; i<agent_count; i++)); do
        if ! is_task_trained "$task_name" "$i"; then
            all_trained=false
            break
        fi
    done
    
    if [ "$all_trained" = true ]; then
        log "${GREEN}✓ Task $task_name is already trained for all agents, skipping...${NC}"
        return 0
    fi
    
    # Check if dataset is available
    if ! is_dataset_available "$task_name"; then
        log "${RED}✗ Dataset not available for $task_name, skipping...${NC}"
        return 1
    fi
    
    log "${YELLOW}→ Starting training for $task_name with $agent_count agents...${NC}"
    
    # Train all agents
    for ((i=0; i<agent_count; i++)); do
        log "  Training Agent $i..."
        if bash policy/Diffusion-Policy/train.sh "$task_name" $NUM_EPISODES $i $SEED $GPU_ID; then
            log "  ✓ Agent $i training completed"
        else
            log "${RED}  ✗ Agent $i training failed for $task_name${NC}"
            return 1
        fi
    done
    
    log "${GREEN}✓ Task $task_name training completed successfully for all $agent_count agents!${NC}"
    return 0
}

# Get all available tasks from zarr data
get_available_tasks() {
    local tasks=()
    for zarr_dir in data/zarr_data/*_Agent0_${NUM_EPISODES}.zarr; do
        if [ -d "$zarr_dir" ]; then
            local task_name=$(basename "$zarr_dir" | sed "s/_Agent0_${NUM_EPISODES}.zarr$//")
            tasks+=("$task_name")
        fi
    done
    echo "${tasks[@]}"
}

# Main execution
main() {
    log "${BLUE}========================================${NC}"
    log "${BLUE}RoboFactory Training Script${NC}"
    log "${BLUE}========================================${NC}"
    log "Number of episodes: $NUM_EPISODES"
    log "Training epochs: $EPOCHS"
    log "Random seed: $SEED"
    log "GPU ID: $GPU_ID"
    log "Log file: $LOG_FILE"
    log ""
    
    # Get all available tasks
    local available_tasks=($(get_available_tasks))
    
    if [ ${#available_tasks[@]} -eq 0 ]; then
        log "${RED}No available tasks found! Please run data preparation first.${NC}"
        exit 1
    fi
    
    log "${BLUE}Found ${#available_tasks[@]} available tasks:${NC}"
    for task in "${available_tasks[@]}"; do
        log "  - $task"
    done
    log ""
    
    local total_tasks=${#available_tasks[@]}
    local completed_tasks=0
    local skipped_tasks=0
    local failed_tasks=0
    
    # Process each task
    for task_name in "${available_tasks[@]}"; do
        log "${BLUE}========================================${NC}"
        log "${BLUE}Processing: $task_name${NC}"
        log "${BLUE}========================================${NC}"
        
        if train_task "$task_name"; then
            if is_task_trained "$task_name" "0"; then
                if [ -f "checkpoints/${task_name}_Agent0_${NUM_EPISODES}/${EPOCHS}.ckpt" ]; then
                    completed_tasks=$((completed_tasks + 1))
                else
                    skipped_tasks=$((skipped_tasks + 1))
                fi
            else
                completed_tasks=$((completed_tasks + 1))
            fi
        else
            failed_tasks=$((failed_tasks + 1))
        fi
        
        log ""  # Empty line for readability
    done
    
    # Summary
    log "${BLUE}========================================${NC}"
    log "${BLUE}Training Summary${NC}"
    log "${BLUE}========================================${NC}"
    log "Total tasks processed: $total_tasks"
    log "${GREEN}Successfully completed: $completed_tasks${NC}"
    log "${YELLOW}Skipped (already trained): $skipped_tasks${NC}"
    log "${RED}Failed: $failed_tasks${NC}"
    log ""
    
    if [ $failed_tasks -eq 0 ]; then
        log "${GREEN}🎉 All training completed successfully!${NC}"
        log ""
        log "${BLUE}Available checkpoints:${NC}"
        for task_name in "${available_tasks[@]}"; do
            if is_task_trained "$task_name" "0"; then
                log "  ✓ $task_name (Agent 0 & 1)"
            fi
        done
        exit 0
    else
        log "${RED}⚠️  Some tasks failed. Check the log file for details.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
