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
    
    # Check if checkpoint exists for both agents
    if [ -f "checkpoints/${task_name}_Agent0_${NUM_EPISODES}/${EPOCHS}.ckpt" ] && \
       [ -f "checkpoints/${task_name}_Agent1_${NUM_EPISODES}/${EPOCHS}.ckpt" ]; then
        return 0  # Task is trained
    else
        return 1  # Task is not trained
    fi
}

# Check if dataset exists for training
is_dataset_available() {
    local task_name="$1"
    
    # Check if zarr data exists for both agents
    if [ -d "data/zarr_data/${task_name}_Agent0_${NUM_EPISODES}.zarr" ] && \
       [ -d "data/zarr_data/${task_name}_Agent1_${NUM_EPISODES}.zarr" ]; then
        return 0  # Dataset is available
    else
        return 1  # Dataset is not available
    fi
}

# Train a single task
train_task() {
    local task_name="$1"
    
    log "${BLUE}Training task: $task_name${NC}"
    
    # Check if already trained
    if is_task_trained "$task_name" "0"; then
        log "${GREEN}✓ Task $task_name is already trained, skipping...${NC}"
        return 0
    fi
    
    # Check if dataset is available
    if ! is_dataset_available "$task_name"; then
        log "${RED}✗ Dataset not available for $task_name, skipping...${NC}"
        return 1
    fi
    
    log "${YELLOW}→ Starting training for $task_name...${NC}"
    
    # Train Agent 0
    log "  Training Agent 0..."
    if bash policy/Diffusion-Policy/train.sh "$task_name" $NUM_EPISODES 0 $SEED $GPU_ID; then
        log "  ✓ Agent 0 training completed"
    else
        log "${RED}  ✗ Agent 0 training failed for $task_name${NC}"
        return 1
    fi
    
    # Train Agent 1
    log "  Training Agent 1..."
    if bash policy/Diffusion-Policy/train.sh "$task_name" $NUM_EPISODES 1 $SEED $GPU_ID; then
        log "  ✓ Agent 1 training completed"
    else
        log "${RED}  ✗ Agent 1 training failed for $task_name${NC}"
        return 1
    fi
    
    log "${GREEN}✓ Task $task_name training completed successfully!${NC}"
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
