#!/bin/bash

# Comprehensive Evaluation Script for RoboFactory
# This script evaluates all available trained tasks
# Usage: ./eval_all_tasks.sh [video_interval]
# Default: video_interval=10 (saves video every 10 episodes)

set -e  # Exit on any error

# Parse command line arguments
if [ $# -gt 0 ]; then
    VIDEO_RECORD_INTERVAL="$1"
else
    VIDEO_RECORD_INTERVAL=10  # Default to every 10 episodes
fi

# Configuration
NUM_EPISODES=150
CHECKPOINT_EPOCH=300
DEBUG_MODE=0  # 0=quiet, 1=verbose
EVAL_EPISODES=100  # Number of evaluation episodes per task
LOG_FILE="evaluation_all_tasks_$(date +%Y%m%d_%H%M%S).log"
RESULTS_DIR="evaluation_results"

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

# Log configuration
log "${BLUE}========================================${NC}"
log "${BLUE}RoboFactory Evaluation Script${NC}"
log "${BLUE}========================================${NC}"
log "Configuration:"
log "  - Episodes per task: $EVAL_EPISODES"
log "  - Video recording interval: $VIDEO_RECORD_INTERVAL"
log "  - Debug mode: $DEBUG_MODE"
log "  - Results directory: $RESULTS_DIR"
log "${BLUE}========================================${NC}"

# Check if task is trained and ready for evaluation
is_task_ready() {
    local task_name="$1"
    
    # Check if checkpoints exist for both agents
    if [ -f "checkpoints/${task_name}_Agent0_${NUM_EPISODES}/${CHECKPOINT_EPOCH}.ckpt" ] && \
       [ -f "checkpoints/${task_name}_Agent1_${NUM_EPISODES}/${CHECKPOINT_EPOCH}.ckpt" ]; then
        return 0  # Task is ready
    else
        return 1  # Task is not ready
    fi
}

# Get config file for a task
get_config_file() {
    local task_name="$1"
    
    # Convert task name to config file name
    # Remove -rf suffix and convert PascalCase to snake_case
    local base_name=$(echo "$task_name" | sed 's/-rf$//' | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
    
    # Try table config first
    local table_config="configs/table/${base_name}.yaml"
    if [ -f "$table_config" ]; then
        echo "$table_config"
        return 0
    fi
    
    # Try robocasa config
    local robocasa_config="configs/robocasa/${base_name}.yaml"
    if [ -f "$robocasa_config" ]; then
        echo "$robocasa_config"
        return 0
    fi
    
    # If not found, return empty
    echo ""
    return 1
}

# Evaluate a single task
evaluate_task() {
    local task_name="$1"
    local config_file="$2"
    
    log "${BLUE}Evaluating task: $task_name${NC}"
    
    # Check if task is ready
    if ! is_task_ready "$task_name"; then
        log "${RED}✗ Task $task_name is not trained, skipping...${NC}"
        return 1
    fi
    
    log "${YELLOW}→ Starting evaluation for $task_name...${NC}"
    
    # Create task-specific log file
    local task_log="eval_results_${task_name}_${NUM_EPISODES}_${CHECKPOINT_EPOCH}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run evaluation
    log "  Running evaluation with $EVAL_EPISODES episodes..."
    log "  Video recording interval: $VIDEO_RECORD_INTERVAL"
    if bash policy/Diffusion-Policy/eval_multi_with_video_control.sh "$config_file" $NUM_EPISODES $CHECKPOINT_EPOCH $DEBUG_MODE "$task_name" $VIDEO_RECORD_INTERVAL 2>&1 | tee -a "$task_log"; then
        log "  ✓ Evaluation completed for $task_name"
        
        # Extract success rate from the log
        local success_rate=$(tail -1 "$task_log" 2>/dev/null | grep -o '[0-9.]*%' | head -1 || echo "N/A")
        log "  Success rate: $success_rate"
        
        # Move results to results directory
        mkdir -p "$RESULTS_DIR/$task_name"
        mv "$task_log" "$RESULTS_DIR/$task_name/"
        
        return 0
    else
        log "${RED}  ✗ Evaluation failed for $task_name${NC}"
        return 1
    fi
}

# Get all available trained tasks
get_available_tasks() {
    local tasks=()
    for checkpoint_dir in checkpoints/*_Agent0_${NUM_EPISODES}; do
        if [ -d "$checkpoint_dir" ]; then
            local task_name=$(basename "$checkpoint_dir" | sed "s/_Agent0_${NUM_EPISODES}$//")
            if is_task_ready "$task_name"; then
                tasks+=("$task_name")
            fi
        fi
    done
    echo "${tasks[@]}"
}

# Generate summary report
generate_summary() {
    local summary_file="$RESULTS_DIR/evaluation_summary.txt"
    
    log "${BLUE}Generating evaluation summary...${NC}"
    
    echo "RoboFactory Evaluation Summary" > "$summary_file"
    echo "Generated: $(date)" >> "$summary_file"
    echo "=========================================" >> "$summary_file"
    echo "" >> "$summary_file"
    
    for task_dir in "$RESULTS_DIR"/*; do
        if [ -d "$task_dir" ]; then
            local task_name=$(basename "$task_dir")
            echo "Task: $task_name" >> "$summary_file"
            
            # Find the latest log file for this task
            local latest_log=$(find "$task_dir" -name "eval_results_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
            
            if [ -f "$latest_log" ]; then
                echo "  Log file: $(basename "$latest_log")" >> "$summary_file"
                
                # Extract success rate
                local success_rate=$(tail -1 "$latest_log" 2>/dev/null | grep -o '[0-9.]*%' | head -1 || echo "N/A")
                echo "  Success rate: $success_rate" >> "$summary_file"
                
                # Extract total episodes
                local total_episodes=$(grep -c "^[0-9]" "$latest_log" 2>/dev/null || echo "N/A")
                echo "  Episodes evaluated: $total_episodes" >> "$summary_file"
            else
                echo "  No log file found" >> "$summary_file"
            fi
            echo "" >> "$summary_file"
        fi
    done
    
    log "Summary saved to: $summary_file"
}

# Main execution
main() {
    log "${BLUE}========================================${NC}"
    log "${BLUE}RoboFactory Evaluation Script${NC}"
    log "${BLUE}========================================${NC}"
    log "Number of episodes: $NUM_EPISODES"
    log "Checkpoint epoch: $CHECKPOINT_EPOCH"
    log "Evaluation episodes per task: $EVAL_EPISODES"
    log "Debug mode: $DEBUG_MODE"
    log "Log file: $LOG_FILE"
    log "Results directory: $RESULTS_DIR"
    log ""
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Get all available tasks
    local available_tasks=($(get_available_tasks))
    
    if [ ${#available_tasks[@]} -eq 0 ]; then
        log "${RED}No trained tasks found! Please train some tasks first.${NC}"
        exit 1
    fi
    
    log "${BLUE}Found ${#available_tasks[@]} trained tasks:${NC}"
    for task in "${available_tasks[@]}"; do
        log "  - $task"
    done
    log ""
    
    local total_tasks=${#available_tasks[@]}
    local completed_tasks=0
    local failed_tasks=0
    local task_results=()
    
    # Process each task
    for task_name in "${available_tasks[@]}"; do
        log "${BLUE}========================================${NC}"
        log "${BLUE}Processing: $task_name${NC}"
        log "${BLUE}========================================${NC}"
        
        # Get config file
        local config_file=$(get_config_file "$task_name")
        if [ -z "$config_file" ]; then
            log "${RED}✗ Config file not found for $task_name, skipping...${NC}"
            failed_tasks=$((failed_tasks + 1))
            continue
        fi
        
        log "Using config: $config_file"
        
        # Evaluate task
        if evaluate_task "$task_name" "$config_file"; then
            completed_tasks=$((completed_tasks + 1))
            task_results+=("$task_name: SUCCESS")
        else
            failed_tasks=$((failed_tasks + 1))
            task_results+=("$task_name: FAILED")
        fi
        
        log ""  # Empty line for readability
    done
    
    # Generate summary
    generate_summary
    
    # Final summary
    log "${BLUE}========================================${NC}"
    log "${BLUE}Evaluation Summary${NC}"
    log "${BLUE}========================================${NC}"
    log "Total tasks processed: $total_tasks"
    log "${GREEN}Successfully completed: $completed_tasks${NC}"
    log "${RED}Failed: $failed_tasks${NC}"
    log ""
    
    log "${BLUE}Task Results:${NC}"
    for result in "${task_results[@]}"; do
        if [[ "$result" == *"SUCCESS"* ]]; then
            log "${GREEN}  ✓ $result${NC}"
        else
            log "${RED}  ✗ $result${NC}"
        fi
    done
    log ""
    
    log "${BLUE}Results saved in: $RESULTS_DIR/${NC}"
    log "${BLUE}Summary report: $RESULTS_DIR/evaluation_summary.txt${NC}"
    
    if [ $failed_tasks -eq 0 ]; then
        log "${GREEN}🎉 All evaluations completed successfully!${NC}"
        exit 0
    else
        log "${YELLOW}⚠️  Some evaluations failed. Check individual task logs for details.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
