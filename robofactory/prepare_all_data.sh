#!/bin/bash

# Comprehensive Data Preparation Script for RoboFactory
# This script prepares data for all available tasks, skipping already prepared ones

set -e  # Exit on any error

# Configuration
NUM_EPISODES=150
SCENE_TYPES=("table" "robocasa")
LOG_FILE="data_preparation_$(date +%Y%m%d_%H%M%S).log"

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

# Check if task is already prepared
is_task_prepared() {
    local task_name="$1"
    local scene_type="$2"
    
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

# Get task name from config file
get_task_name() {
    local config_file="$1"
    local task_name=$(grep "task_name:" "$config_file" | head -1 | awk '{print $2}')
    echo "${task_name}-rf"
}

# Prepare data for a single task
prepare_task_data() {
    local config_file="$1"
    local scene_type="$2"
    local task_name=$(get_task_name "$config_file")
    
    log "${BLUE}Processing task: $task_name${NC}"
    
    # Check if already prepared
    if is_task_prepared "$task_name" "$scene_type"; then
        log "${GREEN}✓ Task $task_name is already prepared, skipping...${NC}"
        return 0
    fi
    
    log "${YELLOW}→ Generating data for $task_name...${NC}"
    
    # Step 1: Generate raw data
    log "  Step 1/4: Generating raw data..."
    if python script/generate_data.py --config "$config_file" --num $NUM_EPISODES --save-video; then
        log "  ✓ Raw data generation completed"
    else
        log "${RED}  ✗ Raw data generation failed for $task_name${NC}"
        return 1
    fi
    
    # Step 2: Move h5 files to data directory
    log "  Step 2/4: Moving h5 files..."
    local demo_dir="demos/${task_name}/motionplanning"
    if [ -d "$demo_dir" ]; then
        # Find the largest h5 file (main data file)
        local main_h5=$(find "$demo_dir" -name "*.h5" -type f -exec ls -la {} \; | sort -k5 -nr | head -1 | awk '{print $NF}')
        local main_json="${main_h5%.h5}.json"
        
        if [ -f "$main_h5" ] && [ -f "$main_json" ]; then
            cp "$main_h5" "data/h5_data/${task_name}.h5"
            cp "$main_json" "data/h5_data/${task_name}.json"
            log "  ✓ H5 files moved to data/h5_data/"
        else
            log "${RED}  ✗ Could not find main h5/json files in $demo_dir${NC}"
            return 1
        fi
    else
        log "${RED}  ✗ Demo directory $demo_dir not found${NC}"
        return 1
    fi
    
    # Get the number of agents for this task
    local agent_count=$(get_agent_count "$task_name")
    
    if [ "$agent_count" -eq 0 ]; then
        log "${RED}  ✗ No agents found for $task_name, skipping...${NC}"
        return 1
    fi
    
    log "  Found $agent_count agents for this task"
    
    # Step 3: Convert h5 to pkl
    log "  Step 3/4: Converting h5 to pkl..."
    if python script/parse_h5_to_pkl_multi.py --task_name "$task_name" --load_num $NUM_EPISODES --agent_num $agent_count; then
        log "  ✓ H5 to PKL conversion completed for $agent_count agents"
    else
        log "${RED}  ✗ H5 to PKL conversion failed for $task_name${NC}"
        return 1
    fi
    
    # Step 4: Convert pkl to zarr
    log "  Step 4/4: Converting pkl to zarr..."
    for ((agent_id=0; agent_id<agent_count; agent_id++)); do
        if python script/parse_pkl_to_zarr_dp.py --task_name "$task_name" --load_num $NUM_EPISODES --agent_id $agent_id; then
            log "  ✓ PKL to ZARR conversion completed for Agent $agent_id"
        else
            log "${RED}  ✗ PKL to ZARR conversion failed for Agent $agent_id${NC}"
            return 1
        fi
    done
    
    log "${GREEN}✓ Task $task_name data preparation completed successfully!${NC}"
    return 0
}

# Main execution
main() {
    log "${BLUE}========================================${NC}"
    log "${BLUE}RoboFactory Data Preparation Script${NC}"
    log "${BLUE}========================================${NC}"
    log "Number of episodes per task: $NUM_EPISODES"
    log "Scene types: ${SCENE_TYPES[*]}"
    log "Log file: $LOG_FILE"
    log ""
    
    # Ensure data directories exist
    log "Creating data directories..."
    mkdir -p data/{h5_data,pkl_data,zarr_data}
    
    local total_tasks=0
    local completed_tasks=0
    local skipped_tasks=0
    local failed_tasks=0
    
    # Process each scene type
    for scene_type in "${SCENE_TYPES[@]}"; do
        log "${BLUE}Processing scene type: $scene_type${NC}"
        
        # Get all config files for this scene type
        local config_dir="configs/$scene_type"
        if [ ! -d "$config_dir" ]; then
            log "${YELLOW}Warning: Config directory $config_dir not found, skipping...${NC}"
            continue
        fi
        
        # Process each config file
        for config_file in "$config_dir"/*.yaml; do
            if [ -f "$config_file" ]; then
                total_tasks=$((total_tasks + 1))
                
                if prepare_task_data "$config_file" "$scene_type"; then
                    if is_task_prepared "$(get_task_name "$config_file")" "$scene_type"; then
                        if [ -d "data/zarr_data/$(get_task_name "$config_file")_Agent0_${NUM_EPISODES}.zarr" ]; then
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
            fi
        done
    done
    
    # Summary
    log "${BLUE}========================================${NC}"
    log "${BLUE}Data Preparation Summary${NC}"
    log "${BLUE}========================================${NC}"
    log "Total tasks processed: $total_tasks"
    log "${GREEN}Successfully completed: $completed_tasks${NC}"
    log "${YELLOW}Skipped (already prepared): $skipped_tasks${NC}"
    log "${RED}Failed: $failed_tasks${NC}"
    log ""
    
    if [ $failed_tasks -eq 0 ]; then
        log "${GREEN}🎉 All data preparation completed successfully!${NC}"
        exit 0
    else
        log "${RED}⚠️  Some tasks failed. Check the log file for details.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
