#!/bin/bash

# Comprehensive Data Preparation Script for RoboFactory
# This script prepares data for all available tasks, skipping already prepared ones
#
# Usage:
#   ./prepare_all_data.sh                           # Generate all tasks
#   ./prepare_all_data.sh --task LiftBarrier        # Generate specific task
#   ./prepare_all_data.sh --task LiftBarrier --scene table  # Specific task and scene
#   ./prepare_all_data.sh --num 50                  # Generate 50 episodes for all tasks
#   ./prepare_all_data.sh --task LiftBarrier --num 100      # 100 episodes for LiftBarrier

export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

set -e  # Exit on any error

# Default Configuration
NUM_EPISODES=160
SCENE_TYPES=("table" "robocasa")
LOG_FILE="data_preparation_$(date +%Y%m%d_%H%M%S).log"
SPECIFIC_TASK=""
SPECIFIC_SCENE=""
# Batch processing configuration for h5 to pkl conversion
H5_BATCH_SIZE=8  # Number of episodes to process in each batch (to avoid OOM)
H5_NUM_WORKERS=16  # Number of parallel workers for h5 to pkl conversion

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Available tasks
AVAILABLE_TASKS=(
    "LiftBarrier"
    "TwoRobotsStackCube"
    "ThreeRobotsStackCube"
    "CameraAlignment"
    "LongPipelineDelivery"
    "TakePhoto"
    "PassShoe"
    "PlaceFood"
    "StackCube"
    "StrikeCube"
    "PickMeat"
)

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --task|-t)
                SPECIFIC_TASK="$2"
                shift 2
                ;;
            --scene|-s)
                SPECIFIC_SCENE="$2"
                shift 2
                ;;
            --num|-n)
                NUM_EPISODES="$2"
                shift 2
                ;;
            --batch-size)
                H5_BATCH_SIZE="$2"
                shift 2
                ;;
            --workers)
                H5_NUM_WORKERS="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --task, -t TASK    Prepare data for specific task only"
                echo "  --scene, -s SCENE  Use specific scene type (table or robocasa)"
                echo "  --num, -n NUM      Number of episodes to generate (default: 160)"
                echo "  --batch-size NUM   Batch size for h5 to pkl conversion (default: 5)"
                echo "  --workers NUM      Number of parallel workers for h5 to pkl (default: 16)"
                echo "  --help, -h         Show this help message"
                echo ""
                echo "Available tasks:"
                for task in "${AVAILABLE_TASKS[@]}"; do
                    echo "  - $task"
                done
                echo ""
                echo "Examples:"
                echo "  $0                              # Generate all tasks"
                echo "  $0 --task LiftBarrier           # Generate LiftBarrier only"
                echo "  $0 --task LiftBarrier --num 50  # 50 episodes of LiftBarrier"
                echo "  $0 --scene table                # All tasks with table scene"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Validate specific task if provided
    if [ -n "$SPECIFIC_TASK" ]; then
        local valid=false
        for task in "${AVAILABLE_TASKS[@]}"; do
            if [ "$task" == "$SPECIFIC_TASK" ]; then
                valid=true
                break
            fi
        done
        if [ "$valid" == "false" ]; then
            echo "Error: Unknown task '$SPECIFIC_TASK'"
            echo "Available tasks: ${AVAILABLE_TASKS[*]}"
            exit 1
        fi
    fi
    
    # Validate scene if provided
    if [ -n "$SPECIFIC_SCENE" ]; then
        if [ "$SPECIFIC_SCENE" != "table" ] && [ "$SPECIFIC_SCENE" != "robocasa" ]; then
            echo "Error: Unknown scene '$SPECIFIC_SCENE'"
            echo "Available scenes: table, robocasa"
            exit 1
        fi
        SCENE_TYPES=("$SPECIFIC_SCENE")
    fi
}

# Convert task name to config filename
task_to_config_name() {
    local task_name="$1"
    # Convert CamelCase to snake_case
    echo "$task_name" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]'
}

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

#==============================================================================
# Step-level existence checks for incremental processing
#==============================================================================

# Check if h5 data exists for a task
has_h5_data() {
    local task_name="$1"
    local h5_file="data/h5_data/${task_name}.h5"
    local json_file="data/h5_data/${task_name}.json"
    
    if [ -f "$h5_file" ] && [ -f "$json_file" ]; then
        return 0
    fi
    return 1
}

# Check if pkl data exists for a task (all agents)
has_pkl_data() {
    local task_name="$1"
    local agent_count="$2"
    
    for ((i=0; i<agent_count; i++)); do
        local pkl_dir="data/pkl_data/${task_name}_Agent${i}"
        if [ ! -d "$pkl_dir" ]; then
            return 1
        fi
    done
    return 0
}

# Check if zarr data exists for a specific agent
has_zarr_data_agent() {
    local task_name="$1"
    local agent_id="$2"
    local zarr_path="data/zarr_data/${task_name}_Agent${agent_id}_${NUM_EPISODES}.zarr"
    
    if [ -d "$zarr_path" ]; then
        return 0
    fi
    return 1
}

# Check if zarr data exists for all agents
has_zarr_data() {
    local task_name="$1"
    local agent_count="$2"
    
    for ((i=0; i<agent_count; i++)); do
        if ! has_zarr_data_agent "$task_name" "$i"; then
            return 1
        fi
    done
    return 0
}

# Check if RLDS data exists for a specific agent
has_rlds_data_agent() {
    local task_name="$1"
    local agent_id="$2"
    local rlds_path="data/rlds_data/${task_name}_Agent${agent_id}"
    
    if [ -d "$rlds_path" ]; then
        return 0
    fi
    return 1
}

# Check if RLDS data exists for all agents
has_rlds_data() {
    local task_name="$1"
    local agent_count="$2"
    
    for ((i=0; i<agent_count; i++)); do
        if ! has_rlds_data_agent "$task_name" "$i"; then
            return 1
        fi
    done
    return 0
}

# Check if LeRobot data exists for a specific agent
has_lerobot_data_agent() {
    local task_name="$1"
    local agent_id="$2"
    local lerobot_path="data/lerobot_data/${task_name}_Agent${agent_id}_${NUM_EPISODES}"
    
    if [ -d "$lerobot_path" ]; then
        return 0
    fi
    return 1
}

# Check if LeRobot data exists for all agents
has_lerobot_data() {
    local task_name="$1"
    local agent_count="$2"
    
    for ((i=0; i<agent_count; i++)); do
        if ! has_lerobot_data_agent "$task_name" "$i"; then
            return 1
        fi
    done
    return 0
}

#==============================================================================
# Prepare data for a single task (with step-level skipping)
#==============================================================================
prepare_task_data() {
    local config_file="$1"
    local scene_type="$2"
    local task_name=$(get_task_name "$config_file")
    
    log "${BLUE}Processing task: $task_name${NC}"
    
    # Get the number of agents for this task (needed for all checks)
    local agent_count=$(get_agent_count "$task_name")
    
    if [ "$agent_count" -eq 0 ]; then
        log "${RED}  ✗ No agents found for $task_name, skipping...${NC}"
        return 1
    fi
    
    log "  Found $agent_count agents for this task"
    
    # Check if fully prepared (all RLDS and LeRobot data exists)
    if has_rlds_data "$task_name" "$agent_count" && has_lerobot_data "$task_name" "$agent_count"; then
        log "${GREEN}✓ Task $task_name is fully prepared (all RLDS and LeRobot data exists), skipping...${NC}"
        return 0
    fi
    
    log "${YELLOW}→ Processing data for $task_name...${NC}"
    
    #--------------------------------------------------------------------------
    # Step 1 & 2: Generate raw data and move h5 files
    #--------------------------------------------------------------------------
    if has_h5_data "$task_name"; then
        log "  Step 1/6: ${GREEN}[SKIP]${NC} H5 data already exists"
        log "  Step 2/6: ${GREEN}[SKIP]${NC} H5 files already in data/h5_data/"
    else
        log "  Step 1/6: Generating raw data..."
        if python script/generate_data.py --config "$config_file" --num $NUM_EPISODES --save-video; then
            log "  ✓ Raw data generation completed"
        else
            log "${RED}  ✗ Raw data generation failed for $task_name${NC}"
            return 1
        fi
        
        log "  Step 2/6: Moving h5 files..."
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
    fi
    
    #--------------------------------------------------------------------------
    # Step 3: Convert h5 to pkl
    #--------------------------------------------------------------------------
    if has_pkl_data "$task_name" "$agent_count"; then
        log "  Step 3/6: ${GREEN}[SKIP]${NC} PKL data already exists for all $agent_count agents"
    else
        log "  Step 3/6: Converting h5 to pkl with batch_size=$H5_BATCH_SIZE and $H5_NUM_WORKERS workers..."
        if python script/parse_h5_to_pkl_multi.py --task_name "$task_name" --load_num $NUM_EPISODES --agent_num $agent_count --batch_size $H5_BATCH_SIZE --num_workers $H5_NUM_WORKERS; then
            log "  ✓ H5 to PKL conversion completed for $agent_count agents + global"
        else
            log "${RED}  ✗ H5 to PKL conversion failed for $task_name${NC}"
            return 1
        fi
    fi
    
    #--------------------------------------------------------------------------
    # Step 4: Convert pkl to zarr (per-agent check)
    #--------------------------------------------------------------------------
    log "  Step 4/6: Converting pkl to zarr..."
    local zarr_skipped=0
    local zarr_converted=0
    
    for ((agent_id=0; agent_id<agent_count; agent_id++)); do
        if has_zarr_data_agent "$task_name" "$agent_id"; then
            log "    Agent $agent_id: ${GREEN}[SKIP]${NC} ZARR data already exists"
            zarr_skipped=$((zarr_skipped + 1))
        else
            if python script/parse_pkl_to_zarr_dp.py --task_name "$task_name" --load_num $NUM_EPISODES --agent_id $agent_id; then
                log "    Agent $agent_id: ✓ PKL to ZARR conversion completed"
                zarr_converted=$((zarr_converted + 1))
            else
                log "${RED}    Agent $agent_id: ✗ PKL to ZARR conversion failed${NC}"
                return 1
            fi
        fi
    done
    
    # Global camera zarr
    local global_zarr="data/zarr_data/${task_name}_global_${NUM_EPISODES}.zarr"
    if [ -d "$global_zarr" ]; then
        log "    Global: ${GREEN}[SKIP]${NC} ZARR data already exists"
    else
        if python script/parse_pkl_to_zarr_dp.py --task_name "$task_name" --load_num $NUM_EPISODES --agent_id -1 2>/dev/null; then
            log "    Global: ✓ PKL to ZARR conversion completed"
        else
            log "    Global: ${YELLOW}[SKIP]${NC} Global camera conversion skipped"
        fi
    fi
    
    log "  ✓ Step 4 complete (converted: $zarr_converted, skipped: $zarr_skipped)"
    
    #--------------------------------------------------------------------------
    # Step 5: Convert zarr to RLDS (per-agent check)
    #--------------------------------------------------------------------------
    log "  Step 5/6: Converting zarr to RLDS..."
    local rlds_skipped=0
    local rlds_converted=0
    
    for ((agent_id=0; agent_id<agent_count; agent_id++)); do
        if has_rlds_data_agent "$task_name" "$agent_id"; then
            log "    Agent $agent_id: ${GREEN}[SKIP]${NC} RLDS data already exists"
            rlds_skipped=$((rlds_skipped + 1))
        else
            local zarr_path="data/zarr_data/${task_name}_Agent${agent_id}_${NUM_EPISODES}.zarr"
            if [ -d "$zarr_path" ]; then
                if python -c "
from robofactory.policy.OpenVLA.openvla_policy.utils.data_conversion import convert_zarr_to_rlds, create_dataset_statistics
import os

# Task instructions mapping
TASK_INSTRUCTIONS = {
    'LiftBarrier-rf': 'Lift the barrier together with the other robot',
    'TwoRobotsStackCube-rf': 'Stack the cubes together with the other robot',
    'ThreeRobotsStackCube-rf': 'Stack the cubes together with the other robots',
    'CameraAlignment-rf': 'Align the camera with the target object',
    'LongPipelineDelivery-rf': 'Pass the object along the robot chain',
    'TakePhoto-rf': 'Take a photo of the target object',
    'PassShoe-rf': 'Pass the shoe to the other robot',
    'PlaceFood-rf': 'Place the food on the plate',
    'StackCube-rf': 'Stack the cube on top of the other cube',
    'StrikeCube-rf': 'Strike the cube to the target location',
    'PickMeat-rf': 'Pick up the meat from the grill',
}

task_name = '$task_name'
agent_id = $agent_id
zarr_path = '$zarr_path'
instruction = TASK_INSTRUCTIONS.get(task_name, 'Complete the task')

output_path = convert_zarr_to_rlds(
    zarr_path=zarr_path,
    output_dir='data/rlds_data',
    task_name=task_name,
    agent_id=agent_id,
    language_instruction=instruction
)

# Create statistics file
stats_path = os.path.join(output_path, 'statistics.json')
create_dataset_statistics(zarr_path, stats_path)
print(f'RLDS conversion successful: {output_path}')
" 2>/dev/null; then
                    log "    Agent $agent_id: ✓ ZARR to RLDS conversion completed"
                    rlds_converted=$((rlds_converted + 1))
                else
                    log "    Agent $agent_id: ${YELLOW}[WARN]${NC} ZARR to RLDS conversion failed"
                fi
            else
                log "    Agent $agent_id: ${YELLOW}[WARN]${NC} ZARR data not found, skipping RLDS"
            fi
        fi
    done
    
    # Convert global camera data to RLDS
    local global_rlds_path="data/rlds_data/${task_name}_global"
    local global_zarr_path="data/zarr_data/${task_name}_global_${NUM_EPISODES}.zarr"
    
    if [ -d "$global_rlds_path" ]; then
        log "    Global: ${GREEN}[SKIP]${NC} RLDS data already exists"
    elif [ -d "$global_zarr_path" ]; then
        if python -c "
from robofactory.policy.OpenVLA.openvla_policy.utils.data_conversion import convert_zarr_to_rlds_global, create_dataset_statistics
import os

# Task instructions mapping
TASK_INSTRUCTIONS = {
    'LiftBarrier-rf': 'Observe the multi-robot task from global view',
    'TwoRobotsStackCube-rf': 'Observe the cube stacking task from global view',
    'ThreeRobotsStackCube-rf': 'Observe the cube stacking task from global view',
    'CameraAlignment-rf': 'Observe the camera alignment task from global view',
    'LongPipelineDelivery-rf': 'Observe the delivery task from global view',
    'TakePhoto-rf': 'Observe the photo taking task from global view',
    'PassShoe-rf': 'Observe the shoe passing task from global view',
    'PlaceFood-rf': 'Observe the food placement task from global view',
    'StackCube-rf': 'Observe the cube stacking task from global view',
    'StrikeCube-rf': 'Observe the cube striking task from global view',
    'PickMeat-rf': 'Observe the meat picking task from global view',
}

task_name = '$task_name'
zarr_path = '$global_zarr_path'
instruction = TASK_INSTRUCTIONS.get(task_name, 'Observe the task from global view')

output_path = convert_zarr_to_rlds_global(
    zarr_path=zarr_path,
    output_dir='data/rlds_data',
    task_name=task_name,
    language_instruction=instruction
)

# Create statistics file (only for image data, no actions)
stats_path = os.path.join(output_path, 'statistics.json')
create_dataset_statistics(zarr_path, stats_path)
print(f'RLDS conversion successful: {output_path}')
" 2>/dev/null; then
            log "    Global: ✓ ZARR to RLDS conversion completed"
        else
            log "    Global: ${YELLOW}[SKIP]${NC} Global RLDS conversion skipped"
        fi
    else
        log "    Global: ${YELLOW}[SKIP]${NC} No global ZARR data found"
    fi
    
    log "  ✓ Step 5 complete (converted: $rlds_converted, skipped: $rlds_skipped)"
    
    #--------------------------------------------------------------------------
    # Step 6: Convert zarr to LeRobot format (for Pi0/Pi0.5) [per-agent check]
    #--------------------------------------------------------------------------
    log "  Step 6/6: Converting zarr to LeRobot format (for Pi0)..."
    local lerobot_skipped=0
    local lerobot_converted=0

    # Only convert agent data, NOT global data (Pi0 trains on agent-level policies)
    for ((agent_id=0; agent_id<agent_count; agent_id++)); do
        if has_lerobot_data_agent "$task_name" "$agent_id"; then
            log "    Agent $agent_id: ${GREEN}[SKIP]${NC} LeRobot data already exists"
            lerobot_skipped=$((lerobot_skipped + 1))
        else
            local zarr_path="data/zarr_data/${task_name}_Agent${agent_id}_${NUM_EPISODES}.zarr"
            if [ -d "$zarr_path" ]; then
                # Execute from OpenMARL root to handle imports correctly
                if cd .. && PYTHONPATH=/workspace/OpenMARL python /workspace/OpenMARL/robofactory/policy/Pi0/pi0_policy/utils/data_conversion.py \
                    --zarr_path="robofactory/$zarr_path" \
                    --output_dir="robofactory/data/lerobot_data" \
                    --task_name="$task_name" \
                    --agent_id=$agent_id \
                    --num_episodes=$NUM_EPISODES; then
                    log "    Agent $agent_id: ✓ ZARR to LeRobot conversion completed"
                    lerobot_converted=$((lerobot_converted + 1))
                    cd robofactory
                else
                    log "    Agent $agent_id: ${YELLOW}[WARN]${NC} ZARR to LeRobot conversion failed"
                    cd robofactory 2>/dev/null || true  # Make sure we return even on error
                fi
            else
                log "    Agent $agent_id: ${YELLOW}[WARN]${NC} ZARR data not found, skipping LeRobot"
            fi
        fi
    done

    # Explicitly SKIP global data for Pi0/Pi0.5
    log "    Global: ${YELLOW}[SKIP]${NC} Global data not needed for Pi0 (action-prediction model)"

    log "  ✓ Step 6 complete (converted: $lerobot_converted, skipped: $lerobot_skipped)"
    
    log "${GREEN}✓ Task $task_name data preparation completed successfully!${NC}"
    return 0
}

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"
    
    log "${BLUE}========================================${NC}"
    log "${BLUE}RoboFactory Data Preparation Script${NC}"
    log "${BLUE}========================================${NC}"
    log "Number of episodes per task: $NUM_EPISODES"
    log "Scene types: ${SCENE_TYPES[*]}"
    if [ -n "$SPECIFIC_TASK" ]; then
        log "Specific task: $SPECIFIC_TASK"
    else
        log "Processing: All tasks"
    fi
    log "Log file: $LOG_FILE"
    log ""

    # Download assets
    log "Downloading assets..."
    python script/download_assets.py
    log "Assets downloaded successfully"

    # Ensure data directories exist
    log "Creating data directories..."
    mkdir -p data/{h5_data,pkl_data,zarr_data,rlds_data,lerobot_data}
    
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
        
        # If specific task is requested, only process that config
        if [ -n "$SPECIFIC_TASK" ]; then
            local config_name=$(task_to_config_name "$SPECIFIC_TASK")
            local config_file="$config_dir/${config_name}.yaml"
            
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
            else
                log "${YELLOW}Warning: Config file $config_file not found for scene $scene_type${NC}"
            fi
        else
            # Process all config files
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
        fi
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
