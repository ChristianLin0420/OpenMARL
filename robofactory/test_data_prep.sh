#!/bin/bash

# Test script for data preparation functions

NUM_EPISODES=150

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

echo "Testing data preparation status:"
echo "TakePhoto-rf: $(is_task_prepared "TakePhoto-rf" "table" && echo "PREPARED" || echo "NOT PREPARED") (needs $(get_agent_count "TakePhoto-rf") agents)"
echo "ThreeRobotsStackCube-rf: $(is_task_prepared "ThreeRobotsStackCube-rf" "table" && echo "PREPARED" || echo "NOT PREPARED") (needs $(get_agent_count "ThreeRobotsStackCube-rf") agents)"
echo "LiftBarrier-rf: $(is_task_prepared "LiftBarrier-rf" "table" && echo "PREPARED" || echo "NOT PREPARED") (needs $(get_agent_count "LiftBarrier-rf") agents)"

echo ""
echo "Available zarr datasets:"
ls data/zarr_data/ | grep -E "(TakePhoto|ThreeRobotsStackCube|LiftBarrier)" | sort

