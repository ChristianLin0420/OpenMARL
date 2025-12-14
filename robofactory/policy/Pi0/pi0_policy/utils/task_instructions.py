"""
Centralized task instruction mappings for Pi0/Pi0.5 policies.

This module provides language instructions for different RoboFactory tasks.
"""

from typing import Dict

# Centralized task instruction mapping
TASK_INSTRUCTIONS: Dict[str, str] = {
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


def get_task_instruction(task_name: str, is_global_view: bool = False) -> str:
    """
    Get language instruction for a task.
    
    Args:
        task_name: Task name (e.g., 'LiftBarrier-rf')
        is_global_view: If True, return instruction for global/coordinator view
        
    Returns:
        Language instruction string
    """
    if is_global_view:
        # For global coordinator views (not used in Pi0 agent training)
        base_instruction = TASK_INSTRUCTIONS.get(
            task_name, 
            f"Observe the {task_name.replace('-rf', '')} task"
        )
        return f"Observe: {base_instruction}"
    
    # Return agent-specific instruction
    return TASK_INSTRUCTIONS.get(
        task_name,
        f"Complete the {task_name.replace('-rf', '')} task"
    )

