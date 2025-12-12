"""
Task instructions formatted for OpenVLA pretrained model.

Instructions should be simple verb phrases that complete:
  "What action should the robot take to {instruction}?"

Examples from OpenVLA pretraining (Bridge/Open X-Embodiment):
  - "pick up the red block"
  - "place it on the table"
  - "flip the pot upright"
  - "move the gripper to the left"

Key principles:
  1. Lowercase - Use lowercase (the template capitalizes the sentence)
  2. Simple verb phrase - Start with verb infinitive: "pick up", "place", "move", "lift"
  3. Concise - Avoid extra context like "together with the other robot"
  4. Object-focused - Mention the main object: "the cube", "the barrier"
"""

# Task instructions in OpenVLA-compatible format
TASK_INSTRUCTIONS = {
    'LiftBarrier-rf': 'lift the barrier',
    'TwoRobotsStackCube-rf': 'stack the cube',
    'ThreeRobotsStackCube-rf': 'stack the cubes',
    'StackCube-rf': 'stack the cube',
    'TakePhoto-rf': 'take a photo',
    'PassShoe-rf': 'pass the shoe',
    'PlaceFood-rf': 'place the food on the plate',
    'CameraAlignment-rf': 'align the camera',
    'LongPipelineDelivery-rf': 'deliver the object',
    'StrikeCube-rf': 'strike the cube',
    'PickMeat-rf': 'pick up the meat',
}

# Global view instructions (for observation-only data)
GLOBAL_VIEW_INSTRUCTIONS = {
    task: f'observe the {task.replace("-rf", "").replace("_", " ").lower()} task'
    for task in TASK_INSTRUCTIONS.keys()
}


def get_task_instruction(task_name: str, is_global_view: bool = False) -> str:
    """
    Get instruction for a task in OpenVLA-compatible format.
    
    Args:
        task_name: Task name (e.g., 'LiftBarrier-rf')
        is_global_view: If True, return observation-style instruction for global camera
        
    Returns:
        Simple verb phrase instruction (lowercase)
    """
    if is_global_view:
        return GLOBAL_VIEW_INSTRUCTIONS.get(
            task_name,
            f'observe the {task_name.replace("-rf", "").replace("_", " ").lower()} task'
        )
    
    return TASK_INSTRUCTIONS.get(
        task_name,
        # Fallback: convert task name to instruction
        task_name.replace('-rf', '').replace('_', ' ').lower()
    )


def get_all_instructions() -> dict:
    """Get all task instructions for debugging/logging."""
    return TASK_INSTRUCTIONS.copy()


if __name__ == "__main__":
    # Print all instructions for verification
    print("Task Instructions (OpenVLA format):")
    print("=" * 60)
    for task, instruction in TASK_INSTRUCTIONS.items():
        full_prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        print(f"\n{task}:")
        print(f"  Instruction: '{instruction}'")
        print(f"  Full prompt: '{full_prompt}'")

