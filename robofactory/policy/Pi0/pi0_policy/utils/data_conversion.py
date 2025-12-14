"""
Data conversion utilities for converting ZARR format to LeRobot format for Pi0/Pi0.5.

This module follows the openpi data pipeline conventions:
- Use LeRobot dataset format (not RLDS)
- Image keys following Pi0 requirements: "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
- State key: "state" (for proprio)
- Action key: "actions"
- Task key: "task" (for language instructions)

RoboFactory camera order in ZARR 'img' array:
[0] head_camera - side view (fixed position) → base_0_rgb
[1] global_camera - overhead view (bird's eye) → left_wrist_0_rgb
[2] wrist_camera - gripper view (end-effector) → right_wrist_0_rgb
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import zarr
from tqdm import tqdm

try:
    # Try new lerobot structure (v0.4+)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        # Fallback to old structure (v0.2-0.3)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Warning: LeRobot not installed. Please install with: pip install lerobot>=0.2.0")
        LeRobotDataset = None

# Handle both module import and direct script execution
try:
    from .task_instructions import get_task_instruction, TASK_INSTRUCTIONS
except ImportError:
    # If running as script, use absolute import
    import sys
    from pathlib import Path
    # Add the OpenMARL root to path
    openmarl_root = Path(__file__).resolve().parents[5]
    if str(openmarl_root) not in sys.path:
        sys.path.insert(0, str(openmarl_root))
    from robofactory.policy.Pi0.pi0_policy.utils.task_instructions import get_task_instruction, TASK_INSTRUCTIONS


def _process_image(image: np.ndarray) -> np.ndarray:
    """
    Process image to ensure correct format (HWC, uint8).
    
    Args:
        image: Input image array
        
    Returns:
        Processed image in HWC uint8 format
    """
    # Ensure image is in HWC format (H, W, C)
    if len(image.shape) == 3 and image.shape[0] == 3:
        # Convert from CHW to HWC
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    return image


def convert_zarr_to_lerobot(
    zarr_path: str,
    output_dir: str,
    task_name: str,
    agent_id: int,
    num_episodes: int,
    language_instruction: Optional[str] = None,
) -> str:
    """
    Convert ZARR dataset to LeRobot format for Pi0/Pi0.5.
    
    RoboFactory camera order (in ZARR 'img' array):
    [0] head_camera - side view (fixed position)
    [1] global_camera - overhead view (bird's eye)
    [2] wrist_camera - gripper view (end-effector)
    
    Maps to Pi0's 3-camera requirement:
    - base_0_rgb <- head_camera [0]
    - left_wrist_0_rgb <- global_camera [1]
    - right_wrist_0_rgb <- wrist_camera [2]
    
    Args:
        zarr_path: Path to input ZARR dataset
        output_dir: Directory to save LeRobot dataset
        task_name: Name of the task (e.g., "LiftBarrier-rf")
        agent_id: Agent ID
        num_episodes: Number of episodes in dataset
        language_instruction: Language instruction for the task
        
    Returns:
        Repo ID of created LeRobot dataset
    """
    if LeRobotDataset is None:
        raise ImportError("LeRobot is required. Install with: pip install lerobot>=0.2.0")
    
    print(f"Loading ZARR data from {zarr_path}...")
    root = zarr.open(zarr_path, mode='r')
    
    data_group = root['data']
    meta_group = root['meta']
    
    # Get episode boundaries
    episode_ends = np.array(meta_group['episode_ends'])
    
    # Check which camera format is used
    has_img_array = 'img' in data_group
    has_separate_cameras = 'head_camera' in data_group or 'wrist_camera' in data_group
    
    if has_img_array:
        # Old format: single 'img' array with all cameras
        # Get image dimensions from first frame
        first_img = data_group['img'][0]  # Shape: (num_cameras, C, H, W) or (num_cameras, H, W, C)
        num_cameras = first_img.shape[0]
        
        # Determine if CHW or HWC format
        if first_img.shape[1] == 3:  # CHW format: (num_cameras, 3, H, W)
            img_h, img_w = first_img.shape[2], first_img.shape[3]
            is_chw = True
        else:  # HWC format: (num_cameras, H, W, 3)
            img_h, img_w = first_img.shape[1], first_img.shape[2]
            is_chw = False
        
        print(f"Found {num_cameras} cameras with resolution {img_h}x{img_w} (img array format)")
        use_separate_cameras = False
        
    elif has_separate_cameras:
        # New format: separate arrays for each camera
        # Get image dimensions from first camera
        first_cam = data_group.get('head_camera', data_group.get('wrist_camera'))
        first_img = first_cam[0]
        
        # Determine if CHW or HWC format
        if first_img.shape[0] == 3:  # CHW format: (3, H, W)
            img_h, img_w = first_img.shape[1], first_img.shape[2]
            is_chw = True
        else:  # HWC format: (H, W, 3)
            img_h, img_w = first_img.shape[0], first_img.shape[1]
            is_chw = False
        
        print(f"Found separate camera arrays with resolution {img_h}x{img_w}")
        print(f"  - head_camera → base_0_rgb")
        if 'global_camera' in data_group:
            print(f"  - global_camera → left_wrist_0_rgb")
        print(f"  - wrist_camera → right_wrist_0_rgb")
        use_separate_cameras = True
        
    else:
        raise ValueError(f"No camera data found in ZARR at {zarr_path}. Expected 'img' array or separate camera arrays.")
    
    # Get action and state dimensions
    action_dim = data_group['action'].shape[-1]
    state_key = 'state' if 'state' in data_group else 'agent_pos'
    state_dim = data_group[state_key].shape[-1]
    
    # Prepare output
    repo_id = f"{task_name}_Agent{agent_id}_{num_episodes}"
    output_path = Path(output_dir) / repo_id
    
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Clean cache if it exists to avoid conflicts with LeRobot's exist_ok=False
    cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if cache_path.exists():
        print(f"Cleaning existing cache at {cache_path}")
        shutil.rmtree(cache_path)
    
    print(f"Creating LeRobot dataset: {repo_id}")
    print(f"  Output path: {output_path}")
    print(f"  Action dim: {action_dim}, State dim: {state_dim}")
    
    # Create LeRobot dataset with Pi0's exact 3-camera format
    # DON'T specify root - let it use default location, then we'll move it
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,  # RoboFactory default
        features={
            # Pi0's 3 required cameras (exact naming convention)
            "base_0_rgb": {  # Head/side camera [0]
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_0_rgb": {  # Global/overhead camera [1]
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_0_rgb": {  # Wrist/gripper camera [2]
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            # Proprio/state (openpi convention: "state")
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            # Actions (openpi convention: "actions")
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Set default language instruction
    if language_instruction is None:
        language_instruction = get_task_instruction(task_name)
    
    print(f"Converting {len(episode_ends)} episodes")
    print(f"Language instruction: '{language_instruction}'")
    
    # Convert episodes
    start_idx = 0
    for ep_idx, end_idx in enumerate(tqdm(episode_ends, desc="Converting episodes")):
        for t in range(start_idx, end_idx):
            if use_separate_cameras:
                # New format: read from separate camera arrays
                head_img = data_group['head_camera'][t]
                wrist_img = data_group['wrist_camera'][t]
                
                # Check if global camera exists, otherwise duplicate head camera
                if 'global_camera' in data_group:
                    global_img = data_group['global_camera'][t]
                else:
                    global_img = head_img.copy()  # Fallback if no global camera
                
                # Convert from CHW to HWC if needed
                if is_chw:
                    head_img = np.transpose(head_img, (1, 2, 0))
                    global_img = np.transpose(global_img, (1, 2, 0))
                    wrist_img = np.transpose(wrist_img, (1, 2, 0))
            else:
                # Old format: extract from 'img' array
                # Camera order: [0]=head/side, [1]=global/overhead, [2]=wrist/gripper
                img_t = data_group['img'][t]  # Shape: (3, ...) for 3 cameras
                
                if is_chw:
                    # Convert from (C, H, W) to (H, W, C)
                    head_img = np.transpose(img_t[0], (1, 2, 0))
                    global_img = np.transpose(img_t[1], (1, 2, 0))
                    wrist_img = np.transpose(img_t[2], (1, 2, 0))
                else:
                    # Already in (H, W, C) format
                    head_img = img_t[0]
                    global_img = img_t[1]
                    wrist_img = img_t[2]
            
            # Ensure uint8 format
            head_img = _process_image(head_img)
            global_img = _process_image(global_img)
            wrist_img = _process_image(wrist_img)
            
            # Get state/action
            state_data = data_group[state_key][t]
            action_data = data_group['action'][t]
            
            frame_data = {
                # Map to Pi0's exact key names IN THE CORRECT ORDER
                "base_0_rgb": head_img,           # [0] Side view
                "left_wrist_0_rgb": global_img,   # [1] Overhead view
                "right_wrist_0_rgb": wrist_img,   # [2] Gripper view
                
                "state": state_data.astype(np.float32),
                "actions": action_data.astype(np.float32),
                "task": language_instruction,
            }
            
            dataset.add_frame(frame_data)
        
        dataset.save_episode()
        start_idx = end_idx
    
    print(f"✅ Successfully converted {len(episode_ends)} episodes to LeRobot format")
    print(f"   Camera mapping:")
    print(f"     base_0_rgb <- head_camera [0] (side view)")
    print(f"     left_wrist_0_rgb <- global_camera [1] (overhead view)")
    print(f"     right_wrist_0_rgb <- wrist_camera [2] (gripper view)")
    
    # Move dataset from cache to output directory
    # LeRobot creates datasets in ~/.cache/huggingface/lerobot by default
    import os
    cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    
    if cache_path.exists():
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Move from cache to target location
        print(f"   Moving dataset from {cache_path} to {output_path}")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(cache_path), str(output_path))
        print(f"   ✓ Dataset moved to: {output_path}")
    else:
        print(f"   Output: {output_path}")
    
    print(f"   Repo ID: {repo_id}")
    
    return repo_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ZARR to LeRobot format for Pi0")
    parser.add_argument(
        "--zarr_path",
        type=str,
        required=True,
        help="Path to ZARR file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/lerobot_data",
        help="Output directory for LeRobot datasets"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name (e.g., LiftBarrier-rf)"
    )
    parser.add_argument(
        "--agent_id",
        type=int,
        required=True,
        help="Agent ID"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        required=True,
        help="Number of episodes in dataset"
    )
    
    args = parser.parse_args()
    
    convert_zarr_to_lerobot(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        task_name=args.task_name,
        agent_id=args.agent_id,
        num_episodes=args.num_episodes,
    )

