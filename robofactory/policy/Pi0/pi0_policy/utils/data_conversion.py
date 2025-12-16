"""
Data conversion utilities for converting ZARR format to LeRobot format for Pi0/Pi0.5.

This module follows the openpi data pipeline conventions:
- Use LeRobot dataset format (not RLDS)
- Image keys following Pi0 requirements: "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
- State key: "state" (for proprio)
- Action key: "actions"
- Task key: "task" (for language instructions)

RoboFactory ZARR Data Structure:
- Agent ZARR: {TASK}_Agent{ID}_{NUM}.zarr
    - head_camera: side view (fixed position) → base_0_rgb
    - wrist_camera: gripper view (end-effector) → right_wrist_0_rgb
    
- Global ZARR: {TASK}_Agent{ID}_global_{NUM}.zarr
    - head_camera: overhead/global view (bird's eye) → left_wrist_0_rgb

Pi0 3-Camera Mapping:
- base_0_rgb      ← head_camera from Agent ZARR (side view)
- left_wrist_0_rgb ← head_camera from Global ZARR (overhead view)
- right_wrist_0_rgb ← wrist_camera from Agent ZARR (gripper view)
"""

import json
import os
import re
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
    global_zarr_path: Optional[str] = None,
) -> str:
    """
    Convert ZARR dataset to LeRobot format for Pi0/Pi0.5.
    """
    if LeRobotDataset is None:
        raise ImportError("LeRobot is required. Install with: pip install lerobot>=0.2.0")
    
    print(f"Loading ZARR data from {zarr_path}...")
    root = zarr.open(zarr_path, mode='r')
    
    data_group = root['data']
    meta_group = root['meta']
    
    # Load global ZARR if provided (contains overhead camera view)
    global_data_group = None
    if global_zarr_path and Path(global_zarr_path).exists():
        print(f"Loading global view ZARR from {global_zarr_path}...")
        global_root = zarr.open(global_zarr_path, mode='r')
        global_data_group = global_root['data']
        print(f"  ✓ Global ZARR loaded (overhead camera view)")
    elif global_zarr_path:
        print(f"  WARNING: Global ZARR not found at {global_zarr_path}, will duplicate head_camera")
    else:
        # Try to auto-detect global ZARR path
        # Agent ZARR: LiftBarrier-rf_Agent0_5.zarr
        # Global ZARR: LiftBarrier-rf_global_5.zarr (no agent ID!)
        zarr_stem = Path(zarr_path).stem  # e.g., "LiftBarrier-rf_Agent0_5"
        
        # Extract task name and number
        # Pattern: {task}_Agent{id}_{num} -> {task}_global_{num}
        match = re.match(r'(.+)_Agent\d+_(\d+)$', zarr_stem)
        if match:
            task_name_part = match.group(1)  # "LiftBarrier-rf"
            num_part = match.group(2)         # "5"
            global_zarr_name = f"{task_name_part}_global_{num_part}.zarr"
            auto_global_path = Path(zarr_path).parent / global_zarr_name
            if auto_global_path.exists():
                print(f"Auto-detected global ZARR: {auto_global_path}")
                global_root = zarr.open(str(auto_global_path), mode='r')
                global_data_group = global_root['data']
                print(f"  ✓ Global ZARR loaded (overhead camera view)")
            else:
                print(f"  INFO: No global ZARR found at {auto_global_path}, will duplicate head_camera")
        else:
            print(f"  INFO: Could not parse ZARR name pattern, will duplicate head_camera")
    
    # Get episode boundaries
    episode_ends = np.array(meta_group['episode_ends'])
    
    # Check which camera format is used
    has_separate_cameras = 'head_camera' in data_group
    
    if has_separate_cameras:
        first_img = data_group['head_camera'][0]
        if first_img.shape[0] == 3:
            img_h, img_w = first_img.shape[1], first_img.shape[2]
            is_chw = True
        else:
            img_h, img_w = first_img.shape[0], first_img.shape[1]
            is_chw = False
        print(f"Found separate camera arrays with resolution {img_h}x{img_w}")
    else:
        first_img = data_group['img'][0]
        if first_img.shape[1] == 3:
            img_h, img_w = first_img.shape[2], first_img.shape[3]
            is_chw = True
        else:
            img_h, img_w = first_img.shape[1], first_img.shape[2]
            is_chw = False
        print(f"Found img array with resolution {img_h}x{img_w}")
    
    # Get action and state dimensions
    action_dim = data_group['action'].shape[-1]
    state_key = 'state' if 'state' in data_group else 'agent_pos'
    state_dim = data_group[state_key].shape[-1]
    
    # Prepare output
    repo_id = f"{task_name}_Agent{agent_id}_{num_episodes}"
    output_path = Path(output_dir) / repo_id
    
    # Remove existing output if it exists
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Clean HuggingFace cache for this repo to avoid conflicts
    cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if cache_path.exists():
        print(f"Cleaning cache at {cache_path}")
        shutil.rmtree(cache_path)
    
    print(f"Creating LeRobot dataset: {repo_id}")
    print(f"  Action dim: {action_dim}, State dim: {state_dim}")
    
    # Create LeRobot dataset with Pi0's 3-camera format
    # Use single-threaded image writing to prevent parquet corruption
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "base_0_rgb": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_0_rgb": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_0_rgb": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=1,
        image_writer_processes=0,
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
            if has_separate_cameras:
                # Read from separate camera arrays
                head_img = data_group['head_camera'][t]
                wrist_img = data_group['wrist_camera'][t]
                
                # Get global view from global ZARR if available
                if global_data_group is not None and 'head_camera' in global_data_group:
                    global_img = global_data_group['head_camera'][t]
                else:
                    global_img = head_img.copy()
                
                # Convert from CHW to HWC if needed
                if is_chw:
                    head_img = np.transpose(head_img, (1, 2, 0))
                    global_img = np.transpose(global_img, (1, 2, 0))
                    wrist_img = np.transpose(wrist_img, (1, 2, 0))
            else:
                # Old format: extract from 'img' array
                img_t = data_group['img'][t]
                
                if is_chw:
                    head_img = np.transpose(img_t[0], (1, 2, 0))
                    wrist_img = np.transpose(img_t[2] if img_t.shape[0] > 2 else img_t[1], (1, 2, 0))
                else:
                    head_img = img_t[0]
                    wrist_img = img_t[2] if img_t.shape[0] > 2 else img_t[1]
                
                # Get global view
                if global_data_group is not None:
                    if 'head_camera' in global_data_group:
                        global_img = global_data_group['head_camera'][t]
                        if is_chw:
                            global_img = np.transpose(global_img, (1, 2, 0))
                    elif 'img' in global_data_group:
                        global_img_t = global_data_group['img'][t]
                        if is_chw:
                            global_img = np.transpose(global_img_t[0], (1, 2, 0))
                        else:
                            global_img = global_img_t[0]
                    else:
                        global_img = head_img.copy()
                else:
                    # No global ZARR, duplicate head
                    if is_chw:
                        global_img = np.transpose(img_t[1] if img_t.shape[0] > 2 else img_t[0], (1, 2, 0))
                    else:
                        global_img = img_t[1] if img_t.shape[0] > 2 else img_t[0]
            
            # Ensure uint8 format
            head_img = _process_image(head_img)
            global_img = _process_image(global_img)
            wrist_img = _process_image(wrist_img)
            
            # Get state/action
            state_data = data_group[state_key][t]
            action_data = data_group['action'][t]
            
            frame_data = {
                "base_0_rgb": head_img,
                "left_wrist_0_rgb": global_img,
                "right_wrist_0_rgb": wrist_img,
                "state": state_data.astype(np.float32),
                "actions": action_data.astype(np.float32),
                "task": language_instruction,
            }
            
            dataset.add_frame(frame_data)
        
        dataset.save_episode()
        
        # Ensure all writes are flushed
        if hasattr(dataset, 'image_writer') and dataset.image_writer is not None:
            try:
                dataset.image_writer.wait_until_done()
            except:
                pass
        
        start_idx = end_idx
    
    # Final flush to ensure all data is written
    if hasattr(dataset, 'consolidate'):
        try:
            dataset.consolidate()
        except:
            pass
    
    print(f"✅ Successfully converted {len(episode_ends)} episodes to LeRobot format")
    
    # Move dataset from cache to output directory
    cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    
    if cache_path.exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"   Moving dataset from {cache_path} to {output_path}")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(cache_path), str(output_path))
    
    # Consolidate parquet files in data/chunk-000 to prevent corruption issues
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    data_chunk_dir = output_path / "data" / "chunk-000"
    actual_total_frames = None
    if data_chunk_dir.exists():
        parquet_files = sorted(data_chunk_dir.glob("*.parquet"))
        if len(parquet_files) >= 1:
            # Read the parquet to get actual row count
            main_table = pq.read_table(parquet_files[0])
            actual_total_frames = len(main_table)
            
            if len(parquet_files) > 1:
                print(f"   Consolidating {len(parquet_files)} parquet files into one...")
                # Read all parquet files and merge
                tables = [main_table]
                for pf in parquet_files[1:]:
                    try:
                        tables.append(pq.read_table(pf))
                    except Exception as e:
                        print(f"   Warning: Could not read {pf.name}: {e}, skipping")
                
                # Merge all tables
                merged_table = pa.concat_tables(tables)
                actual_total_frames = len(merged_table)
                
                # Remove old files
                for pf in parquet_files:
                    pf.unlink()
                
                # Write merged file
                merged_path = data_chunk_dir / "file-000.parquet"
                pq.write_table(merged_table, merged_path)
                print(f"   ✓ Consolidated into single file: {merged_path.name} ({actual_total_frames} rows)")
    
    # Update info.json with actual total_frames if there was data loss
    if actual_total_frames is not None:
        info_path = output_path / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            if info.get('total_frames') != actual_total_frames:
                print(f"   Updating info.json: total_frames {info.get('total_frames')} -> {actual_total_frames}")
                info['total_frames'] = actual_total_frames
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=4)
    
    # Create meta/episodes directory with parquet file (required by LeRobot v3.0+)
    episodes_chunk_dir = output_path / "meta" / "episodes" / "chunk-000"
    episodes_chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate episode boundaries for the parquet file
    # Use actual_total_frames if available (in case of data loss during consolidation)
    episode_data = []
    start_idx = 0
    
    if actual_total_frames is not None and actual_total_frames < episode_ends[-1]:
        # Data was lost - recalculate episode boundaries based on available data
        print(f"   Recalculating episode boundaries (actual frames: {actual_total_frames}, expected: {episode_ends[-1]})")
        remaining_frames = actual_total_frames
        for ep_idx, end_idx in enumerate(episode_ends):
            original_length = end_idx - start_idx
            if remaining_frames <= 0:
                break
            actual_length = min(original_length, remaining_frames)
            actual_end = start_idx + actual_length
            episode_data.append({
                "episode_index": ep_idx,
                "tasks": [language_instruction],
                "length": actual_length,
                "dataset_from_index": start_idx,
                "dataset_to_index": actual_end,
            })
            remaining_frames -= actual_length
            start_idx = actual_end
    else:
        # Normal case - use original episode boundaries
        for ep_idx, end_idx in enumerate(episode_ends):
            episode_length = end_idx - start_idx
            episode_data.append({
                "episode_index": ep_idx,
                "tasks": [language_instruction],
                "length": episode_length,
                "dataset_from_index": start_idx,
                "dataset_to_index": end_idx,
            })
            start_idx = end_idx
    
    table = pa.Table.from_pylist(episode_data)
    parquet_path = episodes_chunk_dir / "file-000.parquet"
    pq.write_table(table, parquet_path)
    print(f"   ✓ Created meta/episodes/chunk-000/file-000.parquet with {len(episode_data)} episodes")
    
    print(f"   Output: {output_path}")
    return repo_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ZARR to LeRobot format for Pi0")
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--global_zarr_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/lerobot_data")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--agent_id", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    
    args = parser.parse_args()
    
    convert_zarr_to_lerobot(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        task_name=args.task_name,
        agent_id=args.agent_id,
        num_episodes=args.num_episodes,
        global_zarr_path=args.global_zarr_path,
    )
