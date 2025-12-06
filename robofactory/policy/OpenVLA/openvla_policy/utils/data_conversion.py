"""
Data conversion utilities for converting ZARR format to RLDS format.

This module provides functions to convert RoboFactory ZARR datasets to RLDS format
compatible with OpenVLA training pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import zarr
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import concurrent.futures
from functools import lru_cache


def load_zarr_data(zarr_path: str) -> Dict[str, np.ndarray]:
    """
    Load data from ZARR file.
    
    Args:
        zarr_path: Path to ZARR dataset
        
    Returns:
        Dictionary containing loaded data arrays
    """
    root = zarr.open(zarr_path, mode='r')
    
    data = {}
    # Load metadata
    if 'meta' in root:
        meta = root['meta']
        data['episode_ends'] = np.array(meta['episode_ends'])
    
    # Load data arrays
    if 'data' in root:
        data_group = root['data']
        for key in data_group.keys():
            data[key] = np.array(data_group[key])
    
    return data


def create_dataset_statistics(
    zarr_path: str,
    output_path: str,
    action_key: str = 'action',
    state_key: str = 'state'
) -> Dict:
    """
    Create dataset statistics JSON file for normalization.
    
    Args:
        zarr_path: Path to ZARR dataset
        output_path: Path to save statistics JSON
        action_key: Key for action data
        state_key: Key for state/proprio data
        
    Returns:
        Dictionary containing dataset statistics
    """
    data = load_zarr_data(zarr_path)
    
    statistics = {}
    
    # Compute action statistics
    if action_key in data:
        actions = data[action_key]
        statistics['action'] = {
            'mean': actions.mean(axis=0).tolist(),
            'std': actions.std(axis=0).tolist(),
            'min': actions.min(axis=0).tolist(),
            'max': actions.max(axis=0).tolist(),
            'q01': np.percentile(actions, 1, axis=0).tolist(),
            'q99': np.percentile(actions, 99, axis=0).tolist(),
        }
    
    # Compute proprio/state statistics  
    if state_key in data:
        states = data[state_key]
        statistics['proprio'] = {
            'mean': states.mean(axis=0).tolist(),
            'std': states.std(axis=0).tolist(),
            'min': states.min(axis=0).tolist(),
            'max': states.max(axis=0).tolist(),
        }
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Saved dataset statistics to {output_path}")
    return statistics


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


def convert_zarr_to_rlds(
    zarr_path: str,
    output_dir: str,
    task_name: str,
    agent_id: int,
    language_instruction: Optional[str] = None,
    action_key: str = 'action',
    state_key: str = 'state'
) -> str:
    """
    Convert ZARR dataset to RLDS format for OpenVLA.
    Includes both wrist_camera (gripper view) and head_camera (side view).
    
    Args:
        zarr_path: Path to input ZARR dataset
        output_dir: Directory to save RLDS dataset
        task_name: Name of the task
        agent_id: Agent ID
        language_instruction: Optional language instruction for the task
        action_key: Key for actions
        state_key: Key for robot state/proprio
        
    Returns:
        Path to created RLDS dataset
    """
    # Load ZARR data
    print(f"Loading ZARR data from {zarr_path}...")
    data = load_zarr_data(zarr_path)
    
    # Check available cameras
    has_wrist_camera = 'wrist_camera' in data
    has_head_camera = 'head_camera' in data
    
    print(f"Available cameras - wrist_camera (gripper): {has_wrist_camera}, head_camera (side): {has_head_camera}")
    
    # Get episode boundaries
    episode_ends = data.get('episode_ends', None)
    if episode_ends is None:
        # Assume single episode if no episode_ends
        total_steps = len(data[action_key])
        episode_ends = np.array([total_steps])
    
    # Prepare output directory
    dataset_name = f"{task_name}_Agent{agent_id}"
    output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Set default language instruction if not provided
    if language_instruction is None:
        language_instruction = f"Complete the {task_name.replace('-rf', '')} task"
    
    print(f"Converting {len(episode_ends)} episodes to RLDS format...")
    
    # Create RLDS dataset builder
    episodes = []
    start_idx = 0
    
    for ep_idx, end_idx in enumerate(tqdm(episode_ends, desc="Converting episodes")):
        episode_data = {
            'steps': []
        }
        
        # Extract episode data
        for step_idx in range(start_idx, end_idx):
            step = {
                'observation': {},
                'action': data[action_key][step_idx].astype(np.float32),
                'is_first': step_idx == start_idx,
                'is_last': step_idx == end_idx - 1,
                'is_terminal': step_idx == end_idx - 1,
                'language_instruction': language_instruction,
            }
            
            # Add wrist_camera (gripper view) as primary image
            if has_wrist_camera:
                wrist_image = _process_image(data['wrist_camera'][step_idx])
                step['observation']['wrist_image'] = wrist_image
            
            # Add head_camera (side view) as secondary image
            if has_head_camera:
                head_image = _process_image(data['head_camera'][step_idx])
                step['observation']['side_image'] = head_image
            
            # For backward compatibility, also set 'image' to wrist if available, else head
            if has_wrist_camera:
                step['observation']['image'] = step['observation']['wrist_image']
            elif has_head_camera:
                step['observation']['image'] = step['observation']['side_image']
            
            # Add proprio/state observation
            if state_key in data:
                step['observation']['proprio'] = data[state_key][step_idx].astype(np.float32)
            
            episode_data['steps'].append(step)
        
        episodes.append(episode_data)
        start_idx = end_idx
    
    # Save as TFRecord format (RLDS compatible)
    tfrecord_path = os.path.join(output_path, 'train.tfrecord')
    
    print(f"Saving to TFRecord format at {tfrecord_path}...")
    _save_episodes_as_tfrecord(episodes, tfrecord_path)
    
    # Create dataset_info.json
    camera_info = []
    if has_wrist_camera:
        camera_info.append('wrist_image (gripper view)')
    if has_head_camera:
        camera_info.append('side_image (side view)')
    
    info = {
        'name': dataset_name,
        'version': '1.0.0',
        'description': f'RoboFactory task {task_name} for agent {agent_id}',
        'num_episodes': len(episodes),
        'num_transitions': int(episode_ends[-1]),
        'action_dim': data[action_key].shape[-1],
        'cameras': camera_info,
        'has_wrist_camera': has_wrist_camera,
        'has_side_camera': has_head_camera,
    }
    
    info_path = os.path.join(output_path, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Successfully converted dataset to {output_path}")
    return output_path


def convert_zarr_to_rlds_global(
    zarr_path: str,
    output_dir: str,
    task_name: str,
    language_instruction: Optional[str] = None,
    image_key: str = 'head_camera',
) -> str:
    """
    Convert global camera ZARR dataset to RLDS format.
    Global camera data has no actions, only observations.
    
    Args:
        zarr_path: Path to input ZARR dataset
        output_dir: Directory to save RLDS dataset
        task_name: Name of the task
        language_instruction: Optional language instruction for the task
        image_key: Key for camera images
        
    Returns:
        Path to created RLDS dataset
    """
    # Load ZARR data
    print(f"Loading global ZARR data from {zarr_path}...")
    data = load_zarr_data(zarr_path)
    
    # Get episode boundaries
    episode_ends = data.get('episode_ends', None)
    if episode_ends is None:
        # Assume single episode if no episode_ends
        total_steps = len(data[image_key])
        episode_ends = np.array([total_steps])
    
    # Prepare output directory
    dataset_name = f"{task_name}_global"
    output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Set default language instruction if not provided
    if language_instruction is None:
        language_instruction = f"Observe the {task_name.replace('-rf', '')} task from global view"
    
    print(f"Converting {len(episode_ends)} episodes to RLDS format (global view)...")
    
    # Create RLDS dataset builder
    episodes = []
    start_idx = 0
    
    for ep_idx, end_idx in enumerate(tqdm(episode_ends, desc="Converting episodes")):
        episode_data = {
            'steps': []
        }
        
        # Extract episode data
        for step_idx in range(start_idx, end_idx):
            step = {
                'observation': {},
                'is_first': step_idx == start_idx,
                'is_last': step_idx == end_idx - 1,
                'is_terminal': step_idx == end_idx - 1,
                'language_instruction': language_instruction,
            }
            
            # Add global image observation
            if image_key in data:
                image = _process_image(data[image_key][step_idx])
                step['observation']['global_image'] = image
                step['observation']['image'] = image  # backward compatibility
            
            episode_data['steps'].append(step)
        
        episodes.append(episode_data)
        start_idx = end_idx
    
    # Save as TFRecord format (RLDS compatible)
    tfrecord_path = os.path.join(output_path, 'train.tfrecord')
    
    print(f"Saving to TFRecord format at {tfrecord_path}...")
    _save_episodes_as_tfrecord_global(episodes, tfrecord_path)
    
    # Create dataset_info.json
    info = {
        'name': dataset_name,
        'version': '1.0.0',
        'description': f'RoboFactory task {task_name} global camera view',
        'num_episodes': len(episodes),
        'num_transitions': int(episode_ends[-1]),
        'is_global_view': True,
        'cameras': ['global_image (global overhead view)'],
        'has_actions': False,
    }
    
    info_path = os.path.join(output_path, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Successfully converted global dataset to {output_path}")
    return output_path


def _encode_image_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    """Encode image to JPEG bytes (much faster than PNG)."""
    return tf.io.encode_jpeg(image, quality=quality).numpy()


def _save_episodes_as_tfrecord_global(episodes: List[Dict], output_path: str, 
                                       num_workers: int = 16, jpeg_quality: int = 95):
    """
    Save global camera episodes as TFRecord format (no actions).
    Uses parallel JPEG encoding for speed.
    
    Args:
        episodes: List of episode dictionaries
        output_path: Path to save TFRecord file
        num_workers: Number of parallel workers for image encoding
        jpeg_quality: JPEG quality (0-100)
    """
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    # Collect all steps and images for parallel encoding
    all_steps = []
    all_images = []  # (step_idx, image_key, image_data)
    
    for episode in episodes:
        for step in episode['steps']:
            step_idx = len(all_steps)
            all_steps.append(step)
            
            if 'global_image' in step['observation']:
                all_images.append((step_idx, 'global_image', step['observation']['global_image']))
            if 'image' in step['observation']:
                # Only add if different from global_image
                if 'global_image' not in step['observation']:
                    all_images.append((step_idx, 'image', step['observation']['image']))
    
    # Parallel encode all images
    print(f"  Encoding {len(all_images)} images with {num_workers} workers (JPEG quality={jpeg_quality})...")
    
    def encode_single(args):
        step_idx, key, image = args
        return (step_idx, key, _encode_image_jpeg(image, jpeg_quality))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        encoded_results = list(executor.map(encode_single, all_images))
    
    # Build lookup for encoded images
    encoded_lookup = {}
    for step_idx, key, encoded_bytes in encoded_results:
        if step_idx not in encoded_lookup:
            encoded_lookup[step_idx] = {}
        encoded_lookup[step_idx][key] = encoded_bytes
    
    # Cache for language instructions
    instruction_cache = {}
    
    # Write TFRecord
    print(f"  Writing TFRecord with {len(all_steps)} steps...")
    with tf.io.TFRecordWriter(output_path) as writer:
        for step_idx, step in enumerate(all_steps):
            # Cache language instruction encoding
            instruction = step['language_instruction']
            if instruction not in instruction_cache:
                instruction_cache[instruction] = instruction.encode('utf-8')
            
            feature = {
                'is_first': _int64_feature(int(step['is_first'])),
                'is_last': _int64_feature(int(step['is_last'])),
                'is_terminal': _int64_feature(int(step['is_terminal'])),
                'language_instruction': _bytes_feature(instruction_cache[instruction]),
            }
            
            # Add pre-encoded images
            if step_idx in encoded_lookup:
                if 'global_image' in encoded_lookup[step_idx]:
                    feature['observation/global_image'] = _bytes_feature(
                        encoded_lookup[step_idx]['global_image']
                    )
                    # Also use for backward compatible 'image' key
                    feature['observation/image'] = _bytes_feature(
                        encoded_lookup[step_idx]['global_image']
                    )
                elif 'image' in encoded_lookup[step_idx]:
                    feature['observation/image'] = _bytes_feature(
                        encoded_lookup[step_idx]['image']
                    )
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def _save_episodes_as_tfrecord(episodes: List[Dict], output_path: str,
                               num_workers: int = 16, jpeg_quality: int = 95):
    """
    Save episodes as TFRecord format with multiple camera views.
    Uses parallel JPEG encoding for speed.
    
    Args:
        episodes: List of episode dictionaries
        output_path: Path to save TFRecord file
        num_workers: Number of parallel workers for image encoding
        jpeg_quality: JPEG quality (0-100)
    """
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    # Collect all steps and images for parallel encoding
    all_steps = []
    all_images = []  # (step_idx, image_key, image_data)
    
    for episode in episodes:
        for step in episode['steps']:
            step_idx = len(all_steps)
            all_steps.append(step)
            
            # Collect all camera images
            if 'wrist_image' in step['observation']:
                all_images.append((step_idx, 'wrist_image', step['observation']['wrist_image']))
            if 'side_image' in step['observation']:
                all_images.append((step_idx, 'side_image', step['observation']['side_image']))
            if 'image' in step['observation']:
                # Only add if not already covered by wrist_image
                if 'wrist_image' not in step['observation']:
                    all_images.append((step_idx, 'image', step['observation']['image']))
    
    # Parallel encode all images
    print(f"  Encoding {len(all_images)} images with {num_workers} workers (JPEG quality={jpeg_quality})...")
    
    def encode_single(args):
        step_idx, key, image = args
        return (step_idx, key, _encode_image_jpeg(image, jpeg_quality))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        encoded_results = list(executor.map(encode_single, all_images))
    
    # Build lookup for encoded images
    encoded_lookup = {}
    for step_idx, key, encoded_bytes in encoded_results:
        if step_idx not in encoded_lookup:
            encoded_lookup[step_idx] = {}
        encoded_lookup[step_idx][key] = encoded_bytes
    
    # Cache for language instructions
    instruction_cache = {}
    
    # Write TFRecord
    print(f"  Writing TFRecord with {len(all_steps)} steps...")
    with tf.io.TFRecordWriter(output_path) as writer:
        for step_idx, step in enumerate(all_steps):
            # Cache language instruction encoding
            instruction = step['language_instruction']
            if instruction not in instruction_cache:
                instruction_cache[instruction] = instruction.encode('utf-8')
            
            feature = {
                'action': _float_feature(step['action'].flatten()),
                'is_first': _int64_feature(int(step['is_first'])),
                'is_last': _int64_feature(int(step['is_last'])),
                'is_terminal': _int64_feature(int(step['is_terminal'])),
                'language_instruction': _bytes_feature(instruction_cache[instruction]),
            }
            
            # Add pre-encoded images
            if step_idx in encoded_lookup:
                if 'wrist_image' in encoded_lookup[step_idx]:
                    feature['observation/wrist_image'] = _bytes_feature(
                        encoded_lookup[step_idx]['wrist_image']
                    )
                    # Also use for backward compatible 'image' key
                    feature['observation/image'] = _bytes_feature(
                        encoded_lookup[step_idx]['wrist_image']
                    )
                if 'side_image' in encoded_lookup[step_idx]:
                    feature['observation/side_image'] = _bytes_feature(
                        encoded_lookup[step_idx]['side_image']
                    )
                if 'image' in encoded_lookup[step_idx] and 'wrist_image' not in encoded_lookup[step_idx]:
                    feature['observation/image'] = _bytes_feature(
                        encoded_lookup[step_idx]['image']
                    )
            
            # Add proprio/state
            if 'proprio' in step['observation']:
                feature['observation/proprio'] = _float_feature(
                    step['observation']['proprio'].flatten()
                )
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def batch_convert_zarr_to_rlds(
    zarr_dir: str,
    output_dir: str,
    task_instructions: Optional[Dict[str, str]] = None
):
    """
    Batch convert all ZARR datasets in a directory to RLDS format.
    
    Args:
        zarr_dir: Directory containing ZARR datasets
        output_dir: Directory to save RLDS datasets
        task_instructions: Dictionary mapping task names to language instructions
    """
    zarr_dir = Path(zarr_dir)
    zarr_files = list(zarr_dir.glob("*.zarr"))
    
    if task_instructions is None:
        task_instructions = {}
    
    print(f"Found {len(zarr_files)} ZARR datasets to convert")
    
    for zarr_file in zarr_files:
        # Parse filename: TaskName-rf_AgentN_M.zarr
        stem = zarr_file.stem
        parts = stem.split('_')
        
        if len(parts) >= 3:
            task_name = parts[0]  # e.g., "LiftBarrier-rf"
            agent_str = parts[1]  # e.g., "Agent0"
            
            # Extract agent ID
            agent_id = int(agent_str.replace('Agent', ''))
            
            # Get language instruction
            instruction = task_instructions.get(
                task_name,
                f"Complete the {task_name.replace('-rf', '')} task"
            )
            
            print(f"\nConverting {zarr_file.name}...")
            try:
                output_path = convert_zarr_to_rlds(
                    zarr_path=str(zarr_file),
                    output_dir=output_dir,
                    task_name=task_name,
                    agent_id=agent_id,
                    language_instruction=instruction
                )
                
                # Also create statistics file
                stats_path = os.path.join(output_path, 'statistics.json')
                create_dataset_statistics(
                    zarr_path=str(zarr_file),
                    output_path=stats_path
                )
                
            except Exception as e:
                print(f"Error converting {zarr_file.name}: {e}")
                continue
    
    print(f"\nBatch conversion complete! RLDS datasets saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ZARR to RLDS format")
    parser.add_argument(
        "--zarr_path",
        type=str,
        help="Path to ZARR file or directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/rlds_data",
        help="Output directory for RLDS datasets"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all ZARR files in directory"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Define task instructions
        task_instructions = {
            'LiftBarrier-rf': 'Lift the barrier together with the other robot',
            'StackCube-rf': 'Stack the cube on the target location',
            'TakePhoto-rf': 'Take a photo of the target object',
            'PassShoe-rf': 'Pass the shoe to the other robot',
            'PlaceFood-rf': 'Place the food on the plate',
            'CameraAlignment-rf': 'Align the camera with the target',
        }
        
        batch_convert_zarr_to_rlds(
            zarr_dir=args.zarr_path,
            output_dir=args.output_dir,
            task_instructions=task_instructions
        )
    else:
        # Single file conversion
        print("Please provide task details for single file conversion")

