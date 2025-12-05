#!/usr/bin/env python3
"""
RLDS Data Visualization Tool

Usage:
    python visualize_rlds.py --data_dir data/rlds_data/CameraAlignment-rf_Agent0
    python visualize_rlds.py --data_dir data/rlds_data/CameraAlignment-rf_Agent0 --save_video
    python visualize_rlds.py --data_dir data/rlds_data/CameraAlignment-rf_Agent0 --episode 0 --show_actions
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_dataset_info(data_dir: Path) -> Dict:
    """Load dataset info from JSON."""
    info_path = data_dir / 'dataset_info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def load_statistics(data_dir: Path) -> Dict:
    """Load dataset statistics."""
    stats_path = data_dir / 'statistics.json'
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {}


def load_episodes(data_dir: Path) -> List[List[Dict]]:
    """Load all episodes from TFRecord file with multiple camera views."""
    tfrecord_path = data_dir / 'train.tfrecord'
    
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
    
    # First pass: detect available features
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    sample_record = next(iter(dataset))
    example = tf.train.Example()
    example.ParseFromString(sample_record.numpy())
    available_features = set(example.features.feature.keys())
    
    # Build feature description based on available features
    feature_description = {
        'is_first': tf.io.FixedLenFeature([], tf.int64),
        'is_last': tf.io.FixedLenFeature([], tf.int64),
        'is_terminal': tf.io.FixedLenFeature([], tf.int64),
        'language_instruction': tf.io.FixedLenFeature([], tf.string),
    }
    
    # Optional features
    if 'action' in available_features:
        feature_description['action'] = tf.io.VarLenFeature(tf.float32)
    if 'observation/proprio' in available_features:
        feature_description['observation/proprio'] = tf.io.VarLenFeature(tf.float32)
    
    # Camera observations - check for all possible camera types
    camera_keys = []
    for key in ['observation/image', 'observation/wrist_image', 'observation/side_image', 'observation/global_image']:
        if key in available_features:
            feature_description[key] = tf.io.FixedLenFeature([], tf.string)
            camera_keys.append(key)
    
    print(f"Detected cameras: {camera_keys}")
    
    # Parse dataset
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    
    episodes = []
    current_episode = []
    
    for raw_record in dataset:
        example = tf.io.parse_single_example(raw_record, feature_description)
        
        # Decode data
        step = {
            'instruction': example['language_instruction'].numpy().decode('utf-8'),
            'is_first': bool(example['is_first'].numpy()),
            'is_last': bool(example['is_last'].numpy()),
        }
        
        # Decode action if available
        if 'action' in feature_description:
            step['action'] = tf.sparse.to_dense(example['action']).numpy().astype(np.float32)
        else:
            step['action'] = np.array([])
        
        # Decode proprio if available
        if 'observation/proprio' in feature_description:
            step['proprio'] = tf.sparse.to_dense(example['observation/proprio']).numpy().astype(np.float32)
        else:
            step['proprio'] = np.array([])
        
        # Decode all available camera images
        for key in camera_keys:
            short_key = key.replace('observation/', '')
            step[short_key] = tf.io.decode_png(example[key].numpy()).numpy()
        
        # Set default 'image' key for backward compatibility
        if 'wrist_image' in step:
            step['image'] = step['wrist_image']
        elif 'side_image' in step:
            step['image'] = step['side_image']
        elif 'global_image' in step:
            step['image'] = step['global_image']
        elif 'image' in step:
            pass  # Already set
        
        current_episode.append(step)
        
        if step['is_last']:
            episodes.append(current_episode)
            current_episode = []
    
    return episodes


def print_dataset_summary(data_dir: Path, episodes: List, info: Dict, stats: Dict):
    """Print dataset summary."""
    print("\n" + "="*60)
    print("RLDS DATASET SUMMARY")
    print("="*60)
    
    print(f"\nDataset Path: {data_dir}")
    
    if info:
        print(f"\n📊 Dataset Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    print(f"\n📁 Loaded Data:")
    print(f"   Episodes: {len(episodes)}")
    total_steps = sum(len(ep) for ep in episodes)
    print(f"   Total steps: {total_steps}")
    
    if episodes:
        ep_lengths = [len(ep) for ep in episodes]
        print(f"   Episode length - Min: {min(ep_lengths)}, Max: {max(ep_lengths)}, Avg: {np.mean(ep_lengths):.1f}")
        
        sample = episodes[0][0]
        
        # Show available cameras
        camera_keys = ['wrist_image', 'side_image', 'global_image', 'image']
        available_cameras = [k for k in camera_keys if k in sample and sample[k] is not None]
        print(f"\n📷 Available Cameras:")
        for cam in available_cameras:
            if cam in sample:
                print(f"   {cam}: shape {sample[cam].shape}")
        
        print(f"\n🎮 Action dim: {len(sample['action'])}")
        print(f"🤖 Proprio dim: {len(sample['proprio'])}")
        print(f"📝 Instruction: \"{sample['instruction']}\"")
    
    if stats:
        print(f"\n📈 Action Statistics:")
        if 'action' in stats:
            action_stats = stats['action']
            print(f"   Mean: {[f'{x:.3f}' for x in action_stats.get('mean', [])]}")
            print(f"   Std:  {[f'{x:.3f}' for x in action_stats.get('std', [])]}")
            print(f"   Min:  {[f'{x:.3f}' for x in action_stats.get('min', [])]}")
            print(f"   Max:  {[f'{x:.3f}' for x in action_stats.get('max', [])]}")
    
    print("\n" + "="*60 + "\n")


def visualize_episode(episodes: List, episode_idx: int = 0, show_actions: bool = True):
    """Visualize frames from a single episode."""
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found. Only {len(episodes)} episodes available.")
        return
    
    episode = episodes[episode_idx]
    n_frames = len(episode)
    
    # Sample frames evenly
    n_show = min(8, n_frames)
    indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
    
    if show_actions:
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, n_show, height_ratios=[3, 1])
    else:
        fig, axes = plt.subplots(1, n_show, figsize=(16, 4))
        axes = [axes] if n_show == 1 else list(axes)
    
    fig.suptitle(f'Episode {episode_idx} - "{episode[0]["instruction"]}" ({n_frames} steps)', fontsize=12)
    
    for i, idx in enumerate(indices):
        step = episode[idx]
        
        # Image subplot
        if show_actions:
            ax_img = fig.add_subplot(gs[0, i])
        else:
            ax_img = axes[i]
        
        ax_img.imshow(step['image'])
        ax_img.set_title(f'Step {idx}', fontsize=10)
        ax_img.axis('off')
        
        # Action subplot
        if show_actions:
            ax_act = fig.add_subplot(gs[1, i])
            action = step['action']
            colors = plt.cm.viridis(np.linspace(0, 1, len(action)))
            ax_act.bar(range(len(action)), action, color=colors, width=0.8)
            ax_act.set_ylim(-3, 3)
            ax_act.set_xlabel('Action dim', fontsize=8)
            if i == 0:
                ax_act.set_ylabel('Value', fontsize=8)
            ax_act.tick_params(axis='both', labelsize=7)
    
    plt.tight_layout()
    plt.show()


def visualize_image_grid(episodes: List, n_episodes: int = 4, n_frames_per_ep: int = 4):
    """Show a grid of images from multiple episodes."""
    n_episodes = min(n_episodes, len(episodes))
    
    fig, axes = plt.subplots(n_episodes, n_frames_per_ep, figsize=(12, 3 * n_episodes))
    if n_episodes == 1:
        axes = [axes]
    
    fig.suptitle('Sample Frames from Multiple Episodes', fontsize=14)
    
    for ep_idx in range(n_episodes):
        episode = episodes[ep_idx]
        indices = np.linspace(0, len(episode) - 1, n_frames_per_ep, dtype=int)
        
        for frame_idx, step_idx in enumerate(indices):
            ax = axes[ep_idx][frame_idx] if n_episodes > 1 else axes[frame_idx]
            ax.imshow(episode[step_idx]['image'])
            if frame_idx == 0:
                ax.set_ylabel(f'Ep {ep_idx}', fontsize=10)
            if ep_idx == 0:
                ax.set_title(f'Step {step_idx}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_action_distribution(episodes: List):
    """Visualize action value distributions."""
    # Collect all actions
    all_actions = []
    for episode in episodes:
        for step in episode:
            all_actions.append(step['action'])
    
    actions = np.array(all_actions)
    n_dims = actions.shape[1]
    
    fig, axes = plt.subplots(2, (n_dims + 1) // 2, figsize=(14, 6))
    axes = axes.flatten()
    
    fig.suptitle('Action Value Distributions', fontsize=14)
    
    dim_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7', 'Gripper']
    
    for i in range(n_dims):
        ax = axes[i]
        ax.hist(actions[:, i], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        name = dim_names[i] if i < len(dim_names) else f'Dim {i}'
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        
        # Add mean line
        mean_val = np.mean(actions[:, i])
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_episode_video(episodes: List, episode_idx: int, output_path: str, fps: int = 10, 
                       camera_view: str = 'all'):
    """
    Save episode as video file.
    
    Args:
        episodes: List of episodes
        episode_idx: Episode index to save
        output_path: Output path for video
        fps: Frames per second
        camera_view: 'all' for side-by-side, or specific camera ('wrist_image', 'side_image', 'global_image')
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return
    
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found.")
        return
    
    episode = episodes[episode_idx]
    sample_step = episode[0]
    
    # Detect available cameras
    camera_keys = ['wrist_image', 'side_image', 'global_image']
    available_cameras = [k for k in camera_keys if k in sample_step and sample_step[k] is not None]
    
    # Fallback to default 'image' if no specific cameras
    if not available_cameras and 'image' in sample_step:
        available_cameras = ['image']
    
    if camera_view != 'all' and camera_view in available_cameras:
        # Single camera view
        sample_img = sample_step[camera_view]
        height, width = sample_img.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for step in episode:
            frame = cv2.cvtColor(step[camera_view], cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        print(f"Saved {camera_view} video to: {output_path}")
    else:
        # All cameras side by side
        if len(available_cameras) == 0:
            print("No camera views available in the dataset.")
            return
        
        # Get dimensions for combined frame
        sample_img = sample_step[available_cameras[0]]
        height, width = sample_img.shape[:2]
        combined_width = width * len(available_cameras)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))
        
        for step in episode:
            frames = []
            for cam in available_cameras:
                if cam in step:
                    frame = cv2.cvtColor(step[cam], cv2.COLOR_RGB2BGR)
                    # Add camera label
                    label = cam.replace('_image', '').replace('_', ' ').title()
                    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    frames.append(frame)
            
            if frames:
                combined = np.hstack(frames)
                out.write(combined)
        
        out.release()
        print(f"Saved combined video ({', '.join(available_cameras)}) to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize RLDS dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to RLDS dataset directory')
    parser.add_argument('--episode', type=int, default=0,
                        help='Episode index to visualize (default: 0)')
    parser.add_argument('--show_actions', action='store_true',
                        help='Show action values alongside images')
    parser.add_argument('--show_grid', action='store_true',
                        help='Show image grid from multiple episodes')
    parser.add_argument('--show_distribution', action='store_true',
                        help='Show action value distributions')
    parser.add_argument('--save_video', action='store_true',
                        help='Save episode as video')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='Video FPS (default: 10)')
    parser.add_argument('--camera_view', type=str, default='all',
                        choices=['all', 'wrist_image', 'side_image', 'global_image'],
                        help='Camera view for video: all (side-by-side), wrist_image (gripper), side_image, or global_image')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only print dataset summary, no visualization')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
    
    # Load data
    print(f"Loading RLDS data from: {data_dir}")
    info = load_dataset_info(data_dir)
    stats = load_statistics(data_dir)
    episodes = load_episodes(data_dir)
    
    # Print summary
    print_dataset_summary(data_dir, episodes, info, stats)
    
    if args.summary_only:
        return
    
    # Visualizations
    if args.show_grid:
        visualize_image_grid(episodes)
    
    if args.show_distribution:
        visualize_action_distribution(episodes)
    
    if args.save_video:
        video_path = str(data_dir / f'episode_{args.episode}.mp4')
        save_episode_video(episodes, args.episode, video_path, args.video_fps, args.camera_view)
    
    # Always show single episode visualization unless summary_only
    if not args.summary_only:
        visualize_episode(episodes, args.episode, args.show_actions)


if __name__ == "__main__":
    main()