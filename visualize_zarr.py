#!/usr/bin/env python3
"""
ZARR Data Visualization Tool

Usage:
    python visualize_zarr.py --data_path data/zarr_data/CameraAlignment-rf_Agent0_150.zarr
    python visualize_zarr.py --data_path data/zarr_data/CameraAlignment-rf_Agent0_150.zarr --episode 0 --save_video
    python visualize_zarr.py --data_path data/zarr_data/CameraAlignment-rf_Agent0_150.zarr --show_distribution
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr


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
        if 'episode_ends' in meta:
            data['episode_ends'] = np.array(meta['episode_ends'])
    
    # Load data arrays
    if 'data' in root:
        data_group = root['data']
        for key in data_group.keys():
            data[key] = np.array(data_group[key])
            print(f"  Loaded '{key}': shape={data[key].shape}, dtype={data[key].dtype}")
    
    return data


def get_episodes(data: Dict) -> List[Tuple[int, int]]:
    """
    Get episode boundaries as (start, end) tuples.
    
    Args:
        data: Loaded zarr data dictionary
        
    Returns:
        List of (start_idx, end_idx) tuples for each episode
    """
    episode_ends = data.get('episode_ends', None)
    
    if episode_ends is None:
        # Assume single episode
        total_steps = len(data.get('action', data.get('head_camera', [])))
        return [(0, total_steps)]
    
    episodes = []
    start_idx = 0
    for end_idx in episode_ends:
        episodes.append((start_idx, int(end_idx)))
        start_idx = int(end_idx)
    
    return episodes


def print_dataset_summary(zarr_path: str, data: Dict, episodes: List[Tuple[int, int]]):
    """Print dataset summary."""
    print("\n" + "="*60)
    print("ZARR DATASET SUMMARY")
    print("="*60)
    
    print(f"\nDataset Path: {zarr_path}")
    
    print(f"\n📁 Data Contents:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
    
    print(f"\n📊 Episode Statistics:")
    print(f"   Total episodes: {len(episodes)}")
    total_steps = sum(end - start for start, end in episodes)
    print(f"   Total steps: {total_steps}")
    
    ep_lengths = [end - start for start, end in episodes]
    print(f"   Episode length - Min: {min(ep_lengths)}, Max: {max(ep_lengths)}, Avg: {np.mean(ep_lengths):.1f}")
    
    # Image info
    if 'head_camera' in data:
        img_data = data['head_camera']
        print(f"\n🖼️  Image Data:")
        print(f"   Shape: {img_data.shape}")
        if len(img_data.shape) == 4:
            if img_data.shape[1] == 3:  # NCHW format
                print(f"   Format: NCHW (N={img_data.shape[0]}, C={img_data.shape[1]}, H={img_data.shape[2]}, W={img_data.shape[3]})")
            else:  # NHWC format
                print(f"   Format: NHWC (N={img_data.shape[0]}, H={img_data.shape[1]}, W={img_data.shape[2]}, C={img_data.shape[3]})")
        print(f"   Value range: [{img_data.min()}, {img_data.max()}]")
    
    # Action info
    if 'action' in data:
        action_data = data['action']
        print(f"\n🎮 Action Data:")
        print(f"   Shape: {action_data.shape}")
        print(f"   Action dim: {action_data.shape[-1]}")
        print(f"   Mean: {action_data.mean(axis=0)}")
        print(f"   Std:  {action_data.std(axis=0)}")
    
    print("\n" + "="*60 + "\n")


def get_image(data: Dict, idx: int) -> np.ndarray:
    """
    Get image at index, converting from NCHW to HWC if needed.
    
    Args:
        data: Loaded zarr data
        idx: Step index
        
    Returns:
        Image as HWC numpy array
    """
    img = data['head_camera'][idx]
    
    # Convert NCHW to HWC if needed
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    return img


def visualize_episode(data: Dict, episodes: List[Tuple[int, int]], 
                      episode_idx: int = 0, show_actions: bool = True):
    """Visualize frames from a single episode."""
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found. Only {len(episodes)} episodes available.")
        return
    
    start_idx, end_idx = episodes[episode_idx]
    n_frames = end_idx - start_idx
    
    # Sample frames evenly
    n_show = min(8, n_frames)
    step_indices = np.linspace(start_idx, end_idx - 1, n_show, dtype=int)
    
    if show_actions:
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, n_show, height_ratios=[3, 1])
    else:
        fig, axes = plt.subplots(1, n_show, figsize=(16, 4))
        axes = [axes] if n_show == 1 else list(axes)
    
    fig.suptitle(f'Episode {episode_idx} ({n_frames} steps, indices {start_idx}-{end_idx-1})', fontsize=12)
    
    for i, step_idx in enumerate(step_indices):
        # Image subplot
        if show_actions:
            ax_img = fig.add_subplot(gs[0, i])
        else:
            ax_img = axes[i]
        
        if 'head_camera' in data:
            img = get_image(data, step_idx)
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, 'No image', ha='center', va='center')
        
        ax_img.set_title(f'Step {step_idx}', fontsize=10)
        ax_img.axis('off')
        
        # Action subplot
        if show_actions and 'action' in data:
            ax_act = fig.add_subplot(gs[1, i])
            action = data['action'][step_idx]
            colors = plt.cm.viridis(np.linspace(0, 1, len(action)))
            ax_act.bar(range(len(action)), action, color=colors, width=0.8)
            ax_act.set_ylim(-3, 3)
            ax_act.set_xlabel('Action dim', fontsize=8)
            if i == 0:
                ax_act.set_ylabel('Value', fontsize=8)
            ax_act.tick_params(axis='both', labelsize=7)
    
    plt.tight_layout()
    plt.show()


def visualize_image_grid(data: Dict, episodes: List[Tuple[int, int]], 
                         n_episodes: int = 4, n_frames_per_ep: int = 4):
    """Show a grid of images from multiple episodes."""
    n_episodes = min(n_episodes, len(episodes))
    
    fig, axes = plt.subplots(n_episodes, n_frames_per_ep, figsize=(12, 3 * n_episodes))
    if n_episodes == 1:
        axes = [axes]
    
    fig.suptitle('Sample Frames from Multiple Episodes', fontsize=14)
    
    for ep_idx in range(n_episodes):
        start_idx, end_idx = episodes[ep_idx]
        step_indices = np.linspace(start_idx, end_idx - 1, n_frames_per_ep, dtype=int)
        
        for frame_idx, step_idx in enumerate(step_indices):
            ax = axes[ep_idx][frame_idx] if n_episodes > 1 else axes[frame_idx]
            
            if 'head_camera' in data:
                img = get_image(data, step_idx)
                ax.imshow(img)
            
            if frame_idx == 0:
                ax.set_ylabel(f'Ep {ep_idx}', fontsize=10)
            if ep_idx == 0:
                ax.set_title(f'Step {step_idx}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_action_distribution(data: Dict):
    """Visualize action value distributions."""
    if 'action' not in data:
        print("No action data found in dataset.")
        return
    
    actions = data['action']
    n_dims = actions.shape[1]
    
    n_rows = 2
    n_cols = (n_dims + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6))
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


def visualize_action_trajectory(data: Dict, episodes: List[Tuple[int, int]], episode_idx: int = 0):
    """Visualize action values over time for an episode."""
    if 'action' not in data:
        print("No action data found in dataset.")
        return
    
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found.")
        return
    
    start_idx, end_idx = episodes[episode_idx]
    actions = data['action'][start_idx:end_idx]
    n_dims = actions.shape[1]
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
    fig.suptitle(f'Action Trajectory - Episode {episode_idx}', fontsize=14)
    
    dim_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7', 'Gripper']
    colors = plt.cm.tab10(np.linspace(0, 1, n_dims))
    
    for i in range(n_dims):
        ax = axes[i]
        name = dim_names[i] if i < len(dim_names) else f'Dim {i}'
        ax.plot(actions[:, i], color=colors[i], linewidth=1.5)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(actions))
    
    axes[-1].set_xlabel('Step', fontsize=12)
    plt.tight_layout()
    plt.show()


def save_episode_video(data: Dict, episodes: List[Tuple[int, int]], 
                       episode_idx: int, output_path: str, fps: int = 10):
    """Save episode as video file."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return
    
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found.")
        return
    
    if 'head_camera' not in data:
        print("No image data found.")
        return
    
    start_idx, end_idx = episodes[episode_idx]
    
    # Get image dimensions
    sample_img = get_image(data, start_idx)
    height, width = sample_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for idx in range(start_idx, end_idx):
        frame = get_image(data, idx)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video to: {output_path}")


def compare_actions(data: Dict):
    """Compare action vs tcp_action if both exist."""
    if 'action' not in data or 'tcp_action' not in data:
        print("Need both 'action' and 'tcp_action' for comparison.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Action (joint)
    ax1 = axes[0]
    action = data['action']
    im1 = ax1.imshow(action[:100].T, aspect='auto', cmap='viridis')
    ax1.set_title('Joint Actions (first 100 steps)', fontsize=12)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Joint Dimension')
    plt.colorbar(im1, ax=ax1)
    
    # TCP action
    ax2 = axes[1]
    tcp = data['tcp_action']
    im2 = ax2.imshow(tcp[:100].T, aspect='auto', cmap='viridis')
    ax2.set_title('TCP Actions (first 100 steps)', fontsize=12)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('TCP Dimension')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize ZARR dataset')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ZARR dataset (.zarr directory)')
    parser.add_argument('--episode', type=int, default=0,
                        help='Episode index to visualize (default: 0)')
    parser.add_argument('--show_actions', action='store_true',
                        help='Show action values alongside images')
    parser.add_argument('--show_grid', action='store_true',
                        help='Show image grid from multiple episodes')
    parser.add_argument('--show_distribution', action='store_true',
                        help='Show action value distributions')
    parser.add_argument('--show_trajectory', action='store_true',
                        help='Show action trajectory over episode')
    parser.add_argument('--compare_actions', action='store_true',
                        help='Compare joint action vs TCP action')
    parser.add_argument('--save_video', action='store_true',
                        help='Save episode as video')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='Video FPS (default: 10)')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only print dataset summary, no visualization')
    
    args = parser.parse_args()
    
    zarr_path = args.data_path
    
    if not os.path.exists(zarr_path):
        print(f"Error: Path not found: {zarr_path}")
        return
    
    # Load data
    print(f"\nLoading ZARR data from: {zarr_path}")
    data = load_zarr_data(zarr_path)
    episodes = get_episodes(data)
    
    # Print summary
    print_dataset_summary(zarr_path, data, episodes)
    
    if args.summary_only:
        return
    
    # Visualizations
    if args.show_grid:
        visualize_image_grid(data, episodes)
    
    if args.show_distribution:
        visualize_action_distribution(data)
    
    if args.show_trajectory:
        visualize_action_trajectory(data, episodes, args.episode)
    
    if args.compare_actions:
        compare_actions(data)
    
    if args.save_video:
        video_path = str(Path(zarr_path).parent / f'episode_{args.episode}.mp4')
        save_episode_video(data, episodes, args.episode, video_path, args.video_fps)
    
    # Default: show single episode
    if not (args.summary_only or args.show_grid or args.show_distribution or 
            args.show_trajectory or args.compare_actions):
        visualize_episode(data, episodes, args.episode, args.show_actions)


if __name__ == "__main__":
    main()