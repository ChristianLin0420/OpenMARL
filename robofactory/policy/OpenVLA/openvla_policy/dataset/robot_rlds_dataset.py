"""RLDS dataset loader for RoboFactory tasks with OpenVLA - Multi-view support."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tensorflow as tf
from .base_dataset import BaseVLADataset


class RobotRLDSDataset(BaseVLADataset):
    """
    RLDS dataset loader with multi-view image support.
    
    Supports loading multiple camera views:
    - primary: head/side camera (third-person view)
    - secondary: global camera (overhead view)  
    - wrist: wrist/gripper camera (end-effector view)
    """
    
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        augment_crop_ratio: float = 0.9,
        val_split: float = 0.1,
        seed: int = 42,
        use_multi_view: bool = False,
        image_views: List[str] = None,  # ['primary', 'secondary', 'wrist']
    ):
        """
        Initialize RLDS dataset with multi-view support.
        
        Args:
            data_dir: Directory containing RLDS dataset
            train: Whether to load training or validation split
            image_size: Target image size (H, W)
            augment: Whether to apply image augmentation
            augment_crop_ratio: Crop ratio for augmentation
            val_split: Validation split ratio
            seed: Random seed for split
            use_multi_view: Whether to load multiple camera views
            image_views: List of view names to load ['primary', 'secondary', 'wrist']
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train = train
        self.image_size = image_size
        self.augment = augment and train
        self.augment_crop_ratio = augment_crop_ratio
        self.val_split = val_split
        self.seed = seed
        self.use_multi_view = use_multi_view
        self.image_views = image_views or ['primary', 'secondary', 'wrist']
        
        # Map view names to TFRecord keys
        self.view_to_key = {
            'primary': 'observation/side_image',      # Head camera (side view)
            'secondary': 'observation/global_image',  # Global camera
            'wrist': 'observation/wrist_image',       # Wrist camera (gripper view)
            'image': 'observation/image',             # Fallback single image
        }
        
        self.info = self._load_dataset_info()
        self.statistics = self._load_statistics()
        self.episodes = self._load_episodes()
        self.indices = self._create_split()
        
        print(f"Loaded {len(self.indices)} samples ({'train' if train else 'val'})")
        if self.use_multi_view:
            print(f"  Multi-view enabled: {self.image_views}")
    
    def _load_dataset_info(self) -> Dict:
        """Load dataset info from JSON file."""
        info_path = self.data_dir / 'dataset_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_statistics(self) -> Dict:
        """Load dataset statistics for normalization."""
        stats_path = self.data_dir / 'statistics.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                for key in stats:
                    for subkey in stats[key]:
                        stats[key][subkey] = np.array(stats[key][subkey], dtype=np.float32)
                return stats
        else:
            print(f"Warning: Statistics file not found at {stats_path}")
            return {}
    
    def _load_episodes(self):
        """Load all episodes from TFRecord files with multi-view support."""
        tfrecord_path = self.data_dir / 'train.tfrecord'
        
        if not tfrecord_path.exists():
            raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
        
        episodes = []
        
        # Feature description with multi-view support
        example_keys = {
            'action': tf.io.VarLenFeature(tf.float32),
            'is_first': tf.io.FixedLenFeature([], tf.int64),
            'is_last': tf.io.FixedLenFeature([], tf.int64),
            'is_terminal': tf.io.FixedLenFeature([], tf.int64),
            'language_instruction': tf.io.FixedLenFeature([], tf.string),
            # Multi-view images (with defaults for missing keys)
            'observation/image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/wrist_image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/side_image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/global_image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'observation/proprio': tf.io.VarLenFeature(tf.float32),
        }
        
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        current_episode = []
        
        for raw_record in dataset:
            example = tf.io.parse_single_example(raw_record, example_keys)
            
            action = tf.sparse.to_dense(example['action']).numpy().astype(np.float32)
            proprio = tf.sparse.to_dense(example['observation/proprio']).numpy().astype(np.float32)
            instruction = example['language_instruction'].numpy().decode('utf-8')
            
            # Load all available images
            images = {}
            
            # Primary: side/head camera
            side_bytes = example['observation/side_image'].numpy()
            if side_bytes:
                images['primary'] = tf.io.decode_png(side_bytes).numpy()
            
            # Secondary: global camera
            global_bytes = example['observation/global_image'].numpy()
            if global_bytes:
                images['secondary'] = tf.io.decode_png(global_bytes).numpy()
            
            # Wrist: gripper camera
            wrist_bytes = example['observation/wrist_image'].numpy()
            if wrist_bytes:
                images['wrist'] = tf.io.decode_png(wrist_bytes).numpy()
            
            # Fallback: single image
            image_bytes = example['observation/image'].numpy()
            if image_bytes:
                images['image'] = tf.io.decode_png(image_bytes).numpy()
            
            step = {
                'images': images,
                'proprio': proprio,
                'action': action,
                'instruction': instruction,
                'is_first': bool(example['is_first'].numpy()),
                'is_last': bool(example['is_last'].numpy()),
            }
            
            current_episode.append(step)
            
            if step['is_last']:
                episodes.append(current_episode)
                current_episode = []
        
        # Log available views from first episode
        if episodes and episodes[0]:
            available_views = list(episodes[0][0]['images'].keys())
            print(f"Available camera views in dataset: {available_views}")
        
        print(f"Loaded {len(episodes)} episodes from {tfrecord_path}")
        return episodes
    
    def _create_split(self):
        """Create train/val split indices."""
        all_indices = []
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode)):
                all_indices.append((ep_idx, step_idx))
        
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_indices)
        
        n_val = int(len(all_indices) * self.val_split)
        if self.train:
            return all_indices[n_val:]
        else:
            return all_indices[:n_val]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with multi-view support.
        
        Returns:
            Dictionary containing:
                - 'image': Dict[str, Tensor] for multi-view or Tensor for single view
                - 'proprio': Proprioceptive state tensor
                - 'action': Action tensor
                - 'instruction': Language instruction string
        """
        ep_idx, step_idx = self.indices[idx]
        step = self.episodes[ep_idx][step_idx]
        
        # Process images
        if self.use_multi_view:
            # Multi-view: return dict of processed images
            processed_images = {}
            for view in self.image_views:
                if view in step['images']:
                    processed_images[view] = torch.from_numpy(
                        self._process_image(step['images'][view])
                    )
                elif 'image' in step['images']:
                    # Fallback to single image for missing views
                    processed_images[view] = torch.from_numpy(
                        self._process_image(step['images']['image'])
                    )
                else:
                    # Create placeholder if no image available
                    processed_images[view] = torch.zeros(3, self.image_size[0], self.image_size[1])
            image_output = processed_images
        else:
            # Single image mode (backward compatible)
            if 'image' in step['images']:
                img = step['images']['image']
            elif 'primary' in step['images']:
                img = step['images']['primary']
            elif step['images']:
                img = list(step['images'].values())[0]
            else:
                # Placeholder if no image
                img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            image_output = torch.from_numpy(self._process_image(img))
        
        # Process proprio
        proprio = step['proprio'].astype(np.float32)
        if 'proprio' in self.statistics:
            proprio = self._normalize(
                proprio,
                self.statistics['proprio']['mean'],
                self.statistics['proprio']['std']
            )
        
        # Process action
        action = step['action'].astype(np.float32)
        if 'action' in self.statistics:
            action = self._normalize(
                action,
                self.statistics['action']['mean'],
                self.statistics['action']['std']
            )
        
        return {
            'image': image_output,  # Dict[str, Tensor] or Tensor
            'proprio': torch.from_numpy(proprio),
            'action': torch.from_numpy(action),
            'instruction': step['instruction'],
        }
    
    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image: resize, crop, normalize.
        
        Args:
            image: Input image (H, W, C) in [0, 255]
            
        Returns:
            Processed image (C, H, W) in [0, 1]
        """
        # Convert to PIL Image
        image_pil = Image.fromarray(image)
        
        # Apply augmentation (random crop) if enabled
        if self.augment:
            w, h = image_pil.size
            crop_w = int(w * self.augment_crop_ratio)
            crop_h = int(h * self.augment_crop_ratio)
            left = np.random.randint(0, w - crop_w + 1)
            top = np.random.randint(0, h - crop_h + 1)
            image_pil = image_pil.crop((left, top, left + crop_w, top + crop_h))
        else:
            # Center crop for validation
            w, h = image_pil.size
            crop_w = int(w * self.augment_crop_ratio)
            crop_h = int(h * self.augment_crop_ratio)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            image_pil = image_pil.crop((left, top, left + crop_w, top + crop_h))
        
        # Resize to target size
        image_pil = image_pil.resize(self.image_size, Image.BILINEAR)
        
        # Convert to numpy array
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        
        # Convert to CHW format
        image_np = np.transpose(image_np, (2, 0, 1))
        
        return image_np
    
    def _normalize(self, data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize data using mean and std."""
        return (data - mean) / (std + eps)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.statistics
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset info."""
        return self.info


def collate_fn(batch):
    """
    Collate function for DataLoader with multi-view support.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with images as dict or tensor
    """
    first_image = batch[0]['image']
    
    if isinstance(first_image, dict):
        # Multi-view: stack each view separately
        images = {}
        for view in first_image.keys():
            images[view] = torch.stack([item['image'][view] for item in batch])
    else:
        # Single image: backward compatible
        images = torch.stack([item['image'] for item in batch])
    
    proprios = torch.stack([item['proprio'] for item in batch])
    actions = torch.stack([item['action'] for item in batch])
    instructions = [item['instruction'] for item in batch]
    
    return {
        'image': images,
        'proprio': proprios,
        'action': actions,
        'instruction': instructions,
    }


if __name__ == "__main__":
    # Test dataset loading with multi-view
    print("Testing multi-view dataset loading...")
    
    dataset = RobotRLDSDataset(
        data_dir="data/rlds_data/LiftBarrier-rf_Agent0",
        train=True,
        augment=True,
        use_multi_view=True,
        image_views=['primary', 'secondary', 'wrist'],
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset info: {dataset.get_info()}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    
    if isinstance(sample['image'], dict):
        print("Multi-view images:")
        for view, img in sample['image'].items():
            print(f"  {view}: shape={img.shape}")
    else:
        print(f"Single image shape: {sample['image'].shape}")
    
    print(f"Proprio shape: {sample['proprio'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Instruction: {sample['instruction']}")
