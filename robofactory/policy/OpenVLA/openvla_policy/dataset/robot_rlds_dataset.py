"""RLDS dataset loader for RoboFactory tasks with OpenVLA."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tensorflow as tf
from .base_dataset import BaseVLADataset


class RobotRLDSDataset(BaseVLADataset):
    """
    RLDS dataset loader for RoboFactory robotic manipulation tasks.
    
    This dataset loads data from TFRecord format (RLDS compatible) and provides
    image observations, proprioceptive states, actions, and language instructions
    for training OpenVLA models.
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
    ):
        """
        Initialize RLDS dataset.
        
        Args:
            data_dir: Directory containing RLDS dataset
            train: Whether to load training or validation split
            image_size: Target image size (H, W)
            augment: Whether to apply image augmentation
            augment_crop_ratio: Crop ratio for augmentation (0.9 = 90% crop)
            val_split: Validation split ratio
            seed: Random seed for split
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train = train
        self.image_size = image_size
        self.augment = augment and train  # Only augment during training
        self.augment_crop_ratio = augment_crop_ratio
        self.val_split = val_split
        self.seed = seed
        
        # Load dataset info
        self.info = self._load_dataset_info()
        
        # Load statistics for normalization
        self.statistics = self._load_statistics()
        
        # Load all episodes into memory
        self.episodes = self._load_episodes()
        
        # Create train/val split
        self.indices = self._create_split()
        
        print(f"Loaded {len(self.indices)} samples ({'train' if train else 'val'})")
    
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
                # Convert lists to numpy arrays
                for key in stats:
                    for subkey in stats[key]:
                        stats[key][subkey] = np.array(stats[key][subkey], dtype=np.float32)
                return stats
        else:
            print(f"Warning: Statistics file not found at {stats_path}")
            return {}
    
    def _load_episodes(self):
        """Load all episodes from TFRecord files."""
        tfrecord_path = self.data_dir / 'train.tfrecord'
        
        if not tfrecord_path.exists():
            raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
        
        episodes = []
        
        # Define feature description - need to figure out the actual feature types
        # We'll parse them dynamically
        feature_description = {}
        
        # Try to infer feature types from first example
        raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        first_example = next(iter(raw_dataset))
        
        # Parse with empty description to get keys
        example_keys = {
            'action': tf.io.VarLenFeature(tf.float32),
            'is_first': tf.io.FixedLenFeature([], tf.int64),
            'is_last': tf.io.FixedLenFeature([], tf.int64),
            'is_terminal': tf.io.FixedLenFeature([], tf.int64),
            'language_instruction': tf.io.FixedLenFeature([], tf.string),
            'observation/image': tf.io.FixedLenFeature([], tf.string),
            'observation/proprio': tf.io.VarLenFeature(tf.float32),
        }
        
        # Read TFRecord file
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        
        current_episode = []
        
        for raw_record in dataset:
            example = tf.io.parse_single_example(raw_record, example_keys)
            
            # Decode action (stored as VarLenFeature)
            action = tf.sparse.to_dense(example['action']).numpy().astype(np.float32)
            
            # Decode proprio (stored as VarLenFeature)
            proprio = tf.sparse.to_dense(example['observation/proprio']).numpy().astype(np.float32)
            
            # Decode image
            image_bytes = example['observation/image'].numpy()
            image = tf.io.decode_png(image_bytes).numpy()
            
            # Decode language instruction
            instruction = example['language_instruction'].numpy().decode('utf-8')
            
            # Create step
            step = {
                'image': image,
                'proprio': proprio,
                'action': action,
                'instruction': instruction,
                'is_first': bool(example['is_first'].numpy()),
                'is_last': bool(example['is_last'].numpy()),
            }
            
            current_episode.append(step)
            
            # If this is the last step, save episode and start new one
            if step['is_last']:
                episodes.append(current_episode)
                current_episode = []
        
        print(f"Loaded {len(episodes)} episodes from {tfrecord_path}")
        return episodes
    
    def _create_split(self):
        """Create train/val split indices."""
        # Count total steps
        all_indices = []
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode)):
                all_indices.append((ep_idx, step_idx))
        
        # Shuffle with seed
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_indices)
        
        # Split
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
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - 'image': Image tensor (C, H, W) in [0, 1]
                - 'proprio': Proprioceptive state tensor
                - 'action': Action tensor
                - 'instruction': Language instruction string
        """
        ep_idx, step_idx = self.indices[idx]
        step = self.episodes[ep_idx][step_idx]
        
        # Process image
        image = step['image']
        image = self._process_image(image)
        
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
            'image': torch.from_numpy(image),
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
            
            # Random crop
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
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary
    """
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
    # Test dataset loading
    dataset = RobotRLDSDataset(
        data_dir="data/rlds_data/LiftBarrier-rf_Agent0",
        train=True,
        augment=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset info: {dataset.get_info()}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Proprio shape: {sample['proprio'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Instruction: {sample['instruction']}")

