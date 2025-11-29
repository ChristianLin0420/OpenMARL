"""Action processing utilities for OpenVLA."""

import numpy as np
import torch
from typing import Dict, Optional


def normalize_action(
    action: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize actions using mean and standard deviation.
    
    Args:
        action: Action array to normalize
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Normalized action array
    """
    return (action - mean) / (std + eps)


def denormalize_action(
    action: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Denormalize actions back to original scale.
    
    Args:
        action: Normalized action array
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized action array
    """
    return action * std + mean


def compute_action_statistics(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute mean and std statistics for actions.
    
    Args:
        actions: Array of actions with shape (N, action_dim)
        
    Returns:
        Dictionary containing 'mean' and 'std' arrays
    """
    return {
        'mean': np.mean(actions, axis=0),
        'std': np.std(actions, axis=0),
        'min': np.min(actions, axis=0),
        'max': np.max(actions, axis=0),
    }


def action_to_7dof(action: np.ndarray) -> np.ndarray:
    """
    Convert action to 7-DoF format (6 joint angles + 1 gripper state).
    
    Args:
        action: Action array, potentially with 8 DoF
        
    Returns:
        7-DoF action array
    """
    if len(action) == 8:
        # Assume last element is gripper, combine first 7 DoF
        return action[:7]
    elif len(action) == 7:
        return action
    else:
        raise ValueError(f"Unexpected action dimension: {len(action)}")

