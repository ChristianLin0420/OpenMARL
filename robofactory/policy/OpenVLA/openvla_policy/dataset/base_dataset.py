"""Base dataset interface for OpenVLA."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch.utils.data import Dataset


class BaseVLADataset(Dataset, ABC):
    """Base class for VLA datasets."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'image': Image observation tensor (C, H, W)
                - 'proprio': Proprioceptive state tensor
                - 'action': Action tensor
                - 'instruction': Language instruction string
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics for normalization.
        
        Returns:
            Dictionary containing mean, std, min, max for actions and proprio
        """
        pass

