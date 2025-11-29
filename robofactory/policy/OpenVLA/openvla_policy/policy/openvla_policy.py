"""OpenVLA policy interface for inference."""

from typing import Dict, Optional
import torch
import numpy as np
from ..model.openvla_wrapper import OpenVLAModel


class OpenVLAPolicy:
    """
    Policy interface for OpenVLA model inference.
    
    This provides a simple interface for loading trained models
    and predicting actions during evaluation.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        action_statistics: Optional[Dict] = None,
    ):
        """
        Initialize policy from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            action_statistics: Dictionary with action mean/std for denormalization
        """
        self.device = device
        
        # Load model
        print(f"Loading OpenVLA policy from {checkpoint_path}")
        self.model = OpenVLAModel.from_pretrained(
            checkpoint_path,
            device=device,
        )
        self.model.eval()
        
        # Set action statistics
        if action_statistics is not None:
            self.model.set_action_statistics(
                mean=action_statistics['mean'],
                std=action_statistics['std'],
            )
    
    def predict(
        self,
        observation: Dict[str, np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        """
        Predict action given observation and instruction.
        
        Args:
            observation: Dictionary containing:
                - 'image': Image observation (H, W, C) or (C, H, W)
                - 'proprio': Optional proprioceptive state
            instruction: Language instruction
            
        Returns:
            Predicted action as numpy array
        """
        # Get image
        image = observation['image']
        
        # Convert to torch tensor if needed
        if isinstance(image, np.ndarray):
            # Check if HWC or CHW format
            if image.shape[-1] == 3:
                # HWC -> CHW
                image = np.transpose(image, (2, 0, 1))
            
            # Normalize to [0, 1] if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            image = torch.from_numpy(image).float()
        
        # Ensure on correct device
        image = image.to(self.device)
        
        # Predict action
        with torch.no_grad():
            action = self.model.predict_action(
                image=image,
                instruction=instruction,
                do_sample=False,
            )
        
        return action
    
    def reset(self):
        """Reset policy state (if any)."""
        pass

