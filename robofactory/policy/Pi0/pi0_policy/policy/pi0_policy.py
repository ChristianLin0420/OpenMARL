"""
Pi0/Pi0.5 policy interface for RoboFactory evaluation.

This module provides the policy interface that connects to RoboFactory's
evaluation pipeline, similar to OpenVLA's policy interface.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any
from omegaconf import OmegaConf

from .model.pi0_wrapper import Pi0Model
from .utils.task_instructions import get_task_instruction


class Pi0Policy:
    """
    Pi0/Pi0.5 policy for RoboFactory evaluation.
    
    Provides a simple interface for action prediction during rollouts.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        task_name: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        Initialize policy from checkpoint.
        
        Args:
            checkpoint_path: Path to trained checkpoint directory
            config_path: Path to config YAML (optional, will try to infer)
            task_name: Task name for language instruction
            device: Device to run inference on
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.task_name = task_name
        self.device = device
        
        # Load config
        if config_path is None:
            # Try to find config in checkpoint directory or parent
            config_candidates = [
                self.checkpoint_path / "config.yaml",
                self.checkpoint_path.parent.parent / "config.yaml",
            ]
            config_path = next((c for c in config_candidates if c.exists()), None)
            
            if config_path is None:
                raise ValueError(f"Could not find config.yaml near {checkpoint_path}")
        
        self.cfg = OmegaConf.load(config_path)
        
        # Initialize model
        self.model = Pi0Model(
            model_variant=self.cfg.model.model_variant,
            paligemma_variant=self.cfg.model.paligemma_variant,
            action_expert_variant=self.cfg.model.action_expert_variant,
            pretrained_checkpoint=None,  # Will load from checkpoint
            action_dim=self.cfg.model.action_dim,
            action_horizon=self.cfg.model.action_horizon,
            max_token_len=self.cfg.model.max_token_len,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            pytorch_training_precision=self.cfg.model.pytorch_training_precision,
            device=device,
            use_gradient_checkpointing=False,  # Disable for inference
        )
        
        # Load checkpoint weights
        self._load_checkpoint()
        
        # Set to eval mode
        self.model.eval()
        
        # Get language instruction
        if task_name:
            self.language_instruction = get_task_instruction(task_name)
        else:
            self.language_instruction = "Complete the task"
        
        print(f"Initialized Pi0 policy from {checkpoint_path}")
        print(f"Language instruction: {self.language_instruction}")
        print(f"Action dim: {self.cfg.model.action_dim}, Horizon: {self.cfg.model.action_horizon}")
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        model_file = self.checkpoint_path / "model.safetensors"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        import safetensors.torch
        state_dict = safetensors.torch.load_file(model_file, device='cpu')
        self.model.model.load_state_dict(state_dict)
        
        print(f"Loaded model weights from {model_file}")
        
        # Load normalization stats if available
        metadata_file = self.checkpoint_path / "metadata.pt"
        if metadata_file.exists():
            metadata = torch.load(metadata_file, map_location='cpu')
            if "normalization_stats" in metadata:
                stats = metadata["normalization_stats"]
                self.model.set_normalization_statistics(
                    action_q01=stats["action_q01"],
                    action_q99=stats["action_q99"],
                    state_q01=stats["state_q01"],
                    state_q99=stats["state_q99"],
                )
    
    def predict_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            observation: Dict with keys:
                - "images": List/array of camera images [num_cameras, H, W, 3] or dict
                - "state": Robot state [state_dim]
                
        Returns:
            Action array [action_dim] (first action from predicted sequence)
        """
        # Prepare images
        if isinstance(observation["images"], dict):
            # Already in dict format with keys
            images = observation["images"]
        else:
            # Convert from array to dict (following RoboFactory -> Pi0 mapping)
            imgs = observation["images"]
            if isinstance(imgs, np.ndarray):
                # Assuming order: [head, global, wrist] as per data conversion
                images = {
                    "base_0_rgb": imgs[0],         # head/side
                    "left_wrist_0_rgb": imgs[1],   # global/overhead
                    "right_wrist_0_rgb": imgs[2],  # wrist/gripper
                }
            else:
                raise ValueError("Unexpected image format")
        
        # Convert to tensors and add batch dimension
        image_tensors = {}
        image_masks = {}
        
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            if key in images:
                img = images[key]
                # Convert to CHW format and normalize
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                if img.ndim == 3 and img.shape[-1] == 3:  # HWC -> CHW
                    img = img.permute(2, 0, 1)
                if img.max() > 1.0:
                    img = img / 255.0
                
                image_tensors[key] = img.unsqueeze(0).to(self.device)
                image_masks[key] = torch.tensor([True]).to(self.device)
            else:
                # Create dummy if missing
                image_tensors[key] = torch.zeros(1, 3, 224, 224).to(self.device)
                image_masks[key] = torch.tensor([False]).to(self.device)
        
        # Prepare state
        state = torch.from_numpy(observation["state"]).float().unsqueeze(0).to(self.device)
        
        # Predict actions
        with torch.no_grad():
            action_sequence = self.model.predict(
                image=image_tensors,
                image_mask=image_masks,
                state=state,
                prompt=self.language_instruction,
            )
        
        # Return first action from sequence
        action = action_sequence[0].cpu().numpy()
        
        return action
    
    def reset(self):
        """Reset policy state (if needed)."""
        pass
    
    def __call__(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Callable interface for compatibility."""
        return self.predict_action(observation)

