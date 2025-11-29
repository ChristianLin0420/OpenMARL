"""OpenVLA model wrapper with LoRA fine-tuning support."""

import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
import numpy as np


class OpenVLAModel(nn.Module):
    """
    Wrapper for OpenVLA model with LoRA fine-tuning support.
    
    This class handles:
    - Loading pretrained OpenVLA from HuggingFace
    - Setting up LoRA for efficient fine-tuning
    - Forward pass for training
    - Action prediction for inference
    """
    
    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initialize OpenVLA model.
        
        Args:
            model_name: HuggingFace model name or path
            use_lora: Whether to use LoRA fine-tuning
            lora_rank: Rank for LoRA layers
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout rate for LoRA layers
            torch_dtype: Data type for model weights
            device: Device to load model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.use_lora = use_lora
        self.torch_dtype = torch_dtype
        self.device = device
        
        # Load processor
        print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        print(f"Loading model from {model_name}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Setup LoRA if enabled
        if use_lora:
            print(f"Setting up LoRA with rank={lora_rank}, alpha={lora_alpha}")
            self._setup_lora(lora_rank, lora_alpha, lora_dropout)
        
        # Move to device
        self.model = self.model.to(device)
        
        # Statistics for action denormalization
        self.action_mean = None
        self.action_std = None
    
    def _setup_lora(self, rank: int, alpha: int, dropout: float):
        """Setup LoRA for efficient fine-tuning."""
        # Configure LoRA for all linear layers
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def set_action_statistics(self, mean: np.ndarray, std: np.ndarray):
        """
        Set action statistics for denormalization.
        
        Args:
            mean: Mean values for actions
            std: Standard deviation values for actions
        """
        self.action_mean = torch.from_numpy(mean).to(
            device=self.device,
            dtype=self.torch_dtype
        )
        self.action_std = torch.from_numpy(std).to(
            device=self.device,
            dtype=self.torch_dtype
        )
    
    def forward(
        self,
        images: torch.Tensor,
        instructions: list[str],
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: Batch of images (B, C, H, W)
            instructions: List of language instructions
            actions: Ground truth actions for training (B, action_dim)
            
        Returns:
            Dictionary containing:
                - 'loss': Training loss (if actions provided)
                - 'logits': Model logits
        """
        # Format prompts
        prompts = [
            f"In: What action should the robot take to {inst}?\nOut:"
            for inst in instructions
        ]
        
        # Process inputs
        # Convert images from (B, C, H, W) to list of PIL Images
        images_list = []
        for img in images:
            # Convert from (C, H, W) to (H, W, C) and to uint8
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images_list.append(Image.fromarray(img_np))
        
        # Process with processor
        inputs = self.processor(
            text=prompts,
            images=images_list,
            return_tensors="pt",
        )
        
        # Move to device and convert to correct dtype
        inputs = {
            k: v.to(device=self.device, dtype=self.torch_dtype) if v.dtype in [torch.float32, torch.float64] else v.to(self.device)
            for k, v in inputs.items()
        }
        
        # If actions provided, compute loss
        if actions is not None:
            # Tokenize actions (OpenVLA uses action tokenization)
            # For now, we'll use the model's forward pass with labels
            outputs = self.model(
                **inputs,
                labels=inputs['input_ids'],  # This is a placeholder
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
            }
        else:
            # Inference mode
            outputs = self.model(**inputs)
            return {
                'logits': outputs.logits,
            }
    
    def predict_action(
        self,
        image: torch.Tensor,
        instruction: str,
        unnorm_key: Optional[str] = None,
        do_sample: bool = False,
    ) -> np.ndarray:
        """
        Predict action for a single observation.
        
        Args:
            image: Single image tensor (C, H, W) or (1, C, H, W)
            instruction: Language instruction
            unnorm_key: Key for denormalization (not used, kept for compatibility)
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Predicted action as numpy array
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Convert to PIL Image
        img_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Format prompt
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        # Process inputs
        inputs = self.processor(prompt, img_pil)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Generate action
        with torch.no_grad():
            action = self.model.predict_action(**inputs, do_sample=do_sample)
        
        # Denormalize if statistics available
        if self.action_mean is not None and self.action_std is not None:
            action = action * self.action_std.cpu().numpy() + self.action_mean.cpu().numpy()
        
        return action
    
    def save_pretrained(self, save_directory: str):
        """
        Save model weights.
        
        Args:
            save_directory: Directory to save model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        if self.use_lora:
            # Save LoRA weights only
            self.model.save_pretrained(save_directory)
        else:
            # Save full model
            self.model.save_pretrained(save_directory)
        
        # Save processor
        self.processor.save_pretrained(save_directory)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Load pretrained model with LoRA weights.
        
        Args:
            model_path: Path to saved model
            device: Device to load on
            torch_dtype: Data type for model
            
        Returns:
            Loaded model instance
        """
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load base model
        base_model_name = "openvla/openvla-7b"
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, model_path)
        
        # Merge LoRA weights for faster inference
        model = model.merge_and_unload()
        
        # Move to device
        model = model.to(device)
        
        # Create wrapper instance
        wrapper = cls.__new__(cls)
        wrapper.model = model
        wrapper.processor = processor
        wrapper.device = device
        wrapper.torch_dtype = torch_dtype
        wrapper.use_lora = False  # Already merged
        wrapper.action_mean = None
        wrapper.action_std = None
        
        return wrapper
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        if self.action_mean is not None:
            self.action_mean = self.action_mean.to(device)
            self.action_std = self.action_std.to(device)
        return self
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()


if __name__ == "__main__":
    # Test model loading
    print("Testing OpenVLA model wrapper...")
    
    model = OpenVLAModel(
        model_name="openvla/openvla-7b",
        use_lora=True,
        lora_rank=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Model loaded successfully!")
    
    # Test forward pass
    batch_size = 2
    images = torch.rand(batch_size, 3, 224, 224)
    instructions = ["pick up the cube", "place the object"]
    actions = torch.rand(batch_size, 7)
    
    if torch.cuda.is_available():
        images = images.cuda()
        actions = actions.cuda()
    
    print("\nTesting forward pass...")
    outputs = model(images, instructions, actions)
    print(f"Loss: {outputs['loss'].item()}")
    
    print("\nModel wrapper test complete!")

