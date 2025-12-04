"""
OpenVLA training workspace with multi-GPU support.

This module implements the training loop for OpenVLA models using PyTorch DDP
for distributed training across multiple GPUs.
"""

import os
import copy
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from ..model.openvla_wrapper import OpenVLAModel
from ..dataset.robot_rlds_dataset import RobotRLDSDataset, collate_fn


class OpenVLAWorkspace:
    """
    Training workspace for OpenVLA models.
    
    Handles:
    - Dataset loading
    - Model initialization with LoRA
    - Multi-GPU training with DDP
    - Wandb logging
    - Checkpoint saving/loading
    """
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """
        Initialize workspace.
        
        Args:
            cfg: Hydra configuration
            output_dir: Output directory for logs and checkpoints
        """
        self.cfg = cfg
        
        if output_dir is None:
            output_dir = cfg.hydra.run.dir if hasattr(cfg, 'hydra') else 'outputs'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize distributed training if needed
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            # Use rank 1 for logging to avoid conflicts with rank 0
            self.is_main_process = self.rank == 1
        else:
            self.rank = 0
            self.world_size = 1
            self.is_main_process = True
        
        # Disable wandb for non-main processes
        if not self.is_main_process:
            os.environ["WANDB_MODE"] = "disabled"
        
        # Set random seed
        self._set_seed(cfg.training.seed)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Model and optimizer (will be initialized in run())
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _init_model(self):
        """Initialize model with LoRA."""
        cfg = self.cfg
        
        self.model = OpenVLAModel(
            model_name=cfg.model.model_name,
            use_lora=cfg.model.use_lora,
            lora_rank=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            torch_dtype=getattr(torch, cfg.model.torch_dtype),
            device=cfg.training.device,
        )
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model.model = DDP(
                self.model.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        cfg = self.cfg
        
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            eps=cfg.training.adam_eps,
        )
        
        # Learning rate scheduler
        if cfg.training.use_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.training.num_epochs,
                eta_min=cfg.training.min_learning_rate,
            )
    
    def _init_dataloaders(self):
        """Initialize train and validation dataloaders."""
        cfg = self.cfg
        
        # Training dataset
        train_dataset = RobotRLDSDataset(
            data_dir=cfg.task.dataset.rlds_path,
            train=True,
            image_size=(cfg.model.image_size, cfg.model.image_size),
            augment=cfg.training.image_aug,
            augment_crop_ratio=cfg.training.augment_crop_ratio,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
        )
        
        # Validation dataset
        val_dataset = RobotRLDSDataset(
            data_dir=cfg.task.dataset.rlds_path,
            train=False,
            image_size=(cfg.model.image_size, cfg.model.image_size),
            augment=False,
            augment_crop_ratio=cfg.training.augment_crop_ratio,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
        )
        
        # Setup action statistics for denormalization
        stats = train_dataset.get_statistics()
        if 'action' in stats:
            self.model.set_action_statistics(
                mean=stats['action']['mean'],
                std=stats['action']['std'],
            )
        
        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.dataloader.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            collate_fn=collate_fn,
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.val_dataloader.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.val_dataloader.num_workers,
            pin_memory=cfg.val_dataloader.pin_memory,
            collate_fn=collate_fn,
        )
    
    def run(self):
        """Main training loop."""
        cfg = self.cfg
        
        # Initialize components
        if self.is_main_process:
            print("Initializing model...")
        self._init_model()
        
        if self.is_main_process:
            print("Initializing optimizer...")
        self._init_optimizer()
        
        if self.is_main_process:
            print("Initializing dataloaders...")
        self._init_dataloaders()
        
        # Resume from checkpoint if exists
        if cfg.training.resume:
            checkpoint_path = self.output_dir / 'checkpoints' / 'latest.ckpt'
            if checkpoint_path.exists():
                if self.is_main_process:
                    print(f"Resuming from {checkpoint_path}")
                self.load_checkpoint(checkpoint_path)
        
        # Initialize wandb
        if self.is_main_process and cfg.logging.mode == "online":
            wandb.init(
                project=cfg.logging.project,
                name=cfg.logging.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(self.output_dir),
                mode=cfg.logging.mode,
            )
        
        # Training loop
        for epoch in range(self.epoch, cfg.training.num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self._train_epoch()
            
            # Validate
            if (epoch + 1) % cfg.training.val_every == 0:
                val_metrics = self._val_epoch()
            else:
                val_metrics = {}
            
            # Log metrics
            if self.is_main_process:
                metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
                if cfg.logging.mode == "online":
                    wandb.log(metrics, step=self.global_step)
                print(f"Epoch {epoch}: {metrics}")
            
            # Save checkpoint
            if self.is_main_process and (epoch + 1) % cfg.training.checkpoint_every == 0:
                self.save_checkpoint()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Finish
        if self.is_main_process and cfg.logging.mode == "online":
            wandb.finish()
    
    def _train_epoch(self):
        """Train for one epoch."""
        cfg = self.cfg
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Train Epoch {self.epoch}",
            disable=not self.is_main_process,
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device (use model's device for correct GPU in distributed training)
            device = self.model.device
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            instructions = batch['instruction']
            
            # Forward pass
            outputs = self.model(images, instructions, actions)
            loss = outputs['loss']
            
            # Backward pass
            loss = loss / cfg.training.gradient_accumulate_every
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                # Gradient clipping
                if cfg.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        cfg.training.max_grad_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            total_loss += loss.item() * cfg.training.gradient_accumulate_every
            num_batches += 1
            
            if self.is_main_process:
                pbar.set_postfix({'loss': loss.item() * cfg.training.gradient_accumulate_every})
            
            # Early exit for debugging
            if cfg.training.debug and batch_idx >= cfg.training.max_train_steps:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def _val_epoch(self):
        """Validate for one epoch."""
        cfg = self.cfg
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.val_dataloader,
            desc=f"Val Epoch {self.epoch}",
            disable=not self.is_main_process,
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device (use model's device for correct GPU in distributed training)
            device = self.model.device
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            instructions = batch['instruction']
            
            # Forward pass
            outputs = self.model(images, instructions, actions)
            loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
            
            # Early exit for debugging
            if cfg.training.debug and batch_idx >= cfg.training.max_val_steps:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, name: Optional[str] = None):
        """Save checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / 'checkpoints' / Path(self.cfg.task.dataset.rlds_path).stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if name is None:
            name = f'epoch_{self.epoch + 1}.ckpt'
        
        checkpoint_path = checkpoint_dir / name
        
        # Save model
        self.model.save_pretrained(str(checkpoint_path.with_suffix('')))
        
        # Save training state
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(state, checkpoint_path)
        
        # Also save as latest
        latest_path = checkpoint_dir.parent / 'latest.ckpt'
        torch.save(state, latest_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        state = torch.load(path, map_location='cpu')
        
        self.epoch = state.get('epoch', 0)
        self.global_step = state.get('global_step', 0)
        
        if 'optimizer_state' in state:
            self.optimizer.load_state_dict(state['optimizer_state'])
        
        if 'scheduler_state' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])
        
        print(f"Loaded checkpoint from {path}")


if __name__ == "__main__":
    print("OpenVLA workspace module loaded successfully!")

