"""
Pi0/Pi0.5 training workspace with multi-GPU support.

This module implements the training loop for Pi0 models using PyTorch DDP
for distributed training across multiple GPUs.

Follows openpi's training patterns from scripts/train_pytorch.py:
- Wandb logging with run ID tracking for resuming
- Periodic logging every N steps (not every batch)
- Checkpoint saving following openpi conventions
- Quantile normalization for actions and states
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
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

from ..model.pi0_wrapper import Pi0Model
from ..dataset.robot_lerobot_dataset import RobotLeRobotDataset, collate_fn

# Suppress verbose OpenPI logs (e.g., "Resizing image..." on every batch)
logging.getLogger("openpi").setLevel(logging.WARNING)


class Pi0Workspace:
    """
    Training workspace for Pi0/Pi0.5 models.
    
    Handles:
    - Dataset loading (LeRobot format)
    - Model initialization from pretrained checkpoints
    - Multi-GPU training with DDP
    - Wandb logging (following openpi pattern)
    - Checkpoint saving/loading
    """
    
    def __init__(self, cfg: OmegaConf = None, output_dir: Optional[str] = None, **kwargs):
        """
        Initialize workspace.
        
        Args:
            cfg: Hydra configuration
            output_dir: Output directory for logs and checkpoints
            **kwargs: Additional arguments from Hydra (ignored, used for config flexibility)
        """
        # If cfg is not passed, reconstruct it from kwargs (Hydra instantiation pattern)
        if cfg is None:
            cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        
        if output_dir is None:
            output_dir = cfg.hydra.run.dir if hasattr(cfg, 'hydra') else 'outputs'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize distributed training (following openpi pattern)
        self.use_ddp, self.local_rank, self.device = self._setup_ddp()
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main_process = (not self.use_ddp) or (dist.get_rank() == 0)
        
        # Set random seed
        self._set_seed(cfg.training.seed + self.local_rank)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Model and optimizer (will be initialized in run())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # For periodic logging (following openpi)
        self.step_infos = []
        self.start_time = time.time()
    
    def _setup_ddp(self):
        """Setup DDP (following openpi's setup_ddp)."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_ddp = world_size > 1
        
        if use_ddp and not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend, init_method="env://")
            
            # Set up debugging environment variables
            if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
        
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        
        return use_ddp, local_rank, device
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _init_model(self):
        """Initialize Pi0 model."""
        cfg = self.cfg
        
        self.model = Pi0Model(
            model_variant=cfg.model.model_variant,
            paligemma_variant=cfg.model.paligemma_variant,
            action_expert_variant=cfg.model.action_expert_variant,
            pretrained_checkpoint=cfg.model.pretrained_checkpoint,
            action_dim=cfg.model.action_dim,
            action_horizon=cfg.model.action_horizon,
            max_token_len=cfg.model.max_token_len,
            torch_dtype=getattr(torch, cfg.model.dtype),
            pytorch_training_precision=cfg.model.pytorch_training_precision,
            device=str(self.device),
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
        )
        
        # Wrap with DDP if distributed
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        cfg = self.cfg
        
        # Get trainable parameters
        model_for_params = self.model.module if isinstance(self.model, DDP) else self.model
        params = [p for p in model_for_params.parameters() if p.requires_grad]
        
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
        
        # Get lerobot_path from the correct location (task.lerobot_path, not task.dataset.lerobot_path)
        # The override from train.sh sets task.lerobot_path
        lerobot_path = cfg.task.lerobot_path if hasattr(cfg.task, 'lerobot_path') else cfg.task.dataset.lerobot_path
        
        if self.is_main_process:
            print(f"Loading LeRobot dataset: {lerobot_path}")
        
        # Training dataset (LeRobot format)
        train_dataset = RobotLeRobotDataset(
            repo_id=lerobot_path,
            action_horizon=cfg.model.action_horizon,
            train=True,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
        )
        
        # Validation dataset
        val_dataset = RobotLeRobotDataset(
            repo_id=lerobot_path,
            action_horizon=cfg.model.action_horizon,
            train=False,
            val_split=cfg.training.val_split,
            seed=cfg.training.seed,
        )
        
        # Setup normalization statistics (following openpi's quantile normalization)
        stats = train_dataset.get_statistics()
        if 'action' in stats and 'state' in stats:
            model_for_stats = self.model.module if isinstance(self.model, DDP) else self.model
            model_for_stats.set_normalization_statistics(
                action_q01=stats['action']['q01'],
                action_q99=stats['action']['q99'],
                state_q01=stats['state']['q01'],
                state_q99=stats['state']['q99'],
            )
            
            if self.is_main_process:
                print(f"Set normalization statistics:")
                print(f"  Action q01: {stats['action']['q01']}")
                print(f"  Action q99: {stats['action']['q99']}")
        
        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=cfg.training.seed,
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
        
        if self.is_main_process:
            print(f"Created dataloaders:")
            print(f"  Train: {len(train_dataset)} samples, {len(self.train_dataloader)} batches")
            print(f"  Val: {len(val_dataset)} samples, {len(self.val_dataloader)} batches")
    
    def _init_wandb(self, resuming: bool = False):
        """Initialize wandb logging (following openpi's init_wandb pattern)."""
        if not self.is_main_process:
            return
        
        cfg = self.cfg
        
        if not cfg.logging.wandb_enabled or cfg.logging.mode == "disabled":
            wandb.init(mode="disabled")
            return
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        wandb_id_file = checkpoint_dir / "wandb_id.txt"
        
        if resuming and wandb_id_file.exists():
            run_id = wandb_id_file.read_text().strip()
            wandb.init(
                id=run_id,
                resume="must",
                project=cfg.logging.project,
            )
            print(f"Resumed wandb run: {run_id}")
        else:
            wandb.init(
                name=cfg.logging.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                project=cfg.logging.project,
                tags=cfg.logging.tags,
                mode=cfg.logging.mode,
            )
            # Save wandb ID for resuming
            wandb_id_file.write_text(wandb.run.id)
            print(f"Started new wandb run: {wandb.run.id}")
    
    def run(self):
        """Main training loop."""
        # Initialize components
        if self.is_main_process:
            print("="*80)
            print("Initializing Pi0 Training")
            print("="*80)
        
        self._init_model()
        self._init_optimizer()
        self._init_dataloaders()
        
        # Check for existing checkpoints to resume
        resuming = self._load_checkpoint_if_exists()
        
        # Initialize wandb
        self._init_wandb(resuming=resuming)
        
        if self.is_main_process:
            print(f"\nStarting training from epoch {self.epoch}")
            print(f"Training for {self.cfg.training.num_epochs} epochs")
            print(f"Global step: {self.global_step}")
        
        # Training loop
        for epoch in range(self.epoch, self.cfg.training.num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.use_ddp and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Train epoch
            train_loss = self._train_epoch()
            
            # Validation
            if (epoch + 1) % self.cfg.training.val_every == 0:
                val_loss = self._validate()
                
                if self.is_main_process:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                    
                    if self.cfg.logging.wandb_enabled:
                        wandb.log({
                            "val_loss": val_loss,
                            "epoch": epoch + 1,
                        }, step=self.global_step)
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.training.checkpoint_every == 0:
                self._save_checkpoint()
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Cleanup
        if self.use_ddp:
            dist.barrier()
            dist.destroy_process_group()
        
        if self.is_main_process and self.cfg.logging.wandb_enabled:
            wandb.finish()
            print("\n" + "="*80)
            print("Training completed!")
            print("="*80)
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Follows openpi's training pattern with periodic logging.
        """
        model_to_train = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_train.model.train()
        
        # Reset periodic logging
        self.step_infos = []
        self.start_time = time.time()
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Train Epoch {self.epoch+1}",
            disable=not self.is_main_process
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass (compute loss)
            loss = model_to_train.compute_loss(batch)
            
            # Backward pass
            loss = loss / self.cfg.training.gradient_accumulate_every
            loss.backward()
            
            # Store for periodic logging
            self.step_infos.append({
                "loss": loss.item() * self.cfg.training.gradient_accumulate_every,
                "lr": self.optimizer.param_groups[0]['lr'],
            })
            
            # Gradient accumulation
            if (batch_idx + 1) % self.cfg.training.gradient_accumulate_every == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_to_train.parameters(),
                    self.cfg.training.max_grad_norm
                )
                
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Periodic logging (following openpi: every N steps, not every batch)
                if self.global_step % self.cfg.logging.log_every_n_steps == 0 and len(self.step_infos) > 0:
                    elapsed = time.time() - self.start_time
                    
                    # Aggregate stats
                    avg_loss = np.mean([info["loss"] for info in self.step_infos])
                    avg_lr = np.mean([info["lr"] for info in self.step_infos])
                    
                    # Log to wandb (openpi pattern)
                    if self.is_main_process and self.cfg.logging.wandb_enabled:
                        log_payload = {
                            "loss": avg_loss,
                            "learning_rate": avg_lr,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "grad_norm": grad_norm.item(),
                            "time_per_step": elapsed / len(self.step_infos) if len(self.step_infos) > 0 else 0,
                        }
                        wandb.log(log_payload, step=self.global_step)
                    
                    # Reset stats collection
                    self.step_infos = []
                    self.start_time = time.time()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4e}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "step": self.global_step
                })
        
        # Return average loss for epoch
        return np.mean([info["loss"] for info in self.step_infos]) if self.step_infos else 0.0
    
    def _validate(self) -> float:
        """Validate the model."""
        model_to_eval = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_eval.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", disable=not self.is_main_process):
                batch = self._move_batch_to_device(batch)
                loss = model_to_eval.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Handle nested dicts (like image dict)
                result[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in value.items()}
            else:
                result[key] = value
        return result
    
    def _save_checkpoint(self):
        """Save model checkpoint (following openpi's save_checkpoint pattern)."""
        if not self.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints" / str(self.global_step)
        tmp_checkpoint_dir = self.output_dir / "checkpoints" / f"tmp_{self.global_step}"
        
        # Create temp directory
        if tmp_checkpoint_dir.exists():
            import shutil
            shutil.rmtree(tmp_checkpoint_dir)
        tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model to save
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Save model using safetensors (following openpi)
        import safetensors.torch
        safetensors.torch.save_model(model_to_save.model, tmp_checkpoint_dir / "model.safetensors")
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), tmp_checkpoint_dir / "optimizer.pt")
        
        # Save training metadata
        metadata = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }
        torch.save(metadata, tmp_checkpoint_dir / "metadata.pt")
        
        # Atomically move temp to final location
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
        tmp_checkpoint_dir.rename(checkpoint_dir)
        
        print(f"Saved checkpoint at step {self.global_step} -> {checkpoint_dir}")
        
        # Log to wandb
        if self.cfg.logging.wandb_enabled:
            wandb.log({"checkpoint_step": self.global_step}, step=self.global_step)
    
    def _load_checkpoint_if_exists(self) -> bool:
        """
        Load the latest checkpoint if it exists.
        
        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.cfg.training.resume:
            return False
        
        checkpoint_dir = self.output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return False
        
        # Find latest checkpoint
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not checkpoints:
            return False
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name))
        
        if self.is_main_process:
            print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Load model
        import safetensors.torch
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        state_dict = safetensors.torch.load_file(latest_checkpoint / "model.safetensors", device='cpu')
        model_to_load.model.load_state_dict(state_dict)
        
        # Load optimizer
        if (latest_checkpoint / "optimizer.pt").exists():
            optimizer_state = torch.load(latest_checkpoint / "optimizer.pt", map_location='cpu')
            self.optimizer.load_state_dict(optimizer_state)
        
        # Load metadata
        if (latest_checkpoint / "metadata.pt").exists():
            metadata = torch.load(latest_checkpoint / "metadata.pt")
            self.global_step = metadata.get("global_step", 0)
            self.epoch = metadata.get("epoch", 0)
            self.best_loss = metadata.get("best_loss", float('inf'))
        
        if self.is_main_process:
            print(f"Resumed from: epoch={self.epoch}, step={self.global_step}")
        
        return True

