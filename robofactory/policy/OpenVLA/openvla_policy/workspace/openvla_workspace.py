"""
OpenVLA training workspace with multi-GPU support.

This module implements the training loop for OpenVLA models using PyTorch DDP
for distributed training across multiple GPUs.
"""

import os
import json
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
from PIL import Image

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
        
        # Initialize distributed training using environment variables set by torchrun
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        
        # Initialize process group if distributed and not already initialized
        if self.is_distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Use rank 1 for logging to avoid conflicts with rank 0
        # (rank 0 often has extra work like gradient aggregation)
        self.is_main_process = (self.world_size == 1) or (self.rank == 1)
        
        # Set random seed
        self._set_seed(cfg.training.seed)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0  # For plateau-based early stopping
        self.wandb_run_id = None  # For resuming wandb runs
        
        # Model and optimizer (will be initialized in run())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Step-level metrics for detailed logging
        self.step_losses = []
        
        # Simulation evaluation environment (lazy init)
        self.eval_env = None
    
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
        
        # Initialize wandb with resume support
        if self.is_main_process and cfg.logging.mode == "online":
            wandb_kwargs = {
                "project": cfg.logging.project,
                "name": cfg.logging.name,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "dir": str(self.output_dir),
                "mode": cfg.logging.mode,
                "tags": list(cfg.logging.tags) if hasattr(cfg.logging, 'tags') else None,
            }
            # Resume wandb run if we have a saved run ID
            if self.wandb_run_id is not None:
                wandb_kwargs["id"] = self.wandb_run_id
                wandb_kwargs["resume"] = "must"
                print(f"Resuming wandb run: {self.wandb_run_id}")
            
            wandb.init(**wandb_kwargs)
            self.wandb_run_id = wandb.run.id
            
            # Log model architecture info
            wandb.config.update({
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }, allow_val_change=True)
        
        # Training loop
        early_stopped = False
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
                
                # Add learning rate to metrics
                if hasattr(cfg.logging, 'log_learning_rate') and cfg.logging.log_learning_rate:
                    metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                if cfg.logging.mode == "online":
                    wandb.log(metrics, step=self.global_step)
                
                print(f"Epoch {epoch}: {metrics}")
            
            # Plateau-based early stopping (stop when loss stops improving)
            current_loss = train_metrics.get('train_loss', float('inf'))
            if hasattr(cfg.training, 'early_stopping') and cfg.training.early_stopping:
                patience = getattr(cfg.training, 'early_stopping_patience', 10)
                min_delta = getattr(cfg.training, 'early_stopping_min_delta', 1e-7)
                
                # Check if loss improved by at least min_delta
                if current_loss < self.best_loss - min_delta:
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if self.is_main_process:
                        print(f"No improvement for {self.epochs_without_improvement}/{patience} epochs. "
                              f"Best: {self.best_loss:.8f}, Current: {current_loss:.8f}")
                    
                    if self.epochs_without_improvement >= patience:
                        if self.is_main_process:
                            print(f"Early stopping triggered! No improvement for {patience} epochs.")
                        early_stopped = True
            
            # Update best loss and save best checkpoint
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                if self.is_main_process:
                    self.save_checkpoint(name='best.ckpt')
                if self.is_distributed:
                    dist.barrier()
            
            # Simulation evaluation (only on main process, with barrier sync)
            if hasattr(cfg.training, 'eval_in_sim') and cfg.training.eval_in_sim:
                eval_every = getattr(cfg.training, 'eval_sim_every_n_epochs', 10)
                if (epoch + 1) % eval_every == 0:
                    sim_metrics = self._evaluate_in_simulation()
                    if self.is_main_process and sim_metrics:
                        metrics.update(sim_metrics)
                        print(f"Simulation eval: {sim_metrics}")
                    # Barrier to ensure all ranks wait for simulation evaluation to complete
                    if self.is_distributed:
                        dist.barrier()
            
            # Save checkpoint (only main process saves, others wait)
            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                if self.is_main_process:
                    self.save_checkpoint()
                if self.is_distributed:
                    dist.barrier()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Break if early stopped
            if early_stopped:
                break
        
        # Final save
        if self.is_main_process:
            self.save_checkpoint(name='final.ckpt')
            if cfg.logging.mode == "online":
                # Log final summary
                wandb.run.summary["final_train_loss"] = train_metrics.get('train_loss', 0)
                wandb.run.summary["best_loss"] = self.best_loss
                wandb.run.summary["total_epochs"] = self.epoch + 1
                wandb.run.summary["early_stopped"] = early_stopped
                wandb.finish()
    
    def _train_epoch(self):
        """Train for one epoch."""
        cfg = self.cfg
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        self.step_losses = []  # Reset step losses
        
        # Get logging frequency
        log_every = getattr(cfg.logging, 'log_every_n_steps', 10)
        
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
                # Compute gradient norm before clipping (for logging)
                grad_norm = None
                if self.is_main_process and hasattr(cfg.logging, 'log_gradients') and cfg.logging.log_gradients:
                    grad_norm = self._compute_grad_norm()
                
                # Gradient clipping
                if cfg.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        cfg.training.max_grad_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Step-level wandb logging
                step_loss = loss.item() * cfg.training.gradient_accumulate_every
                self.step_losses.append(step_loss)
                
                if self.is_main_process and cfg.logging.mode == "online":
                    if self.global_step % log_every == 0:
                        step_metrics = {
                            'train/step_loss': step_loss,
                            'train/global_step': self.global_step,
                        }
                        if grad_norm is not None:
                            step_metrics['train/grad_norm'] = grad_norm
                        if hasattr(cfg.logging, 'log_learning_rate') and cfg.logging.log_learning_rate:
                            step_metrics['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
                        
                        wandb.log(step_metrics, step=self.global_step)
            
            # Logging
            current_loss = loss.item() * cfg.training.gradient_accumulate_every
            total_loss += current_loss
            num_batches += 1
            
            if self.is_main_process:
                pbar.set_postfix({'loss': current_loss})
            
            # Early exit for debugging
            if cfg.training.debug and batch_idx >= cfg.training.max_train_steps:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute additional epoch metrics
        epoch_metrics = {
            'train_loss': avg_loss,
            'train_loss_std': np.std(self.step_losses) if self.step_losses else 0.0,
            'train_loss_min': min(self.step_losses) if self.step_losses else 0.0,
            'train_loss_max': max(self.step_losses) if self.step_losses else 0.0,
        }
        
        return epoch_metrics
    
    def _compute_grad_norm(self) -> float:
        """Compute the total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    @torch.no_grad()
    def _evaluate_in_simulation(self) -> Dict[str, Any]:
        """
        Run model evaluation in simulation environment and log results/videos to wandb.
        
        Returns:
            Dictionary with evaluation metrics (success_rate, etc.)
        """
        if not self.is_main_process:
            return {}
        
        cfg = self.cfg
        
        # Setup headless rendering environment BEFORE importing sim libraries
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["DISPLAY"] = ""
        
        # Create ALL missing EGL directories that SAPIEN might check for headless rendering
        egl_dirs = [
            "/usr/share/glvnd/egl_vendor.d",
            "/etc/glvnd/egl_vendor.d",
            "/usr/share/egl/egl_external_platform.d",
        ]
        for egl_dir in egl_dirs:
            if not os.path.exists(egl_dir):
                try:
                    os.makedirs(egl_dir, exist_ok=True)
                except PermissionError:
                    pass  # Silently skip if no permission
        
        try:
            import gymnasium as gym
            from robofactory.envs.wrappers import RecordEpisodeMA
        except (ImportError, FileNotFoundError, OSError) as e:
            print(f"Warning: Could not import simulation dependencies: {e}")
            print("Skipping simulation evaluation - graphics environment not available")
            return {}
        
        # Get evaluation settings
        num_episodes = getattr(cfg.training, 'eval_sim_episodes', 3)
        max_steps = getattr(cfg.training, 'eval_sim_max_steps', 200)
        
        # Determine environment ID from task
        env_id = cfg.task.name + '-rf' if hasattr(cfg.task, 'name') else None
        if env_id is None:
            print("Warning: Could not determine environment ID for simulation eval")
            return {}
        
        print(f"\nRunning simulation evaluation: {num_episodes} episodes in {env_id}")
        
        try:
            # Create environment
            env_kwargs = dict(
                obs_mode='rgb',
                control_mode='pd_ee_delta_pose',
                render_mode='rgb_array',
                num_envs=1,
                sim_backend='cpu',
            )
            
            # Add config path if available
            if hasattr(cfg.task, 'config_path'):
                env_kwargs['config'] = cfg.task.config_path
            
            env = gym.make(env_id, **env_kwargs)
            
            # Set model to eval mode
            self.model.eval()
            
            # Run evaluation episodes
            success_count = 0
            total_rewards = []
            all_frames = []
            
            for ep in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0.0
                episode_frames = []
                
                for step in range(max_steps):
                    # Get image observation
                    if isinstance(obs, dict):
                        # Handle different observation formats
                        if 'rgb' in obs:
                            image = obs['rgb']
                        elif 'image' in obs:
                            image = obs['image']
                        elif 'sensor_data' in obs:
                            # Get first camera's RGB
                            sensor_data = obs['sensor_data']
                            cam_key = list(sensor_data.keys())[0]
                            image = sensor_data[cam_key].get('rgb', None)
                        else:
                            image = None
                    else:
                        image = obs
                    
                    if image is None:
                        print(f"Warning: Could not extract image from observation")
                        break
                    
                    # Convert to tensor format expected by model
                    if isinstance(image, np.ndarray):
                        # Normalize to [0, 1] and convert to (C, H, W)
                        if image.max() > 1.0:
                            image = image.astype(np.float32) / 255.0
                        if image.ndim == 3 and image.shape[-1] == 3:
                            image = np.transpose(image, (2, 0, 1))
                        image = torch.from_numpy(image).float()
                    
                    # Get instruction from task config or use default
                    instruction = getattr(cfg.task, 'instruction', 'complete the task')
                    
                    # Predict action
                    action = self.model.predict_action(
                        image.to(self.model.device),
                        instruction,
                        do_sample=False
                    )
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    # Capture frame for video (every 5 steps to reduce size)
                    if step % 5 == 0:
                        frame = env.render()
                        if frame is not None:
                            episode_frames.append(frame)
                    
                    if terminated or truncated:
                        break
                
                # Check success
                if info.get('success', False):
                    success_count += 1
                
                total_rewards.append(episode_reward)
                
                # Keep frames from last episode for video
                if ep == num_episodes - 1:
                    all_frames = episode_frames
            
            env.close()
            
            # Compute metrics
            success_rate = success_count / num_episodes
            avg_reward = np.mean(total_rewards)
            
            metrics = {
                'eval/sim_success_rate': success_rate,
                'eval/sim_avg_reward': avg_reward,
                'eval/sim_episodes': num_episodes,
            }
            
            # Log to wandb
            if cfg.logging.mode == "online":
                wandb.log(metrics, step=self.global_step)
                
                # Log video if we have frames
                if all_frames and len(all_frames) > 0:
                    try:
                        # Stack frames into video array (T, H, W, C)
                        video_array = np.stack(all_frames, axis=0)
                        # wandb expects (T, C, H, W) for video
                        if video_array.ndim == 4 and video_array.shape[-1] == 3:
                            video_array = np.transpose(video_array, (0, 3, 1, 2))
                        
                        wandb.log({
                            "eval/sim_video": wandb.Video(video_array, fps=10, format="mp4")
                        }, step=self.global_step)
                    except Exception as e:
                        print(f"Warning: Could not log video to wandb: {e}")
            
            # Set model back to train mode
            self.model.train()
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Simulation evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
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
        """Save checkpoint with full training state for proper resuming."""
        if not self.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / 'checkpoints' / Path(self.cfg.task.dataset.rlds_path).stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if name is None:
            name = f'epoch_{self.epoch + 1}.ckpt'
        
        checkpoint_path = checkpoint_dir / name
        
        # Save model (LoRA weights)
        model_save_path = str(checkpoint_path.with_suffix(''))
        self.model.save_pretrained(model_save_path)
        
        # Save complete training state
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'wandb_run_id': self.wandb_run_id,
            'model_save_path': model_save_path,
        }
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(state, checkpoint_path)
        
        # Also save as latest
        latest_path = checkpoint_dir.parent / 'latest.ckpt'
        torch.save(state, latest_path)
        
        # Save a JSON metadata file for easy inspection
        metadata = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': float(self.best_loss),
            'wandb_run_id': self.wandb_run_id,
            'checkpoint_path': str(checkpoint_path),
        }
        metadata_path = checkpoint_dir.parent / 'training_state.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint and restore complete training state."""
        path = Path(path)
        state = torch.load(path, map_location='cpu')
        
        # Restore training state
        self.epoch = state.get('epoch', 0) + 1  # Start from next epoch
        self.global_step = state.get('global_step', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        # Support both old and new variable names for backwards compatibility
        self.epochs_without_improvement = state.get('epochs_without_improvement', 
                                                     state.get('epochs_below_threshold', 0))
        self.wandb_run_id = state.get('wandb_run_id', None)
        
        # Restore optimizer state
        if 'optimizer_state' in state:
            self.optimizer.load_state_dict(state['optimizer_state'])
            # Move optimizer state to correct device
            for param_state in self.optimizer.state.values():
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        param_state[k] = v.to(self.model.device)
        
        # Restore scheduler state
        if 'scheduler_state' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])
        
        # Load model weights if path is specified
        model_save_path = state.get('model_save_path')
        if model_save_path and Path(model_save_path).exists():
            # For LoRA models, we need to load the adapter weights
            # The base model is already loaded, so we just need to load the LoRA weights
            try:
                from peft import PeftModel
                if hasattr(self.model, 'model'):
                    # Check if already a PeftModel
                    base_model = self.model.model.module if self.is_distributed else self.model.model
                    if hasattr(base_model, 'load_adapter'):
                        base_model.load_adapter(model_save_path, adapter_name="default")
                        print(f"Loaded LoRA adapter from {model_save_path}")
            except Exception as e:
                print(f"Warning: Could not load LoRA adapter: {e}")
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Resuming from epoch {self.epoch}, global_step {self.global_step}")
        if self.wandb_run_id:
            print(f"  Will resume wandb run: {self.wandb_run_id}")


if __name__ == "__main__":
    print("OpenVLA workspace module loaded successfully!")

