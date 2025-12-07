if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import pathlib
import copy
import random
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import wandb

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def is_distributed():
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed():
    """Initialize distributed training if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        
        print(f"[Rank {rank}/{world_size}] Initialized distributed training on GPU {local_rank}")
        return True, local_rank
    return False, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def reduce_tensor(tensor):
    """Reduce tensor across all processes."""
    if not is_distributed():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


# ============================================================================
# Distributed-aware BatchSampler
# ============================================================================

class DistributedBatchSampler:
    """Batch sampler that supports distributed training."""
    
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, 
                 seed: int = 0, drop_last: bool = True, rank: int = 0, world_size: int = 1):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        
        # Calculate batches per process
        self.total_batches = data_size // batch_size
        self.batches_per_rank = self.total_batches // world_size
        self.discard = data_size - batch_size * self.total_batches
        
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling (important for distributed training)."""
        self.epoch = epoch
        
    def __iter__(self):
        # Use epoch-dependent seed for shuffling
        rng = np.random.default_rng(self.seed + self.epoch)
        
        if self.shuffle:
            perm = rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
            
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.total_batches, self.batch_size)
        
        # Each rank gets a subset of batches
        start_batch = self.rank * self.batches_per_rank
        end_batch = start_batch + self.batches_per_rank
        
        for i in range(start_batch, end_batch):
            yield perm[i]
            
    def __len__(self):
        return self.batches_per_rank


class BatchSampler:
    """Original batch sampler for single-GPU training."""
    
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, 
                 seed: int = 0, drop_last: bool = True):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


# ============================================================================
# RobotWorkspace with DDP Support
# ============================================================================

class RobotWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # Setup distributed training if available
        self.distributed, self.local_rank = setup_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # Set seed (different seed per rank for data augmentation diversity)
        seed = cfg.training.seed + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # Configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                if is_main_process():
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        # Create dataloader with distributed support
        train_dataloader, train_sampler = self._create_dataloader(
            dataset, 
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers,
            seed=cfg.training.seed
        )
        normalizer = dataset.get_normalizer()

        # Configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader, _ = self._create_dataloader(
            val_dataset,
            batch_size=cfg.val_dataloader.batch_size,
            shuffle=False,
            num_workers=cfg.val_dataloader.num_workers,
            pin_memory=cfg.val_dataloader.pin_memory,
            persistent_workers=cfg.val_dataloader.persistent_workers,
            seed=cfg.training.seed
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # Configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner = None

        # Configure logging (only on main process)
        if is_main_process():
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update({
                "output_dir": self.output_dir,
                "distributed": self.distributed,
                "world_size": self.world_size,
            })

        # Configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Device transfer
        if self.distributed:
            device = torch.device(f'cuda:{self.local_rank}')
        else:
            device = torch.device(cfg.training.device)
            
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Wrap model with DDP if distributed
        model_for_training = self.model
        if self.distributed:
            model_for_training = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            if is_main_process():
                print(f"Using DistributedDataParallel with {self.world_size} GPUs")

        # Save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path) if is_main_process() else None
        
        try:
            if json_logger:
                json_logger.__enter__()
                
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                # Set epoch for distributed sampler
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    if self.distributed:
                        model_for_training.module.obs_encoder.eval()
                        model_for_training.module.obs_encoder.requires_grad_(False)
                    else:
                        self.model.obs_encoder.eval()
                        self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                
                # Only show progress bar on main process
                dataloader_iter = train_dataloader
                if is_main_process():
                    dataloader_iter = tqdm.tqdm(
                        train_dataloader, 
                        desc=f"Training epoch {self.epoch}", 
                        leave=False, 
                        mininterval=cfg.training.tqdm_interval_sec
                    )
                
                for batch_idx, batch in enumerate(dataloader_iter):
                    batch = dataset.postprocess(batch, device)
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # Compute loss
                    raw_loss = model_for_training.compute_loss(batch) if not self.distributed else model_for_training.module.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # Step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # Update ema (only on main process or sync across all)
                    if cfg.training.use_ema:
                        # Use the underlying model for EMA, not the DDP wrapper
                        ema.step(self.model)

                    # Logging
                    raw_loss_cpu = raw_loss.item()
                    
                    # Reduce loss across all processes for accurate logging
                    if self.distributed:
                        loss_tensor = torch.tensor([raw_loss_cpu], device=device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        raw_loss_cpu = loss_tensor.item() / self.world_size
                    
                    if is_main_process():
                        dataloader_iter.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        if is_main_process():
                            json_logger.log(step_log)
                            wandb.log(step_log)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # At the end of each epoch
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # Run validation (only on main process)
                if is_main_process() and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                # Run diffusion sampling on a training batch (only on main process)
                if is_main_process() and (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # Checkpoint (only on main process)
                if is_main_process() and ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    self.save_checkpoint(f'checkpoints/{save_name}/{self.epoch + 1}.ckpt')
                
                # Synchronize all processes before next epoch
                if self.distributed:
                    dist.barrier()
                
                # ========= eval end for this epoch ==========
                policy.train()

                # End of epoch
                if is_main_process():
                    json_logger.log(step_log)
                    wandb.log(step_log)
                self.global_step += 1
                self.epoch += 1
                
        finally:
            if json_logger:
                json_logger.__exit__(None, None, None)
            cleanup_distributed()

    def _create_dataloader(self, dataset, *, batch_size: int, shuffle: bool, 
                           num_workers: int, pin_memory: bool, 
                           persistent_workers: bool, seed: int = 0):
        """Create dataloader with optional distributed support."""
        
        def collate(x):
            assert len(x) == 1
            return x[0]
        
        sampler = None
        
        if self.distributed and len(dataset) > 0:
            # Use distributed batch sampler
            batch_sampler = DistributedBatchSampler(
                len(dataset), 
                batch_size, 
                shuffle=shuffle, 
                seed=seed, 
                drop_last=True,
                rank=self.rank,
                world_size=self.world_size
            )
            sampler = batch_sampler
        else:
            # Use original batch sampler for single GPU
            batch_sampler = BatchSampler(
                len(dataset), 
                batch_size, 
                shuffle=shuffle, 
                seed=seed, 
                drop_last=True
            )
            sampler = batch_sampler
        
        dataloader = DataLoader(
            dataset, 
            collate_fn=collate, 
            sampler=sampler, 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            persistent_workers=persistent_workers if num_workers > 0 else False
        )
        
        return dataloader, sampler


def create_dataloader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, 
                      pin_memory: bool, persistent_workers: bool, seed: int = 0):
    """Legacy function for backward compatibility."""
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)
    def collate(x):
        assert len(x) == 1
        return x[0]
    dataloader = DataLoader(
        dataset, 
        collate_fn=collate, 
        sampler=batch_sampler, 
        num_workers=num_workers, 
        pin_memory=False, 
        persistent_workers=persistent_workers
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
