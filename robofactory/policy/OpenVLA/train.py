"""
OpenVLA training script for RoboFactory.

Usage:
    Single GPU:
        python train.py --config-name=robot_openvla.yaml
    
    Multi-GPU (torchrun):
        torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --config-name=robot_openvla.yaml
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openvla_policy'))

# Use line-buffering for stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
from pathlib import Path

# Register custom resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / 'openvla_policy' / 'config'),
    config_name='robot_openvla'
)
def main(cfg: OmegaConf):
    """Main training function."""
    # Resolve config
    OmegaConf.resolve(cfg)
    
    # Print config
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Import workspace
    from openvla_policy.workspace.openvla_workspace import OpenVLAWorkspace
    
    # Create workspace
    workspace = OpenVLAWorkspace(cfg)
    
    # Run training
    print(f"\nStarting training for task: {cfg.task_name}")
    workspace.run()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

