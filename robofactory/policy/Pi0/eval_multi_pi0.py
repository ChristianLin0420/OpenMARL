"""
Multi-agent evaluation script for Pi0/Pi0.5 policies.

This script evaluates trained Pi0/Pi0.5 models on multi-agent RoboFactory tasks.
Follows the pattern from OpenVLA's evaluation script.
"""

import sys
sys.path.append('./')
sys.path.insert(0, './policy/Pi0')

import torch
import os
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

import gymnasium as gym
import sapien

from robofactory.tasks import *
from mani_skill.envs.sapien_env import BaseEnv
from robofactory.utils.wrappers.record import RecordEpisodeMA
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from pi0_policy.policy.pi0_policy import Pi0Policy
from pi0_policy.utils.task_instructions import get_task_instruction


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Evaluate Pi0/Pi0.5 multi-agent policies")
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to task config file'
    )
    parser.add_argument(
        '--policy_type',
        type=str,
        default='pi0',
        choices=['pi0', 'pi05'],
        help='Policy type: pi0 or pi05'
    )
    parser.add_argument(
        '--data_num',
        type=int,
        default=150,
        help='Number of training samples used'
    )
    parser.add_argument(
        '--checkpoint_step',
        type=int,
        default=5000,
        help='Checkpoint step number'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=10000,
        help='Random seed'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=250,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--debug',
        type=int,
        default=0,
        help='Debug mode (0=off, 1=on)'
    )
    parser.add_argument(
        '--record_dir',
        type=str,
        default='./eval_video/{env_id}',
        help='Directory to save evaluation videos'
    )
    parser.add_argument(
        '--render_mode',
        type=str,
        default='rgb_array',
        help='Render mode'
    )
    parser.add_argument(
        '--obs_mode',
        type=str,
        default='rgb',
        help='Observation mode'
    )
    parser.add_argument(
        '--control_mode',
        type=str,
        default='pd_joint_pos',
        help='Control mode'
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=1,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=50,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for inference'
    )
    
    return parser.parse_args()


def get_model_input(observation, agent_pos, agent_id, num_cameras=3):
    """
    Extract model input from observation.
    
    Pi0 requires 3 camera views:
    - base_0_rgb: head_camera (side view)
    - left_wrist_0_rgb: global_camera (overhead view)
    - right_wrist_0_rgb: wrist_camera (gripper view)
    
    Args:
        observation: Environment observation
        agent_pos: Agent joint positions
        agent_id: Agent ID
        num_cameras: Number of camera views
        
    Returns:
        Dictionary with 'images' and 'state'
    """
    images = []
    
    # Extract all 3 cameras (head, global, wrist)
    camera_keys = [
        f'head_camera_agent{agent_id}',    # [0] Side view
        f'global_camera_agent{agent_id}',  # [1] Overhead view
        f'wrist_camera_agent{agent_id}',   # [2] Gripper view
    ]
    
    for cam_key in camera_keys[:num_cameras]:
        if cam_key in observation['sensor_data']:
            img = observation['sensor_data'][cam_key]['rgb'].squeeze(0).numpy()
            
            # Convert to HWC format if needed
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            images.append(img)
    
    # Stack images: [num_cameras, H, W, 3]
    images = np.array(images)
    
    return {
        'images': images,
        'state': agent_pos,
    }


def load_task_instruction(task_name: str) -> str:
    """Get language instruction for a task."""
    return get_task_instruction(task_name)


class Pi0PolicyWrapper:
    """Wrapper for Pi0 policy to match evaluation interface."""
    
    def __init__(
        self, 
        task_name: str, 
        checkpoint_step: int, 
        data_num: int, 
        agent_id: int = 0,
        policy_type: str = 'pi0',
        device: str = 'cuda:0'
    ):
        """Initialize policy wrapper."""
        # Construct checkpoint path
        checkpoint_dir = Path(f'data/outputs') / 'checkpoints' / str(checkpoint_step)
        
        # Try to find checkpoint in nested structure
        if not checkpoint_dir.exists():
            # Look in dated output structure
            output_base = Path(f'data/outputs')
            if output_base.exists():
                # Find most recent run
                date_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], reverse=True)
                for date_dir in date_dirs:
                    for run_dir in sorted(date_dir.iterdir(), reverse=True):
                        candidate = run_dir / 'checkpoints' / str(checkpoint_step)
                        if candidate.exists():
                            checkpoint_dir = candidate
                            break
                    if checkpoint_dir.exists():
                        break
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found at step {checkpoint_step}")
        
        print(f"Loading Pi0 checkpoint from {checkpoint_dir}")
        
        # Create policy
        self.policy = Pi0Policy(
            checkpoint_path=str(checkpoint_dir),
            task_name=task_name,
            device=device,
        )
        
        # Action horizon for chunking
        self.action_horizon = self.policy.cfg.model.action_horizon
        self.action_buffer = []
        self.action_idx = 0
        
        # Observation buffer
        self.obs_buffer = []
        
        print(f"Initialized {policy_type} policy for {task_name} (Agent {agent_id})")
        print(f"Action horizon: {self.action_horizon}")
        
    def update_obs(self, observation):
        """Update observation buffer."""
        self.obs_buffer.append(observation)
        if len(self.obs_buffer) > 3:
            self.obs_buffer.pop(0)
    
    def get_action(self, observation=None):
        """
        Get action from policy.
        
        Pi0 predicts action sequences (action chunking), so we:
        1. Predict a new sequence when buffer is empty
        2. Return actions from the buffer one at a time
        """
        if observation is None and len(self.obs_buffer) > 0:
            observation = self.obs_buffer[-1]
        
        # Check if we need to predict a new action sequence
        if len(self.action_buffer) == 0 or self.action_idx >= len(self.action_buffer):
            # Predict new action sequence
            action_sequence = self.policy.predict_action(observation)
            
            # Store in buffer (action_sequence is [action_horizon, action_dim])
            self.action_buffer = [action_sequence[i] for i in range(len(action_sequence))]
            self.action_idx = 0
        
        # Get current action from buffer
        action = self.action_buffer[self.action_idx]
        self.action_idx += 1
        
        # Return as list of repeated actions (for multi-step execution compatibility)
        return [action for _ in range(6)]
    
    def get_last_obs(self):
        """Get last observation."""
        return self.obs_buffer[-1] if self.obs_buffer else None


def main(args):
    """Main evaluation function."""
    np.set_printoptions(suppress=True, precision=5)
    verbose = args.debug == 1
    
    # Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    env_id = config['env_id']
    task_name = env_id.replace('-v1', '')
    
    print(f"Evaluating {args.policy_type} on {task_name}")
    print(f"Data num: {args.data_num}, Checkpoint: {args.checkpoint_step}")
    
    # Get number of agents from config
    num_agents = config.get('num_agents', 1)
    
    # Create environment
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        num_envs=args.num_envs,
    )
    
    env = gym.make(env_id, **env_kwargs)
    
    # Wrap with video recording
    record_dir = args.record_dir.format(env_id=env_id)
    env = RecordEpisodeMA(
        env,
        output_dir=record_dir,
        save_trajectory=True,
        save_video=True,
        info_on_video=True,
        max_steps_per_video=args.max_steps,
    )
    
    # Create motion planners (for invalid action handling)
    motion_planners = []
    for agent_id in range(num_agents):
        planner = PandaArmMotionPlanningSolver(
            env.unwrapped,
            agent_idx=agent_id,
            joint_vel_limits=0.75,
            joint_acc_limits=0.75,
        )
        motion_planners.append(planner)
    
    # Load policies for each agent
    policies = []
    for agent_id in range(num_agents):
        policy = Pi0PolicyWrapper(
            task_name=task_name,
            checkpoint_step=args.checkpoint_step,
            data_num=args.data_num,
            agent_id=agent_id,
            policy_type=args.policy_type,
            device=args.device,
        )
        policies.append(policy)
    
    # Evaluation loop
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    
    for episode_idx in range(args.num_eval_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode_idx + 1}/{args.num_eval_episodes}")
        print(f"{'='*80}")
        
        # Reset environment
        obs, info = env.reset(seed=args.seed + episode_idx)
        
        # Reset policies
        for policy in policies:
            policy.obs_buffer = []
            policy.action_buffer = []
            policy.action_idx = 0
        
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done and step_count < args.max_steps:
            # Get actions from each agent's policy
            actions = []
            
            for agent_id in range(num_agents):
                # Extract observation for this agent
                agent_pos = obs['agent']['qpos'][agent_id].cpu().numpy()
                
                model_input = get_model_input(obs, agent_pos, agent_id)
                
                # Update policy observation buffer
                policies[agent_id].update_obs(model_input)
                
                # Get action
                action_sequence = policies[agent_id].get_action(model_input)
                action = action_sequence[0]  # Take first action from sequence
                
                actions.append(action)
            
            # Stack actions for all agents
            actions = np.array(actions)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(actions)
            
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            if verbose:
                print(f"Step {step_count}: reward={reward:.4f}, done={done}")
        
        # Episode results
        success = info.get('success', False)
        if success:
            success_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"Episode {episode_idx + 1} finished:")
        print(f"  Success: {success}")
        print(f"  Reward: {episode_reward:.4f}")
        print(f"  Length: {step_count}")
        print(f"  Success rate so far: {success_count}/{episode_idx + 1} ({100*success_count/(episode_idx+1):.1f}%)")
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"Task: {task_name}")
    print(f"Policy: {args.policy_type}")
    print(f"Episodes: {args.num_eval_episodes}")
    print(f"Success rate: {success_count}/{args.num_eval_episodes} ({100*success_count/args.num_eval_episodes:.1f}%)")
    print(f"Average reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'='*80}")
    
    env.close()
    
    return {
        'success_rate': success_count / args.num_eval_episodes,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
    }


if __name__ == "__main__":
    args = parse_args()
    main(args)

