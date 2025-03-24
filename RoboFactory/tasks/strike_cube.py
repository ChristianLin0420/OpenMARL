from typing import Any, Dict

import numpy as np
import sapien
import torch
import json
import yaml

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
import utils.scenes

@register_env("StrikeCube-rf", max_episode_steps=350)
class StrikeCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda"]
    cube_half_size = 0.02
    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        assert 'config' in kwargs
        with open(kwargs['config'], 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs['config']
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs =[]
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if pose['type'] == 'pose':
                sensor['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                sensor['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**sensor))
        return all_camera_configs

    @property
    def _default_human_render_camera_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs =[]
        for render in render_cfg:
            pose = render['pose']
            if pose['type'] == 'pose':
                render['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        init_poses = []
        for agent_cfg in self.cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        scene_name = self.cfg['scene']['name']
        scene_builder = getattr(utils.scenes, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=self.cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)

    def evaluate(self):
        local_target_matrix = np.asarray(self.annotation_data['hammer']['functional_matrix'][0])
        local_target_matrix[:3,3] *= self.annotation_data['hammer']['scale']
        target_matrix = self.hammer.pose.to_transformation_matrix()[0] @ local_target_matrix
        target_pos = target_matrix[:3,3]
        delta_x = self.cube.pose.p[:, 0] - target_pos[0]
        delta_y = self.cube.pose.p[:, 1] - target_pos[1]
        delta_z = self.cube.pose.p[:, 2]  + self.cube_half_size - target_pos[2]
        # target_matrix = torch.tensor(target_matrix[:3,3])
        success = torch.sqrt(delta_x**2 + delta_y**2 + delta_z**2) < 0.025
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}
