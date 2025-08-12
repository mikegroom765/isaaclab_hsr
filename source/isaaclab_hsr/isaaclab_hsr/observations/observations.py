
"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, TiledCamera, ContactSensor
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from isaaclab.utils.math import transform_points, project_points, quat_conjugate, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def hsr_base_odom_pos(env: ManagerBasedEnv, base_velocity_action_name: str) -> torch.Tensor:
    """The base odom 2D pose (x, y, rz) of the HSR robot in the environment frame."""
    if base_velocity_action_name not in env.action_manager.active_terms:
        raise ValueError(f"Action term {base_velocity_action_name} is not an active action term.")
    return env.action_manager.get_term(base_velocity_action_name).wheel_odometry

def hsr_base_vel(env: ManagerBasedEnv, base_velocity_action_name: str) -> torch.Tensor:
    """The base velocity (linear x, linear y, angular z) of the HSR robot in the world frame
       calculated using forward dynamics from current joint velocities and steering angle."""
    if base_velocity_action_name not in env.action_manager.active_terms:
        raise ValueError(f"Action term {base_velocity_action_name} is not an active action term.")
    return env.action_manager.get_term(base_velocity_action_name).base_velocity
