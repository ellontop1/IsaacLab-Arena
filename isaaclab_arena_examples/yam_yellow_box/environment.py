# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena composition for a provisional YAM yellow-box placement workspace."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena_examples.yam_yellow_box.config import YellowBoxConfig


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _reset_lift_progress(env, env_ids):
    import torch

    if not hasattr(env, "_yam_box_lifted"):
        env._yam_box_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    selected = slice(None) if env_ids is None else env_ids
    env._yam_box_lifted[selected] = False


def task_success(env, config: YellowBoxConfig):
    """Require a lift followed by the whole upright box on target, at rest, with open grippers."""
    import torch

    box = env.scene["yellow_box"]
    position = _tensor(box.data.root_pos_w) - _tensor(env.scene.env_origins)
    rotation = _tensor(box.data.root_quat_w)
    quaternion_x, quaternion_y, quaternion_z, quaternion_w = rotation.unbind(-1)
    rotation_xx = 1 - 2 * (quaternion_y.square() + quaternion_z.square())
    rotation_xy = 2 * (quaternion_x * quaternion_y - quaternion_w * quaternion_z)
    rotation_xz = 2 * (quaternion_x * quaternion_z + quaternion_w * quaternion_y)
    rotation_yx = 2 * (quaternion_x * quaternion_y + quaternion_w * quaternion_z)
    rotation_yy = 1 - 2 * (quaternion_x.square() + quaternion_z.square())
    rotation_yz = 2 * (quaternion_y * quaternion_z - quaternion_w * quaternion_x)
    half_x = 0.5 * (
        rotation_xx.abs() * config.box_size[0]
        + rotation_xy.abs() * config.box_size[1]
        + rotation_xz.abs() * config.box_size[2]
    )
    half_y = 0.5 * (
        rotation_yx.abs() * config.box_size[0]
        + rotation_yy.abs() * config.box_size[1]
        + rotation_yz.abs() * config.box_size[2]
    )
    on_target = ((position[:, 0] - config.target_xy[0]).abs() + half_x <= config.target_size_xy[0] / 2) & (
        (position[:, 1] - config.target_xy[1]).abs() + half_y <= config.target_size_xy[1] / 2
    )
    expected_height = config.table_height + config.box_size[2] / 2
    if not hasattr(env, "_yam_box_lifted"):
        _reset_lift_progress(env, None)
    env._yam_box_lifted |= position[:, 2] > expected_height + config.minimum_lift_height
    on_table = (position[:, 2] - expected_height).abs() < config.success_height_tolerance
    upright = 1 - 2 * (quaternion_x.square() + quaternion_y.square()) > math.cos(0.15)
    settled = torch.linalg.vector_norm(_tensor(box.data.root_lin_vel_w), dim=-1) < config.success_speed_m_s
    settled &= torch.linalg.vector_norm(_tensor(box.data.root_ang_vel_w), dim=-1) < 0.15
    robot = env.scene["robot"]
    finger_ids, _ = robot.find_joints(["left_joint7", "left_joint8", "right_joint7", "right_joint8"])
    released = (_tensor(robot.data.joint_pos)[:, finger_ids] > 0.04695 * 0.8).all(dim=-1)
    return env._yam_box_lifted & on_target & on_table & upright & settled & released


def make_environment(
    config: YellowBoxConfig, control_mode: str = "joint", enable_cameras: bool = True, enable_metrics: bool = True
):
    """Compose the robot, procedural workspace, reset randomization, and success criterion."""
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.configclass import make_configclass
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_examples.yam_yellow_box.embodiment import YamEmbodiment

    usd_path = Path(config.usd_path).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"YAM USD not found: {usd_path}. Run the prepare_assets module on the GPU host first.")
    material = sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0)

    def cuboid(name, size, position, color, dynamic=False):
        return Object(
            name=name,
            object_type=ObjectType.RIGID if dynamic else ObjectType.BASE,
            spawner_cfg=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg() if dynamic else None,
                mass_props=sim_utils.MassPropertiesCfg(mass=config.box_mass) if dynamic else None,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            initial_pose=Pose(position_xyz=position, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )

    table = cuboid(
        "table",
        config.table_size,
        (*config.table_center_xy, config.table_height - config.table_size[2] / 2),
        (0.55, 0.5, 0.42),
    )
    box = cuboid(
        "yellow_box",
        config.box_size,
        (*config.box_start_xy, config.table_height + config.box_size[2] / 2 + 0.002),
        (0.95, 0.75, 0.03),
        dynamic=True,
    )
    box.disable_reset_pose()
    target = cuboid(
        "target",
        (*config.target_size_xy, 0.001),
        (*config.target_xy, config.table_height - 0.0003),
        (0.04, 0.3, 0.8),
    )
    light = Object(
        name="light",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.95, 0.95, 1.0)),
    )
    ground = Object(
        name="ground",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.GroundPlaneCfg(),
    )
    reset_box = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("yellow_box"),
            "pose_range": {
                "x": (-config.box_randomization_xy[0], config.box_randomization_xy[0]),
                "y": (-config.box_randomization_xy[1], config.box_randomization_xy[1]),
                "yaw": config.box_yaw_range,
            },
            "velocity_range": {},
        },
    )

    class YellowBoxTask(NoTask):
        def __init__(self):
            super().__init__()
            self.episode_length_s = config.episode_length_s
            self.task_description = config.task_description

        def get_events_cfg(self):
            return make_configclass(
                "YellowBoxEvents",
                [
                    ("reset_box", EventTermCfg, reset_box),
                    ("reset_lift_progress", EventTermCfg, EventTermCfg(func=_reset_lift_progress, mode="reset")),
                ],
            )()

        def get_termination_cfg(self):
            return make_configclass(
                "YellowBoxTerminations",
                [
                    ("time_out", TerminationTermCfg, TerminationTermCfg(func=mdp.time_out, time_out=True)),
                    ("success", TerminationTermCfg, TerminationTermCfg(func=task_success, params={"config": config})),
                ],
            )()

        def get_metrics(self):
            return [SuccessRateMetric()] if enable_metrics else []

        def get_viewer_cfg(self):
            return ViewerCfg(eye=(1.6, 1.5, 1.8), lookat=(0.35, 0.0, config.table_height), origin_type="env")

    def configure_runtime(env_cfg):
        env_cfg.sim.dt = 1.0 / (config.fps * 4)
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        return env_cfg

    return IsaacLabArenaEnvironment(
        name="yam_yellow_box",
        scene=Scene(assets=[table, box, target, light, ground]),
        embodiment=YamEmbodiment(
            usd_path=str(usd_path),
            enable_cameras=enable_cameras,
            table_height=config.table_height,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            control_mode=control_mode,
            fps=config.fps,
        ),
        task=YellowBoxTask(),
        env_cfg_callback=configure_runtime,
    )


class YellowBoxEnvironment(ExampleEnvironmentBase):
    """Load this environment with Arena's external environment CLI."""

    name = "yam_yellow_box"

    def get_env(self, args_cli):
        config = YellowBoxConfig.load(args_cli.yam_config)
        return make_environment(config, args_cli.yam_control_mode, args_cli.enable_cameras)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--yam_config", default=None)
        parser.add_argument("--yam_control_mode", choices=("joint", "ik"), default="joint")
