# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose a YAM station with an open carton containing an apple, Jell-O box, and can."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena_examples.yam_yellow_box.config import YellowBoxConfig


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def reset_contents(env, env_ids, config: YellowBoxConfig):
    """Reset the carton and contents as one group, preserving clearance and zeroing velocities."""
    import torch

    from isaaclab.utils import math as math_utils

    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(_tensor(env_ids), device=env.device, dtype=torch.long)
    )
    count = len(ids)
    if not count:
        return
    yaw = torch.empty(count, device=env.device).uniform_(*config.box_yaw_range)
    zeros = torch.zeros_like(yaw)
    rotation = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    translation = torch.zeros((count, 3), device=env.device)
    for axis, radius in enumerate(config.box_randomization_xy):
        translation[:, axis].uniform_(-radius, radius)
    origins = _tensor(env.scene.env_origins)[ids]
    pivot = torch.tensor([*config.box_start_xy, config.table_height + config.spawn_clearance], device=env.device)
    for name in ["cardboard_box", *[p.name for p in config.props]]:
        asset = env.scene[name]
        state = _tensor(asset.data.default_root_state)[ids].clone()
        local = state[:, :3] - pivot
        if name != "cardboard_box":
            for axis, radius in enumerate(config.prop_randomization_xy):
                local[:, axis] += torch.empty(count, device=env.device).uniform_(-radius, radius)
        state[:, :3] = pivot + math_utils.quat_apply(rotation, local) + translation + origins
        state[:, 3:7] = math_utils.quat_mul(rotation, state[:, 3:7])
        asset.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((count, 6), device=env.device), env_ids=ids)


def make_environment(config: YellowBoxConfig, control_mode="joint", enable_cameras=True, enable_metrics=True):
    """Compose the real-scene starting arrangement; task completion is operator-confirmed."""
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import EventTermCfg, TerminationTermCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.configclass import make_configclass
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_examples.yam_yellow_box.embodiment import YamEmbodiment
    from isaaclab_arena_examples.yam_yellow_box.scene_assets import make_prop, prepare_carton

    usd_path = Path(config.usd_path).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"YAM USD not found: {usd_path}. Run prepare_assets on the GPU host first.")
    table = Object(
        name="table",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(
            size=config.table_size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.42)),
        ),
        initial_pose=Pose(
            position_xyz=(*config.table_center_xy, config.table_height - config.table_size[2] / 2),
            rotation_xyzw=(0, 0, 0, 1),
        ),
    )
    carton = Object(
        name="cardboard_box",
        object_type=ObjectType.RIGID,
        usd_path=prepare_carton(config),
        initial_pose=Pose(
            position_xyz=(*config.box_start_xy, config.table_height + config.spawn_clearance),
            rotation_xyzw=(0, 0, 0, 1),
        ),
        spawn_cfg_addon={"rigid_props": sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5)},
    )
    carton.disable_reset_pose()
    props, manifest = [], []
    for prop in config.props:
        asset, source = make_prop(prop, config)
        props.append(asset)
        manifest.append(source)
    light = Object(
        name="light",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.95, 0.95, 1.0)),
    )
    ground = Object(name="ground", object_type=ObjectType.BASE, spawner_cfg=sim_utils.GroundPlaneCfg())

    class CartonTask(NoTask):
        def __init__(self):
            super().__init__()
            self.episode_length_s = config.episode_length_s
            self.task_description = config.task_description or "YAM carton scene preview; collection task not specified"

        def get_events_cfg(self):
            return make_configclass(
                "CartonEvents",
                [(
                    "reset_contents",
                    EventTermCfg,
                    EventTermCfg(func=reset_contents, mode="reset", params={"config": config}),
                )],
            )()

        def get_termination_cfg(self):
            return make_configclass(
                "CartonTerminations",
                [("time_out", TerminationTermCfg, TerminationTermCfg(func=mdp.time_out, time_out=True))],
            )()

        def get_metrics(self):
            # No automatic success metric until the real task and its endpoint are established.
            return []

        def get_viewer_cfg(self):
            return ViewerCfg(eye=(1.6, 1.5, 1.8), lookat=(0.38, 0.0, config.table_height + 0.08), origin_type="env")

    def configure_runtime(env_cfg):
        env_cfg.sim.dt = 1 / (config.fps * 4)
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        return env_cfg

    arena = IsaacLabArenaEnvironment(
        name="yam_yellow_box",
        scene=Scene(assets=[table, carton, *props, light, ground]),
        embodiment=YamEmbodiment(
            usd_path=str(usd_path),
            enable_cameras=enable_cameras,
            table_height=config.table_height,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            control_mode=control_mode,
            fps=config.fps,
        ),
        task=CartonTask(),
        env_cfg_callback=configure_runtime,
    )
    arena.yam_asset_manifest = manifest
    return arena


class YellowBoxEnvironment(ExampleEnvironmentBase):
    """Expose the carton scene through Arena's external-environment runner."""

    name = "yam_yellow_box"

    def get_env(self, args_cli):
        return make_environment(
            YellowBoxConfig.load(args_cli.yam_config), args_cli.yam_control_mode, args_cli.enable_cameras
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--yam_config", default=None)
        parser.add_argument("--yam_control_mode", choices=("joint", "ik"), default="joint")
