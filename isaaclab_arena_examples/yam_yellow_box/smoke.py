# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded GPU smoke check and save all three simulated camera views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args):
    import torch

    import imageio.v2 as imageio

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg
    from isaaclab_arena_examples.yam_yellow_box.config import YellowBoxConfig
    from isaaclab_arena_examples.yam_yellow_box.environment import make_environment
    from isaaclab_arena_examples.yam_yellow_box.exporter import CAMERAS
    from isaaclab_arena_examples.yam_yellow_box.validation import validate_start

    config = YellowBoxConfig.load(args.config)
    if args.steps + config.settle_steps >= config.fps * config.episode_length_s:
        raise ValueError("Smoke steps must be shorter than the configured episode")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    arena = make_environment(config)
    builder = ArenaEnvBuilder(
        arena,
        ArenaEnvBuilderCfg(num_envs=1, seed=config.seed, device=args.device, solve_relations=False),
    )
    env = builder.make_registered()
    try:
        with torch.inference_mode():
            observation, _ = env.reset()
            target = observation["policy"]["state"].clone()
            assert target.shape == (1, 14), f"Unexpected state shape {target.shape}"
            assert bool(torch.isfinite(target).all()), "Nonfinite initial robot state"
            assert abs(env.unwrapped.step_dt - 1 / config.fps) < 1e-6, "Control rate differs from dataset rate"
            assert env.unwrapped.action_manager.total_action_dim == 14
            for _ in range(args.steps + config.settle_steps):
                observation, _, terminated, truncated, _ = env.step(target)
                assert not bool(
                    terminated.any() or truncated.any()
                ), "Unexpected reset during hold-position smoke check"
                assert bool(torch.isfinite(observation["policy"]["state"]).all()), "Nonfinite robot state"
            initial_objects = validate_start(env.unwrapped, config, arena.yam_asset_manifest)
            for name in CAMERAS:
                rgb = observation["camera_obs"][f"{name}_rgb"][0].detach().cpu().numpy()
                assert rgb.shape == (
                    config.camera_height,
                    config.camera_width,
                    3,
                ), f"Unexpected camera shape {rgb.shape}"
                assert rgb.max() > rgb.min(), f"Camera {name} produced a constant image"
                imageio.imwrite(output / f"{name}.png", rgb)
            report = {
                "steps": args.steps,
                "state_dimension": 14,
                "action_dimension": 14,
                "fps": 1 / env.unwrapped.step_dt,
                "cameras": list(CAMERAS),
                "source_dataset_verified": config.source_verified,
                "configuration": config.to_dict(),
                "initial_object_positions": initial_objects,
                "asset_manifest": arena.yam_asset_manifest,
                "settle_steps": config.settle_steps,
            }
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Smoke check passed. Inspect camera PNGs in {output} before collecting data.")
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=60, type=int)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.enable_cameras = True

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        from isaaclab_arena.environments.isaaclab_interop import _purge_leaked_isaaclab_assets_presets

        _purge_leaked_isaaclab_assets_presets()
        run(args)


if __name__ == "__main__":
    main()
