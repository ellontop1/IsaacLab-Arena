# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect successful bimanual YAM demonstrations with an interactive keyboard."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


class BimanualKeyboard:
    """Map keyboard motion to one selected arm while preserving both gripper commands."""

    def __init__(self, translation_step: float, rotation_step: float):
        import carb.input
        import omni.appwindow

        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_event)
        self.translation_step = translation_step
        self.rotation_step = rotation_step
        self.held = set()
        self.arm = 0
        self.grippers = [1.0, 1.0]
        self.reset_requested = False
        self.save_requested = False
        self.quit_requested = False
        self.paused = False

    def _on_event(self, event, *_):
        import carb.input

        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.held.discard(key)
        elif event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.held.add(key)
            if key == "TAB":
                self.arm = 1 - self.arm
                print(f"Controlling {'left' if self.arm == 0 else 'right'} arm")
            elif key == "SPACE":
                self.grippers[self.arm] = 1.0 - self.grippers[self.arm]
            elif key == "R":
                self.reset_requested = True
            elif key in {"ENTER", "NUMPAD_ENTER"}:
                self.save_requested = True
            elif key == "P":
                self.paused = not self.paused
            elif key == "ESCAPE":
                self.quit_requested = True
        return True

    def action(self):
        import numpy as np

        command = np.zeros(14, dtype=np.float32)
        pairs = (("W", "S"), ("A", "D"), ("Q", "E"), ("Z", "C"), ("T", "G"), ("F", "H"))
        for axis, (positive, negative) in enumerate(pairs):
            magnitude = self.translation_step if axis < 3 else self.rotation_step
            command[self.arm * 7 + axis] = magnitude * (int(positive in self.held) - int(negative in self.held))
        command[6], command[13] = self.grippers
        return command

    def reset(self, state) -> None:
        self.held.clear()
        self.grippers = [float(state[6] >= 0.5), float(state[13] >= 0.5)]
        self.reset_requested = False
        self.save_requested = False

    def close(self) -> None:
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)


def run(args, simulation_app) -> None:
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg
    from isaaclab_arena_examples.yam_yellow_box.config import YellowBoxConfig
    from isaaclab_arena_examples.yam_yellow_box.embodiment import canonical_joint_vector
    from isaaclab_arena_examples.yam_yellow_box.environment import make_environment
    from isaaclab_arena_examples.yam_yellow_box.exporter import CAMERAS, EpisodeWriter
    from isaaclab_arena_examples.yam_yellow_box.validation import validate_start

    configuration = YellowBoxConfig.load(args.config)
    if args.seed is not None:
        configuration.seed = args.seed
    if args.task:
        configuration.task_description = args.task
    if not configuration.task_description.strip():
        raise ValueError("Specify the real demonstrated task using --task or task_description in the scene config")
    arena = make_environment(configuration, control_mode="ik", enable_cameras=True, enable_metrics=False)
    builder = ArenaEnvBuilder(
        arena, ArenaEnvBuilderCfg(num_envs=1, device=args.device, seed=configuration.seed, solve_relations=False)
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    for name in vars(env_cfg.terminations):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env = builder.make_registered(env_cfg, env_kwargs)
    base_env = env.unwrapped
    keyboard = None
    try:
        if abs(base_env.step_dt - 1 / configuration.fps) > 1e-6:
            raise ValueError("Simulation step period must match dataset fps")
        keyboard = BimanualKeyboard(args.translation_speed / configuration.fps, args.rotation_speed / configuration.fps)
        task = arena.task.get_task_description()
        print("Click the simulator viewport. TAB switches arm; SPACE toggles gripper.")
        print(
            "W/S X, A/D Y, Q/E Z; Z/C roll, T/G pitch, F/H yaw. P pause; R discard/reset; ENTER confirm success and"
            " save; ESC stop."
        )
        print("ENTER labels the episode successful based on your task judgment; no automatic success is assumed.")
        provenance = configuration.to_dict()
        provenance["asset_manifest"] = arena.yam_asset_manifest
        provenance["success_label_source"] = "operator_confirmed"

        def reset_and_settle():
            observation, _ = env.reset()
            command = torch.zeros((1, 14), device=base_env.device)
            command[:, [6, 13]] = observation["policy"]["state"][:, [6, 13]]
            with torch.inference_mode():
                for _ in range(configuration.settle_steps):
                    observation, _, terminated, truncated, _ = env.step(command)
                    assert not bool(terminated.any() or truncated.any()), "Unexpected reset while settling"
            initial_objects = validate_start(base_env, configuration, arena.yam_asset_manifest)
            writer.pending.attrs["initial_object_positions"] = json.dumps(initial_objects)
            keyboard.reset(observation["policy"]["state"][0].detach().cpu().numpy())
            return observation

        with EpisodeWriter(args.output, configuration.fps, task, provenance) as writer:
            observation = reset_and_settle()
            max_steps = round(configuration.fps * configuration.episode_length_s)
            while simulation_app.is_running() and not keyboard.quit_requested and writer.saved < args.episodes:
                frame_start = time.perf_counter()
                if keyboard.reset_requested:
                    writer.begin_episode()
                    observation = reset_and_settle()
                if keyboard.paused:
                    base_env.sim.render()
                    time.sleep(1 / configuration.fps)
                    continue
                with torch.inference_mode():
                    state = observation["policy"]["state"][0].detach().cpu().numpy().copy()
                    cameras = {
                        name: observation["camera_obs"][f"{name}_rgb"][0].detach().cpu().numpy().copy()
                        for name in CAMERAS
                    }
                    command = torch.as_tensor(keyboard.action(), device=base_env.device).unsqueeze(0)
                    observation, _, terminated, truncated, _ = env.step(command)
                    if bool(terminated.any() or truncated.any()):
                        raise RuntimeError("Unexpected automatic reset: recording requires manual episode boundaries")
                    action = canonical_joint_vector(base_env, target=True)[0].detach().cpu().numpy().copy()
                    writer.append(state, action, cameras)
                if keyboard.save_requested or writer.frames >= max_steps:
                    saved = writer.finish_episode(keyboard.save_requested)
                    print(f"{'Saved' if saved else 'Discarded'} episode; {writer.saved}/{args.episodes} successful")
                    if writer.saved < args.episodes:
                        observation = reset_and_settle()
                remaining = 1 / configuration.fps - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
            print(f"Saved {writer.saved} successful episodes to {args.output}")
    finally:
        if keyboard is not None:
            keyboard.close()
        env.close()


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--task", help="Actual demonstrated task instruction; required unless configured in JSON")
    parser.add_argument("--translation-speed", type=float, default=0.10)
    parser.add_argument("--rotation-speed", type=float, default=0.50)
    parser.add_argument("--seed", type=int)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or args.translation_speed <= 0 or args.rotation_speed <= 0:
        parser.error("Episode counts and control speeds must be positive")
    if getattr(args, "headless", False):
        parser.error("Keyboard collection requires a Kit viewport; use --viz kit")
    if Path(args.output).expanduser().exists():
        parser.error("Output already exists; choose a new HDF5 filename")
    args.enable_cameras = True

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args) as simulation:
        from isaaclab_arena.environments.isaaclab_interop import _purge_leaked_isaaclab_assets_presets

        _purge_leaked_isaaclab_assets_presets()
        run(args, simulation)


if __name__ == "__main__":
    main()
