# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Provisional I2RT YAM station with canonical joint controls."""

from __future__ import annotations

import torch
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg

FINGER_TRAVEL = 0.04695
ARM_JOINT_NAMES = {side: [f"{side}_joint{index}" for index in range(1, 7)] for side in ("left", "right")}
JOINT_NAMES = [f"{side}_joint{index}" for side in ("left", "right") for index in range(1, 9)]
STATE_NAMES = [name for side in ("left", "right") for name in (*ARM_JOINT_NAMES[side], f"{side}_gripper")]
TCP_POSITION = (0.0000968103407, 0.0000388982673, -0.144650259)
ARM_LIMITS = (
    (-2.61799, 3.14159),
    (0.0, 3.66519),
    (0.0, 3.14159),
    (-1.69297, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)


def _robot_link_paths(usd_path: str) -> dict[str, str]:
    """Resolve authored rigid-body paths across importer hierarchy variations."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    assert stage is not None, f"Cannot open YAM USD: {usd_path}"
    root = stage.GetDefaultPrim()
    assert root.IsValid(), "YAM USD must define a default prim"
    expected = {"left_base", "left_gripper", "right_gripper", "left_camera", "right_camera", "top_camera"}
    paths = {}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name in expected and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert name not in paths, f"Ambiguous YAM rigid-body name: {name}"
            relative = str(prim.GetPath().MakeRelativePath(root.GetPath()))
            suffix = "" if relative == "." else f"/{relative}"
            paths[name] = f"{{ENV_REGEX_NS}}/Robot{suffix}"
    assert paths.keys() == expected, f"Missing YAM rigid bodies: {sorted(expected - paths.keys())}"
    return paths


def canonical_joint_vector(env, target: bool = False) -> torch.Tensor:
    """Return left arm/opening then right arm/opening, in radians and normalized opening."""
    robot = env.scene["robot"]
    if not hasattr(env, "_yam_joint_indices"):
        indices, names = robot.find_joints(JOINT_NAMES, preserve_order=True, as_proxy=True)
        assert names == JOINT_NAMES, f"Unexpected YAM joint ordering: {names}"
        env._yam_joint_indices = indices.torch
    values = robot.data.joint_pos_target if target else robot.data.joint_pos
    positions = values.torch[:, env._yam_joint_indices]
    return torch.cat(
        (
            positions[:, :6],
            (positions[:, 6:8].mean(dim=1, keepdim=True) / FINGER_TRAVEL).clamp(0.0, 1.0),
            positions[:, 8:14],
            (positions[:, 14:16].mean(dim=1, keepdim=True) / FINGER_TRAVEL).clamp(0.0, 1.0),
        ),
        dim=1,
    )


class YamGripperAction(ActionTerm):
    """Map one continuous opening fraction to both physical finger sliders."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        indices, names = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        assert len(names) == 2, f"Expected two YAM finger joints, got {names}"
        self._joint_ids = indices.torch
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, 2), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(0.0, 1.0) * FINGER_TRAVEL

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        positions = self._asset.data.default_joint_pos.torch[selected][:, self._joint_ids]
        self._processed_actions[selected] = positions
        self._raw_actions[selected] = positions.mean(dim=1, keepdim=True) / FINGER_TRAVEL


@configclass
class YamGripperActionCfg(ActionTermCfg):
    class_type: type = YamGripperAction
    joint_names: list[str] = []


class YamIKAction(DifferentialInverseKinematicsAction):
    """Solve one bounded joint target per control tick and hold it through all substeps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._held_target = None

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._held_target = None

    def apply_actions(self) -> None:
        if self._held_target is None:
            super().apply_actions()
            targets = self._asset.data.joint_pos_target.torch[:, self._joint_ids]
            limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
            self._held_target = torch.clamp(targets, min=limits[..., 0], max=limits[..., 1])
        self._asset.set_joint_position_target_index(target=self._held_target, joint_ids=self._joint_ids)

    def _compute_frame_jacobian(self) -> torch.Tensor:
        self._jacobian_b[:] = self.jacobian_b
        rotation_root_flange = math_utils.quat_mul(
            math_utils.quat_inv(self._asset.data.root_quat_w.torch),
            self._asset.data.body_quat_w.torch[:, self._body_idx],
        )
        offset_root = math_utils.quat_apply(rotation_root_flange, self._offset_pos)
        self._jacobian_b[:, :3, :] -= torch.bmm(
            math_utils.skew_symmetric_matrix(offset_root), self._jacobian_b[:, 3:, :]
        )
        return self._jacobian_b

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._held_target = None


@configclass
class YamIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = YamIKAction


def _joint_action(side: str) -> JointPositionActionCfg:
    return JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES[side],
        preserve_order=True,
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        clip=dict(zip(ARM_JOINT_NAMES[side], ARM_LIMITS)),
    )


def _ik_action(side: str) -> DifferentialInverseKinematicsActionCfg:
    return YamIKActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES[side],
        body_name=f"{side}_gripper",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_POSITION, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class YamActionsCfg:
    left_arm: ActionTermCfg = _joint_action("left")
    left_gripper: ActionTermCfg = YamGripperActionCfg(asset_name="robot", joint_names=["left_joint7", "left_joint8"])
    right_arm: ActionTermCfg = _joint_action("right")
    right_gripper: ActionTermCfg = YamGripperActionCfg(asset_name="robot", joint_names=["right_joint7", "right_joint8"])


@configclass
class YamObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state: ObservationTermCfg = ObservationTermCfg(func=canonical_joint_vector)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class YamEventsCfg:
    reset_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )


def _frame(side: str) -> FrameTransformerCfg:
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_gripper",
                name=f"{side}_tcp",
                offset=OffsetCfg(pos=TCP_POSITION, rot=(1.0, 0.0, 0.0, 0.0)),
            )
        ],
    )


@configclass
class YamSceneCfg:
    robot: ArticulationCfg | None = None
    left_ee_frame: FrameTransformerCfg = _frame("left")
    right_ee_frame: FrameTransformerCfg = _frame("right")


@configclass
class YamCameraCfg(ArenaCameraCfg):
    top_camera: CameraCfg | None = None
    left_wrist_camera: CameraCfg | None = None
    right_wrist_camera: CameraCfg | None = None


class YamEmbodiment(EmbodimentBase):
    """Dual YAM station with provisional geometry and camera intrinsics requiring calibration."""

    name = "yam_yellow_box"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        usd_path: str,
        enable_cameras: bool = True,
        table_height: float = 0.75,
        camera_width: int = 640,
        camera_height: int = 480,
        control_mode: str = "joint",
        fps: int = 30,
    ):
        assert control_mode in ("joint", "ik"), "control_mode must be 'joint' or 'ik'"
        assert fps > 0 and camera_width > 0 and camera_height > 0, "Camera settings must be positive"
        usd_path = str(Path(usd_path).expanduser().resolve())
        if not Path(usd_path).is_file():
            raise FileNotFoundError(f"Prepare the YAM USD first with prepare_assets.py: {usd_path}")
        super().__init__(enable_cameras=enable_cameras)
        link_paths = _robot_link_paths(usd_path)
        initial_arm = (0.0, 1.5, 1.5, -1.3, 0.0, 0.0)
        initial_joints = {
            name: value for side in ("left", "right") for name, value in zip(ARM_JOINT_NAMES[side], initial_arm)
        }
        initial_joints.update({f"{side}_joint{index}": FINGER_TRAVEL for side in ("left", "right") for index in (7, 8)})
        self.scene_config = YamSceneCfg(
            robot=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                soft_joint_pos_limit_factor=1.0,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.305, table_height), rot=(0.0, 0.0, 0.0, 1.0), joint_pos=initial_joints
                ),
                actuators={
                    "shoulders": ImplicitActuatorCfg(
                        joint_names_expr=[".*_joint[1-3]"],
                        stiffness=80.0,
                        damping=5.0,
                        joint_effort_limit=30.0,
                        joint_velocity_limit=3.0,
                    ),
                    "wrists": ImplicitActuatorCfg(
                        joint_names_expr=[".*_joint[4-6]"],
                        stiffness=20.0,
                        damping=1.5,
                        joint_effort_limit=10.0,
                        joint_velocity_limit=3.0,
                    ),
                    "fingers": ImplicitActuatorCfg(
                        joint_names_expr=[".*_joint[78]"],
                        stiffness=400.0,
                        damping=20.0,
                        joint_effort_limit=20.0,
                        joint_velocity_limit=0.15,
                    ),
                },
            )
        )
        self.action_config = YamActionsCfg()
        for side in ("left", "right"):
            frame = getattr(self.scene_config, f"{side}_ee_frame")
            frame.prim_path = link_paths["left_base"]
            frame.target_frames[0].prim_path = link_paths[f"{side}_gripper"]
        if control_mode == "ik":
            self.action_config.left_arm = _ik_action("left")
            self.action_config.right_arm = _ik_action("right")
        self.observation_config = YamObservationsCfg()
        self.event_config = YamEventsCfg()
        self.camera_config = YamCameraCfg()
        for camera_name, body_name in (
            ("top_camera", "top_camera"),
            ("left_wrist_camera", "left_camera"),
            ("right_wrist_camera", "right_camera"),
        ):
            setattr(
                self.camera_config,
                camera_name,
                CameraCfg(
                    prim_path=f"{link_paths[body_name]}/sensor",
                    width=camera_width,
                    height=camera_height,
                    update_period=1.0 / fps,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=18.0,
                        horizontal_aperture=27.0,
                        vertical_aperture=27.0 * camera_height / camera_width,
                        clipping_range=(0.01, 10.0),
                    ),
                    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="ros"),
                ),
            )
        self.add_camera_variations(self.camera_config)

    def get_ee_frame_transformer_names(self) -> list[str]:
        return ["left_ee_frame", "right_ee_frame"]

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        assert arm_mode in (ArmMode.LEFT, ArmMode.RIGHT), "Select one arm for its end-effector frame"
        return f"{arm_mode.value}_ee_frame"
