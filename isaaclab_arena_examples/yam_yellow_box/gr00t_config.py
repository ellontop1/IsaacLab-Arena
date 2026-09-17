# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.6 modality configuration for the standalone YAM simulation dataset."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

CHANNELS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]

register_modality_config(
    {
        "video": ModalityConfig(
            delta_indices=[0], modality_keys=["top_camera", "left_wrist_camera", "right_wrist_camera"]
        ),
        "state": ModalityConfig(delta_indices=[0], modality_keys=CHANNELS),
        "action": ModalityConfig(
            delta_indices=list(range(30)),
            modality_keys=CHANNELS,
            action_configs=[
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
                for _ in CHANNELS
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
    },
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
