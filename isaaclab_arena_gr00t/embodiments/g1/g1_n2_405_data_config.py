# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Client-side modality config for the fine-tuned GR00T N2 G1 model.

This mirrors the ``new_embodiment`` modality layout the N2 checkpoint
(``.../checkpoint-10000``) was trained with on the ``lerobot_output_405``
dataset. Arena only uses the ``state`` / ``video`` / ``language`` modality
keys on the client side (to know which observation groups to build and send
to the policy server); the ``action`` config is included for fidelity but the
per-step action mapping into the sim is driven by the joint-space YAMLs.

Compared to ``g1_sim_wbc_data_config.py`` (the stock Arena G1 locomanip
config), the two functional differences that matter for N2 are:

* ``state`` includes ``left_leg`` / ``right_leg`` (the N2 model was trained
  with full lower-body proprioception), so Arena must send leg joint state.
* the action horizon is 16 (``delta_indices = range(16)``) rather than 50.
"""

from gr00t.configs.data.embodiment_configs import (
    MODALITY_CONFIGS,
    register_modality_config,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

g1_n2_405_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
            "navigate_command",
            "base_height_command",
            "effort_left_arm",
            "effort_right_arm",
            "effort_left_hand",
            "effort_right_hand",
            "effort_waist",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_arm
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_arm
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_hand
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_hand
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # waist
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # navigate_command
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # base_height_command
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # effort_left_arm
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # effort_right_arm
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # effort_left_hand
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # effort_right_hand
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # effort_waist
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# ``register_modality_config`` asserts the tag isn't already registered, so a
# second import within the same process (pytest re-runs, notebooks) would
# crash. Re-binding is safe here — the new config wins.
MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
register_modality_config(g1_n2_405_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
