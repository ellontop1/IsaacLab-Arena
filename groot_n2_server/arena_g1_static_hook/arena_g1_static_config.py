# Embodiment modality config for nvidia/Arena-G1-Static-PickNPlace-Task
# (unitree_g1, 43-DoF whole-body joint-position teleop data, ego-view camera).
#
# Mirrors lerobot/meta/modality.json of that dataset. Compared to the
# lerobot_output_405 config:
#   * state joint order is leg/leg/waist/arm/hand/arm/hand (arm and hand are
#     interleaved differently than the 405 set).
#   * actions are ABSOLUTE joint desired positions (no effort_* channels and no
#     relative-arm deltas); the teleop base_height/navigate/torso commands are
#     all-zero for this static task, so they are omitted from the trained action
#     space. Legs are proprioceptive-only (in state, not action), matching the
#     405 precedent for a tabletop manipulation policy.

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


config = {
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
            "left_hand",
            "right_arm",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
            "waist",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_arm
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_hand
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_arm
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_hand
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # waist
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# ``register_modality_config`` asserts the tag isn't already in
# ``MODALITY_CONFIGS``, so a second import of this file inside the same Python
# process (stats regen after train, notebooks, pytest re-runs) would crash.
# Re-binding is safe -- the new config wins.
MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
