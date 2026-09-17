# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit, provisional scene parameters until source calibration is available."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class YellowBoxConfig:
    """Configure a YAM station and a yellow-box placement task in metres."""

    usd_path: str = "~/.cache/isaaclab_arena/yam/yam_station.usd"
    source_dataset: str = "nvidia/yam_yellow_box_all"
    source_verified: bool = False
    task_description: str = "Pick up the yellow box and place it on the blue target."
    fps: int = 30
    episode_length_s: float = 60.0
    table_height: float = 0.75
    table_size: tuple[float, float, float] = (0.9, 1.2, 0.05)
    table_center_xy: tuple[float, float] = (0.3, 0.0)
    box_size: tuple[float, float, float] = (0.06, 0.06, 0.05)
    box_mass: float = 0.08
    box_start_xy: tuple[float, float] = (0.35, 0.15)
    box_randomization_xy: tuple[float, float] = (0.05, 0.04)
    box_yaw_range: tuple[float, float] = (-0.4, 0.4)
    target_xy: tuple[float, float] = (0.4, -0.15)
    target_size_xy: tuple[float, float] = (0.16, 0.16)
    camera_width: int = 640
    camera_height: int = 480
    success_hold_s: float = 0.5
    success_speed_m_s: float = 0.03
    success_height_tolerance: float = 0.012
    minimum_lift_height: float = 0.04
    seed: int = 42

    def __post_init__(self):
        for name, length in (
            ("table_size", 3),
            ("table_center_xy", 2),
            ("box_size", 3),
            ("box_start_xy", 2),
            ("box_randomization_xy", 2),
            ("box_yaw_range", 2),
            ("target_xy", 2),
            ("target_size_xy", 2),
        ):
            values = getattr(self, name)
            if len(values) != length or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {length} finite numbers")
            setattr(self, name, tuple(values))
        for name in (
            "fps",
            "episode_length_s",
            "table_height",
            "box_mass",
            "camera_width",
            "camera_height",
            "success_hold_s",
            "success_speed_m_s",
            "success_height_tolerance",
            "minimum_lift_height",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("fps", "camera_width", "camera_height", "seed"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.camera_width % 2 or self.camera_height % 2:
            raise ValueError("Camera dimensions must be even for H.264 export")
        if any(value <= 0 for value in (*self.table_size, *self.box_size, *self.target_size_xy)):
            raise ValueError("Object dimensions must be positive")
        if any(value < 0 for value in self.box_randomization_xy):
            raise ValueError("Box randomization bounds cannot be negative")
        if self.box_yaw_range[0] > self.box_yaw_range[1]:
            raise ValueError("Box yaw bounds must be ordered")
        if self.success_hold_s >= self.episode_length_s:
            raise ValueError("Success hold duration must be shorter than an episode")
        if min(self.target_size_xy) <= math.hypot(*self.box_size[:2]):
            raise ValueError("Target must contain the box footprint at arbitrary yaw")
        for axis in range(2):
            table_min = self.table_center_xy[axis] - self.table_size[axis] / 2
            table_max = self.table_center_xy[axis] + self.table_size[axis] / 2
            box_radius = math.hypot(*self.box_size[:2]) / 2
            start_min = self.box_start_xy[axis] - self.box_randomization_xy[axis] - box_radius
            start_max = self.box_start_xy[axis] + self.box_randomization_xy[axis] + box_radius
            target_min = self.target_xy[axis] - self.target_size_xy[axis] / 2
            target_max = self.target_xy[axis] + self.target_size_xy[axis] / 2
            if min(start_min, target_min) < table_min or max(start_max, target_max) > table_max:
                raise ValueError("Box spawn range and target must fit on the tabletop")
        separation = [
            abs(self.box_start_xy[axis] - self.target_xy[axis])
            > self.box_randomization_xy[axis] + self.target_size_xy[axis] / 2 + math.hypot(*self.box_size[:2]) / 2
            for axis in range(2)
        ]
        if not any(separation):
            raise ValueError("Box spawn range must not overlap the target")
        if not self.task_description.strip():
            raise ValueError("Task description cannot be empty")

    @classmethod
    def load(cls, path: str | Path | None = None) -> YellowBoxConfig:
        """Read a JSON configuration, rejecting misspelled parameter names."""
        if path is None:
            return cls()
        values = json.loads(Path(path).expanduser().read_text())
        if not isinstance(values, dict):
            raise ValueError("Scene configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown scene parameters: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        """Return serializable scene provenance for each recording."""
        return asdict(self)
