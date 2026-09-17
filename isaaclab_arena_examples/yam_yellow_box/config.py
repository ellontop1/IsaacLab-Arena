# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure the user-described starting scene; physical calibration remains provisional."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _vector(value, length, name, positive=False):
    if len(value) != length or not all(math.isfinite(v) and (v > 0 if positive else True) for v in value):
        raise ValueError(f"{name} must contain {length} finite {'positive ' if positive else ''}numbers")
    return tuple(value)


@dataclass
class PropConfig:
    """Describe one textured rigid object relative to the inside floor of the carton."""

    name: str
    asset_name: str
    size: tuple[float, float, float]
    mass: float
    offset_xy: tuple[float, float]
    yaw: float = 0.0
    usd_path: str | None = None

    def __post_init__(self):
        if self.name not in {"apple", "jello_box", "can"}:
            raise ValueError("Object names must be apple, jello_box, or can")
        self.size = _vector(self.size, 3, f"{self.name}.size", positive=True)
        self.offset_xy = _vector(self.offset_xy, 2, f"{self.name}.offset_xy")
        if not math.isfinite(self.mass) or self.mass <= 0 or not math.isfinite(self.yaw):
            raise ValueError("Object mass must be positive and yaw finite")
        if not self.asset_name and not self.usd_path:
            raise ValueError("Provide an asset_name or a compatible rigid USD path")

    @property
    def half_extent_xy(self):
        """Return yaw-rotated footprint half extents for collision-free reset validation."""
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        x, y = self.size[:2]
        return ((c * x + s * y) / 2, (s * x + c * y) / 2)


def default_props():
    return [
        PropConfig("apple", "apple_02_objaverse_robolab", (0.075, 0.075, 0.075), 0.15, (-0.10, -0.045)),
        PropConfig("jello_box", "jello_ycb_robolab", (0.085, 0.028, 0.070), 0.10, (0.0, 0.060)),
        PropConfig("can", "tomato_soup_can_ycb_robolab", (0.066, 0.066, 0.120), 0.35, (0.105, -0.040)),
    ]


@dataclass
class YellowBoxConfig:
    """Define the carton and its contents in metres, with explicit unverified assumptions."""

    usd_path: str = "~/.cache/isaaclab_arena/yam/yam_station.usd"
    source_dataset: str = "nvidia/yam_yellow_box_all"
    source_verified: bool = False
    scene_version: str = "carton_contents_v2"
    layout_source: str = "User description: apple, Jell-O box and can start inside a cardboard box"
    task_description: str = ""
    success_mode: str = "operator_confirmed"
    fps: int = 30
    episode_length_s: float = 120.0
    table_height: float = 0.75
    table_size: tuple[float, float, float] = (0.9, 1.2, 0.05)
    table_center_xy: tuple[float, float] = (0.3, 0.0)
    box_size: tuple[float, float, float] = (0.36, 0.28, 0.14)
    box_wall_thickness: float = 0.004
    box_mass: float = 0.18
    box_start_xy: tuple[float, float] = (0.38, 0.0)
    box_randomization_xy: tuple[float, float] = (0.015, 0.015)
    box_yaw_range: tuple[float, float] = (-0.10, 0.10)
    prop_randomization_xy: tuple[float, float] = (0.003, 0.003)
    spawn_clearance: float = 0.002
    settle_steps: int = 60
    props: list[PropConfig] = field(default_factory=default_props)
    camera_width: int = 640
    camera_height: int = 480
    seed: int = 42

    def __post_init__(self):
        for name, length in (
            ("table_size", 3),
            ("table_center_xy", 2),
            ("box_size", 3),
            ("box_start_xy", 2),
            ("box_randomization_xy", 2),
            ("box_yaw_range", 2),
            ("prop_randomization_xy", 2),
        ):
            setattr(self, name, _vector(getattr(self, name), length, name))
        for name in (
            "fps",
            "episode_length_s",
            "table_height",
            "box_mass",
            "box_wall_thickness",
            "spawn_clearance",
            "camera_width",
            "camera_height",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("fps", "camera_width", "camera_height", "seed", "settle_steps"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.settle_steps < 1:
            raise ValueError("settle_steps must be positive")
        if self.camera_width % 2 or self.camera_height % 2:
            raise ValueError("Camera dimensions must be even for H.264 export")
        if min(*self.table_size, *self.box_size) <= 0:
            raise ValueError("Object dimensions must be positive")
        if min(*self.box_randomization_xy, *self.prop_randomization_xy) < 0:
            raise ValueError("Randomization bounds cannot be negative")
        if self.box_yaw_range[0] > self.box_yaw_range[1]:
            raise ValueError("Box yaw bounds must be ordered")
        if 2 * self.box_wall_thickness >= min(self.box_size):
            raise ValueError("Box walls must leave an open interior")
        if self.success_mode != "operator_confirmed":
            raise ValueError("Only operator_confirmed success is supported until the real task is established")
        self.props = [PropConfig(**p) if isinstance(p, dict) else p for p in self.props]
        if len(self.props) != 3 or {p.name for p in self.props} != {"apple", "jello_box", "can"}:
            raise ValueError("Exactly one apple, jello_box, and can must start inside the carton")
        # A bounding circle guarantees containment on the table for every sampled box yaw.
        radius = math.hypot(*self.box_size[:2]) / 2
        for axis in range(2):
            if (
                abs(self.box_start_xy[axis] - self.table_center_xy[axis]) + self.box_randomization_xy[axis] + radius
                > self.table_size[axis] / 2
            ):
                raise ValueError("Box reset range must fit on the tabletop")
        for prop in self.props:
            for axis in range(2):
                if (
                    abs(prop.offset_xy[axis])
                    + prop.half_extent_xy[axis]
                    + self.prop_randomization_xy[axis]
                    + self.spawn_clearance
                    >= self.box_size[axis] / 2 - self.box_wall_thickness
                ):
                    raise ValueError(f"{prop.name} reset range intersects a carton wall")
            if prop.size[2] + self.box_wall_thickness + self.spawn_clearance > self.box_size[2]:
                raise ValueError(f"{prop.name} must start below the carton rim")
        for i, first in enumerate(self.props):
            for second in self.props[i + 1 :]:
                if not any(
                    abs(first.offset_xy[a] - second.offset_xy[a])
                    > first.half_extent_xy[a]
                    + second.half_extent_xy[a]
                    + 2 * self.prop_randomization_xy[a]
                    + self.spawn_clearance
                    for a in range(2)
                ):
                    raise ValueError(f"{first.name} and {second.name} reset ranges overlap")

    @classmethod
    def load(cls, path: str | Path | None = None) -> YellowBoxConfig:
        """Load explicit scene parameters, rejecting unknown or obsolete fields."""
        if path is None:
            return cls()
        values = json.loads(Path(path).expanduser().read_text())
        if not isinstance(values, dict):
            raise ValueError("Scene configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown scene parameters: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)
