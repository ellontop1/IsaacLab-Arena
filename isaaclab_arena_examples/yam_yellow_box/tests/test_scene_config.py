# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check carton reset geometry and explicit task-label requirements."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from isaaclab_arena_examples.yam_yellow_box.config import YellowBoxConfig
from isaaclab_arena_examples.yam_yellow_box.scene_assets import carton_panels


class SceneConfigurationTests(unittest.TestCase):
    def test_default_contents_and_unverified_task(self):
        config = YellowBoxConfig()
        self.assertEqual([p.name for p in config.props], ["apple", "jello_box", "can"])
        self.assertFalse(config.source_verified)
        self.assertEqual(config.task_description, "")
        self.assertEqual(config.success_mode, "operator_confirmed")
        self.assertNotIn("target_xy", config.to_dict())

    def test_reject_spawn_outside_box_or_overlapping_objects(self):
        for field, value in [("offset_xy", (0.5, 0)), ("size", (0.08, 0.08, 0.3)), ("offset_xy", (0.105, -0.04))]:
            values = YellowBoxConfig().to_dict()
            values["props"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                YellowBoxConfig(**values)

    def test_reject_missing_contents_and_unsafe_randomization(self):
        for values in (
            {"props": []},
            {"prop_randomization_xy": (0.1, 0.1)},
            {"box_start_xy": (0.9, 0.0)},
            {"box_wall_thickness": 0.2},
            {"success_mode": "automatic"},
            {"fps": 29.97},
            {"settle_steps": 0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                YellowBoxConfig(**values)

    def test_carton_has_floor_four_walls_and_empty_interior(self):
        config = YellowBoxConfig()
        panels = carton_panels(config.box_size, config.box_wall_thickness)
        self.assertEqual(len(panels), 5)
        floor = panels[0]
        self.assertAlmostEqual(floor[2][2] + floor[1][2] / 2, config.box_wall_thickness)
        centre = (0, 0, config.box_size[2] / 2)
        for name, size, position in panels:
            self.assertFalse(all(abs(centre[a] - position[a]) < size[a] / 2 for a in range(3)), name)
            self.assertLessEqual(position[2] + size[2] / 2, config.box_size[2] + 1e-9)

    def test_json_roundtrip_and_reject_obsolete_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            config = YellowBoxConfig()
            path.write_text(json.dumps(asdict(config)))
            self.assertEqual(YellowBoxConfig.load(path).to_dict(), config.to_dict())
            path.write_text(json.dumps({"target_xy": [0.4, -0.15]}))
            with self.assertRaisesRegex(ValueError, "Unknown scene parameters"):
                YellowBoxConfig.load(path)


if __name__ == "__main__":
    unittest.main()
