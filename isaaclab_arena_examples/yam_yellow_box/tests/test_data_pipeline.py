# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate episode boundaries, field alignment, and the encoded training dataset."""

import h5py
import json
import numpy as np
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import pandas as pd

from isaaclab_arena_examples.yam_yellow_box.exporter import CAMERAS, EpisodeWriter, export_dataset


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.recording = self.root / "demonstrations.hdf5"
        self.state = np.zeros(14, dtype=np.float32)
        self.action = np.ones(14, dtype=np.float32) * 0.5
        self.cameras = {
            name: np.full((16, 24, 3), 30 + index * 60, dtype=np.uint8) for index, name in enumerate(CAMERAS)
        }

    def test_failed_and_unfinished_episodes_are_excluded(self):
        with EpisodeWriter(self.recording, 30, "Place the box") as writer:
            writer.append(self.state, self.action, self.cameras)
            self.assertFalse(writer.finish_episode(False))
            writer.append(self.state, self.action, self.cameras)
            self.assertTrue(writer.finish_episode(True))
            writer.append(self.action, self.state, self.cameras)
        with h5py.File(self.recording, "r") as recording:
            self.assertEqual(list(recording["episodes"]), ["episode_000000"])
            np.testing.assert_array_equal(recording["episodes/episode_000000/state"][0], self.state)
            np.testing.assert_array_equal(recording["episodes/episode_000000/action"][0], self.action)

    def test_invalid_sample_does_not_advance_recording(self):
        with EpisodeWriter(self.recording, 30, "Place the box") as writer:
            invalid = self.state.copy()
            invalid[0] = np.nan
            with self.assertRaises(ValueError):
                writer.append(invalid, self.action, self.cameras)
            with self.assertRaises(ValueError):
                writer.append(self.state, self.action, {"top_camera": self.cameras["top_camera"]})
            writer.append(self.state, self.action, self.cameras)
            changed = dict(self.cameras)
            changed["right_wrist_camera"] = np.zeros((18, 24, 3), dtype=np.uint8)
            with self.assertRaises(ValueError):
                writer.append(self.state, self.action, changed)
            self.assertEqual(writer.frames, 1)
            self.assertEqual(writer.pending["state"].shape[0], 1)

    def test_export_preserves_timing_targets_and_all_views(self):
        with EpisodeWriter(self.recording, 30, "Place the box") as writer:
            for frame_index in range(3):
                state = self.state.copy()
                state[0] = frame_index
                writer.append(state, self.action, self.cameras)
            writer.finish_episode(True)
            writer.append(self.state, self.state, self.cameras)
        output = export_dataset(self.recording, self.root / "lerobot")
        info = json.loads((output / "meta/info.json").read_text())
        self.assertEqual(info["total_episodes"], 1)
        self.assertEqual(info["total_frames"], 3)
        self.assertEqual(info["total_videos"], 3)
        dataframe = pd.read_parquet(output / "data/chunk-000/episode_000000.parquet")
        np.testing.assert_allclose(dataframe["timestamp"], [0, 1 / 30, 2 / 30])
        np.testing.assert_array_equal(np.stack(dataframe["observation.state"])[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(np.stack(dataframe["action"]), np.tile(self.action, (3, 1)))
        for name in CAMERAS:
            video = output / f"videos/chunk-000/observation.images.{name}/episode_000000.mp4"
            with imageio.get_reader(video) as reader:
                frames = list(reader.iter_data())
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, (16, 24, 3))
            np.testing.assert_allclose(frames[0].mean(), self.cameras[name].mean(), atol=3)
        modality = json.loads((output / "meta/modality.json").read_text())
        self.assertEqual(modality["action"]["right_arm"]["start"], 7)
        stats = json.loads((output / "meta/stats.json").read_text())
        np.testing.assert_allclose(stats["observation.state"]["q01"][0], 0.02)
        np.testing.assert_allclose(stats["observation.state"]["q99"][0], 1.98)
        with self.assertRaises(FileExistsError):
            export_dataset(self.recording, output)

    def test_no_success_does_not_publish_dataset(self):
        with EpisodeWriter(self.recording, 30, "Place the box") as writer:
            writer.append(self.state, self.action, self.cameras)
        output = self.root / "lerobot"
        with self.assertRaises(ValueError):
            export_dataset(self.recording, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".lerobot-*")), [])
        with self.assertRaises(FileExistsError):
            EpisodeWriter(self.recording, 30, "Place the box")

    def test_operator_labels_and_starting_positions_survive_export(self):
        provenance = {"scene_version": "carton_contents_v2", "success_label_source": "operator_confirmed"}
        positions = {"apple": [[0.3, 0, 0.8]], "jello_box": [[0.4, 0, 0.8]], "can": [[0.5, 0, 0.8]]}
        with EpisodeWriter(self.recording, 30, "User-confirmed task", provenance) as writer:
            writer.pending.attrs["initial_object_positions"] = json.dumps(positions)
            writer.append(self.state, self.action, self.cameras)
            writer.finish_episode(True)
        output = export_dataset(self.recording, self.root / "labelled")
        episode = json.loads((output / "meta/episodes.jsonl").read_text().splitlines()[0])
        self.assertEqual(episode["success_label_source"], "operator_confirmed")
        self.assertEqual(episode["initial_object_positions"], positions)
        metadata = json.loads((output / "meta/simulation_provenance.json").read_text())
        self.assertEqual(metadata["configuration"], provenance)


if __name__ == "__main__":
    unittest.main()
