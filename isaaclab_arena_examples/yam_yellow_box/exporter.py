# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record synchronized simulation samples and export a standalone LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import h5py
import json
import math
import numpy as np
import shutil
import tempfile
from pathlib import Path

CAMERAS = ("top_camera", "left_wrist_camera", "right_wrist_camera")
JOINT_NAMES = [f"left_joint{index}" for index in range(1, 7)] + ["left_gripper"]
JOINT_NAMES += [f"right_joint{index}" for index in range(1, 7)] + ["right_gripper"]
GROUPS = {"left_arm": (0, 6), "left_gripper": (6, 7), "right_arm": (7, 13), "right_gripper": (13, 14)}
SCHEMA = "arena_yam_yellow_box_v1"


class EpisodeWriter:
    """Stream pre-action observations and joint-position targets to an exclusive HDF5 file."""

    def __init__(self, path: str | Path, fps: int, task: str, configuration: dict | None = None):
        if fps <= 0 or not task.strip():
            raise ValueError("fps must be positive and task must be nonempty")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")
        self.file.attrs.update(
            schema=SCHEMA,
            fps=fps,
            task=task,
            joint_names=json.dumps(JOINT_NAMES),
            configuration=json.dumps(configuration or {}, default=str),
            action_representation="absolute_joint_radians_and_gripper_open_fraction",
            provenance="simulation of user-described starting scene; real-dataset calibration unverified",
            success_label_source=(configuration or {}).get("success_label_source", "unspecified"),
        )
        self.file.create_group("episodes")
        self.saved = 0
        self.frames = 0
        self.begin_episode()

    def begin_episode(self) -> None:
        """Discard the unfinished episode and begin a new one."""
        if "pending" in self.file:
            del self.file["pending"]
        self.pending = self.file.create_group("pending")
        self.frames = 0

    def append(self, state: np.ndarray, action: np.ndarray, cameras: dict[str, np.ndarray]) -> None:
        """Append one synchronized frame after validating every field."""
        samples = {"state": np.asarray(state, dtype=np.float32), "action": np.asarray(action, dtype=np.float32)}
        for field, values in samples.items():
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(f"{field} must contain 14 finite values")
            if np.any(values[[6, 13]] < 0) or np.any(values[[6, 13]] > 1):
                raise ValueError(f"{field} grippers must be opening fractions in [0, 1]")
        if set(cameras) != set(CAMERAS):
            raise ValueError(f"Expected exactly these RGB cameras: {CAMERAS}")
        for name in CAMERAS:
            frames = np.asarray(cameras[name])
            if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[-1] != 3:
                raise ValueError(f"{name} must be a uint8 HxWx3 RGB image")
            samples[f"cameras/{name}"] = frames
        for name, sample in samples.items():
            if name in self.pending and self.pending[name].shape[1:] != sample.shape:
                raise ValueError(f"Shape of {name} changed within the episode")
        for name, sample in samples.items():
            if name not in self.pending:
                self.pending.create_dataset(
                    name,
                    shape=(0, *sample.shape),
                    maxshape=(None, *sample.shape),
                    chunks=(1, *sample.shape),
                    dtype=sample.dtype,
                    compression="lzf",
                )
            dataset = self.pending[name]
            dataset.resize(self.frames + 1, axis=0)
            dataset[self.frames] = sample
        self.frames += 1
        if self.frames % self.file.attrs["fps"] == 0:
            self.file.flush()

    def finish_episode(self, success: bool) -> bool:
        """Commit a nonempty successful episode; discard a failed attempt."""
        saved = bool(success and self.frames)
        if saved:
            self.pending.attrs["success"] = True
            self.pending.attrs["success_label_source"] = self.file.attrs["success_label_source"]
            self.pending.attrs["length"] = self.frames
            self.file.move("pending", f"episodes/episode_{self.saved:06d}")
            self.saved += 1
        self.begin_episode()
        self.file.flush()
        return saved

    def close(self) -> None:
        """Flush the file; unfinished data remains excluded from exported demonstrations."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def modality_config() -> dict:
    """Describe the canonical simulation channels for GR00T."""
    return {
        "state": {
            name: {"original_key": "observation.state", "start": start, "end": end}
            for name, (start, end) in GROUPS.items()
        },
        "action": {
            name: {"original_key": "action", "start": start, "end": end} for name, (start, end) in GROUPS.items()
        },
        "video": {name: {"original_key": f"observation.images.{name}"} for name in CAMERAS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [len(values)],
    }


def export_dataset(input_path: str | Path, output_path: str | Path) -> Path:
    """Export successful episodes atomically, preserving all three synchronized camera streams."""
    import imageio.v2 as imageio
    import pandas as pd

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}-", dir=output_path.parent))
    try:
        with h5py.File(Path(input_path).expanduser(), "r") as source:
            if source.attrs.get("schema") != SCHEMA:
                raise ValueError("Input must be a YAM simulation collection file, not an arbitrary real dataset")
            episodes = [episode for episode in source["episodes"].values() if episode.attrs.get("success", False)]
            if not episodes:
                raise ValueError("No successful episodes to export")
            fps, task = int(source.attrs["fps"]), str(source.attrs["task"])
            metadata = temporary_path / "meta"
            metadata.mkdir()
            episode_rows, statistics_rows, state_arrays, action_arrays = [], [], [], []
            total_frames = 0
            data_path = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            video_path = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            camera_shapes = {}
            for episode_index, episode in enumerate(episodes):
                state, action = episode["state"][:], episode["action"][:]
                length = len(state)
                if state.shape != (length, 14) or action.shape != (length, 14) or length == 0:
                    raise ValueError("State and action arrays must have matching nonzero Tx14 shapes")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError("Nonfinite state or action values")
                if any(np.any(values[:, [6, 13]] < 0) or np.any(values[:, [6, 13]] > 1) for values in (state, action)):
                    raise ValueError("Gripper values must be opening fractions in [0, 1]")
                fields = {
                    "observation.state": list(state),
                    "action": list(action),
                    "timestamp": np.arange(length, dtype=np.float32) / fps,
                    "frame_index": np.arange(length, dtype=np.int64),
                    "episode_index": np.full(length, episode_index, dtype=np.int64),
                    "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
                    "task_index": np.zeros(length, dtype=np.int64),
                }
                parquet_path = temporary_path / data_path.format(
                    episode_chunk=episode_index // 1000, episode_index=episode_index
                )
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(fields).to_parquet(parquet_path, index=False)
                episode_stats = {
                    "observation.state": _stats(state.astype(np.float64)),
                    "action": _stats(action.astype(np.float64)),
                }
                for name in CAMERAS:
                    camera = episode[f"cameras/{name}"]
                    if len(camera) != length or camera.dtype != np.uint8 or camera.ndim != 4 or camera.shape[-1] != 3:
                        raise ValueError(f"Camera {name} is not synchronized uint8 RGB")
                    shape = tuple(camera.shape[1:])
                    if name in camera_shapes and camera_shapes[name] != shape:
                        raise ValueError(f"Camera resolution changed for {name}")
                    camera_shapes[name] = shape
                    video_key = f"observation.images.{name}"
                    destination = temporary_path / video_path.format(
                        episode_chunk=episode_index // 1000, video_key=video_key, episode_index=episode_index
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if shape[0] % 2 or shape[1] % 2:
                        raise ValueError("H.264 camera dimensions must be even")
                    with imageio.get_writer(
                        destination, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1
                    ) as video:
                        for frame in camera:
                            video.append_data(frame)
                episode_rows.append({
                    "episode_index": episode_index,
                    "tasks": [task],
                    "length": length,
                    "success_label_source": str(episode.attrs.get("success_label_source", "unspecified")),
                    "initial_object_positions": json.loads(episode.attrs.get("initial_object_positions", "{}")),
                })
                statistics_rows.append({"episode_index": episode_index, "stats": episode_stats})
                state_arrays.append(state)
                action_arrays.append(action)
                total_frames += length
            features = {
                "observation.state": {"dtype": "float32", "shape": [14], "names": JOINT_NAMES},
                "action": {"dtype": "float32", "shape": [14], "names": JOINT_NAMES},
                **{
                    key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
                    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
                },
                **{
                    f"observation.images.{name}": {
                        "dtype": "video",
                        "shape": list(shape),
                        "names": ["height", "width", "channel"],
                        "video_info": {
                            "video.width": shape[1],
                            "video.height": shape[0],
                            "video.channels": 3,
                            "video.fps": fps,
                            "video.codec": "h264",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "has_audio": False,
                        },
                    }
                    for name, shape in camera_shapes.items()
                },
            }
            _write_json(
                metadata / "info.json",
                {
                    "codebase_version": "v2.1",
                    "robot_type": "yam_bimanual_sim",
                    "fps": fps,
                    "total_episodes": len(episodes),
                    "total_frames": total_frames,
                    "total_tasks": 1,
                    "total_videos": len(episodes) * len(CAMERAS),
                    "total_chunks": math.ceil(len(episodes) / 1000),
                    "chunks_size": 1000,
                    "splits": {"train": f"0:{len(episodes)}"},
                    "data_path": data_path,
                    "video_path": video_path,
                    "features": features,
                },
            )
            _write_json(metadata / "modality.json", modality_config())
            _write_json(
                metadata / "stats.json",
                {
                    "observation.state": _stats(np.concatenate(state_arrays).astype(np.float64)),
                    "action": _stats(np.concatenate(action_arrays).astype(np.float64)),
                },
            )
            _write_json(
                metadata / "simulation_provenance.json",
                {
                    "schema": SCHEMA,
                    "source": str(Path(input_path).resolve()),
                    "configuration": json.loads(source.attrs["configuration"]),
                    "success_label_source": source.attrs.get("success_label_source", "unspecified"),
                    "source_dataset_compatibility": (
                        "Unverified. Do not concatenate with real data without checking units, ordering, camera"
                        " identities, gripper conventions, and task labels."
                    ),
                    "action_representation": str(source.attrs["action_representation"]),
                },
            )
            for filename, rows in (
                ("episodes.jsonl", episode_rows),
                ("episodes_stats.jsonl", statistics_rows),
                ("tasks.jsonl", [{"task_index": 0, "task": task}]),
            ):
                (metadata / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        temporary_path.rename(output_path)
        return output_path
    except BaseException:
        shutil.rmtree(temporary_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(export_dataset(args.input, args.output))


if __name__ == "__main__":
    main()
