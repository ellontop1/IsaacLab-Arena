# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fetch a small, revision-pinned dataset reference using the current Hugging Face login."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_dataset(repo_id: str, output: Path, extra_files: list[str]) -> dict:
    """Download metadata and explicitly requested files without fetching the full dataset."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repository = HfApi().dataset_info(repo_id=repo_id)
    revision = repository.sha
    metadata_files = ("README.md", "meta/info.json", "meta/modality.json", "meta/tasks.jsonl", "meta/tasks.parquet")
    downloaded = {}
    for filename in dict.fromkeys([*metadata_files, *extra_files]):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a repository-relative path: {filename}")
        try:
            downloaded[filename] = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=filename,
                local_dir=output,
            )
        except EntryNotFoundError:
            if filename in extra_files:
                raise
    info = {}
    if "meta/info.json" in downloaded:
        info = json.loads(Path(downloaded["meta/info.json"]).read_text())
    tasks = []
    if "meta/tasks.jsonl" in downloaded:
        tasks = [
            json.loads(line) for line in Path(downloaded["meta/tasks.jsonl"]).read_text().splitlines() if line.strip()
        ]
    report = {
        "repo_id": repo_id,
        "revision": revision,
        "files_downloaded": list(downloaded),
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "features": info.get("features"),
        "tasks": tasks,
        "calibration_verified": False,
    }
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="nvidia/yam_yellow_box_all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", default=[], help="Optional exact repository file to fetch")
    args = parser.parse_args()
    try:
        report = inspect_dataset(args.repo_id, args.output, args.file)
    except Exception as error:
        from huggingface_hub.errors import HfHubHTTPError

        if isinstance(error, HfHubHTTPError):
            status = error.response.status_code if error.response is not None else "unknown"
            raise SystemExit(
                f"Hugging Face request failed (HTTP {status}). Verify the dataset ID and run hf auth login "
                "with an account granted access. Do not share your token in logs or chat."
            ) from None
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
