#!/usr/bin/env bash
# Fine-tune GR00T N2 on nvidia/Arena-G1-Static-PickNPlace-Task (208 episodes,
# unitree_g1 whole-body joint-position teleop, ego-view camera).
#
# Mirrors the lerobot_output_405 recipe but:
#   * dataset  -> ~/arena_g1_static/lerobot
#   * PYTHONPATH hook -> arena_g1_static_hook (this dataset's modality layout)
#   * output   -> .onboarding/outputs/arena-g1-static-full
#   * save_total_limit=1 (checkpoint rotation holds 3 ckpts transiently with
#     limit=2 -> ~141 GB peak, over the ~115 GB free; limit=1 peaks ~94 GB).
set -euo pipefail

cd "$HOME/isaac-gr00t-n2"

LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$PWD/.onboarding/arena_g1_static_hook:${PYTHONPATH:-}" \
TRAINING_OUTPUT_ROOT="$PWD/.onboarding/outputs" \
uv run --no-sync python -m groot.infra.oss train n2_egoscale_mega_soup_pretrain \
  --dataset-path "$HOME/arena_g1_static/lerobot" \
  --embodiment-tag new_embodiment \
  --output-dir "$PWD/.onboarding/outputs/arena-g1-static-full" \
  --num-gpus 1 \
  --batch-size 1 \
  --max-steps 10000 \
  --set 'data.video_backend="torchcodec"' \
  --set 'data.modality_configs={"new_embodiment":{"video":{"delta_indices":[0,5,10,15,20]}}}' \
  --set 'data.override_pretraining_statistics=false' \
  --set 'training.save_steps=2000' \
  --set 'training.save_total_limit=1' \
  --set 'training.torch_compile=false'
