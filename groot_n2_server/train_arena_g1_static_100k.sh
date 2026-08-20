#!/usr/bin/env bash
# Resume GR00T N2 fine-tune on nvidia/Arena-G1-Static-PickNPlace-Task from
# checkpoint-50000 up to 100k steps.
#
#   * output   -> .onboarding/outputs/arena-g1-static-full (same dir => the
#     trainer auto-resumes from the latest checkpoint present, i.e. 50000)
#   * save_steps=25000       -> saves at 75000 and 100000
#   * save_total_limit=2      -> keep BOTH checkpoint-75000 and checkpoint-100000
#
# Disk plan (247G vol): local checkpoint-50000 (47G) is already fully uploaded
# to HF (nvidia/GR00T-N2-G1-arena-static-picknplace/checkpoint-50000). Once this
# run is confirmed stepping past 50k, the local checkpoint-50000 is deleted to
# make room; the 75k save then peaks well within free space.
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
  --max-steps 100000 \
  --set 'data.video_backend="torchcodec"' \
  --set 'data.modality_configs={"new_embodiment":{"video":{"delta_indices":[0,5,10,15,20]}}}' \
  --set 'data.override_pretraining_statistics=false' \
  --set 'training.save_steps=25000' \
  --set 'training.save_total_limit=2' \
  --set 'training.torch_compile=false'
