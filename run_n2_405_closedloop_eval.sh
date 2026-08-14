#!/usr/bin/env bash
# Closed-loop evaluation of the fine-tuned GR00T N2 G1 model (new_embodiment,
# trained on lerobot_output_405) inside IsaacLab-Arena.
#
# Architecture: the N2 policy server runs from the ~/isaac-gr00t-n2 repo and
# serves inference over ZMQ (port 5555). Arena runs the sim + G1 controller and
# connects to that server as a client. The two ZMQ wire protocols
# (Arena's submodule PolicyClient <-> N2 PolicyServer) are compatible on the
# get_action path, so no custom client is required -- only the config files:
#   * isaaclab_arena_gr00t/policy/config/g1_n2_405_closedloop_config.yaml
#   * isaaclab_arena_gr00t/embodiments/g1/g1_n2_405_data_config.py
#
# Run the two steps below in SEPARATE terminals.
set -euo pipefail

ARENA_DIR="${ARENA_DIR:-$HOME/IsaacLab-Arena}"
N2_DIR="${N2_DIR:-$HOME/isaac-gr00t-n2}"
CHECKPOINT="${CHECKPOINT:-$N2_DIR/.onboarding/outputs/new-embodiment-full/checkpoint-10000}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
NUM_STEPS="${NUM_STEPS:-1500}"

usage() {
  cat <<EOF
Usage: $0 {server|eval}

  server   Launch the GR00T N2 policy server (run in terminal 1, from $N2_DIR)
  eval     Launch the Arena closed-loop rollout (run in terminal 2, from $ARENA_DIR)

Env overrides: ARENA_DIR, N2_DIR, CHECKPOINT, HOST, PORT, NUM_STEPS
EOF
}

case "${1:-}" in
  server)
    cd "$N2_DIR"
    export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$PWD/.onboarding/groot_embodiment_hook:${PYTHONPATH:-}"
    exec uv run --no-sync python -m groot.infra.oss policy-server \
      --checkpoint "$CHECKPOINT" \
      --embodiment-tag new_embodiment \
      --host 0.0.0.0 \
      --port "$PORT" \
      --denoising-steps 4
    ;;
  eval)
    cd "$ARENA_DIR"
    # Requires `uv sync` to have completed (see README). We call the venv's
    # python directly so this works whether or not the venv is activated.
    PY="$ARENA_DIR/.venv/bin/python"
    [ -x "$PY" ] || { echo "ERROR: $PY not found -- run 'uv sync' in $ARENA_DIR first."; exit 1; }
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    # gr00t lives in the submodule and is NOT installed by a native `uv sync`
    # (only the Docker -g flavor installs it); put it on the path so Arena's
    # GR00T policy translation code can import it.
    export PYTHONPATH="$ARENA_DIR/submodules/Isaac-GR00T:${PYTHONPATH:-}"
    # --headless: no GUI (required on a display-less server like brev).
    # --enable_cameras: offscreen RTX rendering, works in headless mode and is
    #   required because the policy consumes the head camera.
    exec "$PY" isaaclab_arena/evaluation/policy_runner.py \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
      --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_n2_405_closedloop_config.yaml \
      --remote_host "$HOST" \
      --remote_port "$PORT" \
      --num_steps "$NUM_STEPS" \
      --headless \
      --enable_cameras \
      galileo_g1_locomanip_pick_and_place \
      --object brown_box \
      --embodiment g1_wbc_joint
    ;;
  *)
    usage
    exit 1
    ;;
esac
