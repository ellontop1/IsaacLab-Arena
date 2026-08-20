#!/usr/bin/env bash
# Closed-loop evaluation of the fine-tuned GR00T N2 G1 model trained on
# nvidia/Arena-G1-Static-PickNPlace-Task, inside IsaacLab-Arena.
#
# Architecture: the N2 policy server runs from the ~/isaac-gr00t-n2 repo and
# serves inference over ZMQ (port 5555). Arena runs the sim + G1 controller and
# connects as a client. The two ZMQ wire protocols (Arena's submodule
# PolicyClient <-> N2 PolicyServer) are compatible on the get_action path, so no
# custom client is required -- only the config files:
#   * isaaclab_arena_gr00t/policy/config/g1_n2_static_closedloop_config.yaml
#   * isaaclab_arena_gr00t/embodiments/g1/g1_n2_static_data_config.py
# and, on the SERVER side, the embodiment hook:
#   * ~/isaac-gr00t-n2/.onboarding/arena_g1_static_hook/{sitecustomize.py,arena_g1_static_config.py}
#
# Run the two steps below in SEPARATE terminals.
set -euo pipefail

ARENA_DIR="${ARENA_DIR:-$HOME/IsaacLab-Arena}"
N2_DIR="${N2_DIR:-$HOME/isaac-gr00t-n2}"
# Checkpoint selection (server side). Precedence:
#   1. positional arg:  `$0 server /path/to/checkpoint-100000`
#   2. CHECKPOINT env:  `CHECKPOINT=/path/... $0 server`
#   3. default below
# Teammates: point this at wherever you `hf download`-ed
# nvidia/GR00T-N2-G1-arena-static-picknplace (e.g. .../checkpoint-100000).
CHECKPOINT="${CHECKPOINT:-$N2_DIR/.onboarding/outputs/arena-g1-static-full/checkpoint-100000}"
# Nested server launcher + embodiment hook. These are vendored into this repo
# under groot_n2_server/ so eval is replicable from just this GitHub repo (plus a
# GR00T N2 install for the `groot` package). Fall back to a separate N2 checkout
# ($N2_DIR) if the vendored copies are absent. Override with SERVE= / HOOK_DIR=.
SERVE="${SERVE:-$ARENA_DIR/groot_n2_server/serve_n2_nested.py}"
[ -f "$SERVE" ] || SERVE="$N2_DIR/serve_n2_nested.py"
HOOK_DIR="${HOOK_DIR:-$ARENA_DIR/groot_n2_server/arena_g1_static_hook}"
[ -f "$HOOK_DIR/sitecustomize.py" ] || HOOK_DIR="$N2_DIR/.onboarding/arena_g1_static_hook"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
NUM_STEPS="${NUM_STEPS:-1500}"
# Eval scene knobs (defaults = shipped G1 WBC pick-place env; see yaml note).
TASK="${TASK:-galileo_g1_locomanip_pick_and_place}"
OBJECT="${OBJECT:-brown_box}"
EMBODIMENT="${EMBODIMENT:-g1_wbc_joint}"

usage() {
  cat <<EOF
Usage: $0 server [CHECKPOINT_PATH]
       $0 eval

  server [CHECKPOINT_PATH]
           Launch the GR00T N2 policy server (terminal 1, runs from \$N2_DIR).
           Pass the checkpoint dir as the 2nd arg, or via CHECKPOINT=... ,
           otherwise it defaults to:
             \$N2_DIR/.onboarding/outputs/arena-g1-static-full/checkpoint-100000
           Examples:
             $0 server ~/gr00t_n2_static/checkpoint-100000
             CHECKPOINT=~/ckpts/checkpoint-75000 $0 server

  eval     Launch the Arena closed-loop rollout (terminal 2, runs from \$ARENA_DIR).

Env overrides: ARENA_DIR, N2_DIR, CHECKPOINT, SERVE, HOOK_DIR, HOST, PORT,
               NUM_STEPS, TASK, OBJECT, EMBODIMENT
EOF
}

case "${1:-}" in
  server)
    cd "$N2_DIR"
    # Checkpoint path can be passed positionally: `$0 server /path/to/checkpoint-100000`
    # (overrides the CHECKPOINT env / default).
    if [ -n "${2:-}" ]; then CHECKPOINT="$2"; fi
    [ -d "$CHECKPOINT" ] || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }
    [ -f "$HOOK_DIR/sitecustomize.py" ] || { echo "ERROR: hook dir missing: $HOOK_DIR"; exit 1; }
    [ -f "$SERVE" ] || { echo "ERROR: nested server launcher not found: $SERVE"; exit 1; }
    export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$HOOK_DIR:${PYTHONPATH:-}"
    echo "[server] serving checkpoint: $CHECKPOINT"
    echo "[server] launcher:          $SERVE"
    # IMPORTANT: use serve_n2_nested.py, NOT the stock `groot.infra.oss policy-server`.
    # The stock server wraps the policy in Gr00tFlatPolicyWrapper, which expects
    # FLAT dotted obs keys (video.ego_view) and returns action.<key>. Arena sends
    # NESTED observations ({"video": {"ego_view": ...}, "state": {...}}) and expects
    # actions keyed by joint group. serve_n2_nested.py swaps in a nested passthrough
    # adapter and injects zero navigate_command/base_height_command for the WBC.
    exec uv run --no-sync python "$SERVE" \
      --checkpoint "$CHECKPOINT" \
      --embodiment-tag new_embodiment \
      --host 0.0.0.0 \
      --port "$PORT" \
      --denoising-steps 4
    ;;
  eval)
    cd "$ARENA_DIR"
    # We call the venv's python directly so this works whether or not the venv
    # is activated. Requires `uv sync` to have completed.
    PY="$ARENA_DIR/.venv/bin/python"
    [ -x "$PY" ] || { echo "ERROR: $PY not found -- run 'uv sync' in $ARENA_DIR first."; exit 1; }
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    # gr00t lives in the submodule and is NOT installed by a native `uv sync`
    # (only the Docker -g flavor installs it); put it on the path so Arena's
    # GR00T policy translation code can import it.
    export PYTHONPATH="$ARENA_DIR/submodules/Isaac-GR00T:${PYTHONPATH:-}"
    # --enable_cameras: offscreen RTX rendering (the policy consumes the head cam).
    # Drop --headless if you want to watch the sim GUI on the workstation.
    exec "$PY" isaaclab_arena/evaluation/policy_runner.py \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
      --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_n2_static_closedloop_config.yaml \
      --remote_host "$HOST" \
      --remote_port "$PORT" \
      --num_steps "$NUM_STEPS" \
      --enable_cameras \
      "$TASK" \
      --object "$OBJECT" \
      --embodiment "$EMBODIMENT"
    ;;
  *)
    usage
    exit 1
    ;;
esac
