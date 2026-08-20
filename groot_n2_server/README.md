# GR00T N2 — G1 Arena Static Pick-and-Place: Replication Guide

Everything needed to **evaluate** (and optionally **train**) the GR00T N2 policy
fine-tuned on `nvidia/Arena-G1-Static-PickNPlace-Task`, in IsaacLab-Arena.

## Architecture

Eval is a **server/client split over ZMQ** (N2 has no LEAPP/ONNX export path):

```
  [ RTX workstation ]                         [ GPU box w/ GR00T N2 ]
  IsaacLab-Arena sim + G1 WBC     ZMQ :5555   serve_n2_nested.py
  Gr00tRemoteClosedloopPolicy  <----------->  (GR00T N2 policy)
  (renders head cam, steps env)               (runs inference)
```

- The **server** loads the checkpoint and runs inference. It needs the GR00T N2
  `groot` package + a CUDA GPU (~24 GB VRAM).
- The **client** is Arena; it renders the head camera (needs an **RTX** GPU) and
  drives the sim. Server and client may be the same box or connected via an SSH
  tunnel.

## 0. Prerequisites

- **Access:** Hugging Face account with access to the (private) model repo
  `nvidia/GR00T-N2-G1-arena-static-picknplace`. Run `hf auth login`.
- **GR00T N2** (`isaac-gr00t-n2`) checked out and installed (provides the `groot`
  package + its `uv` venv). This is the internal GitLab repo.
- **IsaacLab-Arena** checked out and set up (`uv sync`, or the conda
  `env_isaaclab`), with the `submodules/Isaac-GR00T` submodule present.
- **Hardware:** a CUDA GPU for the server; an **RTX**-capable GPU for the Arena
  client (Isaac Sim RTX rendering). Training needs ~1x H100 (80 GB) + ~120 GB disk.

## 1. Get this repo

```bash
git clone https://github.com/ellontop1/IsaacLab-Arena.git
cd IsaacLab-Arena && git checkout groot-n2-eval
```

The eval files:
- `isaaclab_arena_gr00t/embodiments/g1/g1_n2_static_data_config.py` — client modality config
- `isaaclab_arena_gr00t/policy/config/g1_n2_static_closedloop_config.yaml` — closed-loop policy config
- `run_n2_static_closedloop_eval.sh` — server|eval launcher
- `groot_n2_server/` — vendored server launcher + embodiment hook + training scripts

## 2. Download the checkpoint

```bash
hf download nvidia/GR00T-N2-G1-arena-static-picknplace \
  --include 'checkpoint-100000/*' \
  --local-dir ~/gr00t_n2_static
# -> ~/gr00t_n2_static/checkpoint-100000
```
(75000 / 50000 / 10000 also exist. 100000 is the final/best.)

## 3. Run the evaluation

Two terminals. Set `N2_DIR` to your GR00T N2 checkout and `ARENA_DIR` to this repo.

### Terminal 1 — policy server (on the GPU box)
```bash
export N2_DIR=~/isaac-gr00t-n2 ARENA_DIR=~/IsaacLab-Arena
./run_n2_static_closedloop_eval.sh server ~/gr00t_n2_static/checkpoint-100000
```
Wait for `Server is running on 0.0.0.0:5555`.

### (If server and client are on different machines) SSH tunnel
On the **client** box, forward the server's port:
```bash
ssh -N -L 5555:localhost:5555 <user>@<server-host>
```
Then set `HOST=127.0.0.1` for the eval step.

### Terminal 2 — Arena client (on the RTX box)
```bash
export N2_DIR=~/isaac-gr00t-n2 ARENA_DIR=~/IsaacLab-Arena
HOST=127.0.0.1 PORT=5555 ./run_n2_static_closedloop_eval.sh eval
```

**Checkpoint selection** (server): positional arg (above), or `CHECKPOINT=/path ...`,
else defaults to `$N2_DIR/.onboarding/outputs/arena-g1-static-full/checkpoint-100000`.

**Run length:** the script uses `NUM_STEPS` (default 1500). For a fixed number of
completed episodes, run `policy_runner.py` directly with `--num_episodes 100`
(omit `--num_steps`).

**Scene note:** the default eval scene (`galileo_g1_locomanip_pick_and_place`,
`--object brown_box`) differs from the training scene (apple→shelf/plate), so
treat a first run as a generalization/pipeline smoke test. Override `TASK` /
`OBJECT` / `EMBODIMENT` for a matching scene.

## 4. (Optional) Train from scratch

```bash
# dataset (public)
hf download nvidia/Arena-G1-Static-PickNPlace-Task --repo-type dataset \
  --include 'lerobot/*' --local-dir ~/arena_g1_static
# NOTE: patch ~/arena_g1_static/lerobot/meta/info.json so total_episodes and
# total_videos = 208 (the shipped file claims 251), total_chunks = 1.

# train (from the GR00T N2 repo; hook must be on PYTHONPATH)
cp -r groot_n2_server/arena_g1_static_hook ~/isaac-gr00t-n2/.onboarding/
bash groot_n2_server/train_arena_g1_static_10k.sh     # 0 -> 10k
# resume to 100k (same output dir auto-resumes from latest checkpoint):
bash groot_n2_server/train_arena_g1_static_100k.sh
```

Training benchmarks (1x H100, bs=1, full fine-tune): **~0.70 s/step (~1.43
steps/s) ≈ ~20 h per 100k steps.** Each checkpoint is ~47 GB — upload to HF and
rotate local copies (the disk holds only ~2 at a time on a 247 GB volume).

## 5. Troubleshooting (issues we actually hit)

| Symptom | Cause / fix |
|---|---|
| Client/server "flat vs nested" mismatch, or actions ignored | Use `serve_n2_nested.py` (this repo), **not** stock `groot.infra.oss policy-server`. The stock server wraps in `Gr00tFlatPolicyWrapper` (flat `video.ego_view` keys); Arena sends nested. |
| `KeyError: 'left_leg'` at the server | Client must send **leg** joint state. Use `g1_n2_static_data_config.py` (includes `left_leg`/`right_leg`), not the stock config. |
| Robot won't move / holds still | Undertrained checkpoint. 10k = 0% success; use 100k. Check server `[action-values]` logs (span/d0/amax) to confirm real motion. |
| `ModuleNotFoundError: groot` | Run the server from the GR00T N2 venv (`uv run --no-sync ...`, handled by the launcher via `$N2_DIR`). |
| Robot tips over / waist drifts | Expected: the model commands waist=0 and zero base motion (static task). The launcher injects zero `navigate_command`/`base_height_command` for the WBC. |
| Isaac Sim RTX/render errors | The **client** needs an RTX GPU + graphics driver; a compute-only card (e.g. bare H100) can't render. |

## Artifacts reference

- Model (private): `nvidia/GR00T-N2-G1-arena-static-picknplace` (`checkpoint-100000`)
- Dataset (public): `nvidia/Arena-G1-Static-PickNPlace-Task` (208 episodes, ego-view, absolute joint actions, horizon 16)
- Eval configs + server + train scripts: this repo, branch `groot-n2-eval`
