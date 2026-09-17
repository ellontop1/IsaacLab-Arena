# YAM carton scene and demonstration collection

This scene starts with an **apple, Jell-O box, and can inside an open cardboard
box**, next to the two YAM arms on a tabletop. The old solid yellow cube and blue
target have been removed. The carton is a movable rigid body made of a floor and
four separate wall colliders, leaving a real opening for the grippers and props.

The layout follows the user's description of
[nvidia/yam_yellow_box_all](https://huggingface.co/datasets/nvidia/yam_yellow_box_all).
Dataset access failed with HTTP 401 anonymously and HTTP 404 using the available
Hugging Face login. No videos or metadata could be inspected. This is a configurable
starting-scene reconstruction, **not yet a calibrated digital twin**. The actual
manipulation task and destination have not been confirmed. There is no invented
automatic success predicate and no automatic successful-episode saving.

## Assets and dimensions

- YAM: pinned, SHA256-verified official I2RT station geometry prepared by `prepare_assets.py`.
- Apple: Arena's `apple_02_objaverse_robolab` textured USD.
- Jell-O: Arena's `jello_ycb_robolab` textured USD.
- Can: Arena's `tomato_soup_can_ycb_robolab` textured USD; the real can type is unverified.
- Carton: generated brown cardboard material and compound collision geometry.

`scene.example.json` exposes outer carton dimensions, wall thickness, mass,
positions, object sizes/masses, reset variation, and each prop's optional
`usd_path` override. Defaults are estimates: carton 36 × 28 × 14 cm, apple 7.5 cm,
Jell-O box 8.5 × 2.8 × 7 cm, can 6.6 × 6.6 × 12 cm. Objects are scaled to these
extents and placed using their USD bounds, not an assumed centred mesh origin.
Library assets are fetched from the configured Arena asset server; no primitive
fallback hides missing downloads. The selected asset paths and scales are saved
with each recording. Appearance, labels, carton flaps/printing, friction and mass
still need comparison to the real setup. Cardboard deformation is not simulated.

## Run on the simulation host

Use an installed Linux/NVIDIA Isaac Lab Arena environment. This Mac cannot run the
Isaac Sim GPU validation. Fetch the updated `yam-embodiment-assets` branch from
`ellontop1/IsaacLab-Arena`. From the repository root inside its configured container:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.yam_yellow_box.prepare_assets
/isaac-sim/python.sh -m isaaclab_arena_examples.yam_yellow_box.smoke \
  --config isaaclab_arena_examples/yam_yellow_box/scene.example.json \
  --output /datasets/yam_smoke_v2 --steps 60
```

The smoke check settles the scene, verifies the carton and all three contents remain
inside/on the intended support surfaces, checks 14D states/actions, and writes three
camera images plus a report. It does not certify grasping or task success. Inspect
all views and compare them with the real cameras before collecting a large dataset.
Use a new output directory each time. To open an interactive preview, see [LAUNCH.md](LAUNCH.md).

## Collect and export

Set `task_description` to the actual demonstrated instruction in the JSON, or pass
it with `--task`. The collector refuses an empty task rather than assigning the old
box-placement instruction to new demonstrations.

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.yam_yellow_box.collect \
  --config isaaclab_arena_examples/yam_yellow_box/scene.example.json \
  --output /datasets/yam_sim/carton_demos_001.hdf5 --episodes 20 --viz kit \
  --task "REPLACE WITH THE ACTUAL DEMONSTRATED TASK"
```

Click the Kit viewport. Tab selects an arm; W/S, A/D, Q/E translate it; Z/C, T/G,
F/H rotate it. Space toggles the selected gripper. P pauses. **Enter confirms task
success and saves**; R discards/resets; Escape exits. Confirm success only when the
specified real task has been completed. Timeouts and unfinished attempts are not
exported. No automatic success label is inferred from moving the carton.

Each reset samples the carton and contents together. Object reset bounds must fit
inside its walls and remain separated; invalid configs fail early. Settling happens
before the first recorded frame, and containment/stability checks run before each
episode. Operator-confirmed success is recorded explicitly in provenance.

The collector records pre-action RGB/state and the corresponding absolute joint
targets. State/action vectors retain 14 entries: six left-arm radians, left gripper
opening fraction, six right-arm radians, right gripper opening fraction. Three RGB
views (`top_camera`, `left_wrist_camera`, `right_wrist_camera`) are recorded at the
configured rate, default 30 Hz. IK is a teleoperation interface, not the exported
action representation.

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.yam_yellow_box.exporter \
  --input /datasets/yam_sim/carton_demos_001.hdf5 \
  --output /datasets/yam_sim_lerobot_v2
```

The exporter creates a standalone LeRobot v2.1 dataset with synchronized videos,
Parquet samples, task labels and simulation provenance. `gr00t_config.py` describes
these channels for GR00T N1.6; it is not a verified adapter for the inaccessible
reference dataset. Before mixing real and simulated data or training a shared
policy, establish the real task, matching camera identities/calibration, timing,
state/action ordering, units and gripper conventions. Evaluate grasp/contact
behavior and held-out simulated episodes before any real-robot deployment.

## Validation

Inside the Arena runtime, run:

```bash
/isaac-sim/python.sh -m unittest discover \
  -s isaaclab_arena_examples/yam_yellow_box/tests -v
```

Tests cover carton geometry, placement bounds and overlap rejection, recording
alignment, failed-episode exclusion and exported video/metadata. GPU smoke, physical
contacts, real-scene matching and policy training have not been executed on this Mac.
