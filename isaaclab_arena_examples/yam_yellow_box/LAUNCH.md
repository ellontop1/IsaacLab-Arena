# Open the YAM carton environment

The starting scene contains two YAM arms, three cameras, a table, and an open
cardboard box containing an apple, Jell-O box, and can. The former blue target is
removed. Dimensions and camera calibration are provisional; see [README.md](README.md).

On your Linux/NVIDIA Isaac Lab Arena host, fetch the updated branch from your fork:

```bash
git fetch origin
git switch yam-embodiment-assets
git pull --ff-only
```

Here `origin` must point to `ellontop1/IsaacLab-Arena`. Inside the configured Arena
container, from the repository root, prepare the robot once and launch the scene:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.yam_yellow_box.prepare_assets

/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --viz kit --enable_cameras --policy_type zero_action --num_steps 3000 \
  --external_environment_class_path isaaclab_arena_examples.yam_yellow_box.environment:YellowBoxEnvironment \
  yam_yellow_box --yam_control_mode ik \
  --yam_config isaaclab_arena_examples/yam_yellow_box/scene.example.json
```

The preview holds the arm pose. It does not execute a manipulation policy. Kit
needs a graphical session. Scene loading also requires access to Arena's textured
prop USD assets; it will fail explicitly if those assets cannot be loaded.

The robot USD and generated carton USD live in `~/.cache/isaaclab_arena/yam` by
default. Keep robot conversion files together because its USD references adjacent
files. Models are not committed to Git. To collect demonstrations, use the README's
collector command with the actual task instruction and operator-confirmed success.
