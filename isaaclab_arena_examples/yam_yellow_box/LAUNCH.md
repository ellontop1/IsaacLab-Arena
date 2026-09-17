# Load the YAM scene

This scene contains a dual-arm YAM station, a table, one yellow box, a blue
placement target, ground, lighting, and three cameras. Additional props are not
included. Geometry and camera calibration are provisional. Isaac Sim runtime
validation has not been performed on the development Mac.

On your configured Linux/NVIDIA Isaac Lab Arena host, fetch and check out the
`yam-embodiment-assets` branch of `ellontop1/IsaacLab-Arena`. Run the following
from the repository root inside its Isaac Sim container:

```bash
/isaac-sim/python.sh -m isaaclab_arena_examples.yam_yellow_box.prepare_assets

/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --viz kit --enable_cameras --policy_type zero_action --num_steps 3000 \
  --external_environment_class_path isaaclab_arena_examples.yam_yellow_box.environment:YellowBoxEnvironment \
  yam_yellow_box --yam_control_mode ik \
  --yam_config isaaclab_arena_examples/yam_yellow_box/scene.example.json
```

Asset preparation downloads verified I2RT meshes and converts the station URDF
to `~/.cache/isaaclab_arena/yam/yam_station.usd`. Run it once in the same runtime
environment used to launch the scene. Keep the generated directory together:
the USD can reference adjacent files. Generated assets are not stored in Git.

The second command opens the scene with a zero-action policy in IK mode to hold
the arm pose; it does not perform the pick-and-place task. A graphical Kit session
must be available. Edit `scene.example.json` to change object sizes and positions
or point `usd_path` at an already prepared compatible station USD.
