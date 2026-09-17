# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate settled object containment before recording demonstrations."""


def validate_start(env, config, manifest):
    """Require all three objects to remain inside the upright, settled carton."""
    import itertools
    import torch

    from isaaclab.utils import math as math_utils

    from isaaclab_arena_examples.yam_yellow_box.environment import _tensor

    carton = env.scene["cardboard_box"]
    carton_pos = _tensor(carton.data.root_pos_w)
    carton_quat = _tensor(carton.data.root_quat_w)
    up = torch.zeros_like(carton_pos)
    up[:, 2] = 1
    assert bool((math_utils.quat_apply(carton_quat, up)[:, 2] > 0.98).all()), "Carton tipped during settling"
    origins = _tensor(env.scene.env_origins)
    assert bool(
        ((carton_pos[:, 2] - origins[:, 2] - config.table_height).abs() < 0.015).all()
    ), "Carton is not on the table"
    report = {}
    for source in manifest:
        asset = env.scene[source["name"]]
        position, rotation = _tensor(asset.data.root_pos_w), _tensor(asset.data.root_quat_w)
        corners = torch.tensor(
            list(itertools.product(*zip(source["bounds_min"], source["bounds_max"]))), device=env.device
        )
        corners = corners.unsqueeze(0).expand(env.num_envs, -1, -1)
        world = math_utils.quat_apply(rotation[:, None, :].expand(-1, 8, -1), corners) + position[:, None, :]
        relative = math_utils.quat_apply_inverse(
            carton_quat[:, None, :].expand(-1, 8, -1), world - carton_pos[:, None, :]
        )
        for axis in range(2):
            inside = relative[:, :, axis].abs() <= config.box_size[axis] / 2 - config.box_wall_thickness + 0.003
            assert bool(inside.all()), f"{source['name']} escaped/intersects a carton wall"
        assert bool(
            (relative[:, :, 2].amin(dim=1) >= config.box_wall_thickness - 0.005).all()
        ), f"{source['name']} fell through the carton floor"
        assert bool(
            (relative[:, :, 2].amax(dim=1) <= config.box_size[2] + 0.005).all()
        ), f"{source['name']} extends above the carton rim"
        report[source["name"]] = position.detach().cpu().tolist()
    for name in ["cardboard_box", *report]:
        asset = env.scene[name]
        assert bool(
            (torch.linalg.vector_norm(_tensor(asset.data.root_lin_vel_w), dim=-1) < 0.05).all()
        ), f"{name} is still moving; increase settle_steps"
        assert bool(
            (torch.linalg.vector_norm(_tensor(asset.data.root_ang_vel_w), dim=-1) < 0.5).all()
        ), f"{name} is still rotating; increase settle_steps"
    report["cardboard_box"] = carton_pos.detach().cpu().tolist()
    return report
