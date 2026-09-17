# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build an open compound-collision carton and dimension the textured library props."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def carton_panels(size, thickness):
    """Return five nonoverlapping panels with the carton origin at its bottom centre."""
    x, y, z = size
    t = thickness
    return [
        ("floor", (x, y, t), (0, 0, t / 2)),
        ("wall_x_neg", (t, y, z - t), (-(x - t) / 2, 0, (z + t) / 2)),
        ("wall_x_pos", (t, y, z - t), ((x - t) / 2, 0, (z + t) / 2)),
        ("wall_y_neg", (x - 2 * t, t, z - t), (0, -(y - t) / 2, (z + t) / 2)),
        ("wall_y_pos", (x - 2 * t, t, z - t), (0, (y - t) / 2, (z + t) / 2)),
    ]


def prepare_carton(config):
    """Cache a rigid carton USD whose separate panel colliders leave the top and interior open."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    parameters = {"size": config.box_size, "wall": config.box_wall_thickness, "mass": config.box_mass, "version": 1}
    key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:16]
    destination = Path(config.usd_path).expanduser().resolve().parent / f"carton_{key}.usda"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return str(destination)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Carton").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(config.box_mass)
    material = UsdShade.Material.Define(stage, "/Carton/Cardboard")
    shader = UsdShade.Shader.Define(stage, "/Carton/Cardboard/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    from pxr import Sdf

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.63, 0.43, 0.20))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.7)
    physics.CreateDynamicFrictionAttr(0.5)
    physics.CreateRestitutionAttr(0.0)
    for name, size, position in carton_panels(config.box_size, config.box_wall_thickness):
        cube = UsdGeom.Cube.Define(stage, f"/Carton/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(material)
        binding.Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()
    return str(destination)


def make_prop(prop, config):
    """Scale a textured rigid USD to configured extents and place its bottom on the carton floor."""
    import math

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import Apple02ObjaverseRobolab, JelloYcbRobolab, TomatoSoupCanYcbRobolab
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

    library = {cls.name: cls for cls in (Apple02ObjaverseRobolab, JelloYcbRobolab, TomatoSoupCanYcbRobolab)}
    if prop.usd_path:
        usd_path = str(Path(prop.usd_path).expanduser()) if "://" not in prop.usd_path else prop.usd_path
    else:
        assert prop.asset_name in library, f"Unsupported prop asset: {prop.asset_name}; set usd_path for a custom asset"
        usd_path = library[prop.asset_name].usd_path
    bounds = compute_local_bounding_box_from_usd(usd_path)
    native_size = bounds.size[0].tolist()
    assert all(math.isfinite(v) and v > 0 for v in native_size), f"Invalid bounds for {usd_path}"
    scale = tuple(wanted / actual for wanted, actual in zip(prop.size, native_size))
    centre = [v * s for v, s in zip(bounds.center[0].tolist(), scale)]
    bottom = float(bounds.min_point[0, 2]) * scale[2]
    c, s = math.cos(prop.yaw), math.sin(prop.yaw)
    position = (
        config.box_start_xy[0] + prop.offset_xy[0] - (c * centre[0] - s * centre[1]),
        config.box_start_xy[1] + prop.offset_xy[1] - (s * centre[0] + c * centre[1]),
        config.table_height + config.box_wall_thickness + 2 * config.spawn_clearance - bottom,
    )
    obj = Object(
        name=prop.name,
        object_type=ObjectType.RIGID,
        usd_path=usd_path,
        scale=scale,
        initial_pose=Pose(
            position_xyz=position, rotation_xyzw=(0.0, 0.0, math.sin(prop.yaw / 2), math.cos(prop.yaw / 2))
        ),
        spawn_cfg_addon={
            "mass_props": sim_utils.MassPropertiesCfg(mass=prop.mass),
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
        },
    )
    obj.disable_reset_pose()
    return obj, {
        "name": prop.name,
        "usd_path": usd_path,
        "scale": scale,
        "size": prop.size,
        "mass": prop.mass,
        "bounds_min": [v * s for v, s in zip(bounds.min_point[0].tolist(), scale)],
        "bounds_max": [v * s for v, s in zip(bounds.max_point[0].tolist(), scale)],
    }
