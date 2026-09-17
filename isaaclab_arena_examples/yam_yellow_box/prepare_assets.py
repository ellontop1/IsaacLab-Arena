# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Download verified official YAM geometry and convert it on an Isaac Sim host."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

I2RT_REVISION = "5b72c47239bd056d0fa6c1a39edeb0537c89443c"
STATION_PATH = "i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf"
ASSET_SHA256 = {
    "LICENSE": "7cd0a0224a287ebd6c45703eb6623d264c7810f8958d65de8f99e87ccd9f4891",
    "i2rt/robot_models/arm/yam/v1/assets/base.stl": "d95a5aa0c06a5ef6da02788b0df5a14e4202c1ae8eb2771a7f6f8947e3a490c7",
    "i2rt/robot_models/arm/yam/v1/assets/gripper.stl": (
        "e0dad958755f0a9d58e6bea4f16fed2849146ae81c0dbd45487aff308cb49e58"
    ),
    "i2rt/robot_models/arm/yam/v1/assets/link1.stl": "41f7dfbd0471a0f96884c73782e82927f2bbc6aced9b55c0111da1b52d6e0c50",
    "i2rt/robot_models/arm/yam/v1/assets/link2.stl": "067b51567a95a631d50fc02515bd045a2eb7f2f34489d5aec31d9f3f6ce58b30",
    "i2rt/robot_models/arm/yam/v1/assets/link3.stl": "9650a114393881b54a68822ba3fb48fcd3318023ebecaf251e4f7804e91f801c",
    "i2rt/robot_models/arm/yam/v1/assets/link4.stl": "42740ce2bb8693760a35dc6e9e8fe0981999c02d6bf167a465c9936acf41c557",
    "i2rt/robot_models/arm/yam/v1/assets/link5.stl": "f70eb1191d6a1ccbb748a46fe8ba541f38094c9829fd1859300e70360c5a07c7",
    "i2rt/robot_models/arm/yam/v1/assets/tip_left.stl": (
        "281354d77e4914ccd994b6c71fecce49a91527701f268d318f801769a6913685"
    ),
    "i2rt/robot_models/arm/yam/v1/assets/tip_right.stl": (
        "070925233d4b742c05c4d5508e951deba5947c2fabfa336a4e33db599c70ac3a"
    ),
    "i2rt/robot_models/station/assets/d405_top.stl": "480453e27beb82122615d2f898623573bf50f202f54305c2407e69770c2ea173",
    "i2rt/robot_models/station/assets/d405_wrist_linear_4310.stl": (
        "f01bcd92b447c3a44c1ad9345367628d4157fe4c07b545ccbf9bc4ae81115112"
    ),
    "i2rt/robot_models/station/yam_station_linear_4310_d405/README.md": (
        "7abbf958fc416d4bc99a6a99875e698d5a6268f239848a34a47836bc4f27c924"
    ),
    STATION_PATH: "769e074e93cde5a3ed66b9903364b887cbd9fd120834ff9f0defa5a3a12452f3",
}


def fetch_assets(destination: Path) -> Path:
    """Fetch only hash-verified model files, retaining the upstream MIT license."""
    source_dir = destination / "i2rt_source"
    for relative_path, expected_hash in ASSET_SHA256.items():
        local_path = source_dir / relative_path
        if local_path.is_file():
            content = local_path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/i2rt-robotics/i2rt/{I2RT_REVISION}/{relative_path}"
            print(f"Downloading {relative_path}")
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {relative_path}: {actual_hash}")
        if not local_path.is_file():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = local_path.with_suffix(local_path.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(local_path)
    return source_dir / STATION_PATH


def prepare_urdf(source_path: Path, destination: Path) -> Path:
    """Fix placeholder limits/inertials in a separate URDF while preserving camera links."""
    document = ET.parse(source_path)
    model = document.getroot()
    for mesh in model.findall(".//mesh"):
        mesh.set("filename", str((source_path.parent / mesh.attrib["filename"]).resolve()))
    for joint in model.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        if joint.attrib["type"] == "prismatic":
            limit.set("effort", "20")
            limit.set("velocity", "0.15")
        else:
            joint_number = int(joint.attrib["name"].rsplit("joint", 1)[1])
            limit.set("effort", "30" if joint_number <= 3 else "10")
            limit.set("velocity", "3")
    for link in model.findall("link"):
        if link.attrib["name"] not in ("left_camera", "right_camera", "top_camera"):
            continue
        mass = link.find("inertial/mass")
        inertia = link.find("inertial/inertia")
        assert mass is not None and inertia is not None, "Pinned camera inertials are missing"
        mass_ratio = 0.12 / float(mass.attrib["value"])
        mass.set("value", "0.12")
        for component, value in list(inertia.attrib.items()):
            inertia.set(component, str(float(value) * mass_ratio))
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "yam_station.urdf"
    ET.indent(document, space="  ")
    document.write(output_path, encoding="utf-8", xml_declaration=True)
    (destination / "asset_manifest.json").write_text(
        json.dumps(
            {
                "repository": "https://github.com/i2rt-robotics/i2rt",
                "revision": I2RT_REVISION,
                "station": STATION_PATH,
                "source_sha256": ASSET_SHA256,
                "provisional": True,
                "modifications": {
                    "camera_mass_kg": 0.12,
                    "camera_inertia": "CAD tensor rescaled by mass ratio; estimated, not measured",
                    "arm_effort_limits_nm": [30, 30, 30, 10, 10, 10],
                    "arm_velocity_limit_rad_s": 3,
                    "finger_effort_limit_n": 20,
                    "finger_velocity_limit_m_s": 0.15,
                    "collision_geometry": "convex decomposition generated from visual meshes",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def convert_urdf(urdf_path: Path, destination: Path) -> Path:
    """Convert a prepared model after Arena has launched Isaac Sim."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(destination),
            usd_file_name="yam_station.usd",
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            collision_from_visuals=True,
            collision_type="Convex Decomposition",
            self_collision=False,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                target_type="none",
            ),
        )
    )
    return Path(converter.usd_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("~/.cache/isaaclab_arena/yam"))
    parser.add_argument(
        "--download-only", action="store_true", help="Prepare the 8 MB URDF/meshes without starting Isaac Sim"
    )
    arguments = parser.parse_args()
    destination = arguments.output.expanduser().resolve()
    source_path = fetch_assets(destination)
    urdf_path = prepare_urdf(source_path, destination)
    print(f"Prepared URDF: {urdf_path}")
    if arguments.download_only:
        return
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(argparse.Namespace(headless=True, enable_cameras=False, device="cuda:0")):
        usd_path = convert_urdf(urdf_path, destination)
        print(f"Prepared USD: {usd_path}")


if __name__ == "__main__":
    main()
