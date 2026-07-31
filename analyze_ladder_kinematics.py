"""Extract the ladder-only chain from URDF into ladder_kinematics.yaml."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def vector(element, attribute, default):
    return [float(value) for value in element.attrib.get(attribute, default).split()]


def analyze(urdf_path):
    root = ET.parse(urdf_path).getroot()
    by_name = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    joints = []
    expected_parent = "base_mobile"
    for index in range(1, 9):
        name = f"joint{index}_mobile"
        joint = by_name[name]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent != expected_parent:
            raise ValueError(f"Broken ladder chain at {name}: expected parent {expected_parent}, got {parent}")
        origin = joint.find("origin")
        joints.append({
            "name": name,
            "parent": parent,
            "child": child,
            "origin_xyz": vector(origin, "xyz", "0 0 0"),
            "origin_rpy": vector(origin, "rpy", "0 0 0"),
            "axis": vector(joint.find("axis"), "xyz", "1 0 0"),
        })
        expected_parent = child
    if expected_parent != "body8_mobile":
        raise ValueError(f"Expected body8_mobile end link, got {expected_parent}")
    return {
        "format_version": 1,
        "source_urdf": str(urdf_path),
        "chain": {"base_link": "base_mobile", "end_link": "body8_mobile", "joints": joints},
        "control": {
            "translation_speed_mps": 0.16,
            "turn_speed_rps": 0.22,
            "joint2_lift_assist_rad_per_m": 2.0,
            "joint2_forward_assist_rad_per_m": 0.8,
            "orientation_hold_gain": 0.35,
            "orientation_task_weight": 1.5,
            "joint_motion_cost": [1.0, 0.08, 1.0, 1.0, 0.55, 1.0, 1.0, 1.0],
            "damped_least_squares": 0.12,
            "max_joint_delta_rad_per_frame": 0.004,
            "target_smoothing_time_s": 0.18,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default="humanoid_urdf_assemble/urdf/combined_mobile_v4_onlyarm_wheels.urdf")
    parser.add_argument("--output", default="ladder_kinematics.yaml")
    args = parser.parse_args()
    result = analyze(Path(args.urdf))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(result['chain']['joints'])}-joint ladder chain to {args.output}")


if __name__ == "__main__":
    main()
