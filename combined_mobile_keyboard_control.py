"""
Reusable Isaac Sim keyboard task-space controller for combined mobile dual-arm robots.

The module loads one of the wheel-equipped mobile_bucket USD stages and controls
the two end effectors with Lula IK, keyboard input, and optional FOB 6D pose input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Isaac Kit parses sys.argv while SimulationApp is constructed. Keep controller
# arguments separately and hide our short -m option from Kit's own parser.
CONTROLLER_CLI_ARGS = sys.argv[1:].copy()
sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg not in ("-m", "--multi")]

from isaacsim import SimulationApp

# "default" is the Kit viewport token for the Grey Studio lighting preset.
VIEWPORT_LIGHTING_MODE = "default"
simulation_app = SimulationApp({
    "headless": False,
    "extra_args": [
        f"--/exts/omni.kit.viewport.menubar.render/lightingMode={VIEWPORT_LIGHTING_MODE}",
    ],
})

import carb
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.isaac.core.utils.types import ArticulationAction

try:
    from isaacsim.core.api.articulations import Articulation
except Exception:
    from omni.isaac.core.articulations import Articulation

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from painting_simulation import PaintSprayVisualizer
from control_status_ui import ControlHelpWindow, ControlStatusWindow
from ladder_kinematics import LadderKinematics
from target_area_selector import PaintTargetAreaSelector
from FOB.arduino_switch_reader import ArduinoSwitchSource
from FOB.fob_control_bridge import FOBControlBridge

# Keep the viewport preset deterministic even when a persistent user preference
# was saved by a previous Isaac Sim session.
carb.settings.get_settings().set(
    "/exts/omni.kit.viewport.menubar.render/lightingMode",
    VIEWPORT_LIGHTING_MODE,
)

EE_SPEED_MPS = 0.10
ROT_SPEED_RPS = 0.8
MOBILE_JOINT_SPEED_RPS = 0.35
LADDER_KINEMATICS_PATH = "./ladder_kinematics.yaml"
LADDER_TASK_JOINT_SPEED_RPS = 0.18
KEYBOARD_BASE_AXIS_SIGN = np.array([-1.0, 1.0, 1.0], dtype=np.float64)
CONTROL_FRAME = "base"  # "base" or "tool"
# Stronger, less overdamped arm drives keep the links on their IK targets.
KP = 2500.0
KD = 800.0
ARM_JOINT_MAX_FORCE = 500.0
HOLD_SECONDS = 1.0
FORCE_BASE_WXYZ_SWAP = False  # True
DISABLE_GRAVITY = True

USE_AXIS_AUTO_ALIGN = False  # True
ALIGN_PROBE_DIST = 0.035
ALIGN_SETTLE_STEPS = 70
ALIGN_RESTORE_STEPS = 70
KINEMATICS_TUNING_PROFILES = [
    {"name": "default", "probe_dist": 0.035, "settle_steps": 70, "restore_steps": 70, "ik_iters": 80},
    {"name": "long_settle", "probe_dist": 0.035, "settle_steps": 140, "restore_steps": 100, "ik_iters": 120},
    {"name": "larger_probe", "probe_dist": 0.050, "settle_steps": 180, "restore_steps": 120, "ik_iters": 160},
    {"name": "small_probe", "probe_dist": 0.020, "settle_steps": 180, "restore_steps": 120, "ik_iters": 160},
]
MIN_ALIGN_TO_ENABLE = 0.35
MIN_RESPONSE_NORM = 0.0015
MAX_DELTA_PER_STEP = 0.2
MAX_TARGET_OFFSET_FROM_START = np.array([0.60, 0.60, 0.60], dtype=np.float64)
MOBILE_JOINT_LOWER_RAD = -np.pi
MOBILE_JOINT_UPPER_RAD = np.pi
# Keep joint1 gentle while giving joints2-8 enough authority to move either payload.
# These values are mirrored into the URDF limits and the loaded USD drives.
MOBILE_JOINT_MAX_FORCE = 20000.0
MOBILE_JOINT_STIFFNESS = 24000.0
MOBILE_JOINT_DAMPING = 18000.0
MOBILE_JOINT1_MAX_FORCE = 6000.0
MOBILE_LIFT_JOINT_MAX_FORCE = 20000.0
MOBILE_JOINT1_STIFFNESS = 2500.0
MOBILE_JOINT1_DAMPING = 3500.0
EE_TRAIL_MIN_DISTANCE_M = 0.005
EE_TRAIL_MAX_POINTS = 2000
EE_TRAIL_WIDTH_M = 0.008
EE_TRAIL_OPACITY = 0.45
EE_TRAIL_COLOR = (0.1, 1.0, 0.2)
MOBILE_JOINT_COMMAND_SCALE = np.array([0.35, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
# Ladder task-space cooperative profiles, ordered joint1_mobile..joint8_mobile.
# A value of 0.25 represents the user's "slight" compensating motion.
LADDER_TASK_JOINT_PROFILES = np.array([
    [0.0, -1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.25, 1.0, 1.0, -0.25, 0.0, 0.0, 0.0],
], dtype=np.float64)
MOBILE_LIFT_COMMAND_PROFILE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
MOBILE_JOINT_NAMES = [f"joint{i}_mobile" for i in range(1, 9)]
MOBILE_PRIMARY_Z_JOINT_NAMES = ["joint3_mobile", "joint4_mobile", "joint5_mobile", "joint6_mobile"]
MOBILE_PRIMARY_LIFT_JOINT_NAMES = ["joint1_mobile", "joint2_mobile", "joint7_mobile", "joint8_mobile"]
MOBILE_INITIAL_JOINT_POSITIONS = {}
MOBILE_JOINT_KEY_BINDINGS = [
    ("Q", "A"),
    ("W", "S"),
    ("E", "D"),
    ("R", "F"),
    ("T", "G"),
    ("Y", "H"),
    ("U", "J"),
    ("I", "K"),
]

WHEEL_JOINT_NAMES = [
    "front_left_wheel_joint", "front_right_wheel_joint",
    "rear_left_wheel_joint", "rear_right_wheel_joint",
]
WHEEL_SPEED_RPS = 4.0
WHEEL_DRIVE_DAMPING = 200.0
WHEEL_MAX_FORCE = 4000.0


ROBOT_CONFIGS = {
    "humanoid_base": {
        "stage_path": "./mobile_bucket_humanoid.usd",
        "urdf_path": "./humanoid_urdf_assemble/urdf/combined_mobile_humanoid_wheels.urdf",
        "robot_prim_path": "/World/combined_mobile_humanoid_base",
        "articulation_root_link": "base_mobile",
        "kinematics_info_path": "./humanoid_kinematics_info.json",
        "right_desc_yaml": "./combined_mobile_humanoid_base_right_arm_robot_descriptor.yaml",
        "left_desc_yaml": "./combined_mobile_humanoid_base_left_arm_robot_descriptor.yaml",
        "right_ee_frame": "link6_R",
        "left_ee_frame": "link6_L",
        "arm_kp": 1800.0,
        "arm_kd": 750.0,
        "arm_max_force": 250.0,
        "upper_body_link_names": ["base_link"] + [f"link{i}_{side}" for side in ("L", "R") for i in range(1, 7)],
        "upper_body_target_mass": 39.0,
        "initial_joint_positions": {
            "joint2_R": -0.35, "joint4_R": 0.70,
            "joint2_L": 0.35, "joint4_L": 0.70,
        },
        "arm_input_axis_sign": {
            "right": [1.0, -1.0, 1.0],
            "left": [1.0, 1.0, 1.0],
        },
    },
    "v4_onlyarm": {
        "stage_path": "./mobile_bucket_onlyarm.usd",
        "urdf_path": "./humanoid_urdf_assemble/urdf/combined_mobile_v4_onlyarm_wheels.urdf",
        "robot_prim_path": "/World/combined_mobile_v4_onlyarm",
        "articulation_root_link": "base_mobile",
        "kinematics_info_path": "./onlyarm_kinematics_info.json",
        "right_desc_yaml": "./combined_mobile_v4_onlyarm_right_arm_robot_descriptor.yaml",
        "left_desc_yaml": "./combined_mobile_v4_onlyarm_left_arm_robot_descriptor.yaml",
        "right_ee_frame": "link7",
        "left_ee_frame": "link14",
        "arm_kp": 3000.0,
        "arm_kd": 1200.0,
        "arm_max_force": 500.0,
        "paint_nozzle_axis": (0.0, 0.0, 1.0),
        "upper_body_link_names": ["base"] + [f"link{i}" for i in range(1, 15)],
        "upper_body_target_mass": 39.0,
    },
}


MULTI_STAGE_PATH = "./mobile_bucket_multi-robot.usd"
MULTI_ROBOT_SPECS = {
    1: {"name": "combined_mobile_humanoid", "root_scope": "/World/combined_mobile_humanoid", "robot_key": "humanoid_base", "full_control": True},
    2: {"name": "combined_mobile_v4_onlyarm_wheels", "root_scope": "/World/combined_mobile_v4_onlyarm_wheels", "robot_key": "v4_onlyarm", "full_control": True},
    3: {"name": "only_mobile", "root_scope": "/World/only_mobile", "robot_key": None, "full_control": False},
}


def fmt(a):
    return np.array2string(np.asarray(a), precision=5, suppress_small=True)

def log_info(*_args, **_kwargs):
    """Keep routine startup information out of the CLI; warnings remain visible."""
    return None



def vec_to_list(a):
    return np.asarray(a, dtype=np.float64).reshape(-1).tolist()


def load_kinematics_info(path: str) -> Optional[dict]:
    info_path = Path(path)
    if not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load kinematics info {path}: {exc}")
        return None


def save_kinematics_info(path: str, info: dict) -> None:
    info_path = Path(path)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
        f.write("\n")
    log_info(f"[INFO] Kinematics info saved: {info_path}")


def load_mobile_calibration(info: Optional[dict], mobile_joint_count: int):
    signs = np.ones(mobile_joint_count, dtype=np.float64)
    scales = np.ones(mobile_joint_count, dtype=np.float64)
    enabled = np.ones(mobile_joint_count, dtype=bool)
    if not info:
        return signs, scales, enabled, False
    calib = info.get("mobile_joints", {}).get("calibration")
    if not calib:
        return signs, scales, enabled, False
    saved_signs = np.asarray(calib.get("joint_sign", []), dtype=np.float64)
    saved_scales = np.asarray(calib.get("joint_scale", []), dtype=np.float64)
    saved_enabled = np.asarray(calib.get("joint_enabled", []), dtype=bool)
    signs[: min(mobile_joint_count, saved_signs.size)] = saved_signs[:mobile_joint_count]
    scales[: min(mobile_joint_count, saved_scales.size)] = saved_scales[:mobile_joint_count]
    enabled[: min(mobile_joint_count, saved_enabled.size)] = saved_enabled[:mobile_joint_count]
    log_info(f"[INFO] Loaded saved mobile calibration: sign={fmt(signs)} scale={fmt(scales)} enabled={enabled}")
    return signs, scales, enabled, True


def apply_saved_kinematics_to_arms(arms: dict, info: Optional[dict]) -> bool:
    if not info:
        return False
    arm_info = info.get("arms", {})
    applied = False
    for arm_name, arm in arms.items():
        calib = arm_info.get(arm_name, {}).get("axis_calibration")
        if not calib:
            continue
        probes = calib.get("probes", [])
        if probes and any(not bool(probe.get("ik_ok", False)) for probe in probes):
            print(f"[WARN] Rejected saved {arm_name} calibration because at least one axis probe failed IK")
            continue
        saved_response = np.asarray(calib.get("axis_response", []), dtype=np.float64)
        if saved_response.shape == (3, 3):
            response_per_meter = saved_response / max(float(calib.get("probe_distance_m", ALIGN_PROBE_DIST)), 1e-9)
            if np.linalg.matrix_rank(response_per_meter) < 3 or np.linalg.cond(response_per_meter) > 50.0:
                print(f"[WARN] Rejected saved {arm_name} calibration because its response matrix is singular or ill-conditioned")
                continue

        arm.axis_sign = np.asarray(calib.get("axis_sign", arm.axis_sign), dtype=np.float64)
        arm.axis_scale = np.asarray(calib.get("axis_scale", arm.axis_scale), dtype=np.float64)
        arm.axis_enabled = np.asarray(calib.get("axis_enabled", arm.axis_enabled), dtype=bool)
        if "axis_response" in calib:
            arm.axis_response = np.asarray(calib["axis_response"], dtype=np.float64)
        if "axis_quality" in calib:
            arm.axis_quality = np.asarray(calib["axis_quality"], dtype=np.float64)
        if "probe_distance_m" in calib:
            arm.axis_probe_dist = float(calib["probe_distance_m"])
        applied = True
        log_info(f"[INFO] Loaded saved kinematics calibration for {arm_name}: sign={fmt(arm.axis_sign)} scale={fmt(arm.axis_scale)} enabled={arm.axis_enabled}")
    return applied




def as_wxyz_quat(rot):
    r = np.array(rot, dtype=np.float64)
    if r.shape == (4,):
        return r
    if r.shape == (3,):
        roll, pitch, yaw = r
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float64)
    if r.shape == (3, 3):
        m = r
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
        return q / (np.linalg.norm(q) + 1e-12)
    raise ValueError(f"Unsupported rotation shape: {r.shape}")


def to_wxyz_from_xyzw(q):
    q = np.array(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_norm(q):
    return q / (np.linalg.norm(q) + 1e-12)


def quat_from_axis_angle(axis, angle):
    axis = np.array(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle * 0.5
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < 1e-12 else v / n


def ladder_task_joint_delta(task_direction, step_rad, joint_count=8):
    """Map an XYZ ladder command to coordinated mobile-joint increments."""
    direction = np.asarray(task_direction, dtype=np.float64).reshape(3)
    count = min(int(joint_count), LADDER_TASK_JOINT_PROFILES.shape[1])
    return float(step_rad) * (direction @ LADDER_TASK_JOINT_PROFILES[:, :count])


def alignment_score(actual_delta, desired_delta):
    au, du = unit(actual_delta), unit(desired_delta)
    if np.linalg.norm(au) < 1e-12 or np.linalg.norm(du) < 1e-12:
        return 0.0
    return float(np.dot(au, du))


def clamp_vec(v, max_abs):
    return np.clip(np.asarray(v, dtype=np.float64).copy(), -max_abs, max_abs)



def get_dof_names(robot) -> list[str]:
    if hasattr(robot, "dof_names"):
        return list(robot.dof_names)
    if hasattr(robot, "get_dof_names"):
        return list(robot.get_dof_names())
    return []


def resolve_mobile_joint_indices(robot) -> list[int]:
    dof_names = get_dof_names(robot)
    indices = []
    for name in MOBILE_JOINT_NAMES:
        idx = None
        if hasattr(robot, "get_dof_index"):
            try:
                idx = int(robot.get_dof_index(name))
            except Exception:
                idx = None
        if idx is None and name in dof_names:
            idx = dof_names.index(name)
        if idx is None:
            print(f"[WARN] Mobile joint DOF not found in articulation: {name}")
            continue
        indices.append(idx)
    return indices


def resolve_wheel_joint_indices(robot) -> list[int]:
    dof_names = get_dof_names(robot)
    indices = []
    for name in WHEEL_JOINT_NAMES:
        if name not in dof_names:
            print(f"[WARN] Wheel joint DOF not found in articulation: {name}")
            continue
        indices.append(dof_names.index(name))
    return indices


def make_wheel_drive_action(wheel_joint_indices, forward: float, turn: float):
    # Joint order: front-left, front-right, rear-left, rear-right.
    left = np.clip(forward - turn, -1.0, 1.0) * WHEEL_SPEED_RPS
    right = np.clip(forward + turn, -1.0, 1.0) * WHEEL_SPEED_RPS
    velocities = np.asarray([left, right, left, right], dtype=np.float64)
    return ArticulationAction(
        joint_velocities=velocities,
        joint_indices=np.asarray(wheel_joint_indices, dtype=np.int32),
    )


def mobile_primary_lift_joint_indices(mobile_joint_indices):
    indices = []
    for name in MOBILE_PRIMARY_LIFT_JOINT_NAMES:
        if name in MOBILE_JOINT_NAMES:
            local_idx = MOBILE_JOINT_NAMES.index(name)
            if local_idx < len(mobile_joint_indices):
                indices.append(local_idx)
    return indices


def apply_initial_mobile_pose(robot, controller, mobile_joint_indices, world, steps: int = 90):
    if not mobile_joint_indices or not MOBILE_INITIAL_JOINT_POSITIONS:
        return
    current = robot.get_joint_positions()[mobile_joint_indices].copy()
    for name, value in MOBILE_INITIAL_JOINT_POSITIONS.items():
        if name in MOBILE_JOINT_NAMES:
            local_idx = MOBILE_JOINT_NAMES.index(name)
            if local_idx < len(current):
                current[local_idx] = value
    action = make_mobile_joint_action(mobile_joint_indices, current)
    for _ in range(steps):
        safe_apply_action(controller, action)
        world.step(render=False)
    actual = robot.get_joint_positions()[mobile_joint_indices]
    log_info(f"[INFO] Initial mobile ladder pose commanded: requested={MOBILE_INITIAL_JOINT_POSITIONS}, "
          f"target={fmt(current)}, actual={fmt(actual)}")


def apply_initial_joint_pose(robot, controller, world, positions: dict, steps: int = 120):
    if not positions:
        return
    dof_names = get_dof_names(robot)
    indices, targets = [], []
    for name, value in positions.items():
        if name not in dof_names:
            print(f"[WARN] Initial-pose joint not found: {name}")
            continue
        index = dof_names.index(name)
        indices.append(index)
        targets.append(float(value))
    if not indices:
        return
    indices = np.asarray(indices, dtype=np.int32)
    start = robot.get_joint_positions()[indices].copy()
    target = np.asarray(targets, dtype=np.float64)
    for step in range(1, steps + 1):
        alpha = step / steps
        blend = alpha * alpha * (3.0 - 2.0 * alpha)
        command = start + (target - start) * blend
        action = ArticulationAction(joint_positions=command, joint_velocities=np.zeros_like(command), joint_indices=indices)
        safe_apply_action(controller, action)
        world.step(render=False)
    actual = robot.get_joint_positions()[indices]
    log_info(f"[INFO] Initial arm pose applied: target={fmt(target)}, actual={fmt(actual)}")

def make_mobile_joint_action(mobile_joint_indices, mobile_joint_targets):
    return ArticulationAction(
        joint_positions=np.asarray(mobile_joint_targets, dtype=np.float64),
        joint_velocities=np.zeros(len(mobile_joint_indices), dtype=np.float64),
        joint_indices=np.asarray(mobile_joint_indices, dtype=np.int32),
    )

def set_ik_iters(kin, iters: int):
    if hasattr(kin, "set_max_iterations"):
        kin.set_max_iterations(iters)
    elif hasattr(kin, "ccd_max_iterations"):
        kin.ccd_max_iterations = iters


def iter_world_prims(stage):
    from pxr import Usd

    world = stage.GetPrimAtPath("/World")
    search_root = world if world and world.IsValid() else stage.GetPseudoRoot()
    return Usd.PrimRange(search_root)


def find_prim_by_name(stage, prim_name: str):
    matches = [prim for prim in iter_world_prims(stage) if prim.GetName() == prim_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"Multiple prims named '{prim_name}' found: {paths}")
    raise RuntimeError(f"Prim named '{prim_name}' not found in opened USD stage")


def configure_upper_body_masses(stage, cfg) -> None:
    """Scale the loaded USD upper body to the same mass target as the URDF."""
    from pxr import Gf, UsdPhysics

    link_names = cfg.get("upper_body_link_names", [])
    target_mass = float(cfg.get("upper_body_target_mass", 0.0))
    if not link_names or target_mass <= 0.0:
        return

    entries = []
    for link_name in link_names:
        prim = find_prim_by_name(stage, link_name)
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass = mass_api.GetMassAttr().Get()
        if mass is None or float(mass) <= 0.0:
            raise RuntimeError(f"Missing positive authored mass for upper-body link {prim.GetPath()}")
        entries.append((link_name, mass_api, float(mass)))

    authored_total = sum(entry[2] for entry in entries)
    scale = target_mass / authored_total
    for _, mass_api, mass in entries:
        mass_api.CreateMassAttr(mass * scale).Set(mass * scale)
        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        inertia = inertia_attr.Get()
        if inertia is not None:
            scaled = Gf.Vec3f(
                float(inertia[0]) * scale,
                float(inertia[1]) * scale,
                float(inertia[2]) * scale,
            )
            mass_api.CreateDiagonalInertiaAttr(scaled).Set(scaled)
    log_info(
        f"[INFO] Upper-body mass configured: links={len(entries)}, "
        f"authored_total={authored_total:.3f}kg -> target={target_mass:.3f}kg"
    )

def configure_mobile_joint_usd_limits(stage) -> None:
    """Relax mobile joint limits/drives in the loaded USD so targets can move immediately."""
    from pxr import UsdPhysics

    configured = 0
    for prim in iter_world_prims(stage):
        if prim.GetName() not in MOBILE_JOINT_NAMES:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            print(f"[WARN] Mobile joint prim is not a UsdPhysics.RevoluteJoint; limit update skipped: {prim.GetPath()}")
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        # USD RevoluteJoint limits use degrees; ArticulationAction uses radians.
        # Writing +/-pi here used to restrict the ladder to only +/-3.14 degrees.
        lower_deg = float(np.rad2deg(MOBILE_JOINT_LOWER_RAD))
        upper_deg = float(np.rad2deg(MOBILE_JOINT_UPPER_RAD))
        joint.CreateLowerLimitAttr(lower_deg).Set(lower_deg)
        joint.CreateUpperLimitAttr(upper_deg).Set(upper_deg)
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            joint_name = prim.GetName()
            if joint_name == "joint1_mobile":
                stiffness = MOBILE_JOINT1_STIFFNESS
                damping = MOBILE_JOINT1_DAMPING
                max_force = MOBILE_JOINT1_MAX_FORCE
            else:
                stiffness = MOBILE_JOINT_STIFFNESS
                damping = MOBILE_JOINT_DAMPING
                max_force = MOBILE_LIFT_JOINT_MAX_FORCE if joint_name in MOBILE_PRIMARY_LIFT_JOINT_NAMES else MOBILE_JOINT_MAX_FORCE
            drive.CreateStiffnessAttr(stiffness).Set(stiffness)
            drive.CreateDampingAttr(damping).Set(damping)
            drive.CreateMaxForceAttr(max_force).Set(max_force)
        except Exception as exc:
            print(f"[WARN] Could not configure angular drive force for {prim.GetPath()}: {exc}")
        configured += 1

    if configured:
        log_info(
            f"[INFO] Mobile USD joint limits configured for {configured} joints: "
            f"lower={lower_deg:.1f}deg, upper={upper_deg:.1f}deg, "
            f"stiffness={MOBILE_JOINT_STIFFNESS:.1f}, damping={MOBILE_JOINT_DAMPING:.1f}, max_force={MOBILE_JOINT_MAX_FORCE:.1f}"
        )
    else:
        print("[WARN] No mobile joint prims found while configuring USD joint limits")


def configure_arm_joint_usd_drives(stage, cfg) -> None:
    """Author arm drives before reset so gravity cannot drop unpowered links."""
    from pxr import UsdPhysics

    arm_kp = float(cfg.get("arm_kp", KP))
    arm_kd = float(cfg.get("arm_kd", KD))
    arm_max_force = float(cfg.get("arm_max_force", ARM_JOINT_MAX_FORCE))
    configured = 0
    for prim in iter_world_prims(stage):
        if prim.GetName() in MOBILE_JOINT_NAMES or prim.GetName() in WHEEL_JOINT_NAMES:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr(arm_kp).Set(arm_kp)
            drive.CreateDampingAttr(arm_kd).Set(arm_kd)
            drive.CreateMaxForceAttr(arm_max_force).Set(arm_max_force)
            configured += 1
        except Exception as exc:
            print(f"[WARN] Could not configure arm drive for {prim.GetPath()}: {exc}")
    log_info(f"[INFO] Arm USD drives configured before gravity: joints={configured}, kp={arm_kp}, kd={arm_kd}, max_force={arm_max_force}")

def configure_wheel_joint_usd_drives(stage) -> None:
    """Configure wheel joints as velocity drives without position locking."""
    from pxr import UsdPhysics

    configured = 0
    for prim in iter_world_prims(stage):
        if prim.GetName() not in WHEEL_JOINT_NAMES:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            print(f"[WARN] Wheel joint is not revolute: {prim.GetPath()}")
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(0.0).Set(0.0)
        drive.CreateDampingAttr(WHEEL_DRIVE_DAMPING).Set(WHEEL_DRIVE_DAMPING)
        drive.CreateMaxForceAttr(WHEEL_MAX_FORCE).Set(WHEEL_MAX_FORCE)
        configured += 1
    if configured:
        log_info(f"[INFO] Wheel velocity drives configured: joints={configured}, speed={WHEEL_SPEED_RPS:.1f} rad_per_s")


def select_articulation_root_path(stage, preferred_root_link: str) -> str:
    """Select an existing ArticulationRootAPI prim without mutating the authored USD hierarchy."""
    from pxr import UsdPhysics

    articulation_paths = []
    preferred_path = None
    for prim in iter_world_prims(stage):
        if prim.GetName() == preferred_root_link:
            preferred_path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_paths.append(str(prim.GetPath()))

    if preferred_path in articulation_paths:
        if len(articulation_paths) > 1:
            print(
                f"[WARN] Multiple articulation roots are authored in the stage: {articulation_paths}. "
                f"Using preferred existing root: {preferred_path}"
            )
        else:
            log_info(f"[INFO] Articulation root active: {preferred_path}")
        return preferred_path

    if len(articulation_paths) == 1:
        print(
            f"[WARN] Preferred articulation root link '{preferred_root_link}' is not authored as an ArticulationRootAPI. "
            f"Using existing root: {articulation_paths[0]}"
        )
        return articulation_paths[0]

    if len(articulation_paths) > 1:
        raise RuntimeError(
            f"Multiple articulation roots are authored in the stage and preferred root '{preferred_root_link}' "
            f"is not one of them: {articulation_paths}"
        )

    raise RuntimeError(
        f"No ArticulationRootAPI found in the opened USD stage. "
        f"Author exactly one articulation root in USD, preferably on '{preferred_root_link}'."
    )


def find_articulation_prim_path(stage, preferred_path: str) -> str:
    """Return preferred_path when valid, otherwise find one articulation in the opened USD stage."""
    preferred = stage.GetPrimAtPath(preferred_path)
    if preferred and preferred.IsValid():
        return preferred_path

    from pxr import UsdPhysics

    articulation_paths = []
    for prim in iter_world_prims(stage):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_paths.append(str(prim.GetPath()))

    if len(articulation_paths) == 1:
        print(f"[WARN] Preferred robot prim not found: {preferred_path}. Using articulation root: {articulation_paths[0]}")
        return articulation_paths[0]
    if len(articulation_paths) > 1:
        raise RuntimeError(
            f"Preferred robot prim not found: {preferred_path}. "
            f"Multiple articulation roots found; set ROBOT_CONFIGS robot_prim_path explicitly: {articulation_paths}"
        )
    raise RuntimeError(f"Robot prim path not found and no articulation root discovered in stage: {preferred_path}")


def configure_robot_gravity(stage_or_world, robot_prim_path: str, disable: bool) -> None:
    """Explicitly author gravity state on every rigid body in the articulation."""
    from pxr import Usd, UsdPhysics, PhysxSchema

    stage = stage_or_world.stage if hasattr(stage_or_world, "stage") else stage_or_world
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Robot prim path not found: {robot_prim_path}")

    rigid_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        physx_rigid = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_rigid:
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid.CreateDisableGravityAttr(disable).Set(disable)
        rigid_count += 1

    if rigid_count == 0:
        print(f"[WARN] No rigid bodies found under {robot_prim_path}; gravity configuration skipped")
        return
    log_info(f"[INFO] Gravity {'disabled' if disable else 'enabled'} for {rigid_count} rigid bodies under {robot_prim_path}")


def articulation_is_ready(robot) -> bool:
    if hasattr(robot, "is_initialized"):
        try:
            if not robot.is_initialized():
                return False
        except Exception:
            pass
    try:
        joint_positions = robot.get_joint_positions()
    except Exception:
        return False
    return joint_positions is not None


def wait_for_articulation_ready(world: World, robot, max_steps: int = 180) -> None:
    for _ in range(max_steps):
        if articulation_is_ready(robot):
            return
        if hasattr(robot, "initialize"):
            try:
                robot.initialize()
            except Exception:
                pass
        world.step(render=False)
    raise RuntimeError("Robot articulation did not initialize; check the USD ArticulationRootAPI and robot prim path.")


def safe_apply_action(controller, action, fallback_action=None) -> bool:
    selected_action = action if action is not None else fallback_action
    if selected_action is None:
        return False
    if selected_action.joint_positions is None and selected_action.joint_velocities is None:
        return False
    controller.apply_action(selected_action)
    return True


@dataclass
class ArmState:
    name: str
    kin: object
    ik: object
    frame_name: str
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    target_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    start_target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    axis_align_enabled: bool = USE_AXIS_AUTO_ALIGN
    axis_sign: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    axis_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    axis_enabled: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=bool))
    axis_quality: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    axis_response: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    axis_probe_dist: float = ALIGN_PROBE_DIST

def main(robot_key: str) -> None:
    global CONTROL_FRAME
    cfg = dict(ROBOT_CONFIGS[robot_key])
    wheels = True

    open_stage(cfg["stage_path"])
    stage = get_current_stage()
    robot_prim_path = select_articulation_root_path(stage, cfg["articulation_root_link"])
    configure_upper_body_masses(stage, cfg)
    configure_mobile_joint_usd_limits(stage)
    configure_arm_joint_usd_drives(stage, cfg)
    if wheels:
        configure_wheel_joint_usd_drives(stage)
    configure_robot_gravity(stage, robot_prim_path, disable=False if wheels else DISABLE_GRAVITY)

    world = World()
    robot = Articulation(robot_prim_path)
    world.scene.add(robot)
    world.reset()
    wait_for_articulation_ready(world, robot)

    controller = robot.get_articulation_controller()
    initial_mobile_joint_indices = resolve_mobile_joint_indices(robot)
    initial_wheel_joint_indices = resolve_wheel_joint_indices(robot) if wheels else []
    arm_kp = float(cfg.get("arm_kp", KP))
    arm_kd = float(cfg.get("arm_kd", KD))
    arm_max_force = float(cfg.get("arm_max_force", ARM_JOINT_MAX_FORCE))
    if hasattr(robot, "set_max_efforts"):
        try:
            max_efforts = np.ones(robot.num_dof) * arm_max_force
            max_efforts[initial_mobile_joint_indices] = MOBILE_JOINT_MAX_FORCE
            if initial_wheel_joint_indices:
                max_efforts[initial_wheel_joint_indices] = WHEEL_MAX_FORCE
            joint1_mobile_index = initial_mobile_joint_indices[MOBILE_JOINT_NAMES.index("joint1_mobile")]
            for joint_name in MOBILE_PRIMARY_LIFT_JOINT_NAMES:
                if joint_name == "joint1_mobile":
                    continue
                local_index = MOBILE_JOINT_NAMES.index(joint_name)
            max_efforts[joint1_mobile_index] = MOBILE_JOINT1_MAX_FORCE
            robot.set_max_efforts(max_efforts)
        except Exception as exc:
            print(f"[WARN] robot.set_max_efforts skipped: {exc}")
    if hasattr(controller, "set_gains"):
        try:
            stiffness = np.ones(robot.num_dof) * arm_kp
            damping = np.ones(robot.num_dof) * arm_kd
            stiffness[initial_mobile_joint_indices] = MOBILE_JOINT_STIFFNESS
            damping[initial_mobile_joint_indices] = MOBILE_JOINT_DAMPING
            if initial_wheel_joint_indices:
                stiffness[initial_wheel_joint_indices] = 0.0
                damping[initial_wheel_joint_indices] = WHEEL_DRIVE_DAMPING
            joint1_mobile_index = initial_mobile_joint_indices[MOBILE_JOINT_NAMES.index("joint1_mobile")]
            stiffness[joint1_mobile_index] = MOBILE_JOINT1_STIFFNESS
            damping[joint1_mobile_index] = MOBILE_JOINT1_DAMPING

            controller.set_gains(stiffness, damping)
        except Exception as exc:
            print(f"[WARN] controller.set_gains skipped because the articulation physics view is not ready: {exc}")

    apply_initial_joint_pose(robot, controller, world, cfg.get("initial_joint_positions", {}))
    apply_initial_mobile_pose(robot, controller, initial_mobile_joint_indices, world)

    jp0 = robot.get_joint_positions()
    if jp0 is None:
        raise RuntimeError("Robot articulation initialized but returned no joint positions.")
    hold_action = ArticulationAction(joint_positions=jp0, joint_velocities=np.zeros_like(jp0))
    dt = 1.0 / 60.0
    for _ in range(int(max(0.0, HOLD_SECONDS) / dt)):
        safe_apply_action(controller, hold_action)
        world.step(render=False)
    settled_q = robot.get_joint_positions()
    gravity_error = np.abs(settled_q - jp0)
    if initial_wheel_joint_indices:
        gravity_error[initial_wheel_joint_indices] = 0.0
    worst_index = int(np.argmax(gravity_error))
    worst_error_deg = float(np.rad2deg(gravity_error[worst_index]))
    if worst_error_deg > 5.0:
        dof_names = get_dof_names(robot)
        worst_name = dof_names[worst_index] if worst_index < len(dof_names) else str(worst_index)
        print(f"[WARN] Gravity hold error is {worst_error_deg:.2f}deg at {worst_name}; increase that joint's effort/gain or use a safer zero pose.")

    right_kin = LulaKinematicsSolver(robot_description_path=cfg["right_desc_yaml"], urdf_path=cfg["urdf_path"])
    left_kin = LulaKinematicsSolver(robot_description_path=cfg["left_desc_yaml"], urdf_path=cfg["urdf_path"])
    set_ik_iters(right_kin, 80)
    set_ik_iters(left_kin, 80)
    right_ik = ArticulationKinematicsSolver(robot, right_kin, cfg["right_ee_frame"])
    left_ik = ArticulationKinematicsSolver(robot, left_kin, cfg["left_ee_frame"])

    base_t, base_q_raw = robot.get_world_pose()
    base_q_use = to_wxyz_from_xyzw(base_q_raw) if FORCE_BASE_WXYZ_SWAP else np.array(base_q_raw, dtype=np.float64)
    right_kin.set_robot_base_pose(base_t, base_q_use)
    left_kin.set_robot_base_pose(base_t, base_q_use)

    arms = {
        "right": ArmState("right", right_kin, right_ik, cfg["right_ee_frame"]),
        "left": ArmState("left", left_kin, left_ik, cfg["left_ee_frame"]),
    }
    active_arm = "right"
    active_control_target = "right"  # "both"
    trail_enabled = False
    trail_points = {"right": [], "left": []}
    paint_enabled = {"right": False, "left": False}

    from pxr import Gf, Usd, UsdGeom

    UsdGeom.Scope.Define(stage, "/World/ArmEndEffectorTrails")
    ee_prims = {
        arm_name: find_prim_by_name(stage, cfg[f"{arm_name}_ee_frame"])
        for arm_name in arms
    }
    paint_visualizer = PaintSprayVisualizer(
        stage,
        robot_prim_path,
        nozzle_axis=cfg.get("paint_nozzle_axis", (0.0, 1.0, 0.0)),
    )
    status_ui = ControlStatusWindow(robot_key, cfg["stage_path"])
    occupied_slot = 1 if robot_key == "humanoid_base" else 2
    slot_names = {
        1: "humanoid" if occupied_slot == 1 else "EMPTY",
        2: "v4 onlyarm" if occupied_slot == 2 else "EMPTY",
        3: "EMPTY",
        4: "EMPTY",
        5: "EMPTY",
    }
    status_ui.set_slots(" | ".join(f"{slot}: {name}" for slot, name in slot_names.items()))
    help_ui = ControlHelpWindow()
    target_selector = PaintTargetAreaSelector(stage, status_ui.set_event)
    paint_visualizer.coverage_callback = target_selector.record_paint_patch
    paint_visualizer.coverage_clear_callback = target_selector.clear_coverage
    fob = FOBControlBridge()
    fob.start()
    status_ui.set_fob_status("connecting /dev/ttyUSB1")
    arduino_switch = ArduinoSwitchSource()
    arduino_switch.start()
    status_ui.set_arduino_status("connecting /dev/ttyUSB0 @ 9600")
    trail_curves = {}
    for arm_name in arms:
        curve = UsdGeom.BasisCurves.Define(
            stage, f"/World/ArmEndEffectorTrails/{arm_name}"
        )
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateWidthsAttr([EE_TRAIL_WIDTH_M])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
            [Gf.Vec3f(*EE_TRAIL_COLOR)]
        )
        curve.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set(
            [EE_TRAIL_OPACITY]
        )
        curve.CreatePointsAttr([])
        curve.CreateCurveVertexCountsAttr([])
        trail_curves[arm_name] = curve

    def end_effector_world_position(arm_name):
        matrix = UsdGeom.Xformable(ee_prims[arm_name]).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        t = matrix.ExtractTranslation()
        return np.asarray([t[0], t[1], t[2]], dtype=np.float64)

    def update_trail_curve(arm_name):
        points = trail_points[arm_name]
        trail_curves[arm_name].GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
        trail_curves[arm_name].GetCurveVertexCountsAttr().Set([len(points)] if points else [])
    control_mode = "arm"  # "arm", "mobile", "ladder_task", or "drive"
    wheel_joint_indices = resolve_wheel_joint_indices(robot) if wheels else []
    if wheels and len(wheel_joint_indices) != len(WHEEL_JOINT_NAMES):
        raise RuntimeError(f"Wheel mode requires {len(WHEEL_JOINT_NAMES)} wheel DOFs; found {len(wheel_joint_indices)}")
    mobile_joint_indices = resolve_mobile_joint_indices(robot)
    dof_names = get_dof_names(robot)
    arm_joint_indices = [
        i for i, name in enumerate(dof_names)
        if name not in MOBILE_JOINT_NAMES and name not in WHEEL_JOINT_NAMES
    ]
    primary_lift_indices = mobile_primary_lift_joint_indices(mobile_joint_indices)
    active_mobile_joint = primary_lift_indices[0] if primary_lift_indices else 0
    mobile_joint_targets = robot.get_joint_positions()[mobile_joint_indices].copy() if mobile_joint_indices else np.array([], dtype=np.float64)
    mobile_joint_sign = np.ones(len(mobile_joint_targets), dtype=np.float64)
    mobile_joint_scale = np.ones(len(mobile_joint_targets), dtype=np.float64)
    mobile_joint_enabled = np.ones(len(mobile_joint_targets), dtype=bool)
    ladder_kin = LadderKinematics(LADDER_KINEMATICS_PATH)
    help_ui.update_context(robot_key, control_mode, wheels=wheels)

    for arm in arms.values():
        pos, rot = arm.ik.compute_end_effector_pose()
        arm.target_pos = np.array(pos, dtype=np.float64)
        arm.target_quat = as_wxyz_quat(rot)
        arm.start_target_pos = arm.target_pos.copy()

    pressed = set()
    last_pressed = set()
    should_quit = False
    input_iface = carb.input.acquire_input_interface()
    try:
        import omni.appwindow
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    except Exception:
        keyboard = input_iface.get_keyboard() if hasattr(input_iface, "get_keyboard") else None
    if keyboard is None:
        raise RuntimeError("Keyboard device handle not found.")

    def on_keyboard_event(event, *args, **kwargs):
        nonlocal should_quit
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            pressed.add(event.input)
            if event.input == carb.input.KeyboardInput.ESCAPE:
                should_quit = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(event.input)


    def spray_requested():
        keyboard_spray = is_down(carb.input.KeyboardInput.X)
        return keyboard_spray or (control_mode == "fob" and arduino_switch.active)
    input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    def is_down(key):
        return key in pressed

    def update_control_status(mode, command="idle", ik_ok=None):
        status_ui.set_arduino_status(arduino_switch.status)
        if log_counter % 10 != 0:
            return
        positions = robot.get_joint_positions()
        velocities = robot.get_joint_velocities()
        if positions is None:
            return
        positions_deg = np.rad2deg(np.asarray(positions, dtype=np.float64))
        status_ui.set_fob_status(fob.status)

        def rows(indices, values, empty="not available"):
            if not indices:
                return empty
            return "\n".join(f"{dof_names[i]:<28} {values[i]:+9.2f}" for i in indices)

        target_text = "-"
        error_text = "-"
        if mode in ("ARM", "FOB"):
            arm = arms[active_arm]
            current, _ = arm.ik.compute_end_effector_pose()
            error = np.linalg.norm(np.asarray(current) - arm.target_pos)
            target_text = f"{active_control_target.upper()} / view {active_arm.upper()} / EE {fmt(arm.target_pos)}"
            error_text = f"EE position {error:.5f} m"
            if ik_ok is not None:
                error_text += f" / IK {'OK' if ik_ok else 'FAILED'}"
        elif mode == "MOBILE" and mobile_joint_indices:
            local_index = min(active_mobile_joint, len(mobile_joint_indices) - 1)
            dof_index = mobile_joint_indices[local_index]
            target_text = MOBILE_JOINT_NAMES[local_index]
            error_deg = np.rad2deg(mobile_joint_targets[local_index]) - positions_deg[dof_index]
            error_text = f"target {np.rad2deg(mobile_joint_targets[local_index]):+.2f} deg / error {error_deg:+.2f} deg"
        elif mode == "LADDER_TASK" and mobile_joint_indices:
            actual = np.asarray(positions)[mobile_joint_indices]
            target_text = "cooperative ladder joints"
            error_text = f"max joint error {np.max(np.abs(np.rad2deg(mobile_joint_targets - actual))):.2f} deg"
        elif mode == "DRIVE":
            target_text = "four-wheel drive"

        status_ui.update(
            mode=mode,
            target=target_text,
            error=error_text,
            command=command,
            paint=(f"{active_arm.upper()} rectangle={'ON' if paint_enabled[active_arm] else 'OFF'} / "
                   f"spray={'ON' if spray_requested() else 'OFF'} / "
                   f"{target_selector.coverage_status}"),
            arm_joints=rows(arm_joint_indices, positions_deg),
            ladder_joints=rows(mobile_joint_indices, positions_deg),
            wheel_joints=rows(wheel_joint_indices, np.asarray(velocities)) if velocities is not None else "not available",
        )

    def refresh_base_pose():
        base_t_now, base_q_now_raw = robot.get_world_pose()
        base_q_now = to_wxyz_from_xyzw(base_q_now_raw) if FORCE_BASE_WXYZ_SWAP else np.array(base_q_now_raw, dtype=np.float64)
        right_kin.set_robot_base_pose(base_t_now, base_q_now)
        left_kin.set_robot_base_pose(base_t_now, base_q_now)

    def step_to_target_for_arm(arm: ArmState, goal_pos, goal_quat, steps=ALIGN_SETTLE_STEPS):
        ok_last = False
        for _ in range(steps):
            refresh_base_pose()
            action, ok = arm.ik.compute_inverse_kinematics(goal_pos, goal_quat)
            ok_last = ok
            safe_apply_action(controller, action if ok else None, hold_action)
            world.step(render=False)
        return ok_last

    def probe_axiswise_alignment_for_arm(arm_name: str):
        arm = arms[arm_name]
        print(f"\n=== AXIS-WISE AUTO ALIGN PROBE ({arm.name.upper()} / {arm.frame_name}) ===")
        refresh_base_pose()
        start_pos, start_rot = arm.ik.compute_end_effector_pose()
        start_pos = np.array(start_pos, dtype=np.float64)
        start_quat = as_wxyz_quat(start_rot)
        signs = np.ones(3, dtype=np.float64)
        scales = np.ones(3, dtype=np.float64)
        enabled = np.ones(3, dtype=bool)
        responses = np.zeros((3, 3), dtype=np.float64)
        qualities = np.zeros(3, dtype=np.float64)
        for label, desired, idx in [
            ("X", np.array([ALIGN_PROBE_DIST, 0.0, 0.0]), 0),
            ("Y", np.array([0.0, ALIGN_PROBE_DIST, 0.0]), 1),
            ("Z", np.array([0.0, 0.0, ALIGN_PROBE_DIST]), 2),
        ]:
            ok = step_to_target_for_arm(arm, start_pos + desired, start_quat)
            cur_pos, _ = arm.ik.compute_end_effector_pose()
            actual = np.array(cur_pos, dtype=np.float64) - start_pos
            responses[:, idx] = actual
            comp = actual[idx]
            align = alignment_score(actual, desired)
            qualities[idx] = align
            signs[idx] = 1.0 if comp >= 0.0 else -1.0
            scales[idx] = min(3.0, ALIGN_PROBE_DIST / abs(comp)) if abs(comp) > 1e-9 else 1.0
            enabled[idx] = bool(ok) and abs(align) >= MIN_ALIGN_TO_ENABLE and np.linalg.norm(actual) >= MIN_RESPONSE_NORM
            print(f"[{arm.name.upper()}] axis {label}: desired={fmt(desired)} actual={fmt(actual)} align={align:+.4f} enabled={enabled[idx]} ok={ok}")
            step_to_target_for_arm(arm, start_pos, start_quat, steps=ALIGN_RESTORE_STEPS)
        arm.axis_sign = signs
        arm.axis_scale = scales
        arm.axis_enabled = enabled
        arm.axis_quality = qualities
        arm.axis_response = responses
        arm.axis_probe_dist = ALIGN_PROBE_DIST
        cur_pos, cur_rot = arm.ik.compute_end_effector_pose()
        arm.target_pos = np.array(cur_pos, dtype=np.float64)
        arm.target_quat = as_wxyz_quat(cur_rot)
        arm.start_target_pos = arm.target_pos.copy()
        print(f"[{arm.name.upper()}] axis_enabled={arm.axis_enabled} axis_sign={fmt(arm.axis_sign)} axis_scale={fmt(arm.axis_scale)}")

    def apply_axiswise_correction(arm: ArmState, desired_delta_base):
        response = np.asarray(arm.axis_response, dtype=np.float64)
        if response.shape == (3, 3) and arm.axis_probe_dist > 1e-9:
            response_per_meter = response / arm.axis_probe_dist
            full_rank = np.linalg.matrix_rank(response_per_meter) == 3
            well_conditioned = full_rank and np.linalg.cond(response_per_meter) <= 50.0
            if well_conditioned and np.all(arm.axis_enabled):
                corrected = np.linalg.pinv(response_per_meter) @ np.asarray(desired_delta_base, dtype=np.float64)
        corrected = np.zeros(3, dtype=np.float64)
        for i in range(3):
            corrected[i] = arm.axis_sign[i] * arm.axis_scale[i] * desired_delta_base[i] if arm.axis_enabled[i] else 0.0
        return corrected

    saved_kinematics_info = load_kinematics_info(cfg["kinematics_info_path"])
    loaded_saved_calibration = apply_saved_kinematics_to_arms(arms, saved_kinematics_info)
    if loaded_saved_calibration:
        for arm in arms.values():
            arm.axis_align_enabled = True
    mobile_joint_sign, mobile_joint_scale, mobile_joint_enabled, _ = load_mobile_calibration(saved_kinematics_info, len(mobile_joint_targets))
    if USE_AXIS_AUTO_ALIGN and not loaded_saved_calibration:
        probe_axiswise_alignment_for_arm("right")
        probe_axiswise_alignment_for_arm("left")

    print("Controls:")
    print("  V: toggle ARM / MOBILE mode")
    if wheels:
        print("  [DRIVE] B: toggle wheel driving mode | W/S: forward/reverse | A/D: left/right | Space: brake")
    print("  [ARM] TAB: switch active arm view  |  1/2/3: control right/left/both arms")
    print("  Arrow keys: translate X/Y. Shift+Up/Down: Z")
    print("  P: toggle ARM / FOB pose control (/dev/ttyUSB1)")
    print("  I/K: pitch +/-,  J/H: yaw +/-,  U/O: roll +/-")
    print("  L: toggle active-arm end-effector trail (translucent green)")
    print("  [PAINT] Z: toggle rectangular paint (10-50cm ray; green=valid hit, yellow=no valid hit)")
    print("  [PAINT] Hold X: circular spray (10-50cm ray; green=valid hit, yellow=no valid hit) | C: clear paint")
    print("  F: toggle CONTROL_FRAME (base/tool)")
    print("  R: reset selected arm target pose")
    print("  M: toggle axis auto alignment on/off for active arm")
    print("  T/G: re-run axis auto alignment for active/both arms")
    print("  ESC: quit")
    print(f"  ROBOT={robot_key}, STAGE={cfg['stage_path']}, PRIM={robot_prim_path}")
    print("  [V MOBILE] Q/A W/S E/D R/F T/G Y/H U/J I/K: joint1..joint8 +/-")
    print("  [V MOBILE] Arrows: body8 task control (X reversed, Shift=Z, Left/Right=joint1)")
    print(f"  MODE={control_mode}, CONTROL_FRAME={CONTROL_FRAME}, ACTIVE_ARM={active_arm}")

    log_counter = 0
    while simulation_app.is_running() and not should_quit:
        world.step(render=True)
        log_counter += 1
        newly_pressed = pressed - last_pressed
        last_pressed = set(pressed)

        if carb.input.KeyboardInput.KEY_5 in newly_pressed:
            if control_mode == "target_area":
                target_selector.end_selection()
                control_mode = "arm"
                status_ui.set_event("TARGET AREA mode -> ARM")
                help_ui.update_context(robot_key, control_mode, wheels=wheels)
            else:
                if wheels:
                    safe_apply_action(controller, make_wheel_drive_action(wheel_joint_indices, 0.0, 0.0))
                control_mode = "target_area"
                target_selector.begin_selection()
                help_ui.update(
                    robot_key, "TARGET AREA",
                    [
                        "Point at PaintWall_A or PaintWall_B: live camera ray",
                        "F: select first/second diagonal corner",
                        "5: return to ARM mode | ESC: quit",
                    ],
                )

        if control_mode == "target_area":
            if carb.input.KeyboardInput.F in newly_pressed:
                target_selector.select_hover_point()
            target_selector.update()
            update_control_status("TARGET AREA", "live camera-to-mouse raycast")
            continue

        if carb.input.KeyboardInput.V in newly_pressed:
            if control_mode == "drive":
                safe_apply_action(controller, make_wheel_drive_action(wheel_joint_indices, 0.0, 0.0))
            control_mode = "mobile" if control_mode == "arm" else "arm"
            if control_mode == "mobile" and mobile_joint_indices:
                mobile_joint_targets[:] = robot.get_joint_positions()[mobile_joint_indices]
                ladder_kin.reset_orientation_reference(mobile_joint_targets)
            if control_mode == "arm":
                for target_arm in arms.values():
                    cur_pos, cur_rot = target_arm.ik.compute_end_effector_pose()
                    target_arm.target_pos = np.array(cur_pos, dtype=np.float64)
                    target_arm.target_quat = as_wxyz_quat(cur_rot)
                    target_arm.start_target_pos = target_arm.target_pos.copy()
            status_ui.set_event(f"Control mode -> {control_mode.upper()}")
            help_ui.update_context(robot_key, control_mode, wheels=wheels)

        if wheels and carb.input.KeyboardInput.B in newly_pressed:
            safe_apply_action(controller, make_wheel_drive_action(wheel_joint_indices, 0.0, 0.0))
            control_mode = "arm" if control_mode == "drive" else "drive"
            if control_mode == "arm":
                for target_arm in arms.values():
                    cur_pos, cur_rot = target_arm.ik.compute_end_effector_pose()
                    target_arm.target_pos = np.array(cur_pos, dtype=np.float64)
                    target_arm.target_quat = as_wxyz_quat(cur_rot)
                    target_arm.start_target_pos = target_arm.target_pos.copy()
            status_ui.set_event(f"Control mode -> {control_mode.upper()}")
            help_ui.update_context(robot_key, control_mode, wheels=wheels)

        if carb.input.KeyboardInput.P in newly_pressed:
            if control_mode == "fob":
                control_mode = "arm"
                status_ui.set_event("FOB mode -> ARM")
                help_ui.update_context(robot_key, control_mode, wheels=wheels)
            elif control_mode == "arm":
                arm = arms[active_arm]
                cur_pos, cur_rot = arm.ik.compute_end_effector_pose()
                arm.target_pos = np.asarray(cur_pos, dtype=np.float64)
                arm.target_quat = as_wxyz_quat(cur_rot)
                arm.start_target_pos = arm.target_pos.copy()
                ok, fob_status = fob.capture_reference(
                    active_arm, arm.target_pos, arm.target_quat
                )
                status_ui.set_fob_status(fob_status)
                if ok:
                    active_control_target = active_arm
                    control_mode = "fob"
                    status_ui.set_event(f"FOB controls {active_arm.upper()} arm")
                    help_ui.update_context(robot_key, control_mode, wheels=wheels)
                else:
                    status_ui.set_event(f"FOB {fob_status}")

        if control_mode == "arm":
            if carb.input.KeyboardInput.TAB in newly_pressed:
                active_arm = "left" if active_arm == "right" else "right"
                if active_control_target != "both":
                    active_control_target = active_arm
                status_ui.set_event(f"Active arm -> {active_arm.upper()} / target {active_control_target.upper()}")
            if carb.input.KeyboardInput.KEY_1 in newly_pressed:
                active_control_target = "right"; active_arm = "right"; status_ui.set_event("Control target -> RIGHT")
            if carb.input.KeyboardInput.KEY_2 in newly_pressed:
                active_control_target = "left"; active_arm = "left"; status_ui.set_event("Control target -> LEFT")
            if carb.input.KeyboardInput.KEY_3 in newly_pressed:
                active_control_target = "both"; status_ui.set_event("Control target -> BOTH")
            if carb.input.KeyboardInput.L in newly_pressed:
                trail_enabled = not trail_enabled
                if trail_enabled:
                    for arm_name in trail_points:
                        trail_points[arm_name].clear()
                        update_trail_curve(arm_name)
                status_ui.set_event(f"End-effector trail -> {'ON' if trail_enabled else 'OFF'} ({active_control_target.upper()})")
            if carb.input.KeyboardInput.Z in newly_pressed:
                paint_enabled[active_arm] = not paint_enabled[active_arm]
                status_ui.set_event(f"{active_arm.upper()} rectangular paint -> {'ON' if paint_enabled[active_arm] else 'OFF'}")
            if carb.input.KeyboardInput.C in newly_pressed:
                paint_visualizer.clear()
                status_ui.set_event("Paint and spray drawings cleared")
            if carb.input.KeyboardInput.F in newly_pressed:
                CONTROL_FRAME = "tool" if CONTROL_FRAME == "base" else "base"
                status_ui.set_event(f"CONTROL_FRAME -> {CONTROL_FRAME.upper()}")
            if carb.input.KeyboardInput.M in newly_pressed:
                arms[active_arm].axis_align_enabled = not arms[active_arm].axis_align_enabled
                status_ui.set_event(f"{active_arm.upper()} AXIS_AUTO_ALIGN -> {arms[active_arm].axis_align_enabled}")
            if carb.input.KeyboardInput.T in newly_pressed:
                probe_axiswise_alignment_for_arm(active_arm)
            if carb.input.KeyboardInput.G in newly_pressed:
                probe_axiswise_alignment_for_arm("right"); probe_axiswise_alignment_for_arm("left")

        if control_mode == "fob":
            if carb.input.KeyboardInput.Z in newly_pressed:
                paint_enabled[active_arm] = not paint_enabled[active_arm]
                status_ui.set_event(
                    f"{active_arm.upper()} rectangular paint -> {'ON' if paint_enabled[active_arm] else 'OFF'}"
                )
            if carb.input.KeyboardInput.C in newly_pressed:
                paint_visualizer.clear()
                status_ui.set_event("Paint and spray drawings cleared")

        refresh_base_pose()
        control_arms = [arms["right"], arms["left"]] if active_control_target == "both" else [arms[active_control_target]]
        if control_mode == "drive":
            forward = is_down(carb.input.KeyboardInput.W) - is_down(carb.input.KeyboardInput.S)
            turn = is_down(carb.input.KeyboardInput.A) - is_down(carb.input.KeyboardInput.D)
            if is_down(carb.input.KeyboardInput.SPACE):
                forward = 0.0
                turn = 0.0
            safe_apply_action(controller, make_wheel_drive_action(wheel_joint_indices, forward, turn))
            update_control_status("DRIVE", f"forward={forward:+.0f} turn={turn:+.0f}")
            continue

        if control_mode == "ladder_task":
            forward = lift = turn = 0.0
            if len(mobile_joint_targets) > 0:
                shift_ladder = is_down(carb.input.KeyboardInput.LEFT_SHIFT) or is_down(carb.input.KeyboardInput.RIGHT_SHIFT)
                forward = 0.0 if shift_ladder else is_down(carb.input.KeyboardInput.DOWN) - is_down(carb.input.KeyboardInput.UP)
                lift = is_down(carb.input.KeyboardInput.UP) - is_down(carb.input.KeyboardInput.DOWN) if shift_ladder else 0.0
                turn = is_down(carb.input.KeyboardInput.LEFT) - is_down(carb.input.KeyboardInput.RIGHT)
                count = min(len(mobile_joint_targets), len(ladder_kin.joints))
                joint_delta = ladder_kin.command_delta(mobile_joint_targets[:count], forward, lift, turn, dt)
                joint_delta *= mobile_joint_sign[:count] * mobile_joint_scale[:count]
                joint_delta *= mobile_joint_enabled[:count]
                mobile_joint_targets[:count] += joint_delta
                mobile_joint_targets[:] = np.clip(mobile_joint_targets, MOBILE_JOINT_LOWER_RAD, MOBILE_JOINT_UPPER_RAD)
                safe_apply_action(controller, make_mobile_joint_action(mobile_joint_indices, mobile_joint_targets))
            update_control_status("LADDER_TASK", f"body8 forward={forward:+.0f} lift={lift:+.0f} turn={turn:+.0f}")
            continue


        if control_mode == "mobile":
            mobile_step = MOBILE_JOINT_SPEED_RPS * dt
            if len(mobile_joint_targets) > 0:
                shift_mobile = is_down(carb.input.KeyboardInput.LEFT_SHIFT) or is_down(carb.input.KeyboardInput.RIGHT_SHIFT)
                forward = 0.0 if shift_mobile else is_down(carb.input.KeyboardInput.DOWN) - is_down(carb.input.KeyboardInput.UP)
                lift = is_down(carb.input.KeyboardInput.UP) - is_down(carb.input.KeyboardInput.DOWN) if shift_mobile else 0.0
                turn = is_down(carb.input.KeyboardInput.LEFT) - is_down(carb.input.KeyboardInput.RIGHT)
                count = min(len(mobile_joint_targets), len(ladder_kin.joints))
                task_delta = ladder_kin.command_delta(mobile_joint_targets[:count], forward, lift, turn, dt)
                task_delta *= mobile_joint_sign[:count] * mobile_joint_scale[:count]
                task_delta *= mobile_joint_enabled[:count]
                mobile_joint_targets[:count] += task_delta
                # Original per-joint letter controls remain active and are added below.
                key_name_to_input = {name: getattr(carb.input.KeyboardInput, name) for pair in MOBILE_JOINT_KEY_BINDINGS for name in pair}
                for i, (plus_key, minus_key) in enumerate(MOBILE_JOINT_KEY_BINDINGS[:len(mobile_joint_targets)]):
                    if not mobile_joint_enabled[i]:
                        continue
                    direction = is_down(key_name_to_input[plus_key]) - is_down(key_name_to_input[minus_key])
                    mobile_joint_targets[i] += direction * mobile_step * MOBILE_JOINT_COMMAND_SCALE[i] * mobile_joint_sign[i] * mobile_joint_scale[i]
                mobile_joint_targets[:] = np.clip(mobile_joint_targets, MOBILE_JOINT_LOWER_RAD, MOBILE_JOINT_UPPER_RAD)
                safe_apply_action(controller, make_mobile_joint_action(mobile_joint_indices, mobile_joint_targets))

            update_control_status("MOBILE", "body8 task + individual joint control")
            continue

        d = EE_SPEED_MPS * dt
        shift = is_down(carb.input.KeyboardInput.LEFT_SHIFT) or is_down(carb.input.KeyboardInput.RIGHT_SHIFT)
        desired_delta = np.zeros(3, dtype=np.float64)
        if is_down(carb.input.KeyboardInput.UP):
            desired_delta[2 if shift else 0] += d
        if is_down(carb.input.KeyboardInput.DOWN):
            desired_delta[2 if shift else 0] -= d
        if is_down(carb.input.KeyboardInput.LEFT):
            desired_delta[1] += d
        if is_down(carb.input.KeyboardInput.RIGHT):
            desired_delta[1] -= d
        desired_delta = clamp_vec(desired_delta * KEYBOARD_BASE_AXIS_SIGN, MAX_DELTA_PER_STEP)

        for target_arm in control_arms:
            arm_input_sign = np.asarray(cfg.get("arm_input_axis_sign", {}).get(target_arm.name, [1.0, 1.0, 1.0]), dtype=np.float64)
            signed_desired_delta = desired_delta * arm_input_sign
            command_delta = apply_axiswise_correction(target_arm, signed_desired_delta) if target_arm.axis_align_enabled else signed_desired_delta.copy()
            command_delta = clamp_vec(command_delta, MAX_DELTA_PER_STEP)
            if np.linalg.norm(command_delta) > 0:
                if CONTROL_FRAME == "tool":
                    w, x, y, z = target_arm.target_quat
                    rot = np.array([
                        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
                    ], dtype=np.float64)
                    target_arm.target_pos += rot @ command_delta
                else:
                    target_arm.target_pos += command_delta
                target_arm.target_pos = np.minimum(
                    np.maximum(target_arm.target_pos, target_arm.start_target_pos - MAX_TARGET_OFFSET_FROM_START),
                    target_arm.start_target_pos + MAX_TARGET_OFFSET_FROM_START,
                )

        if control_mode == "fob":
            fob_position, fob_quaternion, fob_status = fob.target(active_arm)
            status_ui.set_fob_status(fob_status)
            if fob_position is not None:
                fob_arm = arms[active_arm]
                fob_arm.target_pos = np.minimum(
                    np.maximum(fob_position, fob_arm.start_target_pos - MAX_TARGET_OFFSET_FROM_START),
                    fob_arm.start_target_pos + MAX_TARGET_OFFSET_FROM_START,
                )
                fob_arm.target_quat = fob_quaternion
            else:
                status_ui.set_event(f"FOB {fob_status}")

        if carb.input.KeyboardInput.R in newly_pressed:
            for target_arm in control_arms:
                cur_pos, cur_rot = target_arm.ik.compute_end_effector_pose()
                target_arm.target_pos = np.array(cur_pos, dtype=np.float64)
                target_arm.target_quat = as_wxyz_quat(cur_rot)
                target_arm.start_target_pos = target_arm.target_pos.copy()
            if control_mode == "fob":
                ok, fob_status = fob.capture_reference(
                    active_arm, arms[active_arm].target_pos, arms[active_arm].target_quat
                )
                status_ui.set_fob_status(fob_status)
            status_ui.set_event(f"Target pose reset -> {active_control_target.upper()}")

        da = ROT_SPEED_RPS * dt if control_mode == "arm" else 0.0
        roll = (is_down(carb.input.KeyboardInput.U) - is_down(carb.input.KeyboardInput.O)) * da
        pitch = (is_down(carb.input.KeyboardInput.I) - is_down(carb.input.KeyboardInput.K)) * da
        yaw = (is_down(carb.input.KeyboardInput.J) - is_down(carb.input.KeyboardInput.H)) * da
        for target_arm in control_arms:
            for axis, angle in [([1, 0, 0], roll), ([0, 1, 0], pitch), ([0, 0, 1], yaw)]:
                if abs(angle) > 0:
                    dq = quat_from_axis_angle(axis, angle)
                    target_arm.target_quat = quat_norm(quat_mul(target_arm.target_quat, dq)) if CONTROL_FRAME == "tool" else quat_norm(quat_mul(dq, target_arm.target_quat))

        ok = True
        for target_arm in control_arms:
            action, arm_ok = target_arm.ik.compute_inverse_kinematics(target_arm.target_pos, target_arm.target_quat)
            ok = ok and arm_ok
            safe_apply_action(controller, action if arm_ok else None, hold_action)

        if len(mobile_joint_targets) > 0:
            safe_apply_action(controller, make_mobile_joint_action(mobile_joint_indices, mobile_joint_targets))

        if trail_enabled:
            for target_arm in control_arms:
                point = end_effector_world_position(target_arm.name)
                points = trail_points[target_arm.name]
                if not points or np.linalg.norm(point - points[-1]) >= EE_TRAIL_MIN_DISTANCE_M:
                    points.append(point)
                    if len(points) > EE_TRAIL_MAX_POINTS:
                        del points[:len(points) - EE_TRAIL_MAX_POINTS]
                    update_trail_curve(target_arm.name)

        tool_matrix = UsdGeom.Xformable(ee_prims[active_arm]).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        paint_visualizer.update(
            active_arm, tool_matrix, paint_enabled[active_arm],
            spray_requested(),
        )

        update_control_status(control_mode.upper(), f"frame={CONTROL_FRAME.upper()}", ik_ok=ok)

    fob.close()
    arduino_switch.close()
    simulation_app.close()


def run_kinematics_validation(robot_key: str, output_path: Optional[str] = None) -> None:
    from pxr import Usd, UsdGeom

    cfg = ROBOT_CONFIGS[robot_key]
    output_path = output_path or cfg["kinematics_info_path"]

    open_stage(cfg["stage_path"])
    stage = get_current_stage()
    robot_prim_path = select_articulation_root_path(stage, cfg["articulation_root_link"])
    configure_upper_body_masses(stage, cfg)
    configure_mobile_joint_usd_limits(stage)
    configure_arm_joint_usd_drives(stage, cfg)
    configure_robot_gravity(stage, robot_prim_path, disable=DISABLE_GRAVITY)

    world = World()
    robot = Articulation(robot_prim_path)
    world.scene.add(robot)
    world.reset()
    wait_for_articulation_ready(world, robot)

    controller = robot.get_articulation_controller()
    apply_initial_joint_pose(robot, controller, world, cfg.get("initial_joint_positions", {}))
    jp0 = robot.get_joint_positions()
    hold_action = ArticulationAction(joint_positions=jp0, joint_velocities=np.zeros_like(jp0))

    right_kin = LulaKinematicsSolver(robot_description_path=cfg["right_desc_yaml"], urdf_path=cfg["urdf_path"])
    left_kin = LulaKinematicsSolver(robot_description_path=cfg["left_desc_yaml"], urdf_path=cfg["urdf_path"])
    set_ik_iters(right_kin, 80)
    set_ik_iters(left_kin, 80)
    right_ik = ArticulationKinematicsSolver(robot, right_kin, cfg["right_ee_frame"])
    left_ik = ArticulationKinematicsSolver(robot, left_kin, cfg["left_ee_frame"])

    base_t, base_q_raw = robot.get_world_pose()
    base_q_use = to_wxyz_from_xyzw(base_q_raw) if FORCE_BASE_WXYZ_SWAP else np.array(base_q_raw, dtype=np.float64)
    right_kin.set_robot_base_pose(base_t, base_q_use)
    left_kin.set_robot_base_pose(base_t, base_q_use)

    arms = {
        "right": ArmState("right", right_kin, right_ik, cfg["right_ee_frame"]),
        "left": ArmState("left", left_kin, left_ik, cfg["left_ee_frame"]),
    }

    def refresh_base_pose_for_validation():
        base_t_now, _ = robot.get_world_pose()
        right_kin.set_robot_base_pose(base_t_now, base_q_use)
        left_kin.set_robot_base_pose(base_t_now, base_q_use)

    def step_probe(arm: ArmState, goal_pos, goal_quat, steps=ALIGN_SETTLE_STEPS):
        ok_last = False
        for _ in range(steps):
            refresh_base_pose_for_validation()
            action, ok = arm.ik.compute_inverse_kinematics(goal_pos, goal_quat)
            ok_last = ok
            safe_apply_action(controller, action if ok else None, hold_action)
            world.step(render=False)
        return ok_last

    mobile_payload_prim = find_prim_by_name(stage, "body8_mobile")

    def payload_world_position():
        matrix = UsdGeom.Xformable(mobile_payload_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        return np.asarray([translation[0], translation[1], translation[2]], dtype=np.float64)

    mobile_indices = resolve_mobile_joint_indices(robot)
    mobile_calibration = {"probe_delta_rad": 0.05, "joint_sign": [], "joint_scale": [], "joint_enabled": [], "probes": []}
    if mobile_indices:
        start_q = robot.get_joint_positions()[mobile_indices].copy()
        for i, joint_index in enumerate(mobile_indices):
            start_payload_pos = payload_world_position()
            target_q = start_q.copy()
            target_q[i] += mobile_calibration["probe_delta_rad"]
            action = make_mobile_joint_action(mobile_indices, target_q)
            for _ in range(ALIGN_SETTLE_STEPS):
                safe_apply_action(controller, action, hold_action)
                world.step(render=False)
            cur_q = robot.get_joint_positions()[mobile_indices].copy()
            payload_delta = payload_world_position() - start_payload_pos
            actual = float(cur_q[i] - start_q[i])
            probe_delta = mobile_calibration["probe_delta_rad"]
            reliable = np.isfinite(actual) and actual > 0.1 * probe_delta and actual < 2.0 * probe_delta
            sign = 1.0
            scale = float(np.clip(probe_delta / actual, 0.5, 2.0)) if reliable else 1.0
            enabled = abs(actual) > 1e-4
            mobile_calibration["joint_sign"].append(sign)
            mobile_calibration["joint_scale"].append(scale)
            mobile_calibration["joint_enabled"].append(enabled)
            mobile_calibration["probes"].append({
                "joint": MOBILE_JOINT_NAMES[i],
                "joint_index": int(joint_index),
                "target_delta_rad": mobile_calibration["probe_delta_rad"],
                "actual_delta_rad": actual,
                "payload_delta_m": vec_to_list(payload_delta),
                "payload_vertical_delta_m": float(payload_delta[2]),
                "calibration_reliable": bool(reliable),
                "enabled": enabled,
            })
            safe_apply_action(controller, make_mobile_joint_action(mobile_indices, start_q), hold_action)
            for _ in range(ALIGN_RESTORE_STEPS):
                safe_apply_action(controller, make_mobile_joint_action(mobile_indices, start_q), hold_action)
                world.step(render=False)
        lift_count = min(len(start_q), len(MOBILE_LIFT_COMMAND_PROFILE))
        lift_start_pos = payload_world_position()
        lift_target_q = start_q.copy()
        lift_target_q[:lift_count] += mobile_calibration["probe_delta_rad"] * MOBILE_LIFT_COMMAND_PROFILE[:lift_count]
        lift_action = make_mobile_joint_action(mobile_indices, lift_target_q)
        for _ in range(ALIGN_SETTLE_STEPS):
            safe_apply_action(controller, lift_action, hold_action)
            world.step(render=False)
        lift_payload_delta = payload_world_position() - lift_start_pos
        mobile_calibration["coordinated_lift_probe"] = {
            "profile": vec_to_list(MOBILE_LIFT_COMMAND_PROFILE[:lift_count]),
            "payload_delta_m": vec_to_list(lift_payload_delta),
            "payload_vertical_delta_m": float(lift_payload_delta[2]),
        }
        safe_apply_action(controller, make_mobile_joint_action(mobile_indices, start_q), hold_action)
        print(f"[VALIDATION] mobile: sign={fmt(mobile_calibration['joint_sign'])} scale={fmt(mobile_calibration['joint_scale'])} lift_dz={lift_payload_delta[2]:.6f}m enabled={mobile_calibration['joint_enabled']}")

    info = {
        "robot_key": robot_key,
        "stage_path": cfg["stage_path"],
        "urdf_path": cfg["urdf_path"],
        "robot_prim_path": robot_prim_path,
        "base_pose": {"position": vec_to_list(base_t), "quat_wxyz": vec_to_list(base_q_use)},
        "mobile_joints": {"names": MOBILE_JOINT_NAMES, "indices": mobile_indices, "calibration": mobile_calibration},
        "arms": {},
    }

    def probe_arm_with_profile(arm: ArmState, profile: dict):
        set_ik_iters(arm.kin, int(profile["ik_iters"]))
        refresh_base_pose_for_validation()
        start_pos, start_rot = arm.ik.compute_end_effector_pose()
        start_pos = np.asarray(start_pos, dtype=np.float64)
        start_quat = as_wxyz_quat(start_rot)
        probe_dist = float(profile["probe_dist"])
        responses = np.zeros((3, 3), dtype=np.float64)
        signs = np.ones(3, dtype=np.float64)
        scales = np.ones(3, dtype=np.float64)
        enabled = np.ones(3, dtype=bool)
        qualities = np.zeros(3, dtype=np.float64)
        probes = [("x", np.array([probe_dist, 0.0, 0.0]), 0), ("y", np.array([0.0, probe_dist, 0.0]), 1), ("z", np.array([0.0, 0.0, probe_dist]), 2)]
        probe_rows = []
        total_error = 0.0
        for label, desired, idx in probes:
            ok = step_probe(arm, start_pos + desired, start_quat, steps=int(profile["settle_steps"]))
            cur_pos, _ = arm.ik.compute_end_effector_pose()
            actual = np.asarray(cur_pos, dtype=np.float64) - start_pos
            responses[:, idx] = actual
            comp = actual[idx]
            align = alignment_score(actual, desired)
            signs[idx] = 1.0 if comp >= 0.0 else -1.0
            scales[idx] = min(3.0, probe_dist / abs(comp)) if abs(comp) > 1e-9 else 1.0
            enabled[idx] = bool(ok) and abs(align) >= MIN_ALIGN_TO_ENABLE and np.linalg.norm(actual) >= MIN_RESPONSE_NORM
            qualities[idx] = align
            total_error += float(np.linalg.norm(desired - actual))
            probe_rows.append({
                "axis": label,
                "desired_delta_base": vec_to_list(desired),
                "actual_delta": vec_to_list(actual),
                "ik_ok": bool(ok),
                "alignment": float(align),
                "abs_alignment": float(abs(align)),
                "error_norm": float(np.linalg.norm(desired - actual)),
            })
            step_probe(arm, start_pos, start_quat, steps=int(profile["restore_steps"]))
        score = (int(np.count_nonzero(enabled)), float(np.mean(np.abs(qualities))), -total_error)
        return {
            "profile": profile,
            "score": score,
            "start_pos": start_pos,
            "start_quat": start_quat,
            "responses": responses,
            "signs": signs,
            "scales": scales,
            "enabled": enabled,
            "qualities": qualities,
            "probe_rows": probe_rows,
            "total_error": total_error,
        }

    for arm_name, arm in arms.items():
        attempts = []
        for profile in KINEMATICS_TUNING_PROFILES:
            result = probe_arm_with_profile(arm, profile)
            attempts.append(result)
            print(
                f"[TUNING] {arm_name}/{profile['name']}: enabled={result['enabled']} "
                f"mean_abs_align={np.mean(np.abs(result['qualities'])):.4f} total_error={result['total_error']:.5f}"
            )
            if np.all(result["enabled"]):
                break
        best = max(attempts, key=lambda r: r["score"])
        matrix_rank = int(np.linalg.matrix_rank(best["responses"]))
        if not np.all(best["enabled"]):
            print(f"[WARN] {arm_name}: calibration kept failed axes disabled (response rank={matrix_rank}, enabled={best['enabled']})")
        force_enabled = bool(np.all(best["enabled"]))
        info["arms"][arm_name] = {
            "frame_name": arm.frame_name,
            "start_pose": {"position": vec_to_list(best["start_pos"]), "quat_wxyz": vec_to_list(best["start_quat"])},
            "axis_calibration": {
                "probe_distance_m": float(best["profile"]["probe_dist"]),
                "axis_sign": vec_to_list(best["signs"]),
                "axis_scale": vec_to_list(best["scales"]),
                "axis_enabled": best["enabled"].astype(bool).tolist(),
                "axis_quality": vec_to_list(best["qualities"]),
                "axis_response": best["responses"].tolist(),
                "probes": best["probe_rows"],
                "selected_profile": best["profile"],
                "all_axes_enabled": force_enabled,
                "tuning_attempts": [
                    {
                        "profile": r["profile"],
                        "enabled": r["enabled"].astype(bool).tolist(),
                        "mean_abs_alignment": float(np.mean(np.abs(r["qualities"]))),
                        "total_error": float(r["total_error"]),
                    }
                    for r in attempts
                ],
            },
        }
        print(f"[VALIDATION] {arm_name}: selected={best['profile']['name']} sign={fmt(best['signs'])} scale={fmt(best['scales'])} enabled={best['enabled']} quality={fmt(best['qualities'])}")

    save_kinematics_info(output_path, info)
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--robot", choices=sorted(ROBOT_CONFIGS))
    target.add_argument(
        "-m", "--multi", action="store_true",
        help="Control the three robots in mobile_bucket_multi-robot.usd",
    )
    args = parser.parse_args(CONTROLLER_CLI_ARGS)
    if args.multi or args.robot is None:
        from multi_robot_control import run as run_multi_robot_control
        run_multi_robot_control(sys.modules[__name__])
    else:
        main(args.robot)
