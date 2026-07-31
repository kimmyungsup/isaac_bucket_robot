"""Body8-based kinematics for the ladder mechanism (no Isaac dependencies)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _rotation_from_rpy(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s, v = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [x*x*v+c, x*y*v-z*s, x*z*v+y*s],
        [y*x*v+z*s, y*y*v+c, y*z*v-x*s],
        [z*x*v-y*s, z*y*v+x*s, z*z*v+c],
    ])


def _rotation_vector(rotation):
    """Return the shortest axis-angle vector represented by a rotation matrix."""
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return np.zeros(3)
    vector = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return vector * (0.5 * angle / max(np.sin(angle), 1e-9))


class LadderKinematics:
    """Translational Jacobian and damped IK for base_mobile -> body8_mobile."""

    def __init__(self, path):
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)  # JSON is valid YAML 1.2.
        self.data = data
        self.joints = data["chain"]["joints"]
        control = data["control"]
        self.damping = float(control["damped_least_squares"])
        self.max_joint_delta = float(control["max_joint_delta_rad_per_frame"])
        self.translation_speed = float(control["translation_speed_mps"])
        self.turn_speed = float(control["turn_speed_rps"])
        self.joint2_lift_assist = float(control.get("joint2_lift_assist_rad_per_m", 2.0))
        self.joint2_forward_assist = float(control.get("joint2_forward_assist_rad_per_m", 0.8))
        self.orientation_gain = float(control.get("orientation_hold_gain", 0.35))
        self.orientation_weight = float(control.get("orientation_task_weight", 1.5))
        self.joint_costs = np.asarray(control.get("joint_motion_cost", [1.0, 0.08, 1.0, 1.0, 0.55, 1.0, 1.0, 1.0]), dtype=np.float64)
        self.orientation_reference = None

    def pose_and_jacobian(self, joint_positions):
        q = np.asarray(joint_positions, dtype=np.float64)
        transform = np.eye(4)
        origins, axes = [], []
        for index, joint in enumerate(self.joints):
            origin = np.eye(4)
            origin[:3, :3] = _rotation_from_rpy(joint["origin_rpy"])
            origin[:3, 3] = joint["origin_xyz"]
            transform = transform @ origin
            origins.append(transform[:3, 3].copy())
            axis = transform[:3, :3] @ np.asarray(joint["axis"], dtype=np.float64)
            axes.append(axis)
            rotation = np.eye(4)
            rotation[:3, :3] = _axis_angle(joint["axis"], q[index])
            transform = transform @ rotation
        position = transform[:3, 3].copy()
        linear_jacobian = np.column_stack([np.cross(axis, position - origin) for origin, axis in zip(origins, axes)])
        angular_jacobian = np.column_stack(axes)
        return position, transform[:3, :3].copy(), linear_jacobian, angular_jacobian

    def position_and_jacobian(self, joint_positions):
        position, _, jacobian, _ = self.pose_and_jacobian(joint_positions)
        return position, jacobian

    def reset_orientation_reference(self, joint_positions):
        _, rotation, _, _ = self.pose_and_jacobian(joint_positions)
        self.orientation_reference = rotation.copy()

    def _weighted_solve(self, task_jacobian, task_delta, joint_costs):
        costs = np.maximum(np.asarray(joint_costs, dtype=np.float64), 1e-4)
        inverse_cost = np.diag(1.0 / costs)
        regularized = task_jacobian @ inverse_cost @ task_jacobian.T + self.damping ** 2 * np.eye(task_jacobian.shape[0])
        return inverse_cost @ task_jacobian.T @ np.linalg.solve(regularized, task_delta)

    def command_delta(self, joint_positions, forward, lift, turn, dt):
        """Weighted X/Z IK while holding body8 roll, pitch, and yaw."""
        _, rotation, linear_jacobian, angular_jacobian = self.pose_and_jacobian(joint_positions)
        if self.orientation_reference is None:
            self.orientation_reference = rotation.copy()
        step_m = self.translation_speed * float(dt)
        dq = np.zeros(len(self.joints), dtype=np.float64)
        if forward or lift:
            position_rows, position_delta = [], []
            if forward:
                position_rows.append(linear_jacobian[0, 1:])
                position_delta.append(step_m * float(forward))
            if lift:
                position_rows.append(linear_jacobian[2, 1:])
                position_delta.append(step_m * float(lift))
            orientation_error = _rotation_vector(self.orientation_reference @ rotation.T)
            task_jacobian = np.vstack(position_rows + [self.orientation_weight * angular_jacobian[:, 1:]])
            task_delta = np.concatenate([np.asarray(position_delta), self.orientation_weight * self.orientation_gain * orientation_error])
            # Prescribe joint2 first; joints3..8 compensate its position and attitude effects.
            dq[1] = self.joint2_forward_assist * step_m * float(forward) - self.joint2_lift_assist * step_m * float(lift)
            residual = task_delta - task_jacobian[:, 0] * dq[1]
            dq[2:] = self._weighted_solve(task_jacobian[:, 1:], residual, self.joint_costs[2:])
            if lift and np.linalg.norm(linear_jacobian[2, 1:]) < 1e-3:
                dq[6] += step_m * float(lift)
        dq[0] = self.turn_speed * float(dt) * float(turn)
        if turn and not (forward or lift):
            self.orientation_reference = rotation.copy()
        return np.clip(dq, -self.max_joint_delta, self.max_joint_delta)
