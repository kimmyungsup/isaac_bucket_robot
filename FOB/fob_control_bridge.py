"""Non-blocking FOB pose bridge shared by Isaac Sim controller modes."""

from __future__ import annotations

from argparse import Namespace
import time
import numpy as np

from .fob_6d_pose_visualizer import PoseSource


class FOBControlBridge:
    def __init__(self, port="/dev/ttyUSB1", baud=115200, timeout=0.05,
                 range_in=36.0, position_scale_m_per_cm=(0.01, 0.01, 0.01),
                 position_sign=(1.0, -1.0, 1.0), orientation_sign=(-1.0, 1.0, -1.0),
                 stale_after_s=1.0):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.range_in = range_in
        self.position_scale = np.asarray(position_scale_m_per_cm, dtype=np.float64)
        self.position_sign = np.asarray(position_sign, dtype=np.float64)
        self.orientation_sign = np.asarray(orientation_sign, dtype=np.float64)
        self.stale_after_s = stale_after_s
        self.source = None
        self.references = {}

    def start(self):
        if self.source is not None:
            self.source.stop()
        args = Namespace(demo=False, port=self.port, baud=self.baud,
                         timeout=self.timeout, period=0.02, range_in=self.range_in,
                         set_pos_angles=True)
        self.source = PoseSource(args)
        self.source.start()

    def close(self):
        if self.source is not None:
            self.source.stop()
            self.source = None

    def snapshot(self):
        if self.source is None:
            return None, "connection failed: FOB reader is not started"
        pose, sample_time, _count, connected, error = self.source.snapshot()
        if not connected or pose is None:
            return None, f"connection failed: {error or 'no FOB data'}"
        age = time.monotonic() - sample_time
        if age > self.stale_after_s:
            return None, f"connection failed: FOB data stale ({age:.2f}s)"
        return pose, "connected"

    @property
    def status(self):
        return self.snapshot()[1]

    def capture_reference(self, arm_name, arm_position, arm_quaternion):
        pose, status = self.snapshot()
        if pose is None:
            return False, status
        self.references[arm_name] = {
            "sensor_position_cm": np.array([pose.x_cm, pose.y_cm, pose.z_cm]),
            "sensor_angles_deg": np.array([pose.roll_deg, pose.elevation_deg,
                                            pose.azimuth_deg]),
            "arm_position": np.asarray(arm_position, dtype=np.float64).copy(),
            "arm_quaternion": np.asarray(arm_quaternion, dtype=np.float64).copy(),
        }
        return True, "connected"

    def target(self, arm_name):
        pose, status = self.snapshot()
        reference = self.references.get(arm_name)
        if pose is None or reference is None:
            return None, None, status if pose is None else "reference not captured"
        position_cm = np.array([pose.x_cm, pose.y_cm, pose.z_cm])
        position_delta = ((position_cm - reference["sensor_position_cm"])
                          * self.position_scale * self.position_sign)
        target_position = reference["arm_position"] + position_delta
        angles_deg = np.array([pose.roll_deg, pose.elevation_deg, pose.azimuth_deg])
        angle_delta_deg = ((angles_deg - reference["sensor_angles_deg"] + 180.0)
                           % 360.0 - 180.0)
        angle_delta_rad = np.deg2rad(angle_delta_deg) * self.orientation_sign
        delta_quaternion = self._rpy_quaternion(*angle_delta_rad)
        target_quaternion = self._normalize(
            self._multiply(reference["arm_quaternion"], delta_quaternion)
        )
        return target_position, target_quaternion, "connected"

    @staticmethod
    def _multiply(left, right):
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _normalize(quaternion):
        norm = np.linalg.norm(quaternion)
        return (quaternion / norm if norm > 1e-12
                else np.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def _rpy_quaternion(cls, roll, pitch, yaw):
        qx = np.array([np.cos(roll/2), np.sin(roll/2), 0.0, 0.0])
        qy = np.array([np.cos(pitch/2), 0.0, np.sin(pitch/2), 0.0])
        qz = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])
        return cls._multiply(cls._multiply(qz, qy), qx)
