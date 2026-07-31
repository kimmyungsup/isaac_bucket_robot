#!/usr/bin/env python3
"""Standalone real-time 6D pose visualizer for a Flock of Birds sensor."""

from __future__ import annotations

import argparse
from collections import deque
import math
import threading
import time
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

try:
    from .fob_position_angles_reader import (
        POS_ANGLES_CMD, PoseAngles, open_serial,
        parse_position_angles_record, read_one_position_angles_record,
    )
except ImportError:
    from fob_position_angles_reader import (
        POS_ANGLES_CMD, PoseAngles, open_serial,
        parse_position_angles_record, read_one_position_angles_record,
    )

try:
    from .arduino_switch_reader import ArduinoSwitchSource
except ImportError:
    from arduino_switch_reader import ArduinoSwitchSource


class PoseSource:
    """Thread-safe latest-pose source backed by serial FOB data or demo data."""

    def __init__(self, args):
        self.args = args
        self.pose: Optional[PoseAngles] = None
        self.sample_time = 0.0
        self.sample_count = 0
        self.error = ""
        self.connected = False
        self._serial = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="fob-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.args.timeout * 5.0))
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass

    def snapshot(self):
        with self._lock:
            return self.pose, self.sample_time, self.sample_count, self.connected, self.error

    def _publish(self, pose: PoseAngles) -> None:
        with self._lock:
            self.pose = pose
            self.sample_time = time.monotonic()
            self.sample_count += 1
            self.connected = True
            self.error = ""

    def _set_error(self, text: str) -> None:
        with self._lock:
            self.connected = False
            self.error = text

    def _run(self) -> None:
        if self.args.demo:
            self._run_demo()
            return
        try:
            self._serial = open_serial(self.args.port, self.args.baud, self.args.timeout)
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            if self.args.set_pos_angles:
                self._serial.write(bytes([POS_ANGLES_CMD]))
                self._serial.flush()
                time.sleep(0.05)
                self._serial.reset_input_buffer()
        except Exception as exc:
            self._set_error(f"serial open failed: {exc}")
            return

        while not self._stop.is_set():
            started = time.monotonic()
            try:
                record = read_one_position_angles_record(
                    self._serial, timeout_s=max(0.5, self.args.timeout * 4.0)
                )
                self._publish(parse_position_angles_record(record, self.args.range_in))
            except Exception as exc:
                self._set_error(str(exc))
            remaining = self.args.period - (time.monotonic() - started)
            if remaining > 0.0:
                self._stop.wait(remaining)

    def _run_demo(self) -> None:
        started = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic() - started
            xyz = np.array([
                25.0 * math.sin(0.7 * t),
                18.0 * math.sin(0.43 * t + 0.8),
                12.0 * math.cos(0.55 * t),
            ])
            angles = np.array([
                55.0 * math.sin(0.35 * t),
                35.0 * math.sin(0.51 * t),
                70.0 * math.cos(0.29 * t),
            ])
            self._publish(PoseAngles(
                x_in=xyz[0] / 2.54, y_in=xyz[1] / 2.54, z_in=xyz[2] / 2.54,
                x_cm=xyz[0], y_cm=xyz[1], z_cm=xyz[2],
                azimuth_deg=angles[0], elevation_deg=angles[1], roll_deg=angles[2],
                raw_words=(0, 0, 0, 0, 0, 0),
            ))
            self._stop.wait(self.args.period)


def rotation_matrix_zyx(azimuth_deg: float, elevation_deg: float, roll_deg: float):
    """Return Rz(azimuth) @ Ry(elevation) @ Rx(roll)."""
    azimuth, elevation, roll = np.deg2rad([azimuth_deg, elevation_deg, roll_deg])
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[ce, 0.0, se], [0.0, 1.0, 0.0], [-se, 0.0, ce]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


class PoseVisualizer:
    def __init__(self, source: PoseSource, switch_source: ArduinoSwitchSource, args):
        self.source = source
        self.switch_source = switch_source
        self.args = args
        self.reference_xyz = None
        self.latest_absolute_xyz = None
        self.last_sample_count = -1
        self.start_time = time.monotonic()
        self.times = deque(maxlen=args.history)
        self.positions = deque(maxlen=args.history)
        self.angles = deque(maxlen=args.history)

        self.figure = plt.figure(figsize=(14, 8))
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1.2, 1.0))
        self.axis_3d = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.axis_position = self.figure.add_subplot(grid[0, 1])
        self.axis_angles = self.figure.add_subplot(grid[1, 1])
        self.figure.canvas.manager.set_window_title("FOB 6D Pose Visualizer")
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self.animation = FuncAnimation(
            self.figure, self._update, interval=args.refresh_ms, cache_frame_data=False
        )

    def show(self) -> None:
        print("[INFO] R: reset position reference/trail | Q or ESC: quit")
        plt.show()

    def _on_close(self, _event) -> None:
        self.source.stop()
        self.switch_source.stop()

    def _on_key(self, event) -> None:
        if event.key and event.key.lower() == "r":
            self.reference_xyz = (
                None if self.latest_absolute_xyz is None else self.latest_absolute_xyz.copy()
            )
            self.times.clear()
            self.positions.clear()
            self.angles.clear()
        elif event.key and event.key.lower() in ("q", "escape"):
            plt.close(self.figure)

    def _display_position(self, pose: PoseAngles) -> np.ndarray:
        absolute = np.array([pose.x_cm, pose.y_cm, pose.z_cm], dtype=np.float64)
        self.latest_absolute_xyz = absolute
        if self.args.relative:
            if self.reference_xyz is None:
                self.reference_xyz = absolute.copy()
            return absolute - self.reference_xyz
        return absolute

    def _update(self, _frame):
        pose, sample_time, sample_count, connected, error = self.source.snapshot()
        if pose is not None and sample_count != self.last_sample_count:
            self.last_sample_count = sample_count
            position = self._display_position(pose)
            angles = np.array(
                [pose.azimuth_deg, pose.elevation_deg, pose.roll_deg], dtype=np.float64
            )
            self.times.append(sample_time - self.start_time)
            self.positions.append(position)
            self.angles.append(angles)
        switch_active, _count, switch_connected, switch_status = self.switch_source.snapshot()
        self._draw(pose, connected, error, switch_active, switch_connected, switch_status)

    def _draw(self, pose, connected: bool, error: str, switch_active: bool,
              switch_connected: bool, switch_status: str) -> None:
        self.axis_3d.clear()
        self.axis_position.clear()
        self.axis_angles.clear()
        source_name = "DEMO" if self.args.demo else self.args.port
        state = "CONNECTED" if connected else "DISCONNECTED"
        switch_state = ("ON" if switch_active else "OFF") if switch_connected else "DISCONNECTED"
        self.figure.suptitle(
            f"FOB 6D Pose — {source_name} — {state}\n"
            f"Arduino switch — {self.args.switch_port} — {switch_state}", fontsize=14,
        )
        self.axis_position.text(
            0.02, 0.95, f"SPRAY SWITCH: {switch_state}",
            transform=self.axis_position.transAxes, va="top", weight="bold",
            color=("green" if switch_active and switch_connected
                   else "red" if not switch_connected else "0.35"),
        )

        if self.positions:
            trajectory = np.asarray(self.positions)
            current = trajectory[-1]
            self.axis_3d.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color="0.35", linewidth=1.5, label="trajectory",
            )
            self.axis_3d.scatter(*current, color="black", s=35)
            if pose is not None:
                rotation = rotation_matrix_zyx(
                    pose.azimuth_deg, pose.elevation_deg, pose.roll_deg
                )
                for index, (color, label) in enumerate(
                    (("red", "X"), ("green", "Y"), ("blue", "Z"))
                ):
                    vector = rotation[:, index] * self.args.axis_length_cm
                    self.axis_3d.quiver(
                        *current, *vector, color=color, linewidth=2.5,
                        arrow_length_ratio=0.18, label=label,
                    )
                self.axis_3d.text2D(
                    0.02, 0.98,
                    f"Position [cm]  X {current[0]:+8.3f}  Y {current[1]:+8.3f}  Z {current[2]:+8.3f}\n"
                    f"Angles [deg]  Az {pose.azimuth_deg:+8.3f}  "
                    f"El {pose.elevation_deg:+8.3f}  Roll {pose.roll_deg:+8.3f}",
                    transform=self.axis_3d.transAxes, va="top", family="monospace",
                )

            times = np.asarray(self.times)
            positions = np.asarray(self.positions)
            angles = np.asarray(self.angles)
            for index, label in enumerate(("X", "Y", "Z")):
                self.axis_position.plot(times, positions[:, index], label=label)
            for index, label in enumerate(("Azimuth", "Elevation", "Roll")):
                self.axis_angles.plot(times, angles[:, index], label=label)

        view_range = self.args.view_range_cm
        self.axis_3d.set(
            xlim=(-view_range, view_range), ylim=(-view_range, view_range),
            zlim=(-view_range, view_range), xlabel="X [cm]", ylabel="Y [cm]", zlabel="Z [cm]",
        )
        self.axis_3d.set_box_aspect((1, 1, 1))
        self.axis_3d.legend(loc="lower left")
        self.axis_position.set_ylabel("Position [cm]")
        self.axis_angles.set_ylabel("Angle [deg]")
        self.axis_angles.set_xlabel("Time [s]")
        self.axis_position.grid(True, alpha=0.3)
        self.axis_angles.grid(True, alpha=0.3)
        self.axis_position.legend(loc="upper right", ncol=3)
        self.axis_angles.legend(loc="upper right", ncol=3)
        if error:
            self.axis_3d.text2D(
                0.02, 0.02, error, transform=self.axis_3d.transAxes,
                color="red", va="bottom", wrap=True,
            )
        if not switch_connected:
            self.axis_angles.text(
                0.02, 0.02, f"Arduino: {switch_status}",
                transform=self.axis_angles.transAxes,
                color="red", va="bottom", wrap=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FOB position and angles in real time")
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--period", type=float, default=0.03)
    parser.add_argument("--range-in", type=float, default=36.0)
    parser.add_argument("--set-pos-angles", action="store_true")
    parser.add_argument("--relative", action="store_true", help="show position relative to first/reset sample")
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--refresh-ms", type=int, default=50)
    parser.add_argument("--view-range-cm", type=float, default=100.0)
    parser.add_argument("--axis-length-cm", type=float, default=10.0)
    parser.add_argument("--demo", action="store_true", help="use generated 6D data without a sensor")
    parser.add_argument("--switch-port", default="/dev/ttyUSB0")
    parser.add_argument("--switch-baud", type=int, default=9600)
    parser.add_argument("--switch-timeout", type=float, default=0.1)
    args = parser.parse_args()
    if args.period <= 0.0 or args.timeout <= 0.0:
        parser.error("--period and --timeout must be positive")
    if args.history < 2 or args.refresh_ms < 1:
        parser.error("--history must be >= 2 and --refresh-ms must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    source = PoseSource(args)
    switch_source = ArduinoSwitchSource(
        args.switch_port, args.switch_baud, args.switch_timeout
    )
    source.start()
    switch_source.start()
    try:
        PoseVisualizer(source, switch_source, args).show()
    finally:
        source.stop()
        switch_source.stop()


if __name__ == "__main__":
    main()
