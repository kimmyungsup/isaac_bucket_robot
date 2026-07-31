"""Read-only omni.ui status window for the mobile bucket controller."""

from __future__ import annotations

import omni.ui as ui


class ControlStatusWindow:
    def __init__(self, robot_name: str, stage_path: str):
        self.window = ui.Window("Mobile Bucket Control Status", width=520, height=720)
        self.labels = {}
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    ui.Label("MOBILE BUCKET CONTROL", style={"font_size": 20})
                    self._label("robot", f"Robot: {robot_name}")
                    self._label("stage", f"Stage: {stage_path}")
                    self._label("slots", "Slots: -")
                    ui.Separator()
                    self._label("mode", "Mode: ARM")
                    self._label("target", "Target: -")
                    self._label("error", "Error: -")
                    self._label("command", "Command: idle")
                    self._label("paint", "Paint: OFF")
                    self._label("fob", "FOB: connection failed")
                    self._label("arduino", "Arduino switch: connection failed")
                    self._label("event", "Last event: ready")
                    ui.Separator()
                    ui.Label("ARM JOINTS (deg)", style={"font_size": 17})
                    self._label("arm_joints", "-")
                    ui.Separator()
                    ui.Label("LADDER JOINTS (deg)", style={"font_size": 17})
                    self._label("ladder_joints", "-")
                    ui.Separator()
                    ui.Label("WHEEL VELOCITIES (rad/s)", style={"font_size": 17})
                    self._label("wheel_joints", "not available")

    def _label(self, key: str, text: str):
        self.labels[key] = ui.Label(text, word_wrap=True)

    def set_event(self, text: str) -> None:
        self.labels["event"].text = f"Last event: {text}"

    def set_robot(self, text: str) -> None:
        self.labels["robot"].text = f"Robot: {text}"

    def set_slots(self, text: str) -> None:
        self.labels["slots"].text = f"Slots: {text}"

    def set_fob_status(self, text: str) -> None:
        self.labels["fob"].text = f"FOB: {text}"

    def set_arduino_status(self, text: str) -> None:
        self.labels["arduino"].text = f"Arduino switch: {text}"

    def show_empty(self, slot: int, mode: str, event: str) -> None:
        """Show an unoccupied multi-robot slot without stale robot data."""
        self.labels["robot"].text = f"Robot: {slot}: EMPTY"
        self.labels["mode"].text = f"Mode: {mode}"
        self.labels["target"].text = "Target: EMPTY"
        self.labels["error"].text = "Error: -"
        self.labels["command"].text = "Command: controls disabled"
        self.labels["paint"].text = "Paint: not available"
        self.labels["arm_joints"].text = "not available"
        self.labels["ladder_joints"].text = "not available"
        self.labels["wheel_joints"].text = "not available"
        self.set_event(event)

    def update(self, *, mode: str, target: str, error: str, command: str,
               paint: str, arm_joints: str, ladder_joints: str,
               wheel_joints: str) -> None:
        values = {
            "mode": f"Mode: {mode}",
            "target": f"Target: {target}",
            "error": f"Error: {error}",
            "command": f"Command: {command}",
            "paint": f"Paint: {paint}",
            "arm_joints": arm_joints,
            "ladder_joints": ladder_joints,
            "wheel_joints": wheel_joints,
        }
        for key, value in values.items():
            self.labels[key].text = value


class ControlHelpWindow:
    """Read-only, dynamically updated keyboard command reference."""

    def __init__(self):
        self.window = ui.Window("Mobile Bucket Control Help", width=610, height=520)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8, height=0):
                    ui.Label("MOBILE BUCKET KEYBOARD HELP", style={"font_size": 20})
                    self.robot_label = ui.Label("Selected: -", word_wrap=True)
                    self.mode_label = ui.Label("Current mode: -", word_wrap=True)
                    ui.Separator()
                    self.help_label = ui.Label("-", word_wrap=True)

    def update(self, robot_name: str, mode: str, lines) -> None:
        self.robot_label.text = f"Selected: {robot_name}"
        self.mode_label.text = f"Current mode: {mode.upper()}"
        self.help_label.text = "\n".join(lines)

    def show_free_camera(self, selection_line: str) -> None:
        self.update(
            "FREE CAMERA",
            "CONTROLS DISABLED",
            [
                selection_line,
                "5: select/regenerate rectangular paint target (F at two corners)",
                "",
                "Robot keyboard controls are disabled.",
                "Select robot 1, 2, or 3 to resume control.",
                "",
                "ESC: quit",
            ],
        )

    def update_context(self, robot_name: str, mode: str, *, wheels: bool = True, multi: bool = False) -> None:
        lines = []
        if multi:
            lines.append("1: humanoid | 2: v4 onlyarm | 3: only_mobile | 4: EMPTY (camera) | 5: EMPTY (target area)")
        lines.append("5: select/regenerate rectangular paint target (F at two corners)")
        if wheels:
            lines.append("B: toggle DRIVE / ARM")
        lines.append("V: toggle ARM / MOBILE")
        lines.append("P: toggle ARM / FOB pose control")
        lines.append("")
        if mode == "drive":
            lines.extend([
                "[DRIVE]",
                "W/S: forward / reverse",
                "A/D: turn left / right",
                "Space: brake",
            ])
        elif mode in ("mobile", "ladder_task"):
            lines.extend([
                "[MOBILE / LADDER]",
                "Arrow Up/Down: body8 X forward / back",
                "Shift + Up/Down: body8 Z up / down",
                "Arrow Left/Right: joint1 turn",
                "Q/A W/S E/D R/F T/G Y/H U/J I/K: joint1..joint8 +/-",
            ])
        else:
            lines.extend([
                "[FOB]" if mode == "fob" else "[ARM]",
                "1/2/3: control right / left / both arms",
                "TAB: switch active arm view",
                "Arrow keys: translate X/Y | Shift + Up/Down: Z",
                "I/K: pitch | J/H: yaw | U/O: roll",
                "R: reset target | F: base/tool frame | L: EE trail",
                "",
                "[PAINT] Z: rectangular paint | Hold X: spray | C: clear",
            ])
        lines.extend(["", "ESC: quit"])
        self.update(robot_name, mode, lines)
