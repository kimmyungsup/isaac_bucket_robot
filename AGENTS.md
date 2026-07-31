# Repository Guidelines

## Project Structure & Module Organization

The main entry point is `combined_mobile_keyboard_control.py`; it runs a selected single robot or delegates multi-robot operation to `multi_robot_control.py`. Shared behavior is separated into `ladder_kinematics.py`, `painting_simulation.py`, `target_area_selector.py`, and `control_status_ui.py`.

FOB tracking, serial-switch support, and the Arduino Nano sketch live in `FOB/`. Robot descriptions and meshes are under `humanoid_urdf_assemble/`, while root-level `mobile_bucket_*.usd`, YAML descriptors, and `*_kinematics_info.json` files are runtime assets. `Warehouse/` contains environment assets. Treat `legacy/` and `old/` as archival material; do not import from or edit them for active features.

## Build, Test, and Development Commands

Run commands from the repository root because asset paths are relative.

```bash
uv sync
uv run python combined_mobile_keyboard_control.py --robot humanoid_base
uv run python combined_mobile_keyboard_control.py --robot v4_onlyarm
uv run python combined_mobile_keyboard_control.py --multi
uv run python -m py_compile combined_mobile_keyboard_control.py multi_robot_control.py FOB/*.py
uv run python humanoid_kinematics_test.py
uv run python onlyarm_kinematics_test.py
```

`uv sync` installs the pinned Python 3.11/Isaac Sim 5.1 environment. The simulation commands require a working NVIDIA/Vulkan GUI environment. Use `FOB/arduino_switch_reader.py` to diagnose `/dev/ttyUSB0` input independently.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Add type hints to reusable interfaces and keep hardware I/O non-blocking and fail-safe. Keep robot-specific paths and tuning values in configuration dictionaries, YAML, or JSON rather than duplicating control loops. No formatter or linter is currently configured; preserve the surrounding style and run `py_compile` before submitting.

## Testing Guidelines

Tests are executable Python scripts named `*_test.py`; no coverage threshold is configured. Test pure parsing and kinematics separately, then perform a short single-robot and multi-robot startup check. Hardware changes should document port, baud rate, wiring, and a manual hold/release test. Do not commit generated `__pycache__` files.

## Commit & Pull Request Guidelines

No Git history is available in this checkout. Use concise imperative commits such as `Fix continuous Arduino spray input`. Keep each commit focused. Pull requests should describe affected robot modes, list commands run, identify changed USD/URDF/config assets, and include screenshots or a short recording for UI, motion, or paint-visualization changes. Link relevant issues and call out required hardware.

## Safety & Configuration

Do not commit device credentials or machine-specific secrets. Preserve the default serial assignments (`/dev/ttyUSB0` switch, `/dev/ttyUSB1` FOB) unless configuration support is added. Back up material USD/URDF changes before conversion, and place obsolete copies in `legacy/`.
