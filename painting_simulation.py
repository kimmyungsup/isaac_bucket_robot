"""Modular, visual-only paint and spray simulation for Isaac Sim."""

from __future__ import annotations

from collections import deque
import math
from typing import Optional

import numpy as np
import omni.physx
from pxr import Gf, Sdf, UsdGeom


class PaintSprayVisualizer:
    """Raycast along the corrected nozzle-forward axis and draw non-physical surface patches."""

    RAY_START_OFFSET_M = 0.10
    PAINT_RANGE_M = 0.50
    RAY_HIT_COLOR = (0.1, 1.0, 0.2)
    RAY_MISS_COLOR = (1.0, 0.85, 0.05)

    def __init__(
        self,
        stage,
        robot_prim_path: str,
        scope_path: str = "/World/PaintSimulation",
        nozzle_axis=(0.0, 1.0, 0.0),
    ):
        self.stage = stage
        root_path = Sdf.Path(robot_prim_path)
        # Single-robot mode passes the articulation root link; multi-robot mode
        # passes the robot scope itself.
        self.robot_scope_path = str(
            root_path.GetParentPath() if root_path.name == "base_mobile" else root_path
        ).rstrip("/")
        self.scope_path = scope_path
        UsdGeom.Scope.Define(stage, scope_path)
        self._query = omni.physx.get_physx_scene_query_interface()
        self._paths = deque()
        self._last_points = {}
        self._frame = 0
        self._serial = 0
        self.max_patches = 2500
        self._ray_curves = {}
        self.coverage_callback = None
        self.coverage_clear_callback = None
        self.nozzle_axis = Gf.Vec3d(*nozzle_axis)

    def clear(self) -> None:
        for path in list(self._paths):
            self.stage.RemovePrim(path)
        self._paths.clear()
        self._last_points.clear()
        if self.coverage_clear_callback is not None:
            self.coverage_clear_callback()

    @staticmethod
    def _unit(value) -> Optional[np.ndarray]:
        vector = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else None

    def _raycast(self, matrix, max_distance: float):
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotation()
        direction = self._unit(rotation.TransformDir(self.nozzle_axis))
        tool_long = self._unit(rotation.TransformDir(Gf.Vec3d(-1.0, 0.0, 0.0)))
        if direction is None or tool_long is None:
            return None, None, None
        minimum_distance = self.RAY_START_OFFSET_M
        origin = np.asarray(translation, dtype=np.float64) + direction * minimum_distance
        candidates = []

        def collect_hit(hit):
            hit_path = str(hit.rigid_body or hit.collision)
            is_robot = hit_path == self.robot_scope_path or hit_path.startswith(self.robot_scope_path + "/")
            total_distance = minimum_distance + float(hit.distance)
            if not is_robot and total_distance > minimum_distance + 1e-4:
                candidates.append((total_distance, hit))
            return True

        self._query.raycast_all(
            Gf.Vec3f(*origin), Gf.Vec3f(*direction),
            float(max(0.0, max_distance - minimum_distance)), collect_hit,
        )
        if not candidates:
            return None, origin, np.asarray(translation, dtype=np.float64) + direction * max_distance
        _, hit = min(candidates, key=lambda item: item[0])
        point = np.asarray(hit.position, dtype=np.float64)
        normal = self._unit(hit.normal)
        if normal is None:
            return None, origin, point
        # Paint walls only: reject surfaces whose normal is mostly vertical.
        if abs(float(normal[2])) > 0.35:
            return None, origin, point
        # Reject grazing/back-facing contacts and orient the patch toward the nozzle.
        incidence = float(np.dot(direction, normal))
        if abs(incidence) < 0.35:
            return None, origin, point
        if incidence > 0.0:
            normal = -normal
        long_axis = tool_long - normal * float(np.dot(tool_long, normal))
        long_axis = self._unit(long_axis)
        if long_axis is None:
            long_axis = self._unit(np.array([0.0, 0.0, 1.0]) - normal * normal[2])
        if long_axis is None:
            return None, origin, point
        short_axis = self._unit(np.cross(normal, long_axis))
        if short_axis is None:
            return None, origin, point
        return (point + normal * 0.002, normal, long_axis, short_axis), origin, point

    def _show_raycast(self, arm_name: str, start, end, hit_valid: bool) -> None:
        path = f"{self.scope_path}/raycast_{arm_name}"
        curve = self._ray_curves.get(arm_name)
        if curve is None:
            curve = UsdGeom.BasisCurves.Define(self.stage, path)
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            curve.CreateWidthsAttr([0.008])
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            curve.CreateCurveVertexCountsAttr([2])
            self._ray_curves[arm_name] = curve
        curve.CreatePointsAttr().Set([Gf.Vec3f(*start), Gf.Vec3f(*end)])
        color = self.RAY_HIT_COLOR if hit_valid else self.RAY_MISS_COLOR
        curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        curve.GetPrim().SetActive(True)

    def _hide_raycast(self, arm_name: str) -> None:
        curve = self._ray_curves.get(arm_name)
        if curve is not None:
            curve.GetPrim().SetActive(False)

    def _allow_spacing(self, key: str, point: np.ndarray, minimum: float) -> bool:
        previous = self._last_points.get(key)
        if previous is not None and np.linalg.norm(point - previous) < minimum:
            return False
        self._last_points[key] = point.copy()
        return True

    def _create_patch(self, points, color, opacity: float, label: str) -> None:
        self._serial += 1
        path = f"{self.scope_path}/{label}_{self._serial:06d}"
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        mesh.CreateFaceVertexCountsAttr([len(points)])
        mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(opacity)])
        self._paths.append(Sdf.Path(path))
        if self.coverage_callback is not None:
            self.coverage_callback(points)
        while len(self._paths) > self.max_patches:
            self.stage.RemovePrim(self._paths.popleft())

    def _draw_rectangle(self, arm_name: str, hit) -> None:
        center, _, long_axis, short_axis = hit
        if not self._allow_spacing(f"rectangle:{arm_name}", center, 0.018):
            return
        half_long, half_short = 0.060, 0.020
        points = [
            center - long_axis * half_long - short_axis * half_short,
            center + long_axis * half_long - short_axis * half_short,
            center + long_axis * half_long + short_axis * half_short,
            center - long_axis * half_long + short_axis * half_short,
        ]
        self._create_patch(points, (0.08, 0.45, 1.0), 0.72, "paint")

    def _draw_spray(self, arm_name: str, hit) -> None:
        center, _, axis_v, axis_u = hit
        if not self._allow_spacing(f"spray:{arm_name}", center, 0.012):
            return
        radius = 0.050
        points = []
        for index in range(20):
            angle = 2.0 * math.pi * index / 20.0
            points.append(center + radius * (math.cos(angle) * axis_u + math.sin(angle) * axis_v))
        self._create_patch(points, (1.0, 0.24, 0.05), 0.48, "spray")

    def update(self, arm_name: str, tool_matrix, paint_enabled: bool, spray_active: bool) -> None:
        self._frame += 1
        active = paint_enabled or spray_active
        hit = None
        if active:
            hit, ray_start, ray_end = self._raycast(tool_matrix, self.PAINT_RANGE_M)
            if ray_start is not None and ray_end is not None:
                self._show_raycast(arm_name, ray_start, ray_end, hit is not None)
        else:
            self._hide_raycast(arm_name)

        if paint_enabled and self._frame % 3 == 0:
            if hit is not None:
                self._draw_rectangle(arm_name, hit)
        if spray_active and self._frame % 2 == 0:
            if hit is not None:
                self._draw_spray(arm_name, hit)
