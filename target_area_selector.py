"""Real-time viewport raycast selection of rectangular paint target areas."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import carb
import numpy as np
import omni.appwindow
import omni.physx
import omni.ui as ui
from omni.kit.viewport.utility import get_active_viewport_window
from pxr import Gf, Sdf, Usd, UsdGeom


class PaintTargetAreaSelector:
    """Select two diagonal corners on one configured paint-wall surface."""

    SCOPE_PATH = "/World/PaintTargetArea"
    DEFAULT_TARGET_PRIM_PATHS = (
        "/World/Warehouse01/PaintWall_A",
        "/World/Warehouse01/paintWall_B",
    )
    DEFAULT_TARGET_NAMES = frozenset(("paintwall_a", "paintwall_b"))
    MIN_AREA_SIZE_M = 0.01
    MAX_AREA_SIZE_M = 100.0
    NORMAL_DOT_MIN = 0.995
    PLANE_TOLERANCE_M = 0.01
    COVERAGE_GRID_SIZE = 100

    def __init__(
        self,
        stage,
        event_callback: Optional[Callable[[str], None]] = None,
        target_prim_paths: Optional[Sequence[str]] = None,
    ):
        self.stage = stage
        self.event_callback = event_callback
        self._discover_named_targets = target_prim_paths is None
        self.target_prim_paths = tuple(
            str(Sdf.Path(path)) for path in
            (target_prim_paths or self.DEFAULT_TARGET_PRIM_PATHS)
        )
        self.active = False
        self.first_hit = None
        self.target = None
        self._coverage_mask = None
        self._coverage_centers = None
        self._click_pending = False
        self._hover_hit = None
        self._ray_curve = None
        self._hover_point = None
        self._input = carb.input.acquire_input_interface()
        self._mouse = omni.appwindow.get_default_app_window().get_mouse()
        self._scene_query = omni.physx.get_physx_scene_query_interface()
        UsdGeom.Scope.Define(stage, self.SCOPE_PATH)

    def _refresh_target_prim_paths(self) -> None:
        """Resolve walls by name too, since warehouse variants change parent/case."""
        if not self._discover_named_targets:
            return
        paths = {
            path for path in self.target_prim_paths
            if self.stage.GetPrimAtPath(path).IsValid()
        }
        for prim in self.stage.Traverse():
            if prim.GetName().lower() in self.DEFAULT_TARGET_NAMES:
                paths.add(str(prim.GetPath()))
        self.target_prim_paths = tuple(sorted(paths))

    def _notify(self, text: str) -> None:
        print(f"[TARGET AREA] {text}")
        if self.event_callback is not None:
            self.event_callback(text)

    def begin_selection(self) -> None:
        self.active = True
        self.first_hit = None
        self._click_pending = False
        self._remove_preview()
        self._refresh_target_prim_paths()
        available = [
            path for path in self.target_prim_paths
            if self.stage.GetPrimAtPath(path).IsValid()
        ]
        if available:
            self._notify(
                "TARGET AREA mode: point at two diagonal corners and press F on "
                + ", ".join(available)
                + "."
            )
        else:
            self._notify(
                "TARGET AREA mode: configured paint walls were not found: "
                + ", ".join(self.target_prim_paths)
            )

    def end_selection(self) -> None:
        self.active = False
        self.first_hit = None
        self._click_pending = False
        self._remove_preview()
        self._hide_ray()
        self._hide_hover_point()

    def select_hover_point(self) -> None:
        """Queue the current mouse-ray hit as a corner selection."""
        if self.active:
            self._click_pending = True

    def _clear_viewport_selection(self) -> None:
        window = get_active_viewport_window()
        if window is not None and window.viewport_api is not None:
            window.viewport_api.usd_context.get_selection().set_selected_prim_paths([], False)

    def _target_for_hit_path(self, hit_path: str) -> Optional[str]:
        """Map a collider (often a child Mesh) back to its configured paint wall."""
        if not hit_path:
            return None
        path = str(Sdf.Path(hit_path))
        for target_path in self.target_prim_paths:
            if path == target_path or path.startswith(target_path + "/"):
                return target_path
        return None

    def _viewport_ray(self):
        window = get_active_viewport_window()
        if window is None or window.viewport_api is None:
            return None
        widget = window.frame
        if widget is None or widget.computed_width <= 0 or widget.computed_height <= 0:
            return None
        mouse_x, mouse_y = self._input.get_mouse_coords_normalized(None)
        mouse_x *= ui.Workspace.get_main_window_width()
        mouse_y *= ui.Workspace.get_main_window_height()
        local_x = mouse_x - widget.screen_position_x
        local_y = mouse_y - widget.screen_position_y
        if not (0 <= local_x <= widget.computed_width and 0 <= local_y <= widget.computed_height):
            return None
        ndc_x = (local_x / widget.computed_width - 0.5) * 2.0
        ndc_y = (local_y / widget.computed_height - 0.5) * -2.0
        near = window.viewport_api.ndc_to_world.Transform(Gf.Vec3d(ndc_x, ndc_y, -1.0))
        far = window.viewport_api.ndc_to_world.Transform(Gf.Vec3d(ndc_x, ndc_y, 1.0))
        origin = np.asarray(near, dtype=np.float64)
        direction = np.asarray(far - near, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        return None if norm < 1e-9 else (window, origin, direction / norm)

    def update(self) -> None:
        """Run every rendered frame while TARGET AREA mode is active."""
        if not self.active:
            return
        ray = self._viewport_ray()
        if ray is None:
            self._hover_hit = None
            self._hide_ray()
            self._hide_hover_point()
            return
        window, origin, direction = ray
        # Undo the normal viewport selection continuously in this explicit mode.
        self._clear_viewport_selection()
        hits = []

        def collect_hit(hit):
            collision_path = str(hit.collision or "")
            rigid_body_path = str(hit.rigid_body or "")
            target_path = (
                self._target_for_hit_path(collision_path)
                or self._target_for_hit_path(rigid_body_path)
            )
            if target_path is not None:
                hits.append(
                    (float(hit.distance), target_path, tuple(hit.position), tuple(hit.normal))
                )
            return True

        self._scene_query.raycast_all(
            Gf.Vec3f(*origin), Gf.Vec3f(*direction), 1000.0, collect_hit
        )
        if hits:
            _, path, point, normal = min(hits, key=lambda item: item[0])
            self._hover_hit = self._make_hit(path, point, normal)
            end = np.asarray(point, dtype=np.float64)
        else:
            self._hover_hit = None
            end = origin + direction * 100.0
        self._draw_ray(origin, end, self._hover_hit is not None)
        if self._hover_hit is not None:
            self._draw_hover_point(end, self._hover_hit["normal"])
        else:
            self._hide_hover_point()

        if self._click_pending:
            self._click_pending = False
            if self._hover_hit is None:
                self._notify(
                    "No configured paint-wall collider is under the mouse. "
                    "Check that PaintWall_A/PaintWall_B have collision enabled."
                )
            else:
                self._accept_hit(self._hover_hit)

    def _accept_hit(self, hit) -> None:
        if self.first_hit is None:
            self.first_hit = hit
            self._draw_first_corner(hit["point"], hit["normal"])
            self._notify(f"First corner selected on {hit['prim_path']}. Select the diagonal corner.")
            return
        error = self._validate_pair(self.first_hit, hit)
        if error:
            self._notify(error + " The first corner is unchanged.")
            return
        corners = self._rectangle_corners(self.first_hit, hit)
        if corners is None:
            self._notify("The rectangle is too small or too large. Select the second corner again.")
            return
        self._draw_confirmed_area(corners, hit["normal"])
        self.target = {
            "prim_path": hit["prim_path"],
            "corners": np.asarray(corners, dtype=np.float64),
            "normal": hit["normal"].copy(),
        }
        self._reset_coverage(corners)
        width = float(np.linalg.norm(corners[1] - corners[0]))
        height = float(np.linalg.norm(corners[3] - corners[0]))
        self.first_hit = None
        self._remove_preview()
        self._notify(
            f"Target fixed on {hit['prim_path']} ({width:.3f} m x {height:.3f} m). "
            "TARGET AREA mode remains active."
        )

    def _make_hit(self, prim_path: str, world_position, world_normal):
        if not self.stage.GetPrimAtPath(prim_path).IsValid():
            return None
        point = np.asarray(world_position, dtype=np.float64)
        normal = np.asarray(world_normal, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        return {
            "prim_path": prim_path,
            "point": point,
            "normal": normal / norm,
        }

    @property
    def coverage_percent(self) -> float:
        if self._coverage_mask is None:
            return 0.0
        return 100.0 * float(np.count_nonzero(self._coverage_mask)) / self._coverage_mask.size

    @property
    def coverage_status(self) -> str:
        if self.target is None:
            return "No target area"
        return f"coverage={self.coverage_percent:.1f}%"

    def _reset_coverage(self, corners=None) -> None:
        if corners is None:
            self._coverage_mask = None
            self._coverage_centers = None
            return
        size = self.COVERAGE_GRID_SIZE
        coordinates = (np.arange(size, dtype=np.float64) + 0.5) / size
        u, v = np.meshgrid(coordinates, coordinates, indexing="xy")
        self._coverage_centers = np.column_stack((u.ravel(), v.ravel()))
        self._coverage_mask = np.zeros(size * size, dtype=bool)

    def clear_coverage(self) -> None:
        """Reset coverage while retaining the current target rectangle."""
        if self.target is not None:
            self._reset_coverage(self.target["corners"])

    def record_paint_patch(self, points) -> None:
        """Accumulate the portion of the target covered by a paint polygon."""
        if self.target is None or self._coverage_mask is None:
            return
        corners = self.target["corners"]
        origin = corners[0]
        axis_u = corners[1] - origin
        axis_v = corners[3] - origin
        length_u_sq = float(np.dot(axis_u, axis_u))
        length_v_sq = float(np.dot(axis_v, axis_v))
        polygon = np.asarray(points, dtype=np.float64)
        if length_u_sq < 1e-12 or length_v_sq < 1e-12 or len(polygon) < 3:
            return
        plane_distance = np.abs((polygon - origin) @ self.target["normal"])
        if float(np.max(plane_distance)) > 0.02:
            return
        relative = polygon - origin
        polygon_uv = np.column_stack((
            relative @ axis_u / length_u_sq,
            relative @ axis_v / length_v_sq,
        ))
        x = self._coverage_centers[:, 0]
        y = self._coverage_centers[:, 1]
        inside = np.zeros(len(x), dtype=bool)
        previous = polygon_uv[-1]
        for current in polygon_uv:
            x0, y0 = previous
            x1, y1 = current
            crosses = (y0 > y) != (y1 > y)
            edge_x = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-15) + x0
            inside ^= crosses & (x < edge_x)
            previous = current
        self._coverage_mask |= inside

    def _validate_pair(self, first, second) -> Optional[str]:
        if first["prim_path"] != second["prim_path"]:
            return "Both corners must be on the same configured paint wall."
        if float(np.dot(first["normal"], second["normal"])) < self.NORMAL_DOT_MIN:
            return "Both corners must be on the same paint-wall face."
        plane_error = abs(float(np.dot(second["point"] - first["point"], first["normal"])))
        if plane_error > self.PLANE_TOLERANCE_M:
            return "The second corner is not on the first corner's plane."
        return None

    def _rectangle_corners(self, first, second):
        normal = first["normal"]
        axis_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis_v -= normal * float(np.dot(axis_v, normal))
        if np.linalg.norm(axis_v) < 1e-6:
            axis_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis_v -= normal * float(np.dot(axis_v, normal))
        axis_v /= np.linalg.norm(axis_v)
        axis_u = np.cross(axis_v, normal)
        axis_u /= np.linalg.norm(axis_u)
        p0 = first["point"]
        delta = second["point"] - p0
        du, dv = float(np.dot(delta, axis_u)), float(np.dot(delta, axis_v))
        if not (
            self.MIN_AREA_SIZE_M <= abs(du) <= self.MAX_AREA_SIZE_M
            and self.MIN_AREA_SIZE_M <= abs(dv) <= self.MAX_AREA_SIZE_M
        ):
            return None
        return np.asarray([
            p0,
            p0 + axis_u * du,
            p0 + axis_u * du + axis_v * dv,
            p0 + axis_v * dv,
        ])

    def _draw_ray(self, start, end, valid: bool) -> None:
        if self._ray_curve is None:
            self._ray_curve = UsdGeom.BasisCurves.Define(self.stage, f"{self.SCOPE_PATH}/HoverRay")
            self._ray_curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            self._ray_curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            self._ray_curve.CreateCurveVertexCountsAttr([2])
            self._ray_curve.CreateWidthsAttr([0.012])
            self._ray_curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self._ray_curve.CreatePointsAttr().Set([Gf.Vec3f(*start), Gf.Vec3f(*end)])
        color = Gf.Vec3f(0.1, 1.0, 0.2) if valid else Gf.Vec3f(1.0, 0.25, 0.05)
        self._ray_curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([color])
        self._ray_curve.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([1.0])
        self._ray_curve.GetPrim().SetActive(True)

    def _draw_hover_point(self, point, normal) -> None:
        if self._hover_point is None:
            self._hover_point = UsdGeom.Points.Define(self.stage, f"{self.SCOPE_PATH}/HoverPoint")
            self._hover_point.CreateWidthsAttr([0.05])
            self._hover_point.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                [Gf.Vec3f(0.05, 1.0, 0.15)]
            )
        # Avoid depth fighting with either wall surface.
        visible_point = np.asarray(point) + np.asarray(normal) * 0.008
        self._hover_point.CreatePointsAttr().Set([Gf.Vec3f(*visible_point)])
        self._hover_point.GetPrim().SetActive(True)

    def _hide_hover_point(self) -> None:
        if self._hover_point is not None:
            self._hover_point.GetPrim().SetActive(False)

    def _hide_ray(self) -> None:
        if self._ray_curve is not None:
            self._ray_curve.GetPrim().SetActive(False)

    def _remove_prim(self, path: str) -> None:
        if self.stage.GetPrimAtPath(path).IsValid():
            self.stage.RemovePrim(path)

    def _remove_preview(self) -> None:
        self._remove_prim(f"{self.SCOPE_PATH}/Preview")

    def _draw_first_corner(self, point, normal) -> None:
        self._remove_preview()
        marker = UsdGeom.Points.Define(self.stage, f"{self.SCOPE_PATH}/Preview/FirstCorner")
        marker.CreatePointsAttr([Gf.Vec3f(*(point + normal * 0.006))])
        marker.CreateWidthsAttr([0.045])
        marker.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(1.0, 0.85, 0.05)])

    def _draw_confirmed_area(self, corners, normal) -> None:
        self._remove_prim(f"{self.SCOPE_PATH}/Confirmed")
        points = [Gf.Vec3f(*(point + normal * 0.008)) for point in corners]
        outline = UsdGeom.BasisCurves.Define(self.stage, f"{self.SCOPE_PATH}/Confirmed/Outline")
        outline.CreateTypeAttr(UsdGeom.Tokens.linear)
        outline.CreateWrapAttr(UsdGeom.Tokens.periodic)
        outline.CreatePointsAttr(points)
        outline.CreateCurveVertexCountsAttr([4])
        outline.CreateWidthsAttr([0.018])
        outline.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        outline.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(0.05, 1.0, 0.2)])
        outline.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([1.0])

        markers = UsdGeom.Points.Define(self.stage, f"{self.SCOPE_PATH}/Confirmed/Corners")
        markers.CreatePointsAttr(points)
        markers.CreateWidthsAttr([0.055])
        markers.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
            [Gf.Vec3f(0.05, 1.0, 0.2)]
        )
