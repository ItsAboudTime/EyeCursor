"""
Stereo demo: control the mouse cursor with head pose + wink gestures using two cameras.

Pipeline:
left+right frames -> MediaPipe landmarks -> triangulation of important points ->
yaw/pitch/depth estimation -> cursor + gesture control.

Edit the calibration variables below with your stereo calibration output:
    K1, D1, K2, D2, R, T

Keyboard controls are handled by the Tkinter window:
  - q / Esc: quit
  - c: calibrate current head pose as center
"""

import sys
import threading
import queue
import time

import cv2
import numpy as np

from src.cursor import create_cursor
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.stereo_face_analysis import StereoCalibration, StereoFaceAnalysisPipeline
from src.face_tracking.providers.face_landmarks import FaceLandmarksProvider
from src.ui.settings import SettingsWindow


LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 0
BASELINE_METERS = 0.0788

K1 = np.array([
    [542.975661, 0.000000, 347.621721],
    [0.000000, 542.580855, 266.597383],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D1 = np.array([
    [0.139821, -0.092846, 0.013859, 0.018398, -1.438426],
], dtype=np.float64)

K2 = np.array([
    [550.591389, 0.000000, 354.946646],
    [0.000000, 547.426744, 257.201464],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D2 = np.array([
    [0.061811, 0.096200, 0.009729, 0.016548, -0.325093],
], dtype=np.float64)

R = np.array([
    [0.999955, -0.009529, -0.000044],
    [0.009528, 0.999776, 0.018882],
    [-0.000136, -0.018881, 0.999822],
], dtype=np.float64)

T = np.array([
    [-0.078688],
    [-0.000478],
    [0.004312],
], dtype=np.float64)

BOTH_EYES_SQUINT_SCROLL_THRESHOLD = 0.3
BOTH_EYES_OPEN_SCROLL_THRESHOLD = 0.65
EYE_SCROLL_HOLD_SECONDS = 1.0
EYE_SCROLL_DELTA = 120
WINK_EYE_CLOSED_THRESHOLD = 0.3
WINK_EYE_OPEN_THRESHOLD = 0.3
VIS_WINDOW_NAME = "Stereo Pipeline Visualization"

VIS_LANDMARK_INDICES = [1, 10, 33, 133, 145, 152, 159, 160, 234, 263, 362, 373, 374, 385, 386, 387, 454]
VIS_MAIN_LANDMARK_INDICES = [1, 10, 152, 234, 454]
VIS_3D_X_RANGE_METERS = 0.25
VIS_3D_Y_RANGE_METERS = 0.25
VIS_3D_Z_RANGE_METERS = 0.30
VIS_3D_Z_X_SHEAR = 0.08
VIS_3D_Z_Y_SHEAR = -0.04
VIS_POINT_LABELS = {
    1: "front",
    10: "top",
    152: "chin",
    234: "left",
    454: "right",
}


class StereoProcessVisualizer:
    """Renders a stereo processing dashboard without changing control behavior."""

    def __init__(self, calibration):
        self._calibration = calibration
        self._p1 = np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
        self._p2 = np.hstack((self._calibration.r, self._calibration.t))
        self._show_landmark_ids = False
        self._points_3d_ema = None
        self._world_reference = None
        self._axis_sign_x = 1.0
        self._axis_sign_y = 1.0
        self._axis_sign_z = 1.0
        self._axis_sign_ema_x = 1.0
        self._axis_sign_ema_y = 1.0
        self._axis_sign_ema_z = 1.0
        self._view_zoom_3d = 0.9

    def draw(
        self,
        left_frame,
        right_frame,
        left_landmarks,
        right_landmarks,
        face_analysis,
    ):
        left_raw = left_frame.copy()
        right_raw = right_frame.copy()
        left_overlay = left_frame.copy()
        right_overlay = right_frame.copy()

        points_3d = self._triangulate(left_landmarks, right_landmarks, left_frame.shape, right_frame.shape)
        self._points_3d_ema = self._smooth_points(points_3d)

        left_pts = self._extract_pixel_points(left_landmarks, left_frame.shape)
        right_pts = self._extract_pixel_points(right_landmarks, right_frame.shape)

        self._draw_landmarks(left_overlay, left_pts, "LEFT + landmarks")
        self._draw_landmarks(right_overlay, right_pts, "RIGHT + landmarks")

        displacement_view = self._build_displacement_view(left_frame, right_frame, left_pts, right_pts)
        points3d_view = self._build_3d_view(self._points_3d_ema)
        telemetry_view = self._build_telemetry_view(face_analysis, self._points_3d_ema)

        top_row = self._hstack_fit([left_raw, right_raw], target_h=320)
        mid_row = self._hstack_fit([left_overlay, right_overlay], target_h=320)
        bottom_row = self._hstack_fit([displacement_view, points3d_view, telemetry_view], target_h=320)

        dashboard = self._vstack_fit([top_row, mid_row, bottom_row])
        return dashboard

    def handle_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "QUIT"
        if key == ord("c"):
            return "CALIBRATE"
        if key == ord("i"):
            self._show_landmark_ids = not self._show_landmark_ids
        return None

    def close(self):
        try:
            cv2.destroyWindow(VIS_WINDOW_NAME)
        except cv2.error:
            pass

    def _extract_pixel_points(self, landmarks, frame_shape):
        if landmarks is None:
            return {}
        frame_h, frame_w = frame_shape[:2]
        points = {}
        for idx in VIS_LANDMARK_INDICES:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            x = int(np.clip(float(lm.x) * frame_w, 0, frame_w - 1))
            y = int(np.clip(float(lm.y) * frame_h, 0, frame_h - 1))
            points[idx] = (x, y)
        return points

    def _triangulate(self, left_landmarks, right_landmarks, left_shape, right_shape):
        if left_landmarks is None or right_landmarks is None:
            return {}

        left_h, left_w = left_shape[:2]
        right_h, right_w = right_shape[:2]

        left_points = []
        right_points = []
        valid_indices = []

        for idx in VIS_LANDMARK_INDICES:
            if idx >= len(left_landmarks) or idx >= len(right_landmarks):
                continue
            ll = left_landmarks[idx]
            rr = right_landmarks[idx]
            left_points.append([float(ll.x) * left_w, float(ll.y) * left_h])
            right_points.append([float(rr.x) * right_w, float(rr.y) * right_h])
            valid_indices.append(idx)

        if len(valid_indices) < 5:
            return {}

        left_arr = np.asarray(left_points, dtype=np.float64).reshape(-1, 1, 2)
        right_arr = np.asarray(right_points, dtype=np.float64).reshape(-1, 1, 2)

        undist_left = cv2.undistortPoints(left_arr, self._calibration.k1, self._calibration.d1).reshape(-1, 2)
        undist_right = cv2.undistortPoints(right_arr, self._calibration.k2, self._calibration.d2).reshape(-1, 2)

        points_4d = cv2.triangulatePoints(self._p1, self._p2, undist_left.T, undist_right.T)
        points_3d = (points_4d[:3] / (points_4d[3] + 1e-9)).T
        return {idx: points_3d[i] for i, idx in enumerate(valid_indices)}

    def _smooth_points(self, points_3d):
        if not points_3d:
            return points_3d
        if self._points_3d_ema is None:
            return {idx: p.copy() for idx, p in points_3d.items()}

        alpha = 0.2
        out = {}
        for idx, point in points_3d.items():
            prev = self._points_3d_ema.get(idx)
            if prev is None:
                out[idx] = point.copy()
            else:
                out[idx] = alpha * point + (1.0 - alpha) * prev
        return out

    def _draw_landmarks(self, frame, points, title):
        for idx, (x, y) in points.items():
            cv2.circle(frame, (x, y), 3, (20, 240, 20), -1)
            if idx in VIS_POINT_LABELS:
                cv2.putText(
                    frame,
                    VIS_POINT_LABELS[idx],
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            if self._show_landmark_ids:
                cv2.putText(
                    frame,
                    str(idx),
                    (x + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(frame, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 40), 2, cv2.LINE_AA)

    def _build_displacement_view(self, left_frame, right_frame, left_pts, right_pts):
        canvas_h = 1000
        canvas_w = 1600
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cv2.putText(
            canvas,
            "Stereo displacement (zoomed landmark space)",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (255, 230, 40),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "L=cyan, R=magenta, line=pair displacement",
            (24, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (195, 195, 195),
            2,
            cv2.LINE_AA,
        )

        pairs = []
        for idx in VIS_MAIN_LANDMARK_INDICES:
            if idx in left_pts and idx in right_pts:
                pairs.append((idx, left_pts[idx], right_pts[idx]))

        if not pairs:
            cv2.putText(
                canvas,
                "No paired landmarks detected",
                (24, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (120, 180, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        all_x = []
        all_y = []
        for _, (lx, ly), (rx, ry) in pairs:
            all_x.extend([float(lx), float(rx)])
            all_y.extend([float(ly), float(ry)])

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)

        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)

        margin_left = 60.0
        margin_right = 60.0
        margin_top = 130.0
        margin_bottom = 40.0
        usable_w = max(1.0, canvas_w - margin_left - margin_right)
        usable_h = max(1.0, canvas_h - margin_top - margin_bottom)

        # Zoom so the landmark cloud almost fills the tab while keeping aspect ratio.
        fit_scale = 0.9 * min(usable_w / span_x, usable_h / span_y)
        scaled_w = span_x * fit_scale
        scaled_h = span_y * fit_scale
        offset_x = margin_left + 0.5 * (usable_w - scaled_w)
        offset_y = margin_top + 0.5 * (usable_h - scaled_h)

        def t(pt):
            x, y = pt
            tx = int(round(offset_x + (float(x) - min_x) * fit_scale))
            ty = int(round(offset_y + (float(y) - min_y) * fit_scale))
            return tx, ty

        for idx, left_pt, right_pt in pairs:
            lx, ly = left_pt
            rx, ry = right_pt
            ltx, lty = t(left_pt)
            rtx, rty = t(right_pt)

            dx = float(lx - rx)
            dy = float(ly - ry)
            disp_px = float(np.hypot(dx, dy))

            left_color = (255, 230, 40)
            right_color = (255, 0, 255)
            pair_color = (70, 210, 255) if dx >= 0 else (120, 150, 255)

            cv2.line(canvas, (ltx, lty), (rtx, rty), pair_color, 3, cv2.LINE_AA)
            cv2.circle(canvas, (ltx, lty), 10, left_color, -1)
            cv2.circle(canvas, (rtx, rty), 10, right_color, -1)
            cv2.circle(canvas, (ltx, lty), 12, (0, 0, 0), 2)
            cv2.circle(canvas, (rtx, rty), 12, (0, 0, 0), 2)

            mid_x = (ltx + rtx) // 2
            mid_y = (lty + rty) // 2
            point_name = VIS_POINT_LABELS.get(idx, f"idx{idx}")
            label = f"{point_name} ({idx})  dx={dx:+.1f}px  dy={dy:+.1f}px  d={disp_px:.1f}px"

            text_org = (max(14, min(canvas_w - 420, mid_x + 10)), max(145, mid_y - 8))
            cv2.putText(canvas, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

        return canvas

    def _build_3d_view(self, points_3d):
        canvas_h = 1000
        canvas_w = 1200
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cv2.putText(
            canvas,
            "Triangulated 3D points (fixed world frame)",
            (22, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 230, 40),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "X=red  Y=green  Z=orange  (stable axes, no auto-fit)",
            (22, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

        margin_left = 100.0
        margin_right = 100.0
        margin_top = 150.0
        margin_bottom = 80.0
        usable_w = max(1.0, canvas_w - margin_left - margin_right)
        usable_h = max(1.0, canvas_h - margin_top - margin_bottom)

        center_x = margin_left + 0.5 * usable_w
        center_y = margin_top + 0.5 * usable_h

        range_x = VIS_3D_X_RANGE_METERS
        range_y = VIS_3D_Y_RANGE_METERS
        range_z = VIS_3D_Z_RANGE_METERS
        px_per_meter_x = (0.1 * usable_w) / max(1e-6, range_x)
        px_per_meter_y = (0.1 * usable_h) / max(1e-6, range_y)
        px_per_meter_z = 0.5 * (px_per_meter_x + px_per_meter_y)

        available = []
        if points_3d:
            for idx in VIS_MAIN_LANDMARK_INDICES:
                p = points_3d.get(idx)
                if p is not None:
                    available.append(np.asarray(p, dtype=np.float64))

        # Anchor world coordinates once so the view does not re-center each frame.
        if available and self._world_reference is None:
            stacked = np.vstack(available)
            self._world_reference = np.mean(stacked, axis=0)

        self._update_axis_signs(points_3d)

        if self._world_reference is None:
            self._world_reference = np.zeros(3, dtype=np.float64)

        world_reference = np.asarray(self._world_reference, dtype=np.float64)

        max_abs_x = 0.0
        max_abs_y = 0.0
        max_abs_z = 0.0
        for p in available:
            rel = np.asarray(p, dtype=np.float64) - world_reference
            max_abs_x = max(max_abs_x, abs(float(self._axis_sign_x * rel[0])))
            max_abs_y = max(max_abs_y, abs(float(self._axis_sign_y * rel[1])))
            max_abs_z = max(max_abs_z, abs(float(self._axis_sign_z * rel[2])))

        # Auto-adjust zoom so points remain inside a larger safe border while still readable.
        if available:
            projected_half_w = max_abs_x * px_per_meter_x + abs(VIS_3D_Z_X_SHEAR) * max_abs_z * px_per_meter_z
            projected_half_h = max_abs_y * px_per_meter_y + abs(VIS_3D_Z_Y_SHEAR) * max_abs_z * px_per_meter_z

            safe_half_w = 0.40 * usable_w
            safe_half_h = 0.40 * usable_h
            target_zoom_w = safe_half_w / max(projected_half_w, 1e-6)
            target_zoom_h = safe_half_h / max(projected_half_h, 1e-6)
            target_zoom = float(np.clip(min(target_zoom_w, target_zoom_h), 0.35, 1.6))
            self._view_zoom_3d = 0.9 * self._view_zoom_3d + 0.1 * target_zoom

        def project(p):
            rel = np.asarray(p, dtype=np.float64) - world_reference
            x_m = float(self._axis_sign_x * rel[0])
            y_m = float(self._axis_sign_y * rel[1])
            z_m = float(self._axis_sign_z * rel[2])

            # Intuitive mapping: +X right, -Y up, +Z adds mild fixed oblique shear.
            u = center_x + self._view_zoom_3d * (x_m * px_per_meter_x + (VIS_3D_Z_X_SHEAR * z_m * px_per_meter_z))
            v = center_y - self._view_zoom_3d * (y_m * px_per_meter_y - (VIS_3D_Z_Y_SHEAR * z_m * px_per_meter_z))
            return int(round(u)), int(round(v)), z_m

        safe_left = int(center_x - 0.40 * usable_w)
        safe_right = int(center_x + 0.40 * usable_w)
        safe_top = int(center_y - 0.40 * usable_h)
        safe_bottom = int(center_y + 0.40 * usable_h)
        cv2.rectangle(canvas, (safe_left, safe_top), (safe_right, safe_bottom), (50, 50, 50), 1, cv2.LINE_AA)

        axis_len = 0.04
        origin = world_reference
        axis_points = {
            "X": world_reference + np.array([axis_len, 0.0, 0.0]),
            "Y": world_reference + np.array([0.0, axis_len, 0.0]),
            "Z": world_reference + np.array([0.0, 0.0, axis_len]),
        }
        o2x, o2y, _ = project(origin)
        for label, endpoint in axis_points.items():
            p2x, p2y, _ = project(endpoint)
            color = (80, 80, 255) if label == "X" else (80, 255, 80) if label == "Y" else (255, 160, 80)
            cv2.arrowedLine(canvas, (o2x, o2y), (p2x, p2y), color, 3, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(canvas, label, (p2x + 6, p2y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, (p2x + 6, p2y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        if points_3d:
            for idx in VIS_MAIN_LANDMARK_INDICES:
                p = points_3d.get(idx)
                if p is None:
                    continue
                u, v, z_rel = project(p)
                if 0 <= u < canvas_w and 0 <= v < canvas_h:
                    color = (0, 255, 255)
                    # Make closer/farther depth visible without rotating the whole frame.
                    radius = int(np.clip(12 - 18.0 * z_rel, 8, 16))

                    cv2.circle(canvas, (u, v), radius, color, -1)
                    cv2.circle(canvas, (u, v), radius + 2, (0, 0, 0), 2)

                    name = VIS_POINT_LABELS.get(idx, f"idx{idx}")
                    label = f"{name} ({idx}) z={z_rel:+.3f}m"
                    lx = int(np.clip(u + 12, 8, canvas_w - 240))
                    ly = int(np.clip(v - 12, 95, canvas_h - 12))
                    cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "Directions: left<-X->right, up<-Y->down, depth via z label/size",
            (22, canvas_h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )

        sign_text = f"axis signs: X={self._axis_sign_x:+.0f} Y={self._axis_sign_y:+.0f} Z={self._axis_sign_z:+.0f} zoom={self._view_zoom_3d:.2f}"
        cv2.putText(canvas, sign_text, (22, canvas_h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 2, cv2.LINE_AA)

        return canvas

    def _update_axis_signs(self, points_3d):
        if not points_3d:
            return

        left = points_3d.get(234)
        right = points_3d.get(454)
        top = points_3d.get(10)
        chin = points_3d.get(152)
        front = points_3d.get(1)

        if left is not None and right is not None:
            dx_lr = float(right[0] - left[0])
            if abs(dx_lr) > 1e-6:
                # Right landmark should map to right side on screen.
                candidate_x = 1.0 if dx_lr > 0.0 else -1.0
                self._axis_sign_ema_x = 0.9 * self._axis_sign_ema_x + 0.1 * candidate_x

        if top is not None and chin is not None:
            dy_ct = float(chin[1] - top[1])
            if abs(dy_ct) > 1e-6:
                # Chin should appear below top landmark.
                candidate_y = -1.0 if dy_ct > 0.0 else 1.0
                self._axis_sign_ema_y = 0.9 * self._axis_sign_ema_y + 0.1 * candidate_y

        if front is not None and left is not None and right is not None:
            cheeks_z = 0.5 * float(left[2] + right[2])
            dz = float(front[2] - cheeks_z)
            if abs(dz) > 1e-6:
                # Nose/front should read as more "forward" than cheeks.
                candidate_z = -1.0 if dz > 0.0 else 1.0
                self._axis_sign_ema_z = 0.9 * self._axis_sign_ema_z + 0.1 * candidate_z

        self._axis_sign_x = 1.0 if self._axis_sign_ema_x >= 0.0 else -1.0
        self._axis_sign_y = 1.0 if self._axis_sign_ema_y >= 0.0 else -1.0
        self._axis_sign_z = 1.0 if self._axis_sign_ema_z >= 0.0 else -1.0

    def _build_telemetry_view(self, face_analysis, points_3d):
        canvas = np.zeros((640, 500, 3), dtype=np.uint8)
        cv2.putText(canvas, "Head pose + depth telemetry", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 230, 40), 2, cv2.LINE_AA)

        y = 72
        step = 34

        def put_line(text, color=(230, 230, 230)):
            nonlocal y
            cv2.putText(canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            y += step

        if face_analysis is None or face_analysis.angles is None:
            put_line("Face not detected", (120, 180, 255))
            return canvas

        yaw, pitch = face_analysis.angles
        depth = face_analysis.depth

        put_line(f"yaw   : {yaw:+7.2f} deg")
        put_line(f"pitch : {pitch:+7.2f} deg")
        put_line(f"depth : {depth:+7.3f} m" if depth is not None else "depth : n/a")

        if face_analysis.screen_position is not None:
            sx, sy = face_analysis.screen_position
            put_line(f"screen: ({sx}, {sy})")
        else:
            put_line("screen: n/a")

        put_line(f"left EAR : {face_analysis.left_eye_ratio:.4f}" if face_analysis.left_eye_ratio is not None else "left EAR : n/a")
        put_line(f"right EAR: {face_analysis.right_eye_ratio:.4f}" if face_analysis.right_eye_ratio is not None else "right EAR: n/a")
        put_line(f"wink: {face_analysis.wink_direction or 'none'}")

        if points_3d:
            front = points_3d.get(1)
            if front is not None:
                put_line(f"front xyz: ({front[0]:+.4f}, {front[1]:+.4f}, {front[2]:+.4f})", (160, 255, 210))

        cv2.putText(
            canvas,
            "Keys: q/Esc quit, c calibrate, i IDs",
            (14, 620),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _hstack_fit(self, images, target_h=320):
        resized = []
        for img in images:
            h, w = img.shape[:2]
            if h != target_h:
                scale = target_h / float(h)
                img = cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)
            resized.append(img)
        return np.hstack(resized)

    def _vstack_fit(self, rows):
        if not rows:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        max_w = max(row.shape[1] for row in rows)
        normalized_rows = [self._pad_to_width(row, max_w) for row in rows]
        return np.vstack(normalized_rows)

    def _pad_to_width(self, image, target_w):
        h, w = image.shape[:2]
        if w == target_w:
            return image

        pad = np.zeros((h, target_w - w, 3), dtype=image.dtype)
        return np.hstack([image, pad])


def run_tracking_loop(cursor, stop_queue, control_queue, visualization_queue):
    left_camera = cv2.VideoCapture(LEFT_CAMERA_INDEX)
    right_camera = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

    if not left_camera.isOpened() or not right_camera.isOpened():
        print(
            "Could not open stereo cameras "
            f"(left={LEFT_CAMERA_INDEX}, right={RIGHT_CAMERA_INDEX})."
        )
        stop_queue.put("QUIT")
        return

    try:
        stereo_calibration = StereoCalibration(
            k1=K1,
            d1=D1,
            k2=K2,
            d2=D2,
            r=R,
            t=T,
        )

        analysis_pipeline = StereoFaceAnalysisPipeline(
            stereo_calibration=stereo_calibration,
            yaw_span=40.0,
            pitch_span=20.0,
            ema_alpha=0.08,
            wink_closed_threshold=WINK_EYE_CLOSED_THRESHOLD,
            wink_open_threshold=WINK_EYE_OPEN_THRESHOLD,
        )
        left_vis_provider = FaceLandmarksProvider()
        right_vis_provider = FaceLandmarksProvider()
        visualizer = StereoProcessVisualizer(stereo_calibration)
    except Exception as exc:
        print(f"Failed to initialize stereo analysis pipeline: {exc}")
        left_camera.release()
        right_camera.release()
        stop_queue.put("QUIT")
        return

    minx, miny, maxx, maxy = cursor.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    gesture_controller = GestureController(
        cursor=cursor,
        hold_trigger_seconds=1.0,
        release_missed_frames=5,
        both_eyes_open_threshold=BOTH_EYES_OPEN_SCROLL_THRESHOLD,
        both_eyes_squint_threshold=BOTH_EYES_SQUINT_SCROLL_THRESHOLD,
        scroll_trigger_seconds=EYE_SCROLL_HOLD_SECONDS,
        scroll_delta=EYE_SCROLL_DELTA,
    )
    latest_head_angles = None

    print("Stereo Head+Wink Cursor demo running.")
    print(
        "Focus the settings window and press 'c' to calibrate, 'q' or Esc to quit. "
        f"Using left={LEFT_CAMERA_INDEX}, right={RIGHT_CAMERA_INDEX}."
    )

    try:
        while True:
            should_quit = False
            while not control_queue.empty():
                cmd = control_queue.get()
                if cmd == "QUIT":
                    should_quit = True
                    break
                if cmd == "CALIBRATE" and latest_head_angles is not None:
                    analysis_pipeline.calibrate_to_center(*latest_head_angles)

            if should_quit:
                stop_queue.put("QUIT")
                break

            ok_left, left_frame = left_camera.read()
            ok_right, right_frame = right_camera.read()
            if not ok_left or not ok_right:
                continue

            left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

            face_analysis = analysis_pipeline.analyze(
                left_rgb_frame=left_rgb,
                right_rgb_frame=right_rgb,
                left_frame_width=left_frame.shape[1],
                left_frame_height=left_frame.shape[0],
                right_frame_width=right_frame.shape[1],
                right_frame_height=right_frame.shape[0],
                screen_width=screen_w,
                screen_height=screen_h,
            )

            left_obs = left_vis_provider.get_primary_face_observation(left_rgb)
            right_obs = right_vis_provider.get_primary_face_observation(right_rgb)

            dashboard = visualizer.draw(
                left_frame=left_frame,
                right_frame=right_frame,
                left_landmarks=left_obs.landmarks if left_obs is not None else None,
                right_landmarks=right_obs.landmarks if right_obs is not None else None,
                face_analysis=face_analysis,
            )

            if visualization_queue.full():
                try:
                    visualization_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                visualization_queue.put_nowait(dashboard)
            except queue.Full:
                pass

            if face_analysis is not None:
                latest_head_angles = face_analysis.angles
                gesture_controller.handle_face_analysis(face_analysis, now=time.time())

    finally:
        gesture_controller.shutdown()
        left_camera.release()
        right_camera.release()
        analysis_pipeline.release()
        left_vis_provider.release()
        right_vis_provider.release()


def main():
    cursor = create_cursor()
    message_queue = queue.Queue()
    control_queue = queue.Queue()
    visualization_queue = queue.Queue(maxsize=1)

    def resize_keep_aspect(image, target_w, target_h):
        if image is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    try:
        root = SettingsWindow.create_app(cursor=cursor)
    except Exception as e:
        print(f"Fatal Error: Could not start Tkinter: {e}")
        return 1

    def send_quit(_event=None):
        control_queue.put("QUIT")

    def send_calibrate(_event=None):
        control_queue.put("CALIBRATE")

    root.bind("q", send_quit)
    root.bind("<Escape>", send_quit)
    root.bind("c", send_calibrate)
    root.protocol("WM_DELETE_WINDOW", send_quit)

    def check_queue():
        try:
            message = message_queue.get_nowait()
            if message == "QUIT":
                try:
                    cv2.destroyWindow(VIS_WINDOW_NAME)
                except cv2.error:
                    pass
                root.quit()
                root.destroy()
                return
        except queue.Empty:
            pass

        latest_dashboard = None
        while not visualization_queue.empty():
            try:
                latest_dashboard = visualization_queue.get_nowait()
            except queue.Empty:
                break

        if latest_dashboard is not None:
            dashboard_large = np.asarray(resize_keep_aspect(latest_dashboard, 1400, 900), dtype=np.uint8)
            cv2.imshow(VIS_WINDOW_NAME, dashboard_large)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            control_queue.put("QUIT")
        elif key == ord("c"):
            control_queue.put("CALIBRATE")

        root.after(100, check_queue)

    tracking_thread = threading.Thread(
        target=run_tracking_loop,
        args=(cursor, message_queue, control_queue, visualization_queue),
        daemon=True,
    )
    tracking_thread.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
