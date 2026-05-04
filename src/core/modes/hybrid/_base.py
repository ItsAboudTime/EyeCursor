"""Shared base for experimental hybrid head+gaze cursor modes."""

import pathlib
import time
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.modes._viz_helpers import derive_last_action
from src.core.modes.base import TrackingMode
from src.core.modes.one_camera_head_pose import _build_gesture_controller
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.face_analysis import FaceAnalysisPipeline
from src.face_tracking.signals.blendshapes import (
    compute_smirk_activations,
    pucker_value,
    tuck_value,
)


_VIZ_MIN_INTERVAL = 1.0 / 15.0


class _HybridGazeHeadModeBase(TrackingMode):
    required_camera_count = 1
    requires_head_pose_calibration = True
    requires_gaze_calibration = True
    requires_facial_gesture_calibration = True

    # Future hook: skip the gaze CNN forward pass when its result is unused
    # (e.g. outside the safe zone in Modes 1 and 2, or when alpha == 1 in Mode 3).
    # Left False for v1; both pipelines run every frame.
    _skip_gaze_when_unused: bool = False

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False
        self.visualization_callback: Optional[Callable[[dict], None]] = None
        self._last_viz_emit = 0.0

    def validate_requirements(
        self,
        profile_calibrations: Dict[str, Optional[dict]],
        selected_cameras: List[int],
    ) -> Tuple[bool, str]:
        if len(selected_cameras) < 1:
            return False, "No camera selected."
        if not profile_calibrations.get("one_camera_head_pose"):
            return False, "Head pose calibration required."
        if not profile_calibrations.get("facial_gestures"):
            return False, "Facial gesture calibration required."
        gaze_calib = profile_calibrations.get("eye_gaze")
        if not gaze_calib:
            return False, "Gaze calibration required."
        for key, label in (
            ("weights_path", "Model weights"),
            ("predictor_path", "Face landmark predictor"),
            ("face_model_path", "Face model file"),
        ):
            path_str = gaze_calib.get(key, "")
            if path_str and not pathlib.Path(path_str).exists():
                return False, f"{label} not found at: {path_str}. Please recalibrate."
        return True, ""

    def start(
        self,
        profile_calibrations: Dict[str, Optional[dict]],
        selected_cameras: List[int],
        cursor,
        settings: Optional[dict] = None,
    ) -> None:
        self._should_stop = False
        self._paused = False
        settings = settings or {}

        from src.eye_tracking.pipelines.eth_xgaze_inference import ETHXGazeInference
        from src.eye_tracking.controllers.gaze_cursor_controller import GazeCursorController

        calib_head = profile_calibrations["one_camera_head_pose"]
        calib_gaze = profile_calibrations["eye_gaze"]
        gesture_calib = profile_calibrations.get("facial_gestures")

        camera = cv2.VideoCapture(selected_cameras[0])
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open Camera {selected_cameras[0]}. "
                "Try selecting another camera or closing other apps that may be using it."
            )

        head_pipeline = FaceAnalysisPipeline(
            yaw_span=calib_head["yaw_span"],
            pitch_span=calib_head["pitch_span"],
            ema_alpha=calib_head.get("ema_alpha", 0.1),
        )
        head_pipeline.calibrate_to_center(calib_head["center_yaw"], calib_head["center_pitch"])

        inference_kwargs = {"weights": pathlib.Path(calib_gaze["weights_path"])}
        if calib_gaze.get("predictor_path"):
            inference_kwargs["predictor_path"] = pathlib.Path(calib_gaze["predictor_path"])
        if calib_gaze.get("face_model_path"):
            inference_kwargs["face_model_path"] = pathlib.Path(calib_gaze["face_model_path"])
        inference = ETHXGazeInference(**inference_kwargs)

        # target_from_gaze() guards on cursor_enabled and self.cursor; we only
        # want the gaze->screen math (not to drive the cursor), so satisfy the
        # guard with the real cursor and never call update_cursor on this controller.
        gaze_controller = GazeCursorController(cursor_enabled=True)
        gaze_controller.cursor = cursor
        gaze_controller.cursor_bounds = cursor.get_virtual_bounds()
        if calib_gaze.get("affine"):
            gaze_controller.affine = np.array(calib_gaze["affine"], dtype=np.float64)
        if calib_gaze.get("norm_bounds"):
            gaze_controller.norm_bounds = tuple(calib_gaze["norm_bounds"])
        gaze_controller.calibration_yaw = calib_gaze.get("calibration_yaw", 0.0)
        gaze_controller.calibration_pitch = calib_gaze.get("calibration_pitch", 0.0)

        minx, miny, maxx, maxy = cursor.get_virtual_bounds()
        screen_w = maxx - minx + 1
        screen_h = maxy - miny + 1

        gesture_controller = _build_gesture_controller(cursor, gesture_calib)
        gesture_controller.click_enabled = settings.get("click_enabled", True)
        gesture_controller.scroll_enabled = settings.get("scroll_enabled", True)

        try:
            while not self._should_stop:
                if self._paused:
                    time.sleep(0.05)
                    continue

                ok, frame = camera.read()
                if not ok:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = head_pipeline.analyze(
                    rgb_frame=rgb,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    screen_width=screen_w,
                    screen_height=screen_h,
                )
                if result is None:
                    continue

                head_xy = result.screen_position
                yaw_deg, pitch_deg = result.angles or (0.0, 0.0)

                gaze_xy = None
                gaze_pitch_rad = None
                gaze_yaw_rad = None
                face_patch_bgr = None
                gz = inference.infer_from_frame(frame)
                if gz is not None:
                    gaze_pitch_rad, gaze_yaw_rad, face_patch_bgr, _ = gz
                    t = gaze_controller.target_from_gaze(
                        yaw_rad=gaze_yaw_rad, pitch_rad=gaze_pitch_rad
                    )
                    if t is not None:
                        gaze_xy = (t[0] - minx, t[1] - miny)

                blended_xy = head_xy
                debug_meta: dict = {"active": "head_only"}
                if head_xy is not None:
                    blended_xy, debug_meta = self._blend_targets(
                        head_xy=head_xy,
                        gaze_xy=gaze_xy,
                        yaw_deg=yaw_deg,
                        pitch_deg=pitch_deg,
                        screen_w=screen_w,
                        screen_h=screen_h,
                    )

                pre_scroll = gesture_controller.active_scroll_gesture
                if blended_xy is not None:
                    result.screen_position = (int(blended_xy[0]), int(blended_xy[1]))
                gesture_controller.handle_face_analysis(result, now=time.time())

                self._maybe_emit_visualization(
                    frame_bgr=frame,
                    result=result,
                    head_xy=head_xy,
                    gaze_xy=gaze_xy,
                    blended_xy=blended_xy,
                    debug_meta=debug_meta,
                    gaze_pitch_rad=gaze_pitch_rad,
                    gaze_yaw_rad=gaze_yaw_rad,
                    face_patch_bgr=face_patch_bgr,
                    inference=inference,
                    gesture_controller=gesture_controller,
                    pre_scroll=pre_scroll,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    virtual_bounds=(minx, miny, maxx, maxy),
                )
        finally:
            gesture_controller.shutdown()
            camera.release()
            head_pipeline.release()

    @abstractmethod
    def _blend_targets(
        self,
        head_xy: Tuple[int, int],
        gaze_xy: Optional[Tuple[int, int]],
        yaw_deg: float,
        pitch_deg: float,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[Tuple[float, float], dict]:
        ...

    @staticmethod
    def _in_safe_zone(
        yaw_deg: float,
        pitch_deg: float,
        yaw_thr: float,
        pitch_thr: float,
    ) -> bool:
        return abs(yaw_deg) <= yaw_thr and abs(pitch_deg) <= pitch_thr

    def _maybe_emit_visualization(
        self,
        frame_bgr,
        result,
        head_xy: Optional[Tuple[int, int]],
        gaze_xy: Optional[Tuple[int, int]],
        blended_xy: Optional[Tuple[float, float]],
        debug_meta: dict,
        gaze_pitch_rad: Optional[float],
        gaze_yaw_rad: Optional[float],
        face_patch_bgr,
        inference,
        gesture_controller: GestureController,
        pre_scroll: Optional[str],
        screen_w: int,
        screen_h: int,
        virtual_bounds: Tuple[int, int, int, int],
    ) -> None:
        callback = self.visualization_callback
        if callback is None:
            return
        now = time.monotonic()
        if (now - self._last_viz_emit) < _VIZ_MIN_INTERVAL:
            return
        self._last_viz_emit = now

        last_action = derive_last_action(
            last_click_side=gesture_controller._last_click_side,
            pre_scroll=pre_scroll,
            post_scroll=gesture_controller.active_scroll_gesture,
        )

        angles = result.angles
        yaw_deg = float(angles[0]) if angles is not None else None
        pitch_deg = float(angles[1]) if angles is not None else None

        blendshapes = result.blendshapes or {}
        smirk_left, smirk_right = compute_smirk_activations(blendshapes)
        pucker = pucker_value(blendshapes)
        tuck = tuck_value(blendshapes)

        gesture_state = {
            "active_scroll_gesture": gesture_controller.active_scroll_gesture,
            "click_enabled": gesture_controller.click_enabled,
            "scroll_enabled": gesture_controller.scroll_enabled,
            "click_armed": gesture_controller._click_armed,
            "last_click_side": gesture_controller._last_click_side,
            "smirk_left_activation": smirk_left,
            "smirk_right_activation": smirk_right,
            "pucker_value": pucker,
            "tuck_value": tuck,
            "held_button": gesture_controller._held_button,
            "is_held": gesture_controller._held_button is not None,
            "last_action": last_action,
            "last_action_at": now if last_action else None,
        }

        minx, miny, _, _ = virtual_bounds

        if blended_xy is not None:
            target_screen_xy = (
                int(blended_xy[0]) + int(minx),
                int(blended_xy[1]) + int(miny),
            )
        elif result.screen_position is not None:
            target_screen_xy = (
                int(result.screen_position[0]) + int(minx),
                int(result.screen_position[1]) + int(miny),
            )
        else:
            target_screen_xy = None

        if head_xy is not None:
            head_screen_xy = (
                int(head_xy[0]) + int(minx),
                int(head_xy[1]) + int(miny),
            )
        else:
            head_screen_xy = None

        if gaze_xy is not None:
            gaze_screen_xy = (
                int(gaze_xy[0]) + int(minx),
                int(gaze_xy[1]) + int(miny),
            )
        else:
            gaze_screen_xy = None

        dlib_landmarks = getattr(inference, "last_dlib_landmarks", None)
        face_box = getattr(inference, "last_face_box", None)

        payload = {
            "mode_id": self.id,
            "frame_bgr": frame_bgr.copy(),
            "frame_width": int(frame_bgr.shape[1]),
            "frame_height": int(frame_bgr.shape[0]),
            "screen_width": int(screen_w),
            "screen_height": int(screen_h),
            "screen_bounds": tuple(int(v) for v in virtual_bounds),
            "landmarks": list(result.landmarks) if result.landmarks is not None else None,
            "facial_transformation_matrix": result.facial_transformation_matrix,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "screen_position": result.screen_position,
            "target_screen_xy": target_screen_xy,
            "head_screen_xy": head_screen_xy,
            "gaze_screen_xy": gaze_screen_xy,
            "blend_debug": debug_meta,
            "blendshapes": blendshapes,
            "gesture_state": gesture_state,
            "pitch_rad": float(gaze_pitch_rad) if gaze_pitch_rad is not None else None,
            "yaw_rad": float(gaze_yaw_rad) if gaze_yaw_rad is not None else None,
            "face_patch_bgr": face_patch_bgr.copy() if face_patch_bgr is not None else None,
            "dlib_landmarks_68": dlib_landmarks.copy() if dlib_landmarks is not None else None,
            "dlib_face_box": face_box,
            "paused": self._paused,
        }
        try:
            callback(payload)
        except Exception:
            pass

    def stop(self) -> None:
        self._should_stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
