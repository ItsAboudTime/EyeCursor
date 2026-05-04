"""Mode: Bubble Lock -- gaze drives a bubble; pucker freezes it; head pose
fine-tunes the cursor inside the frozen bubble; another click commits and
returns to gaze-following."""

import pathlib
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.modes._viz_helpers import derive_last_action
from src.core.modes.base import TrackingMode
from src.core.modes.eye_gaze import _apply_gaze_controller_settings
from src.core.modes.one_camera_head_pose import (
    _apply_cursor_settings,
    _build_gesture_controller,
)
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.face_analysis import FaceAnalysisPipeline
from src.face_tracking.signals.blendshapes import (
    compute_smirk_activations,
    pucker_value,
    tuck_value,
)
from src.ui.overlays.gaze_bubble_overlay import BUBBLE_RADIUS_PX


_VIZ_MIN_INTERVAL = 1.0 / 15.0

_STATE_GAZE_FOLLOW = "gaze_follow"
_STATE_FROZEN = "frozen"


def _apply_bubble_lock_gesture_settings(gesture_controller, settings: dict) -> None:
    """Apply scroll_enabled only -- click_enabled is owned by the state machine."""
    if gesture_controller is None or not settings:
        return
    if "scroll_enabled" in settings:
        gesture_controller.scroll_enabled = bool(settings["scroll_enabled"])


class HybridBubbleLockMode(TrackingMode):
    id = "hybrid_bubble_lock"
    display_name = "Hybrid: Bubble Lock (experimental)"
    description = (
        "Gaze drives a bubble; pucker freezes it and switches to head pose for "
        "fine cursor control inside the bubble. Click again (left or right) to "
        "commit and resume gaze-following."
    )
    required_camera_count = 1
    requires_head_pose_calibration = True
    requires_gaze_calibration = True
    requires_facial_gesture_calibration = True

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False
        self.gaze_target_callback: Optional[Callable[[int, int], None]] = None
        self.visualization_callback: Optional[Callable[[dict], None]] = None
        self._last_viz_emit = 0.0
        self._cursor = None
        self._gesture_controller: Optional[GestureController] = None
        self._gaze_controller = None

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

        from src.eye_tracking.controllers.gaze_cursor_controller import GazeCursorController
        from src.eye_tracking.pipelines.eth_xgaze_inference import ETHXGazeInference

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
        head_pipeline.calibrate_to_center(
            calib_head["center_yaw"], calib_head["center_pitch"]
        )

        inference_kwargs = {"weights": pathlib.Path(calib_gaze["weights_path"])}
        if calib_gaze.get("predictor_path"):
            inference_kwargs["predictor_path"] = pathlib.Path(calib_gaze["predictor_path"])
        if calib_gaze.get("face_model_path"):
            inference_kwargs["face_model_path"] = pathlib.Path(calib_gaze["face_model_path"])
        inference = ETHXGazeInference(**inference_kwargs)

        # Gaze controller used only for math (target_from_gaze); never drives the cursor.
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

        # Calibrated pucker thresholds (fall back to controller defaults).
        pucker_baseline = gesture_controller.pucker_baseline
        pucker_trigger_high = gesture_controller.pucker_trigger_high
        pucker_release = gesture_controller.pucker_release

        self._cursor = cursor
        self._gesture_controller = gesture_controller
        self._gaze_controller = gaze_controller
        _apply_cursor_settings(cursor, settings)
        _apply_gaze_controller_settings(gaze_controller, settings)
        # Apply user settings first, then force click_enabled=False so the
        # GAZE_FOLLOW state suppresses OS clicks. The state machine flips
        # click_enabled True/False on its own; settings flow through
        # _apply_bubble_lock_gesture_settings (scroll_enabled only).
        _apply_bubble_lock_gesture_settings(gesture_controller, settings)
        gesture_controller.click_enabled = False

        # Mode-owned state.
        state = _STATE_GAZE_FOLLOW
        entry_click_armed = True       # rising-edge gate for the entry pucker
        last_bubble_target: Optional[Tuple[int, int]] = None
        frozen_center: Optional[Tuple[int, int]] = None

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

                # Gaze inference -> screen target (used for the bubble target).
                gaze_target: Optional[Tuple[int, int]] = None
                gaze_pitch_rad: Optional[float] = None
                gaze_yaw_rad: Optional[float] = None
                face_patch_bgr = None
                gz = inference.infer_from_frame(frame)
                if gz is not None:
                    gaze_pitch_rad, gaze_yaw_rad, face_patch_bgr, _ = gz
                    t = gaze_controller.target_from_gaze(
                        yaw_rad=gaze_yaw_rad, pitch_rad=gaze_pitch_rad
                    )
                    if t is not None:
                        gaze_target = t

                blendshapes = result.blendshapes or {}
                pucker_now = max(0.0, pucker_value(blendshapes) - pucker_baseline)

                if state == _STATE_GAZE_FOLLOW:
                    # Bubble follows gaze.
                    if gaze_target is not None and self.gaze_target_callback is not None:
                        self.gaze_target_callback(gaze_target[0], gaze_target[1])
                        last_bubble_target = gaze_target

                    # Entry rising-edge: pucker crosses trigger while armed.
                    if pucker_now < pucker_release:
                        entry_click_armed = True
                    elif (
                        pucker_now >= pucker_trigger_high
                        and entry_click_armed
                        and last_bubble_target is not None
                    ):
                        entry_click_armed = False
                        cx, cy = last_bubble_target
                        cx, cy = cursor.clamp_target(int(cx), int(cy))
                        cursor.set_pos(cx, cy)
                        # Reset step_towards's internal dt so the next head-driven
                        # step computes a clean delta from the snap point.
                        cursor._last_step_time = None
                        frozen_center = (cx, cy)
                        state = _STATE_FROZEN
                        gesture_controller.click_enabled = True
                        # Force re-arm via release: user must relax pucker before
                        # the gesture controller fires the next OS click.
                        gesture_controller._click_armed = False

                else:  # _STATE_FROZEN
                    # Convert full-screen head-pose target to a bubble-local one
                    # by re-normalizing, then offset from the frozen bubble center.
                    if (
                        result.screen_position is not None
                        and frozen_center is not None
                        and screen_w > 1
                        and screen_h > 1
                    ):
                        norm_x = result.screen_position[0] / float(screen_w - 1)
                        norm_y = result.screen_position[1] / float(screen_h - 1)
                        norm_x = min(1.0, max(0.0, norm_x))
                        norm_y = min(1.0, max(0.0, norm_y))
                        bubble_dx = (norm_x - 0.5) * 2.0 * BUBBLE_RADIUS_PX
                        bubble_dy = (norm_y - 0.5) * 2.0 * BUBBLE_RADIUS_PX
                        target_x = int(round(frozen_center[0] + bubble_dx))
                        target_y = int(round(frozen_center[1] + bubble_dy))
                        target_x, target_y = cursor.clamp_target(target_x, target_y)
                        cursor.step_towards(target_x, target_y)

                    # Exit on rising-edge of any click fired by the gesture controller.
                    if (
                        gesture_controller._last_click_side is not None
                        and not gesture_controller._last_click_consumed
                    ):
                        gesture_controller._last_click_consumed = True
                        state = _STATE_GAZE_FOLLOW
                        gesture_controller.click_enabled = False
                        # Require the user to release the pucker before another
                        # entry can fire.
                        entry_click_armed = False
                        frozen_center = None
                        cursor._last_step_time = None

                pre_scroll = gesture_controller.active_scroll_gesture
                # This mode owns cursor placement in both states (none in
                # GAZE_FOLLOW; bubble-relative in FROZEN), so strip the
                # full-screen head-pose target before passing to the gesture
                # controller -- otherwise its built-in cursor follow would
                # clobber us. Restored after so the viz payload is unchanged.
                saved_pos = result.screen_position
                result.screen_position = None
                gesture_controller.handle_face_analysis(result, now=time.time())
                result.screen_position = saved_pos

                self._maybe_emit_visualization(
                    frame_bgr=frame,
                    result=result,
                    state=state,
                    frozen_center=frozen_center,
                    last_bubble_target=last_bubble_target,
                    gaze_target=gaze_target,
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
            self._cursor = None
            self._gesture_controller = None
            self._gaze_controller = None

    def _maybe_emit_visualization(
        self,
        frame_bgr,
        result,
        state: str,
        frozen_center: Optional[Tuple[int, int]],
        last_bubble_target: Optional[Tuple[int, int]],
        gaze_target: Optional[Tuple[int, int]],
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
            "blendshapes": blendshapes,
            "gesture_state": gesture_state,
            "pitch_rad": float(gaze_pitch_rad) if gaze_pitch_rad is not None else None,
            "yaw_rad": float(gaze_yaw_rad) if gaze_yaw_rad is not None else None,
            "face_patch_bgr": face_patch_bgr.copy() if face_patch_bgr is not None else None,
            "dlib_landmarks_68": dlib_landmarks.copy() if dlib_landmarks is not None else None,
            "dlib_face_box": face_box,
            "bubble_active": True,
            "bubble_target_xy": tuple(last_bubble_target) if last_bubble_target is not None else None,
            "bubble_lock_state": state,
            "bubble_lock_frozen_center": tuple(frozen_center) if frozen_center is not None else None,
            "bubble_lock_radius_px": int(BUBBLE_RADIUS_PX),
            "target_screen_xy": tuple(gaze_target) if gaze_target is not None else None,
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

    def update_settings(self, settings: dict) -> None:
        _apply_cursor_settings(self._cursor, settings)
        _apply_bubble_lock_gesture_settings(self._gesture_controller, settings)
        _apply_gaze_controller_settings(self._gaze_controller, settings)
