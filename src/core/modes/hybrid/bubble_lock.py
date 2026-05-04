"""Mode: Bubble Lock -- gaze drives a bubble; pucker freezes it; stereo head
pose fine-tunes the cursor inside the frozen bubble; another click commits and
returns to gaze-following."""

import pathlib
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.devices.camera_identity import match_stereo_cameras
from src.core.modes._viz_helpers import derive_last_action
from src.core.modes.base import TrackingMode
from src.core.modes.eye_gaze import _apply_gaze_controller_settings
from src.core.modes.one_camera_head_pose import (
    _apply_cursor_settings,
    _build_gesture_controller,
)
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.stereo_face_analysis import (
    StereoCalibration,
    StereoFaceAnalysisPipeline,
)
from src.face_tracking.signals.blendshapes import (
    compute_smirk_activations,
    pucker_value,
    tuck_value,
)
from src.ui.overlays.gaze_bubble_overlay import BUBBLE_RADIUS_PX


_VIZ_MIN_INTERVAL = 1.0 / 15.0

_STATE_GAZE_FOLLOW = "gaze_follow"
_STATE_FROZEN = "frozen"

# Gesture must stay above trigger threshold for this long before it fires.
# Eliminates single-frame blendshape spikes caused by head movement.
_GESTURE_HOLD_SEC = 0.15


def _apply_bubble_lock_gesture_settings(gesture_controller, settings: dict) -> None:
    """Apply scroll_enabled only -- click_enabled is owned by the state machine."""
    if gesture_controller is None or not settings:
        return
    if "scroll_enabled" in settings:
        gesture_controller.scroll_enabled = bool(settings["scroll_enabled"])


class HybridBubbleLockMode(TrackingMode):
    id = "hybrid_bubble_lock"
    display_name = "Bubble Lock"
    description = (
        "Gaze drives a bubble; pucker freezes it and switches to stereo head pose "
        "for fine cursor control inside the bubble. Click again (left or right) to "
        "commit and resume gaze-following. Requires two cameras."
    )
    required_camera_count = 2
    requires_head_pose_calibration = True
    requires_stereo_calibration = True
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
        if len(selected_cameras) < 2:
            return False, "Two cameras are required."
        stereo = profile_calibrations.get("stereo")
        if not stereo:
            return False, "Stereo calibration required."
        match = match_stereo_cameras(stereo, selected_cameras)
        if not match.ok:
            return False, match.reason
        selected_cameras[:] = match.resolved_indices
        if not profile_calibrations.get("two_camera_head_pose"):
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

        stereo_data = profile_calibrations["stereo"]
        calib_head = (
            profile_calibrations.get("two_camera_head_pose")
            or profile_calibrations["one_camera_head_pose"]
        )
        calib_gaze = profile_calibrations["eye_gaze"]
        gesture_calib = profile_calibrations.get("facial_gestures")

        stereo_calib = StereoCalibration(
            k1=np.array(stereo_data["K1"], dtype=np.float64),
            d1=np.array(stereo_data["D1"], dtype=np.float64),
            k2=np.array(stereo_data["K2"], dtype=np.float64),
            d2=np.array(stereo_data["D2"], dtype=np.float64),
            r=np.array(stereo_data["R"], dtype=np.float64),
            t=np.array(stereo_data["T"], dtype=np.float64).reshape(3, 1),
        )

        left_cam = cv2.VideoCapture(selected_cameras[0])
        right_cam = cv2.VideoCapture(selected_cameras[1])
        if not left_cam.isOpened() or not right_cam.isOpened():
            left_cam.release()
            right_cam.release()
            raise RuntimeError(
                "Could not open one or both cameras. "
                "Try closing other apps that may be using them."
            )

        head_pipeline = StereoFaceAnalysisPipeline(
            stereo_calibration=stereo_calib,
            yaw_span=calib_head["yaw_span"],
            pitch_span=calib_head["pitch_span"],
            ema_alpha=calib_head.get("ema_alpha", 0.25),
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

        # Calibrated thresholds for mode-owned gesture detection.
        pucker_baseline = gesture_controller.pucker_baseline
        pucker_trigger_high = gesture_controller.pucker_trigger_high
        pucker_release = gesture_controller.pucker_release
        tuck_baseline = gesture_controller.tuck_baseline
        tuck_trigger_high = gesture_controller.tuck_trigger_high
        tuck_release = gesture_controller.tuck_release

        self._cursor = cursor
        self._gesture_controller = gesture_controller
        self._gaze_controller = gaze_controller
        _apply_cursor_settings(cursor, settings)
        _apply_gaze_controller_settings(gaze_controller, settings)
        _apply_bubble_lock_gesture_settings(gesture_controller, settings)
        # Mode owns all click detection and fires OS clicks manually for the
        # exit gesture. click_enabled stays False so the gesture controller
        # only handles smirk scrolls.
        gesture_controller.click_enabled = False

        # Mode-owned state.
        state = _STATE_GAZE_FOLLOW
        last_bubble_target: Optional[Tuple[int, int]] = None
        frozen_center: Optional[Tuple[int, int]] = None

        # Gesture detection -- separate flags so a high resting tuck value
        # can't block the entry re-arm (entry is pucker-only).
        entry_armed = True           # re-arms when pucker < pucker_release
        exit_armed = False           # starts False; re-arms when pucker relaxes in FROZEN
        entry_hold_start: Optional[float] = None   # for 150ms debounce
        exit_hold_start: Optional[float] = None

        try:
            while not self._should_stop:
                if self._paused:
                    time.sleep(0.05)
                    continue

                ok_l, frame_l = left_cam.read()
                ok_r, frame_r = right_cam.read()
                if not ok_l or not ok_r:
                    continue

                rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
                rgb_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
                result = head_pipeline.analyze(
                    left_rgb_frame=rgb_l,
                    right_rgb_frame=rgb_r,
                    left_frame_width=frame_l.shape[1],
                    left_frame_height=frame_l.shape[0],
                    right_frame_width=frame_r.shape[1],
                    right_frame_height=frame_r.shape[0],
                    screen_width=screen_w,
                    screen_height=screen_h,
                )
                if result is None:
                    continue

                # Gaze inference -> screen target (bubble target).
                # Skipped when frozen: the bubble is stationary and the CNN is
                # the most expensive operation in the loop.
                gaze_target: Optional[Tuple[int, int]] = None
                gaze_pitch_rad: Optional[float] = None
                gaze_yaw_rad: Optional[float] = None
                face_patch_bgr = None
                if state == _STATE_GAZE_FOLLOW:
                    gz = inference.infer_from_frame(frame_l)
                    if gz is not None:
                        gaze_pitch_rad, gaze_yaw_rad, face_patch_bgr, _ = gz
                        t = gaze_controller.target_from_gaze(
                            yaw_rad=gaze_yaw_rad, pitch_rad=gaze_pitch_rad
                        )
                        if t is not None:
                            gaze_target = t

                now = time.time()
                blendshapes = result.blendshapes or {}
                pucker_now = max(0.0, pucker_value(blendshapes) - pucker_baseline)
                tuck_now = max(0.0, tuck_value(blendshapes) - tuck_baseline)

                if state == _STATE_GAZE_FOLLOW:
                    # Bubble follows gaze.
                    if gaze_target is not None and self.gaze_target_callback is not None:
                        self.gaze_target_callback(gaze_target[0], gaze_target[1])
                        last_bubble_target = gaze_target

                    # Entry re-arm: only depends on pucker so a high resting
                    # tuck value doesn't prevent the user from ever freezing.
                    if pucker_now < pucker_release:
                        entry_armed = True

                    # Entry: pucker held for _GESTURE_HOLD_SEC while armed.
                    if entry_armed and pucker_now >= pucker_trigger_high:
                        if entry_hold_start is None:
                            entry_hold_start = now
                        if now - entry_hold_start >= _GESTURE_HOLD_SEC:
                            entry_hold_start = None
                            entry_armed = False
                            exit_armed = False   # require pucker release before exit arms
                            if last_bubble_target is not None:
                                cx, cy = cursor.clamp_target(
                                    int(last_bubble_target[0]), int(last_bubble_target[1])
                                )
                            else:
                                cx, cy = cursor.get_pos()
                            cursor.set_pos(cx, cy)
                            cursor._last_step_time = None
                            frozen_center = (cx, cy)
                            state = _STATE_FROZEN
                    else:
                        entry_hold_start = None   # gesture dropped, reset timer

                else:  # _STATE_FROZEN
                    # Convert full-screen head-pose target to bubble-local coords
                    # by re-normalizing, then offset from the frozen center.
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

                    # Exit re-arm: pucker relaxing re-arms exit (pucker-only check
                    # avoids being blocked by a high resting tuck value).
                    if pucker_now < pucker_release:
                        exit_armed = True

                    # Exit: pucker (left click) or tuck (right click) held for
                    # _GESTURE_HOLD_SEC while armed. Hold debounce prevents brief
                    # blendshape spikes from head movement triggering a false exit.
                    exit_gesture = (
                        pucker_now >= pucker_trigger_high or tuck_now >= tuck_trigger_high
                    )
                    if exit_armed and exit_gesture:
                        if exit_hold_start is None:
                            exit_hold_start = now
                        if now - exit_hold_start >= _GESTURE_HOLD_SEC:
                            exit_hold_start = None
                            exit_armed = False
                            if pucker_now >= pucker_trigger_high:
                                cursor.left_click()
                            else:
                                cursor.right_click()
                            state = _STATE_GAZE_FOLLOW
                            frozen_center = None
                            cursor._last_step_time = None
                    else:
                        exit_hold_start = None   # gesture dropped or not armed, reset timer

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
                    frame_bgr=frame_l,
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
            left_cam.release()
            right_cam.release()
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
