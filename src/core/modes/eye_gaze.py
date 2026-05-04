import pathlib
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.capture.frame_capture import SINGLE_CAM_ID
from src.capture.session import assert_capture_alive, capture_session
from src.core.devices.camera_identity import warn_if_single_camera_mismatch
from src.core.modes._viz_helpers import derive_last_action
from src.core.modes.base import TrackingMode
from src.core.modes.idle import IdleController, apply_idle_settings
from src.core.modes.one_camera_head_pose import (
    _apply_cursor_settings,
    _apply_gesture_settings,
    _build_gesture_controller,
)
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.face_analysis import FaceAnalysisResult
from src.face_tracking.providers.face_landmarks import FaceLandmarksProvider
from src.face_tracking.signals.blendshapes import (
    compute_smirk_activations,
    extract_blendshapes,
    pucker_value,
    tuck_value,
)


_VIZ_MIN_INTERVAL = 1.0 / 15.0


def _apply_gaze_controller_settings(controller, settings: dict) -> None:
    """Push live ema_alpha onto the gaze controller. Tolerant of bad input."""
    if controller is None or not settings:
        return
    if "ema_alpha" in settings:
        try:
            controller.cursor_ema_alpha = float(settings["ema_alpha"])
        except (TypeError, ValueError) as exc:
            print(f"warning: bad ema_alpha, ignored: {exc}")


class EyeGazeMode(TrackingMode):
    id = "eye_gaze"
    display_name = "Eye Gaze"
    description = (
        "Control the cursor with gaze direction. "
        "Lip gestures for clicks; smirks for scrolls. Requires gaze calibration and model weights."
    )
    required_camera_count = 1
    requires_gaze_calibration = True
    requires_facial_gesture_calibration = True

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False
        self.visualization_callback: Optional[Callable[[dict], None]] = None
        self._last_viz_emit = 0.0
        self._cursor = None
        self._gaze_controller = None
        self._gesture_controller: Optional[GestureController] = None
        self._idle: Optional[IdleController] = None

    def validate_requirements(
        self,
        profile_calibrations: Dict[str, Optional[dict]],
        selected_cameras: List[int],
    ) -> Tuple[bool, str]:
        if len(selected_cameras) < 1:
            return False, "No camera selected."
        gaze_calib = profile_calibrations.get("eye_gaze")
        if not gaze_calib:
            return False, "Gaze calibration required."
        if not profile_calibrations.get("facial_gestures"):
            return False, "Facial gesture calibration required."
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

        calib = profile_calibrations["eye_gaze"]
        gesture_calib = profile_calibrations.get("facial_gestures")

        for cal_obj, cal_label in (
            (calib, "gaze calibration"),
            (gesture_calib, "facial gesture calibration"),
        ):
            warning = warn_if_single_camera_mismatch(
                cal_obj, selected_cameras[0], label=cal_label
            )
            if warning:
                print(f"[mode] {warning}")

        inference_kwargs = {"weights": pathlib.Path(calib["weights_path"])}
        if calib.get("predictor_path"):
            inference_kwargs["predictor_path"] = pathlib.Path(calib["predictor_path"])
        if calib.get("face_model_path"):
            inference_kwargs["face_model_path"] = pathlib.Path(calib["face_model_path"])
        inference = ETHXGazeInference(**inference_kwargs)

        controller = GazeCursorController(cursor_enabled=False)
        controller.cursor = cursor
        controller.cursor_enabled = True
        controller.cursor_bounds = cursor.get_virtual_bounds()

        if calib.get("affine"):
            controller.affine = np.array(calib["affine"], dtype=np.float64)
        if calib.get("norm_bounds"):
            controller.norm_bounds = tuple(calib["norm_bounds"])
        controller.calibration_yaw = calib.get("calibration_yaw", 0.0)
        controller.calibration_pitch = calib.get("calibration_pitch", 0.0)

        gesture_controller = _build_gesture_controller(cursor, gesture_calib)
        idle = IdleController(
            idle_after_frames=settings.get("idle_after_frames", 30),
            idle_sleep_s=settings.get("idle_sleep_s", 0.15),
        )

        # Capture references and apply initial settings.
        self._cursor = cursor
        self._gaze_controller = controller
        self._gesture_controller = gesture_controller
        self._idle = idle
        _apply_cursor_settings(cursor, settings)
        _apply_gaze_controller_settings(controller, settings)
        _apply_gesture_settings(gesture_controller, settings)

        landmarks_provider = FaceLandmarksProvider()
        screen_bounds = controller.cursor_bounds

        try:
            with capture_session([selected_cameras[0]]) as supervisor:
                last_ts = 0.0
                while not self._should_stop:
                    if self._paused:
                        time.sleep(0.05)
                        continue

                    assert_capture_alive(supervisor)

                    got = supervisor.receiver.get_latest_bgr(
                        cam_id=SINGLE_CAM_ID, since=last_ts, timeout=0.5
                    )
                    if got is None:
                        continue
                    frame, last_ts = got

                    result = inference.infer_from_frame(frame)
                    transitioned = idle.observe(result is not None)
                    if result is None:
                        # Skip MediaPipe + cursor work entirely. Only emit a
                        # visualization payload when the badge state changes
                        # or while we're throttled.
                        if transitioned or idle.is_idle:
                            self._emit_idle_visualization(
                                frame_bgr=frame,
                                idle=idle,
                                screen_bounds=screen_bounds,
                                force=transitioned,
                            )
                        idle.maybe_sleep()
                        continue

                    blendshapes = None
                    pre_scroll = None
                    if (
                        gesture_controller.click_enabled
                        or gesture_controller.scroll_enabled
                        or gesture_controller._held_button is not None
                    ):
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        observation = landmarks_provider.get_primary_face_observation(rgb)
                        if observation is not None:
                            blendshapes = extract_blendshapes(observation.blendshapes)
                            pre_scroll = gesture_controller.active_scroll_gesture
                            face_analysis = FaceAnalysisResult(
                                landmarks=observation.landmarks,
                                facial_transformation_matrix=observation.facial_transformation_matrix,
                                screen_position=None,
                                angles=None,
                                blendshapes=blendshapes,
                            )
                            gesture_controller.handle_face_analysis(face_analysis, now=time.time())

                    pitch_rad, yaw_rad, face_patch_bgr, _ = result
                    target = controller.target_from_gaze(
                        yaw_rad=yaw_rad, pitch_rad=pitch_rad
                    )
                    if target is not None and controller.cursor is not None:
                        controller.cursor.step_towards(*target)
                    self._maybe_emit_visualization(
                        frame_bgr=frame,
                        pitch_rad=pitch_rad,
                        yaw_rad=yaw_rad,
                        face_patch_bgr=face_patch_bgr,
                        target=target,
                        screen_bounds=screen_bounds,
                        inference=inference,
                        gesture_controller=gesture_controller,
                        blendshapes=blendshapes,
                        pre_scroll=pre_scroll,
                        idle=idle,
                        force=transitioned,
                    )
        finally:
            gesture_controller.shutdown()
            landmarks_provider.release()
            self._cursor = None
            self._gaze_controller = None
            self._gesture_controller = None
            self._idle = None

    def _maybe_emit_visualization(
        self,
        frame_bgr,
        pitch_rad: float,
        yaw_rad: float,
        face_patch_bgr,
        target: Optional[Tuple[int, int]],
        screen_bounds,
        inference,
        gesture_controller: Optional[GestureController] = None,
        blendshapes: Optional[dict] = None,
        pre_scroll: Optional[str] = None,
        idle: Optional[IdleController] = None,
        force: bool = False,
    ) -> None:
        callback = self.visualization_callback
        if callback is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_viz_emit) < _VIZ_MIN_INTERVAL:
            return
        self._last_viz_emit = now

        dlib_landmarks = getattr(inference, "last_dlib_landmarks", None)
        face_box = getattr(inference, "last_face_box", None)

        gesture_state = None
        if gesture_controller is not None and blendshapes is not None:
            last_action = derive_last_action(
                last_click_side=gesture_controller._last_click_side,
                pre_scroll=pre_scroll,
                post_scroll=gesture_controller.active_scroll_gesture,
            )
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

        payload = {
            "mode_id": self.id,
            "frame_bgr": frame_bgr.copy(),
            "frame_width": int(frame_bgr.shape[1]),
            "frame_height": int(frame_bgr.shape[0]),
            "pitch_rad": float(pitch_rad),
            "yaw_rad": float(yaw_rad),
            "face_patch_bgr": face_patch_bgr.copy() if face_patch_bgr is not None else None,
            "dlib_landmarks_68": dlib_landmarks.copy() if dlib_landmarks is not None else None,
            "dlib_face_box": face_box,
            "target_screen_xy": tuple(target) if target is not None else None,
            "screen_bounds": tuple(screen_bounds) if screen_bounds is not None else None,
            "blendshapes": blendshapes,
            "gesture_state": gesture_state,
            "paused": self._paused,
            "idle": bool(idle.is_idle) if idle is not None else False,
            "idle_streak_frames": int(idle.streak_frames) if idle is not None else 0,
        }
        try:
            callback(payload)
        except Exception:
            pass

    def _emit_idle_visualization(
        self,
        frame_bgr,
        idle: IdleController,
        screen_bounds,
        force: bool = False,
    ) -> None:
        """Minimal payload while no face is being tracked. See OneCameraHeadPoseMode."""
        callback = self.visualization_callback
        if callback is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_viz_emit) < _VIZ_MIN_INTERVAL:
            return
        self._last_viz_emit = now

        payload = {
            "mode_id": self.id,
            "frame_bgr": frame_bgr.copy(),
            "frame_width": int(frame_bgr.shape[1]),
            "frame_height": int(frame_bgr.shape[0]),
            "pitch_rad": None,
            "yaw_rad": None,
            "face_patch_bgr": None,
            "dlib_landmarks_68": None,
            "dlib_face_box": None,
            "target_screen_xy": None,
            "screen_bounds": tuple(screen_bounds) if screen_bounds is not None else None,
            "blendshapes": None,
            "gesture_state": None,
            "paused": self._paused,
            "idle": True,
            "idle_streak_frames": int(idle.streak_frames),
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
        _apply_gaze_controller_settings(self._gaze_controller, settings)
        _apply_gesture_settings(self._gesture_controller, settings)
        apply_idle_settings(self._idle, settings)
