import pathlib
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.capture.frame_capture import SINGLE_CAM_ID
from src.capture.session import assert_capture_alive, capture_session
from src.core.devices.camera_identity import warn_if_single_camera_mismatch
from src.core.modes.base import TrackingMode
from src.core.modes.eye_gaze import _apply_gaze_controller_settings
from src.core.modes.one_camera_head_pose import _apply_cursor_settings


_VIZ_MIN_INTERVAL = 1.0 / 15.0


class EyeGazeBubbleMode(TrackingMode):
    id = "eye_gaze_bubble"
    display_name = "Eye Gaze (Bubble)"
    description = (
        "Show a translucent bubble where you're looking, on top of all windows. "
        "The mouse stays fully under your control. Reuses gaze calibration."
    )
    required_camera_count = 1
    requires_gaze_calibration = True

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False
        self.gaze_target_callback: Optional[Callable[[int, int], None]] = None
        self.visualization_callback: Optional[Callable[[dict], None]] = None
        self._last_viz_emit = 0.0
        self._cursor = None
        self._gaze_controller = None

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

        warning = warn_if_single_camera_mismatch(
            calib, selected_cameras[0], label="gaze calibration"
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

        self._cursor = cursor
        self._gaze_controller = controller
        _apply_cursor_settings(cursor, settings)
        _apply_gaze_controller_settings(controller, settings)

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
                    if result is not None:
                        pitch_rad, yaw_rad, face_patch_bgr, _ = result
                        target = controller.target_from_gaze(yaw_rad=yaw_rad, pitch_rad=pitch_rad)
                        if target is not None and self.gaze_target_callback is not None:
                            self.gaze_target_callback(target[0], target[1])
                        self._maybe_emit_visualization(
                            frame_bgr=frame,
                            pitch_rad=pitch_rad,
                            yaw_rad=yaw_rad,
                            face_patch_bgr=face_patch_bgr,
                            target=target,
                            screen_bounds=screen_bounds,
                            inference=inference,
                        )
        finally:
            self._cursor = None
            self._gaze_controller = None

    def _maybe_emit_visualization(
        self,
        frame_bgr,
        pitch_rad: float,
        yaw_rad: float,
        face_patch_bgr,
        target: Optional[Tuple[int, int]],
        screen_bounds,
        inference,
    ) -> None:
        callback = self.visualization_callback
        if callback is None:
            return
        now = time.monotonic()
        if (now - self._last_viz_emit) < _VIZ_MIN_INTERVAL:
            return
        self._last_viz_emit = now

        dlib_landmarks = getattr(inference, "last_dlib_landmarks", None)
        face_box = getattr(inference, "last_face_box", None)

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
            "bubble_active": True,
            "bubble_target_xy": tuple(target) if target is not None else None,
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
        _apply_gaze_controller_settings(self._gaze_controller, settings)
