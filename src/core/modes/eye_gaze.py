import pathlib
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.modes.base import TrackingMode


class EyeGazeMode(TrackingMode):
    id = "eye_gaze"
    display_name = "Eye Gaze"
    description = (
        "Control the cursor with gaze direction. "
        "Requires gaze calibration and model weights."
    )
    required_camera_count = 1
    requires_gaze_calibration = True

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False

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

        from src.eye_tracking.pipelines.eth_xgaze_inference import ETHXGazeInference
        from src.eye_tracking.controllers.gaze_cursor_controller import GazeCursorController

        calib = profile_calibrations["eye_gaze"]

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

        camera = cv2.VideoCapture(selected_cameras[0])
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open Camera {selected_cameras[0]}. "
                "Try selecting another camera or closing other apps that may be using it."
            )

        try:
            while not self._should_stop:
                if self._paused:
                    time.sleep(0.05)
                    continue

                ok, frame = camera.read()
                if not ok:
                    continue

                result = inference.infer_from_frame(frame)
                if result is not None:
                    pitch_rad, yaw_rad, _, _ = result
                    controller.update_cursor(yaw_rad, pitch_rad)
        finally:
            camera.release()

    def stop(self) -> None:
        self._should_stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
