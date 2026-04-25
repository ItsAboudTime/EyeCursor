import time
from typing import Dict, List, Optional, Tuple

import cv2

from src.core.modes.base import TrackingMode
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.face_analysis import FaceAnalysisPipeline


class OneCameraHeadPoseMode(TrackingMode):
    id = "one_camera_head_pose"
    display_name = "One-Camera Head Pose"
    description = (
        "Control the cursor with head movement using one webcam. "
        "Eye gestures for clicks and scrolls."
    )
    required_camera_count = 1
    requires_head_pose_calibration = True
    requires_eye_gesture_calibration = True

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
        if not profile_calibrations.get("one_camera_head_pose"):
            return False, "Head pose calibration required."
        if not profile_calibrations.get("eye_gestures"):
            return False, "Eye gesture calibration required."
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

        calib = profile_calibrations["one_camera_head_pose"]
        gesture_calib = profile_calibrations["eye_gestures"]

        camera = cv2.VideoCapture(selected_cameras[0])
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open Camera {selected_cameras[0]}. "
                "Try selecting another camera or closing other apps that may be using it."
            )

        pipeline = FaceAnalysisPipeline(
            yaw_span=calib["yaw_span"],
            pitch_span=calib["pitch_span"],
            ema_alpha=calib.get("ema_alpha", 0.1),
        )
        pipeline.calibrate_to_center(calib["center_yaw"], calib["center_pitch"])

        minx, miny, maxx, maxy = cursor.get_virtual_bounds()
        screen_w = maxx - minx + 1
        screen_h = maxy - miny + 1

        gesture_controller = GestureController(
            cursor=cursor,
            hold_trigger_seconds=gesture_calib.get("hold_duration_click", 1.0),
            release_missed_frames=5,
            both_eyes_open_threshold=gesture_calib["both_eyes_open_threshold"],
            both_eyes_squint_threshold=gesture_calib["both_eyes_squint_threshold"],
            scroll_trigger_seconds=1.0,
            scroll_delta=gesture_calib.get("scroll_delta", 120),
        )
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
                result = pipeline.analyze(
                    rgb_frame=rgb,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    screen_width=screen_w,
                    screen_height=screen_h,
                )
                if result is not None:
                    gesture_controller.handle_face_analysis(result, now=time.time())
        finally:
            gesture_controller.shutdown()
            camera.release()
            pipeline.release()

    def stop(self) -> None:
        self._should_stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
