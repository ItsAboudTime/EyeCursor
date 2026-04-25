import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.modes.base import TrackingMode
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.stereo_face_analysis import (
    StereoCalibration,
    StereoFaceAnalysisPipeline,
)


class TwoCameraHeadPoseMode(TrackingMode):
    id = "two_camera_head_pose"
    display_name = "Two-Camera Head Pose"
    description = (
        "Stereo head tracking with depth estimation. "
        "Requires two cameras and stereo calibration."
    )
    required_camera_count = 2
    requires_head_pose_calibration = True
    requires_eye_gesture_calibration = True
    requires_stereo_calibration = True

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False

    def validate_requirements(
        self,
        profile_calibrations: Dict[str, Optional[dict]],
        selected_cameras: List[int],
    ) -> Tuple[bool, str]:
        if len(selected_cameras) < 2:
            return False, "Two cameras are required."
        if selected_cameras[0] == selected_cameras[1]:
            return False, "Left and right cameras cannot be the same."

        stereo = profile_calibrations.get("stereo")
        if not stereo:
            return False, "Stereo calibration required. Go to Calibration to calibrate."

        if (stereo.get("left_camera_id") != selected_cameras[0] or
                stereo.get("right_camera_id") != selected_cameras[1]):
            return (
                False,
                "Stereo calibration was created for different cameras. Please recalibrate.",
            )

        if not profile_calibrations.get("two_camera_head_pose"):
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

        stereo_data = profile_calibrations["stereo"]
        head_calib = (
            profile_calibrations.get("two_camera_head_pose")
            or profile_calibrations["one_camera_head_pose"]
        )
        gesture_calib = profile_calibrations["eye_gestures"]

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

        pipeline = StereoFaceAnalysisPipeline(
            stereo_calibration=stereo_calib,
            yaw_span=head_calib["yaw_span"],
            pitch_span=head_calib["pitch_span"],
            ema_alpha=head_calib.get("ema_alpha", 0.25),
            wink_closed_threshold=gesture_calib.get("wink_eye_closed_threshold", 0.3),
            wink_open_threshold=gesture_calib.get("wink_eye_open_threshold", 0.3),
        )
        pipeline.calibrate_to_center(head_calib["center_yaw"], head_calib["center_pitch"])

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

                ok_l, frame_l = left_cam.read()
                ok_r, frame_r = right_cam.read()
                if not ok_l or not ok_r:
                    continue

                rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
                rgb_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)

                result = pipeline.analyze(
                    left_rgb_frame=rgb_l,
                    right_rgb_frame=rgb_r,
                    left_frame_width=frame_l.shape[1],
                    left_frame_height=frame_l.shape[0],
                    right_frame_width=frame_r.shape[1],
                    right_frame_height=frame_r.shape[0],
                    screen_width=screen_w,
                    screen_height=screen_h,
                )
                if result is not None:
                    gesture_controller.handle_face_analysis(result, now=time.time())
        finally:
            gesture_controller.shutdown()
            left_cam.release()
            right_cam.release()
            pipeline.release()

    def stop(self) -> None:
        self._should_stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
