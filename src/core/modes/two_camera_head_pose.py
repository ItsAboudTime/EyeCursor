import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.modes._viz_helpers import derive_last_action
from src.core.modes.base import TrackingMode
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.stereo_face_analysis import (
    StereoCalibration,
    StereoFaceAnalysisPipeline,
)


_VIZ_MIN_INTERVAL = 1.0 / 15.0


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
        self.visualization_callback: Optional[Callable[[dict], None]] = None
        self._last_viz_emit = 0.0

    def validate_requirements(
        self,
        profile_calibrations: Dict[str, Optional[dict]],
        selected_cameras: List[int],
    ) -> Tuple[bool, str]:
        if len(selected_cameras) < 2:
            return False, "Two cameras are required."

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
                    pre_blink = gesture_controller.active_blink_side
                    pre_scroll = gesture_controller.active_scroll_gesture
                    gesture_controller.handle_face_analysis(result, now=time.time())
                    self._maybe_emit_visualization(
                        frame_left=frame_l,
                        frame_right=frame_r,
                        result=result,
                        gesture_controller=gesture_controller,
                        pre_blink=pre_blink,
                        pre_scroll=pre_scroll,
                        screen_w=screen_w,
                        screen_h=screen_h,
                        virtual_bounds=(minx, miny, maxx, maxy),
                    )
        finally:
            gesture_controller.shutdown()
            left_cam.release()
            right_cam.release()
            pipeline.release()

    def _maybe_emit_visualization(
        self,
        frame_left,
        frame_right,
        result,
        gesture_controller: GestureController,
        pre_blink: Optional[str],
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
            pre_blink=pre_blink,
            post_blink=gesture_controller.active_blink_side,
            pre_scroll=pre_scroll,
            post_scroll=gesture_controller.active_scroll_gesture,
        )

        angles = result.angles
        yaw_deg = float(angles[0]) if angles is not None else None
        pitch_deg = float(angles[1]) if angles is not None else None

        gesture_state = {
            "active_blink_side": gesture_controller.active_blink_side,
            "active_scroll_gesture": gesture_controller.active_scroll_gesture,
            "click_enabled": gesture_controller.click_enabled,
            "scroll_enabled": gesture_controller.scroll_enabled,
            "left_is_down": gesture_controller.left_is_down,
            "right_is_down": gesture_controller.right_is_down,
            "last_action": last_action,
            "last_action_at": now if last_action else None,
        }

        minx, miny, _, _ = virtual_bounds
        if result.screen_position is not None:
            target_screen_xy = (
                int(result.screen_position[0]) + int(minx),
                int(result.screen_position[1]) + int(miny),
            )
        else:
            target_screen_xy = None

        payload = {
            "mode_id": self.id,
            "frame_left_bgr": frame_left.copy(),
            "frame_right_bgr": frame_right.copy(),
            "left_frame_width": int(frame_left.shape[1]),
            "left_frame_height": int(frame_left.shape[0]),
            "right_frame_width": int(frame_right.shape[1]),
            "right_frame_height": int(frame_right.shape[0]),
            "screen_width": int(screen_w),
            "screen_height": int(screen_h),
            "screen_bounds": tuple(int(v) for v in virtual_bounds),
            "landmarks_left": list(result.landmarks) if result.landmarks is not None else None,
            "landmarks_right": list(result.right_landmarks) if result.right_landmarks is not None else None,
            "points_3d": result.points_3d,
            "facial_transformation_matrix": result.facial_transformation_matrix,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "screen_position": result.screen_position,
            "target_screen_xy": target_screen_xy,
            "depth": result.depth,
            "wink_direction": result.wink_direction,
            "left_eye_ratio": result.left_eye_ratio,
            "right_eye_ratio": result.right_eye_ratio,
            "gesture_state": gesture_state,
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
