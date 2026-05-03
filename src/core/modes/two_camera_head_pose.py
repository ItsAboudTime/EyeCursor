import json
import socket
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.capture.frame_capture import STEREO_LEFT_CAM_ID, STEREO_RIGHT_CAM_ID
from src.capture.supervisor import CaptureSupervisor
from src.core.devices.camera_identity import match_stereo_cameras
from src.core.modes._viz_helpers import derive_last_action
from src.core.modes.base import TrackingMode
from src.core.modes.one_camera_head_pose import (
    _apply_cursor_settings,
    _apply_gesture_settings,
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


_VIZ_MIN_INTERVAL = 1.0 / 15.0


class DepthBroadcaster:
    _MIN_INTERVAL = 1.0 / 30.0  # 30 Hz throttle

    def __init__(self, host: str = "127.0.0.1", port: int = 7345) -> None:
        self._addr = (host, port)
        self._sock: Optional[socket.socket] = None
        self._last_send = 0.0

    def start(self) -> None:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError:
            self._sock = None

    def send(self, depth: Optional[float]) -> None:
        if self._sock is None or depth is None:
            return
        now = time.monotonic()
        if now - self._last_send < self._MIN_INTERVAL:
            return
        self._last_send = now
        try:
            payload = json.dumps({"depth": float(depth)}).encode("utf-8")
            self._sock.sendto(payload, self._addr)
        except OSError:
            pass

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class TwoCameraHeadPoseMode(TrackingMode):
    id = "two_camera_head_pose"
    display_name = "Two-Camera Head Pose"
    description = (
        "Stereo head tracking with depth estimation. "
        "Lip gestures for clicks; smirks for scrolls. Requires two cameras and stereo calibration."
    )
    required_camera_count = 2
    requires_head_pose_calibration = True
    requires_facial_gesture_calibration = True
    requires_stereo_calibration = True

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False
        self.visualization_callback: Optional[Callable[[dict], None]] = None
        self._last_viz_emit = 0.0
        self._cursor = None
        self._gesture_controller: Optional[GestureController] = None

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

        match = match_stereo_cameras(stereo, selected_cameras)
        if not match.ok:
            return False, match.reason
        # Re-map the user's selection to the indices the calibration was
        # built for, in case the cameras moved to different /dev/videoN
        # nodes since calibration time. ``selected_cameras`` is mutated in
        # place so ``start()`` (called later by the worker) sees the
        # remapped values.
        selected_cameras[:] = match.resolved_indices

        if not profile_calibrations.get("two_camera_head_pose"):
            if not profile_calibrations.get("one_camera_head_pose"):
                return False, "Head pose calibration required."

        if not profile_calibrations.get("facial_gestures"):
            return False, "Facial gesture calibration required."
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
        gesture_calib = profile_calibrations.get("facial_gestures")

        stereo_calib = StereoCalibration(
            k1=np.array(stereo_data["K1"], dtype=np.float64),
            d1=np.array(stereo_data["D1"], dtype=np.float64),
            k2=np.array(stereo_data["K2"], dtype=np.float64),
            d2=np.array(stereo_data["D2"], dtype=np.float64),
            r=np.array(stereo_data["R"], dtype=np.float64),
            t=np.array(stereo_data["T"], dtype=np.float64).reshape(3, 1),
        )

        supervisor = CaptureSupervisor(
            camera_indices=[selected_cameras[0], selected_cameras[1]]
        )
        try:
            supervisor.start(timeout=5.0)
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not open one or both cameras. "
                "Try closing other apps that may be using them. "
                f"Detail: {exc}"
            ) from exc

        pipeline = StereoFaceAnalysisPipeline(
            stereo_calibration=stereo_calib,
            yaw_span=head_calib["yaw_span"],
            pitch_span=head_calib["pitch_span"],
            ema_alpha=head_calib.get("ema_alpha", 0.25),
        )
        pipeline.calibrate_to_center(head_calib["center_yaw"], head_calib["center_pitch"])

        minx, miny, maxx, maxy = cursor.get_virtual_bounds()
        screen_w = maxx - minx + 1
        screen_h = maxy - miny + 1

        gesture_controller = _build_gesture_controller(cursor, gesture_calib)
        self._cursor = cursor
        self._gesture_controller = gesture_controller
        _apply_cursor_settings(cursor, settings)
        _apply_gesture_settings(gesture_controller, settings)

        broadcaster = DepthBroadcaster()
        broadcaster.start()

        try:
            since_left = 0.0
            since_right = 0.0
            while not self._should_stop:
                if self._paused:
                    time.sleep(0.05)
                    continue

                if not supervisor.is_alive():
                    tail = supervisor.last_stderr_lines(3)
                    raise RuntimeError(
                        f"Capture process exited unexpectedly. Last stderr: {tail}"
                    )

                pair = supervisor.receiver.get_latest_pair(
                    cam_id_left=STEREO_LEFT_CAM_ID,
                    cam_id_right=STEREO_RIGHT_CAM_ID,
                    timeout=0.5,
                    max_skew=0.05,
                    since_left=since_left,
                    since_right=since_right,
                )
                if pair is None:
                    continue
                frame_l, frame_r, since_left, since_right = pair

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
                    pre_scroll = gesture_controller.active_scroll_gesture
                    gesture_controller.handle_face_analysis(result, now=time.time())
                    broadcaster.send(result.depth)
                    self._maybe_emit_visualization(
                        frame_left=frame_l,
                        frame_right=frame_r,
                        result=result,
                        gesture_controller=gesture_controller,
                        pre_scroll=pre_scroll,
                        screen_w=screen_w,
                        screen_h=screen_h,
                        virtual_bounds=(minx, miny, maxx, maxy),
                    )
        finally:
            broadcaster.stop()
            gesture_controller.shutdown()
            supervisor.stop()
            pipeline.release()
            self._cursor = None
            self._gesture_controller = None

    def _maybe_emit_visualization(
        self,
        frame_left,
        frame_right,
        result,
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
            "blendshapes": blendshapes,
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

    def update_settings(self, settings: dict) -> None:
        _apply_cursor_settings(self._cursor, settings)
        _apply_gesture_settings(self._gesture_controller, settings)
