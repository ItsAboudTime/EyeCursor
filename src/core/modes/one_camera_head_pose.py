import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from src.capture.frame_capture import SINGLE_CAM_ID
from src.capture.supervisor import CaptureSupervisor
from src.core.devices.camera_identity import warn_if_single_camera_mismatch
from src.core.modes._viz_helpers import derive_last_action
from src.core.modes.base import TrackingMode
from src.face_tracking.controllers.blendshape_gesture_constants import (
    PUCKER_MAX,
    PUCKER_RELEASE,
    PUCKER_TRIGGER_HIGH,
    PUCKER_TRIGGER_LOW,
    SCROLL_INTENT_DELAY_SEC,
    SMIRK_RELAX_DIFF,
    SMIRK_TRIGGER_DIFF,
    TUCK_RELEASE,
    TUCK_TRIGGER_HIGH,
    TUCK_TRIGGER_LOW,
)
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.face_analysis import FaceAnalysisPipeline
from src.face_tracking.signals.blendshapes import (
    compute_smirk_activations,
    pucker_value,
    tuck_value,
)


_VIZ_MIN_INTERVAL = 1.0 / 15.0


_CURRENT_GESTURE_CALIB_VERSION = 5


def _apply_cursor_settings(cursor, settings: dict) -> None:
    """Push the live-mutable cursor settings (move_speed / frame_rate /
    scroll_speed) onto the platform cursor instance. Tolerant of missing
    keys so it can be reused by every mode.
    """
    if cursor is None or not settings:
        return
    try:
        if "move_speed" in settings:
            cursor.move_px_per_sec = float(settings["move_speed"])
        if "frame_rate" in settings:
            # Cursor uses int frame_rate.
            cursor.frame_rate = int(settings["frame_rate"])
        if "scroll_speed" in settings:
            cursor.scroll_units_per_sec = float(settings["scroll_speed"])
    except (TypeError, ValueError) as exc:
        print(f"warning: bad cursor settings, ignored: {exc}")


def _apply_gesture_settings(gesture_controller, settings: dict) -> None:
    """Push the live-mutable gesture flags onto the gesture controller."""
    if gesture_controller is None or not settings:
        return
    if "click_enabled" in settings:
        gesture_controller.click_enabled = bool(settings["click_enabled"])
    if "scroll_enabled" in settings:
        gesture_controller.scroll_enabled = bool(settings["scroll_enabled"])


def _build_gesture_controller(cursor, gesture_calib: Optional[dict]) -> GestureController:
    """Construct the gesture controller, honoring v5 calibration if present."""
    if gesture_calib and gesture_calib.get("version") == _CURRENT_GESTURE_CALIB_VERSION:
        return GestureController(
            cursor=cursor,
            smirk_trigger_diff=gesture_calib.get("smirk_trigger_diff", SMIRK_TRIGGER_DIFF),
            smirk_relax_diff=gesture_calib.get("smirk_relax_diff", SMIRK_RELAX_DIFF),
            smirk_baseline_left=gesture_calib.get("smirk_baseline_left", 0.0),
            smirk_baseline_right=gesture_calib.get("smirk_baseline_right", 0.0),
            pucker_release=gesture_calib.get("pucker_release", PUCKER_RELEASE),
            pucker_trigger_low=gesture_calib.get("pucker_trigger_low", PUCKER_TRIGGER_LOW),
            pucker_trigger_high=gesture_calib.get("pucker_trigger_high", PUCKER_TRIGGER_HIGH),
            pucker_max=gesture_calib.get("pucker_max", PUCKER_MAX),
            pucker_baseline=gesture_calib.get("pucker_baseline", 0.0),
            scroll_intent_delay_sec=gesture_calib.get("scroll_intent_delay_sec", SCROLL_INTENT_DELAY_SEC),
            tuck_release=gesture_calib.get("tuck_release", TUCK_RELEASE),
            tuck_trigger_low=gesture_calib.get("tuck_trigger_low", TUCK_TRIGGER_LOW),
            tuck_trigger_high=gesture_calib.get("tuck_trigger_high", TUCK_TRIGGER_HIGH),
            tuck_baseline=gesture_calib.get("tuck_baseline", 0.0),
        )
    if gesture_calib is not None:
        print(
            "[gesture] facial_gestures calibration is from a previous version; "
            "using default smirk/pucker/tuck thresholds (no baseline subtraction). "
            "Please recalibrate for best results."
        )
    return GestureController(cursor=cursor)


class OneCameraHeadPoseMode(TrackingMode):
    id = "one_camera_head_pose"
    display_name = "One-Camera Head Pose"
    description = (
        "Control the cursor with head movement using one webcam. "
        "Lip gestures for clicks; smirks for scrolls."
    )
    required_camera_count = 1
    requires_head_pose_calibration = True
    requires_facial_gesture_calibration = True

    def __init__(self) -> None:
        self._should_stop = False
        self._paused = False
        self.visualization_callback: Optional[Callable[[dict], None]] = None
        self._last_viz_emit = 0.0
        # Live-mutable references captured in start() so update_settings()
        # can push changes into them while the loop is running.
        self._cursor = None
        self._gesture_controller: Optional["GestureController"] = None

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
        gesture_calib = profile_calibrations.get("facial_gestures")

        # Surface a non-blocking warning if either calibration was created
        # on a different physical camera than the one the user picked.
        for cal_obj, cal_label in (
            (calib, "head pose calibration"),
            (gesture_calib, "facial gesture calibration"),
        ):
            warning = warn_if_single_camera_mismatch(
                cal_obj, selected_cameras[0], label=cal_label
            )
            if warning:
                print(f"[mode] {warning}")

        supervisor = CaptureSupervisor(camera_indices=[selected_cameras[0]])
        try:
            supervisor.start(timeout=5.0)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not open Camera {selected_cameras[0]}. "
                f"Try selecting another camera or closing other apps that may be using it. "
                f"Detail: {exc}"
            ) from exc

        pipeline = FaceAnalysisPipeline(
            yaw_span=calib["yaw_span"],
            pitch_span=calib["pitch_span"],
            ema_alpha=calib.get("ema_alpha", 0.1),
        )
        pipeline.calibrate_to_center(calib["center_yaw"], calib["center_pitch"])

        minx, miny, maxx, maxy = cursor.get_virtual_bounds()
        screen_w = maxx - minx + 1
        screen_h = maxy - miny + 1

        gesture_controller = _build_gesture_controller(cursor, gesture_calib)
        # Apply initial settings to both cursor and gesture controller.
        self._cursor = cursor
        self._gesture_controller = gesture_controller
        _apply_cursor_settings(cursor, settings)
        _apply_gesture_settings(gesture_controller, settings)

        try:
            last_ts = 0.0
            while not self._should_stop:
                if self._paused:
                    time.sleep(0.05)
                    continue

                if not supervisor.is_alive():
                    tail = supervisor.last_stderr_lines(3)
                    raise RuntimeError(
                        f"Capture process exited unexpectedly. Last stderr: {tail}"
                    )

                got = supervisor.receiver.get_latest_bgr(
                    cam_id=SINGLE_CAM_ID, since=last_ts, timeout=0.5
                )
                if got is None:
                    continue
                frame, last_ts = got

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pipeline.analyze(
                    rgb_frame=rgb,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    screen_width=screen_w,
                    screen_height=screen_h,
                )
                if result is not None:
                    pre_scroll = gesture_controller.active_scroll_gesture
                    gesture_controller.handle_face_analysis(result, now=time.time())
                    self._maybe_emit_visualization(
                        frame_bgr=frame,
                        result=result,
                        gesture_controller=gesture_controller,
                        pre_scroll=pre_scroll,
                        screen_w=screen_w,
                        screen_h=screen_h,
                        virtual_bounds=(minx, miny, maxx, maxy),
                    )
        finally:
            gesture_controller.shutdown()
            supervisor.stop()
            pipeline.release()
            self._cursor = None
            self._gesture_controller = None

    def _maybe_emit_visualization(
        self,
        frame_bgr,
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
            "blendshapes": blendshapes,
            "gesture_state": gesture_state,
            "paused": self._paused,
        }
        try:
            callback(payload)
        except Exception:
            # Never let a viz failure kill the tracking loop.
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
