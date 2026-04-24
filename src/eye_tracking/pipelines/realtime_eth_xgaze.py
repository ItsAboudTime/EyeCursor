from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import cv2

from src.eye_tracking.calibration.cursor_calibration import run_cursor_calibration
from src.eye_tracking.controllers.gaze_cursor_controller import GazeCursorController
from src.eye_tracking.pipelines.eth_xgaze_inference import ETHXGazeInference
from src.eye_tracking.visualization.overlays import draw_gaze_arrow


class RealtimeETHXGaze:
    def __init__(
        self,
        weights: pathlib.Path,
        camera_index: int = 0,
        device: str = "auto",
        roi_size: int = 224,
        focal_norm: float = 960.0,
        distance_norm: float = 600.0,
        print_interval: float = 0.2,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        predictor_path: Optional[pathlib.Path] = None,
        face_model_path: Optional[pathlib.Path] = None,
        camera_calib_path: Optional[pathlib.Path] = None,
        cursor_enabled: bool = True,
        cursor_yaw_span: float = 0.6,
        cursor_pitch_span: float = 0.4,
        cursor_ema_alpha: float = 0.1,
    ) -> None:
        self.camera_index = int(camera_index)
        self.print_interval = float(print_interval)

        self.inference = ETHXGazeInference(
            weights=weights,
            device=device,
            roi_size=roi_size,
            focal_norm=focal_norm,
            distance_norm=distance_norm,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            predictor_path=predictor_path,
            face_model_path=face_model_path,
            camera_calib_path=camera_calib_path,
        )

        self.cursor_controller = GazeCursorController(
            cursor_enabled=cursor_enabled,
            cursor_yaw_span=cursor_yaw_span,
            cursor_pitch_span=cursor_pitch_span,
            cursor_ema_alpha=cursor_ema_alpha,
        )

        self.cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def _to_app_pitch(model_pitch_rad: float) -> float:
        # ETH-XGaze model pitch uses opposite sign from app-level head-pose mapping.
        return -float(model_pitch_rad)

    def _infer_for_calibration(
        self,
        frame_bgr,
    ) -> Optional[Tuple[float, float, object, object]]:
        result = self.inference.infer_from_frame(frame_bgr)
        if result is None:
            return None
        model_pitch_rad, yaw_rad, face_patch, landmarks_normalized = result
        app_pitch_rad = self._to_app_pitch(model_pitch_rad)
        return app_pitch_rad, yaw_rad, face_patch, landmarks_normalized

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {self.camera_index})")

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def run(self) -> int:
        self.start()
        if self.cap is None:
            raise RuntimeError("Webcam was not initialized.")
        cap = self.cap

        print("ETH-XGaze-style realtime demo started.")
        print("Controls: q / ESC to quit, c to recalibrate cursor mapping")
        print("Showing normalized face patch with gaze arrow.")

        if self.cursor_controller.cursor_enabled and self.cursor_controller.cursor is not None:
            calibrated = run_cursor_calibration(
                cap=cap,
                infer_gaze_from_frame=self._infer_for_calibration,
                cursor_controller=self.cursor_controller,
            )
            if not calibrated:
                print("Using fallback span-based cursor mapping.")

        latest_pitch_yaw: Optional[Tuple[float, float]] = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue

                try:
                    result = self.inference.infer_from_frame(frame_bgr)
                except Exception:
                    result = None

                if result is None:
                    cv2.imshow("ETH-XGaze Realtime", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                model_pitch_rad, yaw_rad, face_patch, landmarks_normalized = result
                pitch_rad = self._to_app_pitch(model_pitch_rad)

                latest_pitch_yaw = (pitch_rad, yaw_rad)

                print(f"pitch: {pitch_rad:.6f}, yaw: {yaw_rad:.6f}", flush=True)

                self.cursor_controller.update_cursor(yaw_rad=yaw_rad, pitch_rad=pitch_rad)

                vis_patch = draw_gaze_arrow(face_patch, pitch_rad, yaw_rad)
                for x, y in landmarks_normalized.astype(int):
                    cv2.circle(vis_patch, (int(x), int(y)), 3, (0, 255, 0), -1)

                cv2.putText(
                    vis_patch,
                    f"pitch: {pitch_rad:.3f}, yaw: {yaw_rad:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("ETH-XGaze Realtime", vis_patch)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c") and latest_pitch_yaw is not None:
                    recalibrated = run_cursor_calibration(
                        cap=cap,
                        infer_gaze_from_frame=self._infer_for_calibration,
                        cursor_controller=self.cursor_controller,
                    )
                    if not recalibrated:
                        self.cursor_controller.calibrate_center(
                            yaw_rad=latest_pitch_yaw[1],
                            pitch_rad=latest_pitch_yaw[0],
                        )
                if key in (27, ord("q")):
                    break
        finally:
            self.stop()

        return 0
