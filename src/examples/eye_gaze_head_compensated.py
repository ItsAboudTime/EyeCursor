from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from src.eye_tracking.calibration.cursor_calibration import run_cursor_calibration
from src.eye_tracking.controllers.gaze_cursor_controller import GazeCursorController
from src.eye_tracking.pipelines.eth_xgaze_inference import ETHXGazeInference
from src.eye_tracking.visualization.overlays import draw_gaze_arrow


class HeadCompensatedETHXGaze:
    """ETH-XGaze realtime demo with head-rotation compensation for cursor mapping."""

    WINDOW_NAME = "ETH-XGaze Head-Compensated"
    HEAD_RESET_BUTTON_TOP_LEFT = (10, 132)
    HEAD_RESET_BUTTON_BOTTOM_RIGHT = (214, 160)

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
        cursor_ema_alpha: float = 0.2,
        head_compensation_yaw: float = 0.3,
        head_compensation_pitch: float = 0.25,
        head_compensation_max_angle_deg: float = 20.0,
        head_compensation_yaw_step: float = 0.02,
        head_compensation_pitch_step: float = 0.02,
    ) -> None:
        if head_compensation_yaw < 0.0:
            raise ValueError("head_compensation_yaw must be >= 0")
        if head_compensation_pitch < 0.0:
            raise ValueError("head_compensation_pitch must be >= 0")
        if head_compensation_yaw_step <= 0.0:
            raise ValueError("head_compensation_yaw_step must be > 0")
        if head_compensation_pitch_step <= 0.0:
            raise ValueError("head_compensation_pitch_step must be > 0")

        self.camera_index = int(camera_index)
        self.print_interval = float(print_interval)
        self.head_compensation_yaw = float(head_compensation_yaw)
        self.head_compensation_pitch = float(head_compensation_pitch)
        self.head_compensation_max_angle_rad = float(np.deg2rad(head_compensation_max_angle_deg))
        self.head_compensation_yaw_step = float(head_compensation_yaw_step)
        self.head_compensation_pitch_step = float(head_compensation_pitch_step)
        self.head_zero_pitch = 0.0
        self.head_zero_yaw = 0.0
        self._latest_head_pose: Optional[Tuple[float, float]] = None

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
    def _head_angles_from_rvec(rvec: np.ndarray) -> Tuple[float, float]:
        rotation = cv2.Rodrigues(rvec)[0]

        # Match the head_pose.py convention: forward points roughly along -Z at neutral.
        forward = -rotation[:, 2]
        norm = float(np.linalg.norm(forward))
        if norm < 1e-9:
            return 0.0, 0.0

        forward = forward / norm
        if forward[2] > 0.0:
            forward = -forward

        head_yaw = float(np.arctan2(float(forward[0]), -float(forward[2])))
        head_pitch = float(np.arctan2(float(forward[1]), -float(forward[2])))
        return head_yaw, head_pitch

    def _apply_compensation(
        self,
        eye_pitch: float,
        eye_yaw: float,
        head_pitch: float,
        head_yaw: float,
    ) -> Tuple[float, float]:
        # Re-center head pose around the latest neutral calibration.
        head_pitch, head_yaw = self._center_head_angles(head_pitch=head_pitch, head_yaw=head_yaw)

        # Clamp head-angle contribution so brief pose spikes do not cause fast cursor jumps.
        max_angle = self.head_compensation_max_angle_rad
        head_yaw = float(np.clip(head_yaw, -max_angle, max_angle))
        head_pitch = float(np.clip(head_pitch, -max_angle, max_angle))

        # Use subtractive compensation because ETH-XGaze output already carries head-related motion.
        compensated_yaw = eye_yaw - (self.head_compensation_yaw * head_yaw)
        compensated_pitch = eye_pitch - (self.head_compensation_pitch * head_pitch)
        return compensated_pitch, compensated_yaw

    def _center_head_angles(self, head_pitch: float, head_yaw: float) -> Tuple[float, float]:
        centered_pitch = float(head_pitch - self.head_zero_pitch)
        centered_yaw = float(head_yaw - self.head_zero_yaw)
        return centered_pitch, centered_yaw

    def _set_head_compensation_yaw(self, new_value: float) -> None:
        self.head_compensation_yaw = max(0.0, float(new_value))
        print(
            f"Updated yaw compensation gain: {self.head_compensation_yaw:.3f}",
            flush=True,
        )

    def _set_head_compensation_pitch(self, new_value: float) -> None:
        self.head_compensation_pitch = max(0.0, float(new_value))
        print(
            f"Updated pitch compensation gain: {self.head_compensation_pitch:.3f}",
            flush=True,
        )

    def _recalibrate_head_pose_zero(self, head_pitch: float, head_yaw: float) -> None:
        self.head_zero_pitch = float(head_pitch)
        self.head_zero_yaw = float(head_yaw)
        print(
            "Recalibrated head neutral pose to current frame: "
            f"pitch={self.head_zero_pitch:.3f}, yaw={self.head_zero_yaw:.3f}",
            flush=True,
        )

    def _is_in_head_reset_button(self, x: int, y: int) -> bool:
        x1, y1 = self.HEAD_RESET_BUTTON_TOP_LEFT
        x2, y2 = self.HEAD_RESET_BUTTON_BOTTOM_RIGHT
        return x1 <= x <= x2 and y1 <= y <= y2

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self._is_in_head_reset_button(x, y):
            return
        if self._latest_head_pose is None:
            return
        head_pitch, head_yaw = self._latest_head_pose
        self._recalibrate_head_pose_zero(head_pitch=head_pitch, head_yaw=head_yaw)

    def _infer_compensated_from_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[float, float, np.ndarray, np.ndarray, float, float, float, float, float, float]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detected_faces = self.inference.face_detector(rgb, 0)
        if len(detected_faces) == 0:
            return None

        shape = self.inference.shape_predictor(frame_bgr, detected_faces[0])
        landmarks = self.inference._shape_to_np(shape)

        landmarks_sub = landmarks[self.inference.LANDMARK_SUBSET, :].astype(np.float64)
        landmarks_sub = landmarks_sub.reshape(6, 1, 2)

        camera_matrix = self.inference._camera_matrix(frame_w, frame_h)
        rvec, tvec = self.inference._estimate_head_pose(
            landmarks_sub=landmarks_sub,
            face_model_pts=self.inference.face_model_pts,
            camera_matrix=camera_matrix,
            camera_distortion=self.inference.camera_distortion,
        )

        face_patch, landmarks_normalized = self.inference._normalize_face_patch(
            frame_bgr=frame_bgr,
            landmarks_2d=landmarks_sub,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
        )

        with torch.no_grad():
            input_tensor = self.inference._preprocess(face_patch, self.inference.device)
            pred = self.inference.model(input_tensor).squeeze(0).detach().cpu().numpy()

        eye_pitch = float(pred[0])
        eye_yaw = float(pred[1])
        raw_head_yaw, raw_head_pitch = self._head_angles_from_rvec(rvec)
        centered_head_pitch, centered_head_yaw = self._center_head_angles(
            head_pitch=raw_head_pitch,
            head_yaw=raw_head_yaw,
        )

        compensated_pitch, compensated_yaw = self._apply_compensation(
            eye_pitch=eye_pitch,
            eye_yaw=eye_yaw,
            head_pitch=raw_head_pitch,
            head_yaw=raw_head_yaw,
        )

        return (
            compensated_pitch,
            compensated_yaw,
            face_patch,
            landmarks_normalized,
            eye_pitch,
            eye_yaw,
            centered_head_pitch,
            centered_head_yaw,
            raw_head_pitch,
            raw_head_yaw,
        )

    def _infer_for_calibration(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[float, float, np.ndarray, np.ndarray]]:
        result = self._infer_compensated_from_frame(frame_bgr)
        if result is None:
            return None
        compensated_pitch, compensated_yaw, face_patch, landmarks_normalized, *rest = result
        if len(rest) >= 6:
            raw_head_pitch = float(rest[4])
            raw_head_yaw = float(rest[5])
            self._latest_head_pose = (raw_head_pitch, raw_head_yaw)
        return compensated_pitch, compensated_yaw, face_patch, landmarks_normalized

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

        print("ETH-XGaze head-compensated realtime demo started.")
        print(
            "Controls: q / ESC to quit, c to recalibrate cursor mapping, "
            "h or click 'Set head 0,0' to recalibrate head neutral pose, "
            "[ and ] to tune yaw compensation, 0 to reset yaw compensation, "
            "- and = to tune pitch compensation, 9 to reset pitch compensation"
        )
        print(
            "Using compensation weights: "
            f"yaw={self.head_compensation_yaw:.3f}, pitch={self.head_compensation_pitch:.3f}, "
            f"max_head_angle_deg={np.rad2deg(self.head_compensation_max_angle_rad):.1f}"
        )

        if self.cursor_controller.cursor_enabled and self.cursor_controller.cursor is not None:
            calibrated = run_cursor_calibration(
                cap=cap,
                infer_gaze_from_frame=self._infer_for_calibration,
                cursor_controller=self.cursor_controller,
            )
            if not calibrated:
                print("Using fallback span-based cursor mapping.")
            if self._latest_head_pose is not None:
                self._recalibrate_head_pose_zero(
                    head_pitch=self._latest_head_pose[0],
                    head_yaw=self._latest_head_pose[1],
                )
                print("Auto head calibration complete (head0 set from startup calibration).", flush=True)

        latest_comp_pitch_yaw: Optional[Tuple[float, float]] = None
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue

                try:
                    result = self._infer_compensated_from_frame(frame_bgr)
                except Exception:
                    result = None

                if result is None:
                    cv2.imshow(self.WINDOW_NAME, frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                (
                    comp_pitch,
                    comp_yaw,
                    face_patch,
                    landmarks_normalized,
                    eye_pitch,
                    eye_yaw,
                    head_pitch,
                    head_yaw,
                    raw_head_pitch,
                    raw_head_yaw,
                ) = result

                latest_comp_pitch_yaw = (comp_pitch, comp_yaw)
                self._latest_head_pose = (raw_head_pitch, raw_head_yaw)

                eye_pitch_deg = float(np.rad2deg(eye_pitch))
                eye_yaw_deg = float(np.rad2deg(eye_yaw))
                head_pitch_deg = float(np.rad2deg(head_pitch))
                head_yaw_deg = float(np.rad2deg(head_yaw))
                raw_head_pitch_deg = float(np.rad2deg(raw_head_pitch))
                raw_head_yaw_deg = float(np.rad2deg(raw_head_yaw))
                comp_pitch_deg = float(np.rad2deg(comp_pitch))
                comp_yaw_deg = float(np.rad2deg(comp_yaw))

                print(
                    "eye_deg(p={:.2f}, y={:.2f})  head0_deg(p={:.2f}, y={:.2f})  raw_head_deg(p={:.2f}, y={:.2f})  comp_deg(p={:.2f}, y={:.2f})".format(
                        eye_pitch_deg,
                        eye_yaw_deg,
                        head_pitch_deg,
                        head_yaw_deg,
                        raw_head_pitch_deg,
                        raw_head_yaw_deg,
                        comp_pitch_deg,
                        comp_yaw_deg,
                    ),
                    flush=True,
                )

                self.cursor_controller.update_cursor(yaw_rad=comp_yaw, pitch_rad=comp_pitch)

                vis_patch = draw_gaze_arrow(face_patch, comp_pitch, comp_yaw)
                for x, y in landmarks_normalized.astype(int):
                    cv2.circle(vis_patch, (int(x), int(y)), 3, (0, 255, 0), -1)

                cv2.putText(
                    vis_patch,
                    f"comp deg pitch: {comp_pitch_deg:.1f}, yaw: {comp_yaw_deg:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_patch,
                    f"head0 deg pitch: {head_pitch_deg:.1f}, yaw: {head_yaw_deg:.1f}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (180, 255, 180),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_patch,
                    f"raw head deg pitch: {raw_head_pitch_deg:.1f}, yaw: {raw_head_yaw_deg:.1f}",
                    (10, 138),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (200, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_patch,
                    f"yaw gain: {self.head_compensation_yaw:.3f} ([ / ] / 0)",
                    (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (150, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_patch,
                    f"pitch gain: {self.head_compensation_pitch:.3f} (- / = / 9)",
                    (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (170, 240, 200),
                    2,
                    cv2.LINE_AA,
                )

                cv2.rectangle(
                    vis_patch,
                    self.HEAD_RESET_BUTTON_TOP_LEFT,
                    self.HEAD_RESET_BUTTON_BOTTOM_RIGHT,
                    (40, 120, 250),
                    -1,
                )
                cv2.rectangle(
                    vis_patch,
                    self.HEAD_RESET_BUTTON_TOP_LEFT,
                    self.HEAD_RESET_BUTTON_BOTTOM_RIGHT,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    vis_patch,
                    "Set head 0,0 (h)",
                    (16, 157),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(self.WINDOW_NAME, vis_patch)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("["):
                    self._set_head_compensation_yaw(self.head_compensation_yaw - self.head_compensation_yaw_step)
                elif key == ord("]"):
                    self._set_head_compensation_yaw(self.head_compensation_yaw + self.head_compensation_yaw_step)
                elif key == ord("0"):
                    self._set_head_compensation_yaw(0.0)
                elif key == ord("-"):
                    self._set_head_compensation_pitch(
                        self.head_compensation_pitch - self.head_compensation_pitch_step
                    )
                elif key == ord("="):
                    self._set_head_compensation_pitch(
                        self.head_compensation_pitch + self.head_compensation_pitch_step
                    )
                elif key == ord("9"):
                    self._set_head_compensation_pitch(0.0)
                elif key == ord("h") and self._latest_head_pose is not None:
                    self._recalibrate_head_pose_zero(
                        head_pitch=self._latest_head_pose[0],
                        head_yaw=self._latest_head_pose[1],
                    )
                if key == ord("c") and latest_comp_pitch_yaw is not None:
                    recalibrated = run_cursor_calibration(
                        cap=cap,
                        infer_gaze_from_frame=self._infer_for_calibration,
                        cursor_controller=self.cursor_controller,
                    )
                    if not recalibrated:
                        self.cursor_controller.calibrate_center(
                            yaw_rad=latest_comp_pitch_yaw[1],
                            pitch_rad=latest_comp_pitch_yaw[0],
                        )
                if key in (27, ord("q")):
                    break
        finally:
            self.stop()

        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime ETH-XGaze gaze tracking with head compensation.")
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        required=True,
        help="Path to ETH-XGaze checkpoint (e.g. epoch_24_ckpt.pth.tar).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument("--roi-size", type=int, default=224, help="Normalized face patch size.")
    parser.add_argument("--focal-norm", type=float, default=960.0, help="Virtual camera focal length.")
    parser.add_argument(
        "--distance-norm",
        type=float,
        default=600.0,
        help="Virtual camera distance used in normalization.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.2,
        help="Reserved for compatibility.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Camera focal length fx in pixels.")
    parser.add_argument("--fy", type=float, default=None, help="Camera focal length fy in pixels.")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point cx in pixels.")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point cy in pixels.")
    parser.add_argument(
        "--predictor-path",
        type=pathlib.Path,
        default=pathlib.Path("./modules/shape_predictor_68_face_landmarks.dat"),
        help="Path to dlib shape_predictor_68_face_landmarks.dat",
    )
    parser.add_argument(
        "--face-model-path",
        type=pathlib.Path,
        default=pathlib.Path("./face_model.txt"),
        help="Path to ETH-XGaze face_model.txt",
    )
    parser.add_argument(
        "--camera-calib-path",
        type=pathlib.Path,
        default=None,
        help="Optional OpenCV XML calibration file containing Camera_Matrix and Distortion_Coefficients",
    )
    parser.add_argument(
        "--disable-cursor-control",
        action="store_true",
        help="Disable cursor control and run visualization-only mode.",
    )
    parser.add_argument(
        "--cursor-yaw-span",
        type=float,
        default=0.6,
        help="Half-range in radians mapped to full screen width.",
    )
    parser.add_argument(
        "--cursor-pitch-span",
        type=float,
        default=0.4,
        help="Half-range in radians mapped to full screen height.",
    )
    parser.add_argument(
        "--cursor-ema-alpha",
        type=float,
        default=0.2,
        help="Smoothing factor for cursor mapping in (0, 1].",
    )
    parser.add_argument(
        "--head-compensation-yaw",
        type=float,
        default=0.3,
        help="Head yaw compensation gain applied subtractively (>= 0).",
    )
    parser.add_argument(
        "--head-compensation-pitch",
        type=float,
        default=0.0,
        help="Head pitch compensation gain applied subtractively (>= 0).",
    )
    parser.add_argument(
        "--head-compensation-max-angle-deg",
        type=float,
        default=20.0,
        help="Clamp absolute head pitch/yaw used for compensation to this angle in degrees.",
    )
    parser.add_argument(
        "--head-compensation-yaw-step",
        type=float,
        default=0.02,
        help="Amount added/subtracted from yaw compensation gain per key press.",
    )
    parser.add_argument(
        "--head-compensation-pitch-step",
        type=float,
        default=0.02,
        help="Amount added/subtracted from pitch compensation gain per key press.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        runner = HeadCompensatedETHXGaze(
            weights=args.weights,
            camera_index=args.camera,
            device=args.device,
            roi_size=args.roi_size,
            focal_norm=args.focal_norm,
            distance_norm=args.distance_norm,
            print_interval=args.print_interval,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            predictor_path=args.predictor_path,
            face_model_path=args.face_model_path,
            camera_calib_path=args.camera_calib_path,
            cursor_enabled=not args.disable_cursor_control,
            cursor_yaw_span=args.cursor_yaw_span,
            cursor_pitch_span=args.cursor_pitch_span,
            cursor_ema_alpha=args.cursor_ema_alpha,
            head_compensation_yaw=args.head_compensation_yaw,
            head_compensation_pitch=args.head_compensation_pitch,
            head_compensation_max_angle_deg=args.head_compensation_max_angle_deg,
            head_compensation_yaw_step=args.head_compensation_yaw_step,
            head_compensation_pitch_step=args.head_compensation_pitch_step,
        )
    except Exception as exc:
        print(f"Failed to initialize ETH-XGaze head-compensated runner: {exc}")
        return 1

    try:
        return runner.run()
    except Exception as exc:
        print(f"Runtime error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
