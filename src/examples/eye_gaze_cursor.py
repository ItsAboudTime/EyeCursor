"""
Realtime ETH-XGaze-style gaze estimation from webcam.

This implementation is intentionally close to ETH-XGaze demo.py:
1) detect 68-point landmarks with dlib,
2) estimate head pose via solvePnP,
3) normalize/crop a face patch,
4) run a ResNet gaze model,
5) output gaze yaw/pitch.

Usage:
  python -m src.examples.eth_xgaze_realtime --weights /path/to/epoch_24_ckpt.pth.tar

Controls:
  - q / ESC: quit
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional, Tuple

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models

from src.cursor import create_cursor


TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class XGazeNetwork(nn.Module):
    """ResNet50 + linear head that matches ETH-XGaze baseline structure."""

    def __init__(self) -> None:
        super().__init__()
        self.gaze_network = models.resnet50(weights=None)
        self.gaze_fc = nn.Sequential(nn.Linear(2048, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep layer flow explicit to match ETH-XGaze style backbone usage.
        x = self.gaze_network.conv1(x)
        x = self.gaze_network.bn1(x)
        x = self.gaze_network.relu(x)
        x = self.gaze_network.maxpool(x)
        x = self.gaze_network.layer1(x)
        x = self.gaze_network.layer2(x)
        x = self.gaze_network.layer3(x)
        x = self.gaze_network.layer4(x)
        feature = self.gaze_network.avgpool(x)
        feature = torch.flatten(feature, 1)
        gaze = self.gaze_fc(feature)
        return gaze


class RealtimeETHXGaze:
    # Dlib 68-point indices: eye corners + nose corners.
    _LANDMARK_SUBSET = [36, 39, 42, 45, 31, 35]

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
        cursor_invert_x: bool = False,
        cursor_invert_y: bool = False,
    ) -> None:
        self.weights = pathlib.Path(weights).expanduser().resolve()
        if not self.weights.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights}")

        self.camera_index = int(camera_index)
        self.roi_size = int(roi_size)
        self.focal_norm = float(focal_norm)
        self.distance_norm = float(distance_norm)
        self.print_interval = float(print_interval)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.cursor_enabled = bool(cursor_enabled)
        self.cursor_yaw_span = float(cursor_yaw_span)
        self.cursor_pitch_span = float(cursor_pitch_span)
        self.cursor_ema_alpha = float(cursor_ema_alpha)
        self.cursor_invert_x = bool(cursor_invert_x)
        self.cursor_invert_y = bool(cursor_invert_y)

        if self.cursor_yaw_span <= 0.0:
            raise ValueError("cursor_yaw_span must be > 0")
        if self.cursor_pitch_span <= 0.0:
            raise ValueError("cursor_pitch_span must be > 0")
        if not (0.0 < self.cursor_ema_alpha <= 1.0):
            raise ValueError("cursor_ema_alpha must be in (0, 1]")

        self.cursor = None
        self._cursor_bounds: Optional[Tuple[int, int, int, int]] = None
        self._cursor_calibration_yaw = 0.0
        self._cursor_calibration_pitch = 0.0
        self._cursor_ema_yaw: Optional[float] = None
        self._cursor_ema_pitch: Optional[float] = None

        self.predictor_path = (
            pathlib.Path(predictor_path).expanduser().resolve()
            if predictor_path is not None
            else pathlib.Path("./modules/shape_predictor_68_face_landmarks.dat").expanduser().resolve()
        )
        self.face_model_path = (
            pathlib.Path(face_model_path).expanduser().resolve()
            if face_model_path is not None
            else pathlib.Path("./face_model.txt").expanduser().resolve()
        )
        self.camera_calib_path = (
            pathlib.Path(camera_calib_path).expanduser().resolve() if camera_calib_path is not None else None
        )

        if not self.predictor_path.exists():
            raise FileNotFoundError(
                f"Dlib shape predictor not found: {self.predictor_path}. "
                "Provide --predictor-path to shape_predictor_68_face_landmarks.dat"
            )
        if not self.face_model_path.exists():
            raise FileNotFoundError(
                f"face_model.txt not found: {self.face_model_path}. "
                "Provide --face-model-path to ETH-XGaze face_model.txt"
            )

        self.device = self._select_device(device)
        self.model = self._load_model(self.weights)

        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(str(self.predictor_path))

        self.face_model = np.loadtxt(str(self.face_model_path)).astype(np.float64)
        # ETH-XGaze uses these 6 points from face_model.txt.
        self.face_model = self.face_model[[20, 23, 26, 29, 15, 19], :]
        self.face_model_pts = self.face_model.reshape(6, 1, 3)

        self.camera_matrix: Optional[np.ndarray] = None
        self.camera_distortion = np.zeros((5, 1), dtype=np.float64)
        self._load_camera_calibration()

        self._init_cursor()

        self.cap: Optional[cv2.VideoCapture] = None

    def _init_cursor(self) -> None:
        if not self.cursor_enabled:
            return

        try:
            self.cursor = create_cursor()
            self._cursor_bounds = self.cursor.get_virtual_bounds()
            print(f"cursor control enabled; virtual bounds={self._cursor_bounds}")
        except Exception as exc:
            self.cursor = None
            self._cursor_bounds = None
            self.cursor_enabled = False
            print(f"warning: failed to initialize cursor control: {exc}")

    @staticmethod
    def _select_device(name: str) -> torch.device:
        choice = str(name).strip().lower()
        if choice == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if choice == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _load_model(self, weights_path: pathlib.Path) -> nn.Module:
        model = XGazeNetwork().to(self.device)
        checkpoint = torch.load(str(weights_path), map_location=self.device)

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

        if not isinstance(state_dict, dict):
            raise RuntimeError("Unsupported checkpoint format. Expected a state_dict-like object.")

        cleaned = {}
        for k, v in state_dict.items():
            name = str(k)
            if name.startswith("module."):
                name = name[len("module.") :]
            cleaned[name] = v

        try:
            model.load_state_dict(cleaned, strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load checkpoint strictly. Ensure weights match ETH-XGaze model architecture."
            ) from exc

        model.eval()
        return model

    def _load_camera_calibration(self) -> None:
        if self.camera_calib_path is None:
            return
        if not self.camera_calib_path.exists():
            print(f"warning: camera calibration file not found at {self.camera_calib_path}; using estimated intrinsics")
            return

        fs = cv2.FileStorage(str(self.camera_calib_path), cv2.FILE_STORAGE_READ)
        try:
            cam_mtx = fs.getNode("Camera_Matrix").mat()
            dist = fs.getNode("Distortion_Coefficients").mat()
            if cam_mtx is not None:
                self.camera_matrix = np.asarray(cam_mtx, dtype=np.float64)
            if dist is not None:
                self.camera_distortion = np.asarray(dist, dtype=np.float64)
            print(f"loaded camera calibration from: {self.camera_calib_path}")
        finally:
            fs.release()

    def _camera_matrix(self, width: int, height: int) -> np.ndarray:
        if self.camera_matrix is not None:
            return self.camera_matrix

        focal = float(max(width, height))
        fx = float(self.fx) if self.fx is not None else focal
        fy = float(self.fy) if self.fy is not None else focal
        cx = float(self.cx) if self.cx is not None else (float(width) * 0.5)
        cy = float(self.cy) if self.cy is not None else (float(height) * 0.5)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def _shape_to_np(shape) -> np.ndarray:
        pts = np.zeros((68, 2), dtype=np.float64)
        for i in range(68):
            pts[i] = (shape.part(i).x, shape.part(i).y)
        return pts

    @staticmethod
    def _estimate_head_pose(
        landmarks_sub: np.ndarray,
        face_model_pts: np.ndarray,
        camera_matrix: np.ndarray,
        camera_distortion: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        success, rvec, tvec = cv2.solvePnP(
            face_model_pts,
            landmarks_sub,
            camera_matrix,
            camera_distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            raise RuntimeError("solvePnP initial fit failed")

        success, rvec, tvec = cv2.solvePnP(
            face_model_pts,
            landmarks_sub,
            camera_matrix,
            camera_distortion,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise RuntimeError("solvePnP iterative fit failed")
        return rvec, tvec

    def _normalize_face_patch(
        self,
        frame_bgr: np.ndarray,
        landmarks_2d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        hR = cv2.Rodrigues(rvec)[0]
        face_model = self.face_model

        Fc = np.dot(hR, face_model.T) + tvec.reshape((3, 1))
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

        distance = float(np.linalg.norm(face_center))
        if distance < 1e-6:
            raise RuntimeError("Invalid face-center distance during normalization.")

        z_scale = self.distance_norm / distance

        cam_norm = np.array(
            [
                [self.focal_norm, 0.0, self.roi_size / 2.0],
                [0.0, self.focal_norm, self.roi_size / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]], dtype=np.float64)

        hRx = hR[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)

        R = np.c_[right, down, forward].T
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))

        img_warped = cv2.warpPerspective(frame_bgr, W, (self.roi_size, self.roi_size))
        landmarks_warped = cv2.perspectiveTransform(landmarks_2d.astype(np.float64), W)
        landmarks_warped = landmarks_warped.reshape(landmarks_2d.shape[0], 2)
        return img_warped, landmarks_warped

    @staticmethod
    def _preprocess(face_patch_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
        input_rgb = face_patch_bgr[:, :, [2, 1, 0]]
        tensor = TRANSFORM(input_rgb)
        tensor = tensor.float().to(device)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))
        return tensor

    @staticmethod
    def _draw_gaze_arrow(image: np.ndarray, pitch_rad: float, yaw_rad: float) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        length = float(min(h, w)) * 0.45
        center = np.array([w * 0.5, h * 0.5], dtype=np.float32)

        dx = -length * np.sin(yaw_rad) * np.cos(pitch_rad)
        dy = -length * np.sin(pitch_rad)

        start = tuple(np.round(center).astype(np.int32))
        end = tuple(np.round(center + np.array([dx, dy], dtype=np.float32)).astype(np.int32))
        cv2.arrowedLine(out, start, end, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)
        return out

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _calibrate_cursor_center(self, yaw_rad: float, pitch_rad: float) -> None:
        self._cursor_calibration_yaw = -yaw_rad
        self._cursor_calibration_pitch = -pitch_rad
        self._cursor_ema_yaw = None
        self._cursor_ema_pitch = None
        print(
            f"cursor calibrated: yaw_offset={self._cursor_calibration_yaw:.4f}, "
            f"pitch_offset={self._cursor_calibration_pitch:.4f}"
        )

    def _cursor_target_from_gaze(self, yaw_rad: float, pitch_rad: float) -> Optional[Tuple[int, int]]:
        if not self.cursor_enabled or self.cursor is None or self._cursor_bounds is None:
            return None

        yaw_adj = yaw_rad + self._cursor_calibration_yaw
        pitch_adj = pitch_rad + self._cursor_calibration_pitch

        if self.cursor_invert_x:
            yaw_adj = -yaw_adj
        if self.cursor_invert_y:
            pitch_adj = -pitch_adj

        if self._cursor_ema_yaw is None:
            self._cursor_ema_yaw = yaw_adj
            self._cursor_ema_pitch = pitch_adj
        else:
            if self._cursor_ema_pitch is None:
                self._cursor_ema_pitch = pitch_adj
            alpha = self.cursor_ema_alpha
            ema_yaw = float(self._cursor_ema_yaw)
            ema_pitch = float(self._cursor_ema_pitch)
            self._cursor_ema_yaw = alpha * yaw_adj + (1.0 - alpha) * ema_yaw
            self._cursor_ema_pitch = alpha * pitch_adj + (1.0 - alpha) * ema_pitch

        norm_x = self._clip01((self._cursor_ema_yaw + self.cursor_yaw_span) / (2.0 * self.cursor_yaw_span))
        norm_y = self._clip01((self._cursor_ema_pitch + self.cursor_pitch_span) / (2.0 * self.cursor_pitch_span))

        minx, miny, maxx, maxy = self._cursor_bounds
        width = maxx - minx + 1
        height = maxy - miny + 1

        target_x = minx + int(round(norm_x * (width - 1)))
        target_y = miny + int(round(norm_y * (height - 1)))
        return target_x, target_y

    def _update_cursor(self, yaw_rad: float, pitch_rad: float) -> None:
        if self.cursor is None:
            return

        target = self._cursor_target_from_gaze(yaw_rad=yaw_rad, pitch_rad=pitch_rad)
        if target is None:
            return

        self.cursor.step_towards(*target)

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
        print("Controls: q / ESC to quit, c to calibrate cursor center")
        print("Showing normalized face patch with gaze arrow.")

        latest_pitch_yaw: Optional[Tuple[float, float]] = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue

                frame_h, frame_w = frame_bgr.shape[:2]
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detected_faces = self.face_detector(rgb, 0)
                if len(detected_faces) == 0:
                    cv2.imshow("ETH-XGaze Realtime", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                shape = self.shape_predictor(frame_bgr, detected_faces[0])
                landmarks = self._shape_to_np(shape)

                landmarks_sub = landmarks[self._LANDMARK_SUBSET, :].astype(np.float64)
                landmarks_sub = landmarks_sub.reshape(6, 1, 2)

                camera_matrix = self._camera_matrix(frame_w, frame_h)

                try:
                    rvec, tvec = self._estimate_head_pose(
                        landmarks_sub=landmarks_sub,
                        face_model_pts=self.face_model_pts,
                        camera_matrix=camera_matrix,
                        camera_distortion=self.camera_distortion,
                    )
                    face_patch, landmarks_normalized = self._normalize_face_patch(
                        frame_bgr=frame_bgr,
                        landmarks_2d=landmarks_sub,
                        rvec=rvec,
                        tvec=tvec,
                        camera_matrix=camera_matrix,
                    )
                except Exception:
                    cv2.imshow("ETH-XGaze Realtime", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                with torch.no_grad():
                    input_tensor = self._preprocess(face_patch, self.device)
                    pred = self.model(input_tensor).squeeze(0).detach().cpu().numpy()

                pitch_rad = float(pred[0])
                yaw_rad = float(pred[1])

                latest_pitch_yaw = (pitch_rad, yaw_rad)

                print(f"pitch: {pitch_rad:.6f}, yaw: {yaw_rad:.6f}", flush=True)

                self._update_cursor(yaw_rad=yaw_rad, pitch_rad=pitch_rad)

                vis_patch = self._draw_gaze_arrow(face_patch, pitch_rad, yaw_rad)
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
                    self._calibrate_cursor_center(
                        yaw_rad=latest_pitch_yaw[1],
                        pitch_rad=latest_pitch_yaw[0],
                    )
                if key in (27, ord("q")):
                    break
        finally:
            self.stop()

        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime ETH-XGaze-style webcam gaze tracking.")
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
        "--cursor-invert-x",
        action="store_true",
        help="Invert horizontal gaze-to-cursor mapping.",
    )
    parser.add_argument(
        "--cursor-invert-y",
        action="store_true",
        help="Invert vertical gaze-to-cursor mapping.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        runner = RealtimeETHXGaze(
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
            cursor_invert_x=args.cursor_invert_x,
            cursor_invert_y=args.cursor_invert_y,
        )
    except Exception as exc:
        print(f"Failed to initialize ETH-XGaze realtime runner: {exc}")
        return 1

    try:
        return runner.run()
    except Exception as exc:
        print(f"Runtime error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
