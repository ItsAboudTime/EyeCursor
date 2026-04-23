from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional, Tuple

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
    _CALIB_WINDOW_NAME = "Gaze Calibration"

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
        self._cursor_affine: Optional[np.ndarray] = None
        self._cursor_norm_bounds: Optional[Tuple[float, float, float, float]] = None

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

    def _infer_gaze_from_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[float, float, np.ndarray, np.ndarray]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detected_faces = self.face_detector(rgb, 0)
        if len(detected_faces) == 0:
            return None

        shape = self.shape_predictor(frame_bgr, detected_faces[0])
        landmarks = self._shape_to_np(shape)

        landmarks_sub = landmarks[self._LANDMARK_SUBSET, :].astype(np.float64)
        landmarks_sub = landmarks_sub.reshape(6, 1, 2)

        camera_matrix = self._camera_matrix(frame_w, frame_h)
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

        with torch.no_grad():
            input_tensor = self._preprocess(face_patch, self.device)
            pred = self.model(input_tensor).squeeze(0).detach().cpu().numpy()

        pitch_rad = float(pred[0])
        yaw_rad = float(pred[1])
        return pitch_rad, yaw_rad, face_patch, landmarks_normalized

    @staticmethod
    def _calibration_points() -> List[Tuple[float, float]]:
        # Start from center to settle the user, then spread to all edges/corners.
        return [
            (0.50, 0.50),
            (0.08, 0.08),
            (0.50, 0.08),
            (0.92, 0.08),
            (0.08, 0.50),
            (0.92, 0.50),
            (0.08, 0.92),
            (0.50, 0.92),
            (0.92, 0.92),
        ]

    @staticmethod
    def _target_abs_point(
        bounds: Tuple[int, int, int, int],
        target_norm: Tuple[float, float],
    ) -> Tuple[int, int]:
        minx, miny, maxx, maxy = bounds
        width = maxx - minx + 1
        height = maxy - miny + 1
        tx = minx + int(round(target_norm[0] * (width - 1)))
        ty = miny + int(round(target_norm[1] * (height - 1)))
        return tx, ty

    def _draw_calibration_screen(
        self,
        screen_size: Tuple[int, int],
        target_norm: Tuple[float, float],
        step_index: int,
        step_total: int,
        msg: str,
    ) -> np.ndarray:
        width, height = screen_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        tx = int(round(target_norm[0] * (width - 1)))
        ty = int(round(target_norm[1] * (height - 1)))

        cv2.circle(canvas, (tx, ty), 34, (80, 80, 80), 2, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 16, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (tx - 45, ty), (tx + 45, ty), (0, 180, 220), 2, cv2.LINE_AA)
        cv2.line(canvas, (tx, ty - 45), (tx, ty + 45), (0, 180, 220), 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"Calibration {step_index}/{step_total}",
            (36, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Look at the marker, then press SPACE to capture.",
            (36, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "ESC/q: quit  |  s: skip calibration",
            (36, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            msg,
            (36, height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _capture_gaze_average(
        self,
        cap: cv2.VideoCapture,
        sample_count: int = 24,
        max_frames: int = 140,
    ) -> Optional[Tuple[float, float, float, float]]:
        samples: List[Tuple[float, float]] = []
        for _ in range(max_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            try:
                result = self._infer_gaze_from_frame(frame_bgr)
            except Exception:
                result = None
            if result is None:
                continue

            pitch_rad, yaw_rad, _, _ = result
            yaw_adj = yaw_rad
            pitch_adj = -pitch_rad
            samples.append((yaw_adj, pitch_adj))

            if len(samples) >= sample_count:
                break

        if len(samples) < max(8, sample_count // 2):
            return None

        arr = np.asarray(samples, dtype=np.float64)
        med = np.median(arr, axis=0)
        delta = arr - med.reshape(1, 2)
        dist = np.linalg.norm(delta, axis=1)
        keep = dist <= np.percentile(dist, 80)
        robust = arr[keep]
        if robust.shape[0] < 6:
            robust = arr

        mean_yaw = float(np.mean(robust[:, 0]))
        mean_pitch = float(np.mean(robust[:, 1]))
        std_yaw = float(np.std(robust[:, 0]))
        std_pitch = float(np.std(robust[:, 1]))
        return mean_yaw, mean_pitch, std_yaw, std_pitch

    def _prepare_calibration_window(self, screen_w: int, screen_h: int) -> None:
        cv2.namedWindow(self._CALIB_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._CALIB_WINDOW_NAME, screen_w, screen_h)
        cv2.moveWindow(self._CALIB_WINDOW_NAME, 0, 0)
        try:
            cv2.setWindowProperty(self._CALIB_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

    def _fit_cursor_calibration(
        self,
        gaze_samples: np.ndarray,
        target_points: np.ndarray,
    ) -> bool:
        affine, inlier_mask = cv2.estimateAffine2D(
            gaze_samples,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.06,
            maxIters=3000,
            confidence=0.995,
        )
        if affine is None:
            return False

        ones = np.ones((gaze_samples.shape[0], 1), dtype=np.float64)
        augmented = np.concatenate([gaze_samples, ones], axis=1)
        pred = np.dot(augmented, affine.T)

        errors = np.linalg.norm(pred - target_points, axis=1)
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))

        inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else gaze_samples.shape[0]
        min_inliers = max(6, int(round(gaze_samples.shape[0] * 0.67)))
        if inliers < min_inliers or mean_err > 0.12 or max_err > 0.24:
            return False

        min_x = float(np.min(pred[:, 0]))
        max_x = float(np.max(pred[:, 0]))
        min_y = float(np.min(pred[:, 1]))
        max_y = float(np.max(pred[:, 1]))
        if (max_x - min_x) < 1e-4 or (max_y - min_y) < 1e-4:
            return False

        self._cursor_affine = affine.astype(np.float64)
        self._cursor_norm_bounds = (min_x, max_x, min_y, max_y)
        self._cursor_ema_yaw = None
        self._cursor_ema_pitch = None
        print(
            f"cursor calibration complete: inliers={inliers}/{gaze_samples.shape[0]}, "
            f"mean_err={mean_err:.4f}, max_err={max_err:.4f}"
        )
        return True

    def _run_cursor_calibration(self, cap: cv2.VideoCapture) -> bool:
        if not self.cursor_enabled or self.cursor is None or self._cursor_bounds is None:
            return False

        minx, miny, maxx, maxy = self._cursor_bounds
        screen_w = maxx - minx + 1
        screen_h = maxy - miny + 1
        if screen_w <= 10 or screen_h <= 10:
            print("warning: invalid screen bounds for calibration; skipping")
            return False

        target_norms = self._calibration_points()
        collected_gaze: List[Tuple[float, float]] = []
        collected_targets: List[Tuple[float, float]] = []

        self._prepare_calibration_window(screen_w, screen_h)
        print("Starting startup gaze calibration. Follow the on-screen points.")

        for idx, target_norm in enumerate(target_norms, start=1):
            msg = "Hold gaze on target, press SPACE."
            while True:
                canvas = self._draw_calibration_screen(
                    screen_size=(screen_w, screen_h),
                    target_norm=target_norm,
                    step_index=idx,
                    step_total=len(target_norms),
                    msg=msg,
                )
                cv2.imshow(self._CALIB_WINDOW_NAME, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return False
                if key == ord("s"):
                    print("Calibration skipped by user.")
                    cv2.destroyWindow(self._CALIB_WINDOW_NAME)
                    return False
                if key != ord(" "):
                    continue

                sample = self._capture_gaze_average(cap=cap)
                if sample is None:
                    msg = "Face/gaze not stable. Re-align and press SPACE again."
                    continue

                mean_yaw, mean_pitch, std_yaw, std_pitch = sample
                stability = max(std_yaw, std_pitch)
                if stability > 0.06:
                    msg = f"Too noisy (stability={stability:.3f}). Keep still and retry."
                    continue

                collected_gaze.append((mean_yaw, mean_pitch))
                collected_targets.append(target_norm)

                target_abs = self._target_abs_point(self._cursor_bounds, target_norm)
                self.cursor.step_towards(*target_abs)

                msg = f"Captured: yaw={mean_yaw:.3f}, pitch={mean_pitch:.3f}"
                break

        cv2.destroyWindow(self._CALIB_WINDOW_NAME)

        gaze_np = np.asarray(collected_gaze, dtype=np.float64)
        target_np = np.asarray(collected_targets, dtype=np.float64)
        if gaze_np.shape[0] < 6:
            print("warning: insufficient calibration points; falling back to span mapping")
            return False

        ok = self._fit_cursor_calibration(gaze_samples=gaze_np, target_points=target_np)
        if not ok:
            print("warning: calibration quality was not sufficient; falling back to span mapping")
            return False
        return True

    def _calibrate_cursor_center(self, yaw_rad: float, pitch_rad: float) -> None:
        self._cursor_calibration_yaw = -yaw_rad
        self._cursor_calibration_pitch = pitch_rad
        self._cursor_ema_yaw = None
        self._cursor_ema_pitch = None
        print(
            f"cursor calibrated: yaw_offset={self._cursor_calibration_yaw:.4f}, "
            f"pitch_offset={self._cursor_calibration_pitch:.4f}"
        )

    def _cursor_target_from_gaze(self, yaw_rad: float, pitch_rad: float) -> Optional[Tuple[int, int]]:
        if not self.cursor_enabled or self.cursor is None or self._cursor_bounds is None:
            return None

        yaw_adj = yaw_rad
        pitch_adj = -pitch_rad
        yaw_adj += self._cursor_calibration_yaw
        pitch_adj += self._cursor_calibration_pitch

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

        if self._cursor_affine is not None:
            point = np.array([self._cursor_ema_yaw, self._cursor_ema_pitch, 1.0], dtype=np.float64)
            mapped = np.dot(self._cursor_affine, point)
            norm_x = float(mapped[0])
            norm_y = float(mapped[1])

            if self._cursor_norm_bounds is not None:
                min_x, max_x, min_y, max_y = self._cursor_norm_bounds
                norm_x = (norm_x - min_x) / (max_x - min_x)
                norm_y = (norm_y - min_y) / (max_y - min_y)
        else:
            norm_x = (self._cursor_ema_yaw + self.cursor_yaw_span) / (2.0 * self.cursor_yaw_span)
            norm_y = (self._cursor_ema_pitch + self.cursor_pitch_span) / (2.0 * self.cursor_pitch_span)

        norm_x = self._clip01(norm_x)
        norm_y = self._clip01(norm_y)

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
        print("Controls: q / ESC to quit, c to recalibrate cursor mapping")
        print("Showing normalized face patch with gaze arrow.")

        if self.cursor_enabled and self.cursor is not None:
            calibrated = self._run_cursor_calibration(cap)
            if not calibrated:
                print("Using fallback span-based cursor mapping.")

        latest_pitch_yaw: Optional[Tuple[float, float]] = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue

                try:
                    result = self._infer_gaze_from_frame(frame_bgr)
                except Exception:
                    result = None

                if result is None:
                    cv2.imshow("ETH-XGaze Realtime", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                pitch_rad, yaw_rad, face_patch, landmarks_normalized = result

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
                    recalibrated = self._run_cursor_calibration(cap)
                    if not recalibrated:
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
