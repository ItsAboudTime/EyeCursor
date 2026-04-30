from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.eye_tracking.models.xgaze_network import XGazeNetwork


TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ETHXGazeInference:
    # Dlib 68-point indices: eye corners + nose corners.
    LANDMARK_SUBSET = [36, 39, 42, 45, 31, 35]

    def __init__(
        self,
        weights: pathlib.Path,
        device: str = "auto",
        roi_size: int = 224,
        focal_norm: float = 960.0,
        distance_norm: float = 600.0,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        predictor_path: Optional[pathlib.Path] = None,
        face_model_path: Optional[pathlib.Path] = None,
        camera_calib_path: Optional[pathlib.Path] = None,
    ) -> None:
        self.weights = pathlib.Path(weights).expanduser().resolve()
        if not self.weights.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights}")

        self.roi_size = int(roi_size)
        self.focal_norm = float(focal_norm)
        self.distance_norm = float(distance_norm)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

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

        # Stashed for visualizers to read after each infer_from_frame call.
        self.last_dlib_landmarks: Optional[np.ndarray] = None
        self.last_face_box: Optional[Tuple[int, int, int, int]] = None

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

    def infer_from_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[float, float, np.ndarray, np.ndarray]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detected_faces = self.face_detector(rgb, 0)
        if len(detected_faces) == 0:
            self.last_dlib_landmarks = None
            self.last_face_box = None
            return None

        face_rect = detected_faces[0]
        self.last_face_box = (
            int(face_rect.left()),
            int(face_rect.top()),
            int(face_rect.right()),
            int(face_rect.bottom()),
        )
        shape = self.shape_predictor(frame_bgr, face_rect)
        landmarks = self._shape_to_np(shape)
        self.last_dlib_landmarks = landmarks.copy()

        landmarks_sub = landmarks[self.LANDMARK_SUBSET, :].astype(np.float64)
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
