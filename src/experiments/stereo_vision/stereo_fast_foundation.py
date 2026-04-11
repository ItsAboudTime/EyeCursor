from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch


def _validate_required_path(raw_path: str, label: str) -> Path:
    p = Path(raw_path).expanduser()
    placeholder_tokens = ("/path/to/", "<", ">")
    if any(token in raw_path for token in placeholder_tokens):
        raise FileNotFoundError(
            f"{label} looks like a placeholder: {raw_path}. Please replace it with a real path."
        )
    if not p.exists():
        raise FileNotFoundError(f"{label} does not exist: {p.resolve()}")
    return p.resolve()


def _probe_available_cameras(max_index: int = 10) -> list[int]:
    available: list[int] = []
    for idx in range(max(1, int(max_index))):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                available.append(idx)
        cap.release()
    return available


@dataclass
class StereoIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float

    @classmethod
    def from_file(cls, intrinsic_file: str | Path) -> "StereoIntrinsics":
        file_path = _validate_required_path(str(intrinsic_file), "Intrinsic file")
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            raise ValueError(
                "Intrinsic file must contain 2 lines: flattened 3x3 K matrix and baseline in meters."
            )

        k_values = [float(v) for v in lines[0].split()]
        if len(k_values) != 9:
            raise ValueError("First line in intrinsic file must contain exactly 9 float values.")

        baseline_m = float(lines[1].strip())
        k = np.asarray(k_values, dtype=np.float32).reshape(3, 3)
        return cls(
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            baseline_m=baseline_m,
        )


class _Padder:
    """Pad tensors to make height/width divisible by a factor, then unpad output."""

    def __init__(self, height: int, width: int, divis_by: int = 32) -> None:
        pad_h = (divis_by - (height % divis_by)) % divis_by
        pad_w = (divis_by - (width % divis_by)) % divis_by
        self.top = 0
        self.bottom = pad_h
        self.left = 0
        self.right = pad_w
        self.height = height
        self.width = width

    def pad2(self, left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = (self.left, self.right, self.top, self.bottom)
        left = torch.nn.functional.pad(left, pad, mode="replicate")
        right = torch.nn.functional.pad(right, pad, mode="replicate")
        return left, right

    def unpad(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., : self.height, : self.width]


class FastFoundationStereoRealtime:
    def __init__(
        self,
        fast_foundation_repo: str | Path,
        model_path: str | Path,
        intrinsic_file: str | Path,
        scale: float = 0.5,
        valid_iters: int = 4,
        max_disp: int = 192,
        remove_invisible: bool = True,
        use_amp: bool = True,
    ) -> None:
        self.fast_foundation_repo = _validate_required_path(
            str(fast_foundation_repo), "Fast-FoundationStereo repo"
        )
        self.model_path = _validate_required_path(str(model_path), "Model file")
        self.intrinsics = StereoIntrinsics.from_file(intrinsic_file)
        self.scale = float(scale)
        self.valid_iters = int(valid_iters)
        self.max_disp = int(max_disp)
        self.remove_invisible = bool(remove_invisible)
        self.use_amp = bool(use_amp)

        if str(self.fast_foundation_repo) not in sys.path:
            sys.path.insert(0, str(self.fast_foundation_repo))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = self.device.type == "cuda"
        torch.autograd.set_grad_enabled(False)

        self.model = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        self.model.args.valid_iters = self.valid_iters
        self.model.args.max_disp = self.max_disp
        self.model = self.model.to(self.device).eval()

    def infer_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if left_bgr is None or right_bgr is None:
            raise ValueError("Both left and right frames are required.")

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        if self.scale != 1.0:
            left_rgb = cv2.resize(left_rgb, dsize=None, fx=self.scale, fy=self.scale)
            right_rgb = cv2.resize(right_rgb, dsize=(left_rgb.shape[1], left_rgb.shape[0]))

        left_t = torch.from_numpy(left_rgb).to(self.device).float().permute(2, 0, 1)[None]
        right_t = torch.from_numpy(right_rgb).to(self.device).float().permute(2, 0, 1)[None]

        padder = _Padder(height=left_t.shape[2], width=left_t.shape[3], divis_by=32)
        left_t, right_t = padder.pad2(left_t, right_t)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                    disp = self.model.forward(
                        left_t,
                        right_t,
                        iters=self.valid_iters,
                        test_mode=True,
                        optimize_build_volume="pytorch1",
                    )
            else:
                disp = self.model.forward(
                    left_t,
                    right_t,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )

        disp = padder.unpad(disp.float())
        disp_np = disp.squeeze().detach().cpu().numpy().astype(np.float32)
        disp_np = np.clip(disp_np, 0.0, None)
        return left_rgb, right_rgb, disp_np

    def disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        fx_scaled = self.intrinsics.fx * self.scale
        invalid = disp <= 1e-6
        depth = (fx_scaled * self.intrinsics.baseline_m) / np.where(invalid, 1.0, disp)

        if self.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
            invalid |= (xx - disp) < 0

        depth[invalid] = np.inf
        return depth.astype(np.float32)

    @staticmethod
    def colorize_disparity(disp: np.ndarray) -> np.ndarray:
        valid = np.isfinite(disp) & (disp > 0.0)
        vis = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
        if not np.any(valid):
            return vis

        lo = float(np.percentile(disp[valid], 2.0))
        hi = float(np.percentile(disp[valid], 98.0))
        if hi <= lo:
            hi = lo + 1.0

        norm = ((disp - lo) / (hi - lo)).clip(0.0, 1.0)
        vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~valid] = 0
        return vis

    @staticmethod
    def colorize_depth(depth: np.ndarray, z_far: float) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0.0)
        vis = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        if not np.any(valid):
            return vis

        clipped = np.clip(depth, 0.0, z_far)
        inv = 1.0 - (clipped / max(1e-6, z_far))
        vis = cv2.applyColorMap((inv * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        vis[~valid] = 0
        return vis


def run_fast_foundation_stereo_demo(
    fast_foundation_repo: str | Path,
    model_path: str | Path,
    intrinsic_file: str | Path,
    left_camera_index: int = 6,
    right_camera_index: int = 4,
    width: int = 640,
    height: int = 480,
    scale: float = 0.5,
    valid_iters: int = 4,
    max_disp: int = 192,
    z_far: float = 5.0,
    infer_every_n_frames: int = 2,
    show_depth_panel: bool = False,
    camera_probe_max_index: int = 10,
) -> int:
    cv2.setUseOptimized(True)

    engine = FastFoundationStereoRealtime(
        fast_foundation_repo=fast_foundation_repo,
        model_path=model_path,
        intrinsic_file=intrinsic_file,
        scale=scale,
        valid_iters=valid_iters,
        max_disp=max_disp,
        remove_invisible=True,
        use_amp=True,
    )

    left_cap = cv2.VideoCapture(int(left_camera_index))
    right_cap = cv2.VideoCapture(int(right_camera_index))
    for cap in (left_cap, right_cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not left_cap.isOpened() or not right_cap.isOpened():
        left_cap.release()
        right_cap.release()
        available = _probe_available_cameras(max_index=camera_probe_max_index)
        available_text = ", ".join(str(i) for i in available) if available else "none"
        raise RuntimeError(
            "Could not open stereo cameras "
            f"(left={left_camera_index}, right={right_camera_index}). "
            f"Detected camera indices: [{available_text}]."
        )

    print("Fast-FoundationStereo live stereo running.")
    print("Press 'q' or ESC to quit.")

    fps_ema: Optional[float] = None
    prev_t = time.perf_counter()
    frame_idx = 0
    cached_left_rgb: Optional[np.ndarray] = None
    cached_right_rgb: Optional[np.ndarray] = None
    cached_disp: Optional[np.ndarray] = None
    cached_depth: Optional[np.ndarray] = None

    infer_every_n_frames = max(1, int(infer_every_n_frames))

    try:
        while True:
            ok_l, frame_l = left_cap.read()
            ok_r, frame_r = right_cap.read()
            if not ok_l or not ok_r:
                continue

            should_infer = (
                cached_disp is None
                or cached_depth is None
                or (frame_idx % infer_every_n_frames == 0)
            )

            if should_infer:
                left_rgb, right_rgb, disp = engine.infer_disparity(frame_l, frame_r)
                depth = engine.disparity_to_depth(disp)
                cached_left_rgb = left_rgb
                cached_right_rgb = right_rgb
                cached_disp = disp
                cached_depth = depth
            else:
                left_rgb = cached_left_rgb
                right_rgb = cached_right_rgb
                disp = cached_disp
                depth = cached_depth

            if left_rgb is None or right_rgb is None or disp is None or depth is None:
                continue

            disp_vis = engine.colorize_disparity(disp)
            depth_vis = None
            if show_depth_panel:
                depth_vis = engine.colorize_depth(depth, z_far=float(z_far))

            # Lightweight display path for higher FPS.
            display_items = [
                cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR),
                disp_vis,
            ]
            if depth_vis is not None:
                display_items.append(depth_vis)
            display = np.hstack(display_items)

            now = time.perf_counter()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)

            cv2.putText(
                display,
                f"FPS: {fps_ema:.1f} | scale={scale} | iters={valid_iters} | infer_every={infer_every_n_frames}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Fast-FoundationStereo: left | right | disparity | depth", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_idx += 1
    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time stereo depth visualization using NVIDIA Fast-FoundationStereo"
    )
    parser.add_argument(
        "--fast-foundation-repo",
        type=str,
        required=True,
        help="Path to local Fast-FoundationStereo repository root",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model_best_bp2_serialize.pth file",
    )
    parser.add_argument(
        "--intrinsic-file",
        type=str,
        required=True,
        help="Path to stereo intrinsic text file (line1: flattened K, line2: baseline meters)",
    )
    parser.add_argument("--left-camera-index", type=int, default=6)
    parser.add_argument("--right-camera-index", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--valid-iters", type=int, default=2)
    parser.add_argument("--max-disp", type=int, default=128)
    parser.add_argument("--z-far", type=float, default=5.0)
    parser.add_argument(
        "--infer-every-n-frames",
        type=int,
        default=2,
        help="Run heavy model inference every N frames and reuse last depth in-between",
    )
    parser.add_argument(
        "--show-depth-panel",
        type=int,
        default=0,
        help="Show depth color panel in 2D GUI (1=yes, 0=no). Turning off is faster.",
    )
    parser.add_argument(
        "--probe-cameras",
        type=int,
        default=0,
        help="Probe and print available camera indices, then exit (1=yes, 0=no)",
    )
    parser.add_argument(
        "--camera-probe-max-index",
        type=int,
        default=10,
        help="Maximum camera index range for probing",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if bool(args.probe_cameras):
        available = _probe_available_cameras(max_index=args.camera_probe_max_index)
        if available:
            print("Detected camera indices:", ", ".join(str(i) for i in available))
        else:
            print("No camera indices detected in probe range.")
        return 0

    return run_fast_foundation_stereo_demo(
        fast_foundation_repo=args.fast_foundation_repo,
        model_path=args.model_path,
        intrinsic_file=args.intrinsic_file,
        left_camera_index=args.left_camera_index,
        right_camera_index=args.right_camera_index,
        width=args.width,
        height=args.height,
        scale=args.scale,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        z_far=args.z_far,
        infer_every_n_frames=args.infer_every_n_frames,
        show_depth_panel=bool(args.show_depth_panel),
        camera_probe_max_index=args.camera_probe_max_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
