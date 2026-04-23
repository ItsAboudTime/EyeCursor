from __future__ import annotations

import argparse
import pathlib
import sys

from src.eye_tracking.pipelines.realtime_eth_xgaze import RealtimeETHXGaze


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
