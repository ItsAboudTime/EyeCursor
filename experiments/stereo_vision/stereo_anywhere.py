"""
Stereo Anywhere disparity map experiment.

Captures rectified stereo frames from two webcams (indices 4 and 6)
and computes a dense disparity map using the Stereo Anywhere model.

Requires:
  1. Clone https://github.com/bartn8/stereoanywhere somewhere on disk.
     Also install its deps:  pip install -r requirements.txt

  2. Download pretrained weights:
     - Stereo model (from Stereo Anywhere Google Drive):
         stereoanywhere_sceneflow.pth
     - Depth Anything V2 Large (from Hugging Face):
         wget -O depth_anything_v2_vitl.pth \
           "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

  3. Run this script, pointing at the repo and weight files:

     python experiments/stereo_vision/stereo_anywhere.py \
         --stereo-anywhere-repo /path/to/stereoanywhere \
         --stereo-weights /path/to/stereoanywhere_sceneflow.pth \
         --mono-weights /path/to/depth_anything_v2_vitl.pth

Controls:
  q / Esc  — quit
  s        — save current disparity frame to disk
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Camera hardware indices
# ---------------------------------------------------------------------------
LEFT_CAMERA_INDEX = 4
RIGHT_CAMERA_INDEX = 6

# ---------------------------------------------------------------------------
# Stereo calibration (from the project's two_camera_final.py)
# ---------------------------------------------------------------------------
K1 = np.array([
    [542.975661, 0.000000, 347.621721],
    [0.000000, 542.580855, 266.597383],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D1 = np.array([
    [0.139821, -0.092846, 0.013859, 0.018398, -1.438426],
], dtype=np.float64)

K2 = np.array([
    [550.591389, 0.000000, 354.946646],
    [0.000000, 547.426744, 257.201464],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D2 = np.array([
    [0.061811, 0.096200, 0.009729, 0.016548, -0.325093],
], dtype=np.float64)

R = np.array([
    [0.999955, -0.009529, -0.000044],
    [0.009528, 0.999776, 0.018882],
    [-0.000136, -0.018881, 0.999822],
], dtype=np.float64)

T = np.array([
    [-0.078688],
    [-0.000478],
    [0.004312],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stereo rectification helpers
# ---------------------------------------------------------------------------
def compute_rectification_maps(image_size: tuple[int, int]):
    """Return (map1_left, map2_left), (map1_right, map2_right) for cv2.remap."""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1,
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_32FC1,
    )
    return (map1_left, map2_left), (map1_right, map2_right)


# ---------------------------------------------------------------------------
# Model loading (Stereo Anywhere + Depth Anything V2)
# ---------------------------------------------------------------------------
def load_models(args):
    """Import from the Stereo Anywhere repo and return (wrapper, mono_model, device)."""
    repo = os.path.expanduser(args.stereo_anywhere_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from models.stereoanywhere import StereoAnywhere
    from models.depth_anything_v2 import get_depth_anything_v2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Build a minimal namespace that StereoAnywhere's constructor expects.
    model_args = argparse.Namespace(
        n_downsample=2,
        n_additional_hourglass=0,
        volume_channels=8,
        vol_downsample=0,
        vol_n_masks=8,
        use_truncate_vol=True,
        mirror_conf_th=0.98,
        mirror_attenuation=0.9,
        use_aggregate_stereo_vol=False,
        use_aggregate_mono_vol=True,
        normal_gain=10,
        lrc_th=1.0,
        iters=args.iters,
    )

    # --- stereo model ---
    stereo_net = StereoAnywhere(model_args)
    stereo_net = nn.DataParallel(stereo_net)
    ckpt = torch.load(args.stereo_weights, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    stereo_net.load_state_dict(state, strict=True)
    stereo_net = stereo_net.module.eval().to(dtype).to(device)

    # --- mono model (Depth Anything V2 Large) ---
    mono_model = get_depth_anything_v2(args.mono_weights)
    mono_model = mono_model.eval().to(dtype).to(device)

    # --- wrapper (handles padding + mono inference) ---
    wrapper = _StereoAnywhereWrapper(model_args, stereo_net, mono_model)
    wrapper = wrapper.eval().to(dtype).to(device)

    return wrapper, device


class _StereoAnywhereWrapper(nn.Module):
    """Thin wrapper matching the upstream fast_demo_utils.StereoAnywhereWrapper."""

    def __init__(self, args, stereo_model, mono_model):
        super().__init__()
        self.args = args
        self.stereo_model = stereo_model
        self.mono_model = mono_model

    @torch.no_grad()
    def forward(self, left_img, right_img):
        # Monocular depth priors
        mono_left = self.mono_model.infer_image(
            left_img, input_size_width=518, input_size_height=518,
        )
        mono_right = self.mono_model.infer_image(
            right_img, input_size_width=518, input_size_height=518,
        )
        mono_depths = torch.cat([mono_left, mono_right], 0)
        mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
        mono_left = mono_depths[0].unsqueeze(0)
        mono_right = mono_depths[1].unsqueeze(0)

        # Pad to multiple of 32
        ht, wt = left_img.shape[-2], left_img.shape[-1]
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        _pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                pad_ht // 2, pad_ht - pad_ht // 2]

        left_img = F.pad(left_img, _pad, mode="replicate")
        right_img = F.pad(right_img, _pad, mode="replicate")
        mono_left = F.pad(mono_left, _pad, mode="replicate")
        mono_right = F.pad(mono_right, _pad, mode="replicate")

        pred_disps, _ = self.stereo_model(
            left_img, right_img, mono_left, mono_right,
            test_mode=True, iters=self.args.iters,
        )
        pred_disp = -pred_disps.squeeze(1)

        # Remove padding
        hd, wd = pred_disp.shape[-2:]
        c = [_pad[2], hd - _pad[3], _pad[0], wd - _pad[1]]
        pred_disp = pred_disp[:, c[0]:c[1], c[2]:c[3]]
        return pred_disp


# ---------------------------------------------------------------------------
# Frame pre-processing (numpy BGR -> torch tensor on device)
# ---------------------------------------------------------------------------
def frame_to_tensor(bgr_frame: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)


# ---------------------------------------------------------------------------
# Colorize a disparity map for display
# ---------------------------------------------------------------------------
def colorize_disparity(disp: np.ndarray) -> np.ndarray:
    if disp.max() - disp.min() > 1e-6:
        norm = (disp - disp.min()) / (disp.max() - disp.min())
    else:
        norm = np.zeros_like(disp)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Stereo Anywhere disparity experiment (live dual-camera)",
    )
    p.add_argument(
        "--stereo-anywhere-repo", required=True,
        help="Path to cloned https://github.com/bartn8/stereoanywhere",
    )
    p.add_argument(
        "--stereo-weights", required=True,
        help="Path to stereoanywhere_sceneflow.pth",
    )
    p.add_argument(
        "--mono-weights", required=True,
        help="Path to depth_anything_v2_vitl.pth",
    )
    p.add_argument("--left-camera", type=int, default=LEFT_CAMERA_INDEX)
    p.add_argument("--right-camera", type=int, default=RIGHT_CAMERA_INDEX)
    p.add_argument(
        "--iters", type=int, default=32,
        help="Number of GRU refinement iterations (lower = faster, less accurate)",
    )
    p.add_argument(
        "--iscale", type=float, default=2.0,
        help="Down-scale factor applied to frames before inference (default 2x for speed)",
    )
    p.add_argument(
        "--save-dir", default="output/stereo_anywhere",
        help="Directory to save snapshots when pressing 's'",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---- open cameras ----
    print(f"Opening cameras: left={args.left_camera}, right={args.right_camera} ...")
    left_cap = cv2.VideoCapture(args.left_camera)
    right_cap = cv2.VideoCapture(args.right_camera)
    if not left_cap.isOpened() or not right_cap.isOpened():
        print("ERROR: could not open one or both cameras.")
        return 1

    # Warm up (auto-exposure settle)
    for _ in range(10):
        left_cap.read()
        right_cap.read()

    # Grab a frame to get resolution
    ok_l, sample_l = left_cap.read()
    ok_r, sample_r = right_cap.read()
    if not ok_l or not ok_r:
        print("ERROR: could not read initial frames.")
        return 1

    h, w = sample_l.shape[:2]
    print(f"Frame size: {w}x{h}")

    # ---- compute rectification maps ----
    print("Computing stereo rectification maps ...")
    left_maps, right_maps = compute_rectification_maps((w, h))

    # ---- load Stereo Anywhere ----
    print("Loading Stereo Anywhere model (this may take a moment) ...")
    wrapper, device = load_models(args)
    print(f"Model loaded on {device}.")

    frame_idx = 0
    cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Rectified Stereo", cv2.WINDOW_NORMAL)

    print("Running — press 'q' or Esc to quit, 's' to save a snapshot.")

    try:
        while True:
            t0 = time.time()

            ok_l, left_bgr = left_cap.read()
            ok_r, right_bgr = right_cap.read()
            if not ok_l or not ok_r:
                continue

            # Rectify
            left_rect = cv2.remap(left_bgr, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_bgr, right_maps[0], right_maps[1], cv2.INTER_LINEAR)

            # Optionally down-scale for faster inference
            if args.iscale != 1.0:
                new_w = round(w / args.iscale)
                new_h = round(h / args.iscale)
                left_input = cv2.resize(left_rect, (new_w, new_h))
                right_input = cv2.resize(right_rect, (new_w, new_h))
            else:
                left_input = left_rect
                right_input = right_rect

            # To tensor
            left_t = frame_to_tensor(left_input, device)
            right_t = frame_to_tensor(right_input, device)

            # Inference
            with torch.no_grad():
                pred_disp = wrapper(left_t, right_t)

            disp_np = pred_disp.squeeze().cpu().numpy()

            # Up-scale disparity back to original resolution if needed
            if args.iscale != 1.0:
                disp_np = cv2.resize(disp_np, (w, h)) * args.iscale

            fps = 1.0 / max(time.time() - t0, 1e-9)

            # Visualise
            disp_color = colorize_disparity(disp_np)
            cv2.putText(
                disp_color, f"FPS: {fps:.1f}  iters={args.iters}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imshow("Disparity", disp_color)

            # Side-by-side rectified view
            stereo_vis = np.hstack([left_rect, right_rect])
            # Draw horizontal lines to verify rectification alignment
            for y in range(0, stereo_vis.shape[0], 40):
                cv2.line(stereo_vis, (0, y), (stereo_vis.shape[1], y), (0, 255, 0), 1)
            cv2.imshow("Rectified Stereo", stereo_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                os.makedirs(args.save_dir, exist_ok=True)
                tag = f"{frame_idx:06d}"
                cv2.imwrite(os.path.join(args.save_dir, f"{tag}_left.png"), left_rect)
                cv2.imwrite(os.path.join(args.save_dir, f"{tag}_right.png"), right_rect)
                cv2.imwrite(os.path.join(args.save_dir, f"{tag}_disp.png"), disp_color)
                np.save(os.path.join(args.save_dir, f"{tag}_disp.npy"), disp_np)
                print(f"Saved snapshot {tag} to {args.save_dir}/")

            frame_idx += 1

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
