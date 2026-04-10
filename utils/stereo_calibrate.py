"""
Stereo camera calibration using a checkerboard pattern.

Usage:
    1. Print a checkerboard pattern (default: 9x6 inner corners).
       A standard one can be found at: https://docs.opencv.org/4.x/pattern.png
    2. Measure the side length of one square in meters (default: 0.025 = 25mm).
    3. Run this script:
           python utils/stereo_calibrate.py
    4. Hold the checkerboard in front of both cameras. Press SPACE to capture
       a frame pair when corners are detected in both views. Capture at least
       15-20 pairs from different angles and positions.
    5. Press 'q' when done capturing. The script will run stereo calibration
       and save the result to stereo_calibration.npz.
    6. Update examples/two_camera_final.py to load the .npz file, or copy the
       printed values into the script.

Keyboard controls:
    SPACE  - capture current frame pair (only if corners found in both)
    q/ESC  - finish capturing and run calibration
"""

import argparse
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
    parser.add_argument("--left", type=int, default=4, help="Left camera index")
    parser.add_argument("--right", type=int, default=6, help="Right camera index")
    parser.add_argument(
        "--rows", type=int, default=7,
        help="Number of inner corner rows in the checkerboard",
    )
    parser.add_argument(
        "--cols", type=int, default=9,
        help="Number of inner corner columns in the checkerboard",
    )
    parser.add_argument(
        "--square-size", type=float, default=0.020,
        help="Side length of one checkerboard square in meters",
    )
    parser.add_argument(
        "--output", type=str, default="stereo_calibration.npz",
        help="Output path for the calibration file",
    )
    parser.add_argument(
        "--min-pairs", type=int, default=15,
        help="Minimum number of image pairs before calibration is allowed",
    )
    args = parser.parse_args()

    board_size = (args.cols, args.rows)

    # Prepare the object points grid: (0,0,0), (1,0,0), ... scaled by square_size.
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_size

    left_cam = cv2.VideoCapture(args.left)
    right_cam = cv2.VideoCapture(args.right)

    if not left_cam.isOpened() or not right_cam.isOpened():
        print(f"Could not open cameras (left={args.left}, right={args.right}).")
        return 1

    obj_points = []       # 3D points in real-world space
    img_points_left = []  # 2D points in left image
    img_points_right = [] # 2D points in right image
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"Checkerboard: {args.cols}x{args.rows} inner corners, "
          f"square size: {args.square_size*1000:.1f}mm")
    print(f"Cameras: left={args.left}, right={args.right}")
    print(f"Capture at least {args.min_pairs} pairs from different angles/positions.")
    print("SPACE = capture, q/ESC = finish and calibrate\n")

    pair_count = 0

    while True:
        ok_l, frame_l = left_cam.read()
        ok_r, frame_r = right_cam.read()
        if not ok_l or not ok_r:
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

        found_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, None)

        display_l = frame_l.copy()
        display_r = frame_r.copy()

        if found_l:
            cv2.drawChessboardCorners(display_l, board_size, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(display_r, board_size, corners_r, found_r)

        # Status overlay
        status = f"Pairs: {pair_count}"
        if found_l and found_r:
            status += " | BOTH DETECTED - press SPACE"
        elif found_l:
            status += " | Left only"
        elif found_r:
            status += " | Right only"
        else:
            status += " | No corners"

        cv2.putText(display_l, status, (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack((display_l, display_r))
        # Resize for display if too wide
        max_display_width = 1600
        if combined.shape[1] > max_display_width:
            scale = max_display_width / combined.shape[1]
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow("Stereo Calibration - Left | Right", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" ") and found_l and found_r:
            # Refine corner locations to sub-pixel accuracy
            corners_l_refined = cv2.cornerSubPix(
                gray_l, corners_l, (11, 11), (-1, -1), criteria
            )
            corners_r_refined = cv2.cornerSubPix(
                gray_r, corners_r, (11, 11), (-1, -1), criteria
            )

            obj_points.append(objp)
            img_points_left.append(corners_l_refined)
            img_points_right.append(corners_r_refined)
            pair_count += 1
            print(f"  Captured pair #{pair_count}")

    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()

    if pair_count < args.min_pairs:
        print(f"\nOnly {pair_count} pairs captured (minimum {args.min_pairs}). "
              "Calibration may be inaccurate.")
        if pair_count < 5:
            print("Too few pairs for reliable calibration. Exiting.")
            return 1
        print("Proceeding anyway...\n")

    print(f"\nRunning stereo calibration with {pair_count} image pairs...")
    print(f"Image size: {image_size}")

    # Step 1: Calibrate each camera individually for good initial estimates.
    flags_individual = 0
    print("  Calibrating left camera...")
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(
        obj_points, img_points_left, image_size, None, None, flags=flags_individual,
    )
    print(f"    Left RMS reprojection error: {rms_l:.4f}")

    print("  Calibrating right camera...")
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(
        obj_points, img_points_right, image_size, None, None, flags=flags_individual,
    )
    print(f"    Right RMS reprojection error: {rms_r:.4f}")

    # Step 2: Stereo calibration to find R, T between the cameras.
    # Use the individual calibrations as initial guesses and refine intrinsics.
    flags_stereo = cv2.CALIB_USE_INTRINSIC_GUESS
    print("  Running stereo calibration...")
    rms_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        K1, D1,
        K2, D2,
        image_size,
        flags=flags_stereo,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    print(f"    Stereo RMS reprojection error: {rms_stereo:.4f}")

    if rms_stereo > 1.0:
        print("\n  WARNING: RMS error > 1.0 pixel. Calibration quality may be poor.")
        print("  Tips: use more image pairs, ensure the checkerboard is flat,")
        print("  and cover different positions/angles across the field of view.")

    # Save calibration
    np.savez(
        args.output,
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T,
        E=E, F=F,
        image_size=np.array(image_size),
        rms_stereo=np.array([rms_stereo]),
    )
    print(f"\nCalibration saved to: {args.output}")

    # Print the values for direct use in code
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS — copy into two_camera_final.py")
    print("=" * 70)

    def format_matrix(name, mat):
        rows = []
        for row in mat:
            if hasattr(row, "__len__"):
                rows.append("    [" + ", ".join(f"{v:.6f}" for v in row) + "],")
            else:
                rows.append(f"    [{row:.6f}],")
        return f"{name} = np.array([\n" + "\n".join(rows) + "\n], dtype=np.float64)"

    print()
    print(format_matrix("K1", K1))
    print()
    print(format_matrix("D1", D1))
    print()
    print(format_matrix("K2", K2))
    print()
    print(format_matrix("D2", D2))
    print()
    print(format_matrix("R", R))
    print()
    print(format_matrix("T", T))

    baseline = np.linalg.norm(T)
    print(f"\nBaseline (||T||): {baseline:.4f} m = {baseline*100:.2f} cm")
    print(f"Stereo RMS error: {rms_stereo:.4f} px")
    print()
    print("Alternatively, load from file in your code:")
    print(f'  calib = StereoCalibration.from_npz("{args.output}")')

    return 0


if __name__ == "__main__":
    sys.exit(main())
