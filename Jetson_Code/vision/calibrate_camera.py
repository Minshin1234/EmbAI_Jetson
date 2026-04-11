"""Capture checkerboard images and solve OpenCV camera calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.config import CAMERA_DEVICE, CALIBRATION_PATH


def _open_camera() -> cv2.VideoCapture:
    """Open the first working camera, preferring CAMERA_DEVICE."""
    camera = None
    for idx in [CAMERA_DEVICE] + [i for i in range(5) if i != CAMERA_DEVICE]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Using /dev/video{idx}")
                camera = cap
                break
        cap.release()

    if camera is None:
        raise RuntimeError("No working camera found on /dev/video0-4")

    return camera


def _build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """Generate checkerboard corner coordinates on the Z=0 plane."""
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= square_size
    return grid


def _iter_images(directory: Path) -> Iterable[Path]:
    """Yield image files in a stable order."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in exts:
            yield path


def capture_images(output_dir: Path, width: int | None, height: int | None) -> None:
    """Interactively capture calibration images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = _open_camera()

    if width:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("Press SPACE to save a frame, or Q to quit.")
    saved = 0
    gui_available = True

    try:
        cv2.namedWindow("camera-calibration-capture", cv2.WINDOW_NORMAL)
    except cv2.error:
        gui_available = False

    if gui_available:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue

            preview = frame.copy()
            text = f"Saved: {saved}  SPACE=capture  Q=quit"
            cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("camera-calibration-capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                path = output_dir / f"calibration_{saved:03d}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                print(f"Saved {path}")
            elif key in (ord("q"), 27):
                break
    else:
        print("No GUI backend detected. Falling back to terminal capture mode.")
        print("Press ENTER to save the next frame, or type q then ENTER to quit.")
        while True:
            command = input("> ").strip().lower()
            if command == "q":
                break

            ok, frame = camera.read()
            if not ok or frame is None:
                print("Failed to read a frame. Try again.")
                continue

            path = output_dir / f"calibration_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            print(f"Saved {path}")

    camera.release()
    if gui_available:
        cv2.destroyAllWindows()
    print(f"Captured {saved} image(s) into {output_dir}")


def calibrate_from_images(
    image_dir: Path,
    cols: int,
    rows: int,
    square_size: float,
    output_path: Path,
    preview: bool,
) -> None:
    """Solve camera calibration from checkerboard images."""
    board_size = (cols, rows)
    object_points_template = _build_object_points(cols, rows, square_size)
    objpoints = []
    imgpoints = []
    image_size = None
    used_images = 0

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for path in _iter_images(image_dir):
        image = cv2.imread(str(path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size)
        if not found:
            print(f"Skipped {path.name}: checkerboard not found")
            continue

        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

        objpoints.append(object_points_template.copy())
        imgpoints.append(refined)
        image_size = gray.shape[::-1]
        used_images += 1
        print(f"Used {path.name}")

        if preview:
            shown = image.copy()
            cv2.drawChessboardCorners(shown, board_size, refined, found)
            cv2.imshow("camera-calibration-detections", shown)
            cv2.waitKey(250)

    cv2.destroyAllWindows()

    if used_images < 8 or image_size is None:
        raise RuntimeError("Need at least 8 good checkerboard images to calibrate reliably.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    total_error = 0.0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        reprojected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        total_error += cv2.norm(imgp, reprojected, cv2.NORM_L2) / len(reprojected)
    mean_error = total_error / len(objpoints)

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "checkerboard_cols": cols,
        "checkerboard_rows": rows,
        "square_size": square_size,
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error": float(mean_error),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"Saved calibration to {output_path}")
    print(f"RMS reprojection error: {rms:.4f}")
    print(f"Mean reprojection error: {mean_error:.4f}")
    print("Paste-free option: the app will load this file automatically on next start.")
    print("camera_matrix =")
    print(camera_matrix)
    print("dist_coeffs =")
    print(dist_coeffs.reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture", help="Capture checkerboard images from the camera")
    capture_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration_images"),
        help="Directory where captured images will be saved",
    )
    capture_parser.add_argument("--width", type=int, default=None, help="Requested capture width")
    capture_parser.add_argument("--height", type=int, default=None, help="Requested capture height")

    calibrate_parser = subparsers.add_parser("solve", help="Solve calibration from saved checkerboard images")
    calibrate_parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("calibration_images"),
        help="Directory containing captured checkerboard images",
    )
    calibrate_parser.add_argument("--cols", type=int, required=True, help="Checkerboard inner corners across")
    calibrate_parser.add_argument("--rows", type=int, required=True, help="Checkerboard inner corners down")
    calibrate_parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Checkerboard square size in real-world units (for example 25.0 for 25 mm)",
    )
    calibrate_parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_PATH,
        help="Where to save the solved calibration JSON",
    )
    calibrate_parser.add_argument(
        "--preview",
        action="store_true",
        help="Show detected checkerboard corners while solving",
    )

    args = parser.parse_args()

    if args.command == "capture":
        capture_images(args.output_dir, args.width, args.height)
    elif args.command == "solve":
        calibrate_from_images(
            args.image_dir,
            args.cols,
            args.rows,
            args.square_size,
            args.output,
            args.preview,
        )


if __name__ == "__main__":
    main()
