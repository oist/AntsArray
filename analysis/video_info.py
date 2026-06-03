import argparse
import os
import sys
import time

import cv2


def fourcc_to_str(fourcc_int: int) -> str:
    # OpenCV returns FOURCC packed in an int; decode to 4 chars.
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])


def print_video_properties(path: str, scan_if_missing: bool = False, scan_max_seconds: float | None = None) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {path}")

    # Reported (container/decoder) properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = fourcc_to_str(fourcc_int) if fourcc_int != 0 else "unknown"

    # Note: these may be 0/NaN depending on backend/container
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"

    # Compute duration if possible
    duration_sec = None
    if fps and fps > 0 and frame_count and frame_count > 0:
        duration_sec = frame_count / fps

    print(f"Path:          {path}")
    print(f"Backend:       {backend}")
    print(f"FOURCC:        {fourcc} (int={fourcc_int})")
    print(f"Resolution:    {int(width)} x {int(height)}")
    print(f"FPS (reported): {fps:.6g}" if fps else "FPS (reported): unknown/0")
    print(f"Frames (reported): {int(frame_count)}" if frame_count else "Frames (reported): unknown/0")
    if duration_sec is not None:
        print(f"Duration (from reported): {duration_sec:.3f} s")
    else:
        print("Duration (from reported): unknown")

    # Optional scan if values are missing or questionable
    needs_scan = (
        scan_if_missing
        and (not fps or fps <= 0 or not frame_count or frame_count <= 0)
    )

    if needs_scan:
        print("\nScanning video to estimate properties (this reads frames)...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        counted = 0
        t0 = time.time()
        last_report = t0

        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            counted += 1

            # Optional time limit for scan
            if scan_max_seconds is not None and (time.time() - t0) >= scan_max_seconds:
                break

            # Lightweight progress output every ~1s
            now = time.time()
            if now - last_report >= 1.0:
                print(f"  scanned frames: {counted}", end="\r")
                last_report = now

        t1 = time.time()
        elapsed = max(t1 - t0, 1e-9)

        # If FPS was missing, estimate “decode FPS” (not necessarily true video FPS)
        est_decode_fps = counted / elapsed

        print("\nScan results:")
        print(f"  Frames scanned: {counted}")
        print(f"  Scan time:      {elapsed:.3f} s")
        print(f"  Decode rate:    {est_decode_fps:.3f} fps (estimate)")

        # If container FPS exists but frame_count missing, you can infer duration:
        if fps and fps > 0:
            print(f"  Duration (using reported FPS): {counted / fps:.3f} s")
        else:
            print("  Duration: cannot infer true duration without reliable FPS")

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Print video properties using OpenCV.")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--scan-if-missing",
        action="store_true",
        help="If FPS or frame count is missing/0, scan frames to estimate.",
    )
    parser.add_argument(
        "--scan-max-seconds",
        type=float,
        default=None,
        help="Limit scan time (seconds). Useful for very long videos.",
    )
    args = parser.parse_args()

    try:
        print_video_properties(args.video_path, args.scan_if_missing, args.scan_max_seconds)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
