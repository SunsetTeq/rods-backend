import argparse
from pathlib import Path
from time import perf_counter

import cv2


def inspect_usb_cameras(max_index: int) -> None:
    print(f"Scanning USB camera indexes 0..{max_index}")

    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        opened = capture.isOpened()

        width = None
        height = None
        fps = None

        if opened:
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS) or None

        print(
            f"[{index}] opened={opened} "
            f"resolution={width or '-'}x{height or '-'} fps={round(fps, 2) if fps else '-'}"
        )
        capture.release()


def inspect_source(source_type: str, source: str, width: int, height: int, fps: int) -> int:
    resolved_source = int(source) if source_type == "usb" else source
    capture = cv2.VideoCapture(resolved_source)

    if not capture.isOpened():
        print(f"Failed to open source: {source}")
        return 1

    if source_type == "usb":
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)

    start = perf_counter()
    frames_read = 0
    frame = None

    while perf_counter() - start < 3:
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frames_read += 1

    capture.release()

    if frame is None or frames_read == 0:
        print("Source opened, but no frames were read.")
        return 2

    actual_height, actual_width = frame.shape[:2]
    elapsed = perf_counter() - start
    actual_fps = frames_read / elapsed if elapsed else 0.0

    print(f"Source type: {source_type}")
    print(f"Source: {source}")
    print(f"Frames read: {frames_read}")
    print(f"Measured FPS: {actual_fps:.2f}")
    print(f"Last frame size: {actual_width}x{actual_height}")

    preview_path = Path("scripts") / "camera_test_preview.jpg"
    cv2.imwrite(str(preview_path), frame)
    print(f"Saved preview frame to: {preview_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick camera/source diagnostic for RODS backend.")
    parser.add_argument("--scan-usb", action="store_true", help="Scan USB camera indexes before testing.")
    parser.add_argument("--max-index", type=int, default=5, help="Maximum USB camera index for scanning.")
    parser.add_argument("--source-type", default="usb", choices=["usb", "rtsp", "file"])
    parser.add_argument("--source", default="0", help="USB index, RTSP URL, or file path.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.scan_usb:
        inspect_usb_cameras(args.max_index)
        print()

    return inspect_source(
        source_type=args.source_type,
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
