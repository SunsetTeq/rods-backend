import argparse
from pathlib import Path
from time import perf_counter

import cv2

from app.core.config import settings


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


def inspect_source(
    source_type: str,
    source: str,
    width: int,
    height: int,
    fps: int,
    preview_path: Path,
    warmup_seconds: float,
) -> int:
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

    while perf_counter() - start < warmup_seconds:
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

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), frame)
    print(f"Saved preview frame to: {preview_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick camera/source diagnostic for RODS backend.")
    parser.add_argument("--scan-usb", action="store_true", help="Scan USB camera indexes before testing.")
    parser.add_argument(
        "--max-index",
        type=int,
        default=settings.camera_discovery_max_index,
        help="Maximum USB camera index for scanning.",
    )
    parser.add_argument(
        "--source-type",
        default=settings.camera_source_type,
        choices=["usb", "rtsp", "file"],
    )
    parser.add_argument(
        "--source",
        default=str(settings.camera_source),
        help="USB index, RTSP URL, or file path.",
    )
    parser.add_argument("--width", type=int, default=settings.camera_width)
    parser.add_argument("--height", type=int, default=settings.camera_height)
    parser.add_argument("--fps", type=int, default=settings.camera_fps)
    parser.add_argument(
        "--preview-path",
        type=Path,
        default=Path("scripts") / "camera_test_preview.jpg",
        help="Where to save the last successfully read frame.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=3.0,
        help="How long to read frames before printing the result.",
    )
    args = parser.parse_args()

    print("RODS camera test")
    print(f"Using defaults from .env where arguments are omitted.")
    print(f"Configured source: {args.source_type}:{args.source}")
    print(f"Target size: {args.width}x{args.height} @ {args.fps} FPS")
    print()

    if args.scan_usb:
        inspect_usb_cameras(args.max_index)
        print()

    return inspect_source(
        source_type=args.source_type,
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview_path=args.preview_path,
        warmup_seconds=args.warmup_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
