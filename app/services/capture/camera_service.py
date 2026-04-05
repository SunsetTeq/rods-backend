import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class CameraService:
    def __init__(
        self,
        source_type: str,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        jpeg_quality: int = 85,
    ) -> None:
        self.source_type = source_type
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None

        self._frames_read = 0
        self._read_failures = 0
        self._actual_fps = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_counter = 0
        self._is_running = False

    def _resolve_source(self):
        if self.source_type == "usb":
            return int(self.source)
        if self.source_type in {"rtsp", "file"}:
            return self.source
        raise ValueError(f"Unsupported source_type: {self.source_type}")

    def start(self) -> None:
        if self._is_running:
            logger.info("Camera service already running")
            return

        resolved_source = self._resolve_source()
        logger.info(
            "Starting camera service | source_type=%s | source=%s",
            self.source_type,
            self.source,
        )

        self._capture = cv2.VideoCapture(resolved_source)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")

        if self.source_type == "usb":
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self._is_running = True

        logger.info("Camera service started")

    def stop(self) -> None:
        logger.info("Stopping camera service")

        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._is_running = False
        logger.info("Camera service stopped")

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._capture is None:
                time.sleep(0.1)
                continue

            ok, frame = self._capture.read()

            if not ok or frame is None:
                self._read_failures += 1
                time.sleep(0.03)
                continue

            self._frames_read += 1
            self._fps_counter += 1

            now = time.time()
            elapsed = now - self._last_fps_calc_time
            if elapsed >= 1.0:
                self._actual_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._last_fps_calc_time = now

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )

            if not success:
                continue

            with self._lock:
                self._latest_frame = frame.copy()
                self._latest_jpeg = encoded.tobytes()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> dict:
        actual_width = None
        actual_height = None
        target_fps = self.fps

        if self._capture is not None:
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        return {
            "is_running": self._is_running,
            "source_type": self.source_type,
            "source": self.source,
            "frame_width": actual_width,
            "frame_height": actual_height,
            "target_fps": target_fps,
            "actual_fps": round(self._actual_fps, 2),
            "frames_read": self._frames_read,
            "read_failures": self._read_failures,
        }
