import logging
import platform
import threading
import time
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)

_CAPTURE_BACKEND_ALIASES = {
    "any": None,
    "auto": None,
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", None),
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "msmf": getattr(cv2, "CAP_MSMF", None),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", None),
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
    "v4l2": getattr(cv2, "CAP_V4L2", None),
}


def resolve_capture_backend(source_type: str, preferred_backend: str = "auto") -> tuple[int | None, str]:
    backend_name = preferred_backend.strip().lower() if preferred_backend else "auto"

    if backend_name not in _CAPTURE_BACKEND_ALIASES:
        raise ValueError(f"Unsupported camera backend: {preferred_backend}")

    explicit_backend = _CAPTURE_BACKEND_ALIASES[backend_name]
    if backend_name not in {"auto", "any"}:
        if explicit_backend is None:
            raise ValueError(
                f"Camera backend '{preferred_backend}' is not available in this OpenCV build"
            )
        return explicit_backend, backend_name

    if source_type == "usb":
        system_name = platform.system().lower()
        if system_name == "darwin" and getattr(cv2, "CAP_AVFOUNDATION", None) is not None:
            return cv2.CAP_AVFOUNDATION, "avfoundation"
        if system_name == "windows" and getattr(cv2, "CAP_DSHOW", None) is not None:
            return cv2.CAP_DSHOW, "dshow"

    return None, "default"


def build_video_capture(
    source_type: str,
    source: str,
    preferred_backend: str = "auto",
) -> tuple[cv2.VideoCapture, str]:
    resolved_source: int | str = int(source) if source_type == "usb" else source
    backend, backend_name = resolve_capture_backend(
        source_type=source_type,
        preferred_backend=preferred_backend,
    )

    if backend is None:
        return cv2.VideoCapture(resolved_source), backend_name

    capture = cv2.VideoCapture(resolved_source, backend)
    if capture.isOpened():
        return capture, backend_name

    logger.warning(
        "Failed to open source with preferred backend, falling back to OpenCV default | "
        "source_type=%s | source=%s | backend=%s",
        source_type,
        source,
        backend_name,
    )
    capture.release()
    return cv2.VideoCapture(resolved_source), "default"


class CameraService:
    def __init__(
        self,
        source_type: str,
        source: str,
        capture_backend: str = "auto",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        jpeg_quality: int = 85,
    ) -> None:
        self.source_type = source_type
        self.source = source
        self.capture_backend = capture_backend
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state_lock = threading.RLock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._last_error: Optional[str] = None

        self._frames_read = 0
        self._read_failures = 0
        self._actual_fps = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_counter = 0
        self._is_running = False
        self._active_backend_name = "default"

    def _reset_runtime_stats(self) -> None:
        self._frames_read = 0
        self._read_failures = 0
        self._actual_fps = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_counter = 0

    def start(self) -> None:
        with self._state_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._is_running:
            logger.info("Camera service already running")
            return

        logger.info(
            "Starting camera service | source_type=%s | source=%s | backend=%s",
            self.source_type,
            self.source,
            self.capture_backend,
        )

        self._reset_runtime_stats()
        self._last_error = None
        self._capture, self._active_backend_name = build_video_capture(
            source_type=self.source_type,
            source=self.source,
            preferred_backend=self.capture_backend,
        )

        if not self._capture.isOpened():
            self._last_error = f"Failed to open camera source: {self.source}"
            self._capture.release()
            self._capture = None
            self._is_running = False
            with self._lock:
                self._latest_frame = None
                self._latest_jpeg = None
            raise RuntimeError(self._last_error)

        if self.source_type == "usb":
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self._is_running = True

        logger.info("Camera service started | backend=%s", self._active_backend_name)

    def stop(self) -> None:
        with self._state_lock:
            self._stop_locked(clear_error=False)

    def _stop_locked(self, clear_error: bool) -> None:
        logger.info("Stopping camera service")

        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._is_running = False
        self._active_backend_name = "default"
        if clear_error:
            self._last_error = None
        logger.info("Camera service stopped")

    def switch_source(self, source_type: str, source: str) -> dict:
        with self._state_lock:
            previous_source_type = self.source_type
            previous_source = self.source
            previous_was_running = self._is_running

            if self._is_running:
                self._stop_locked(clear_error=False)

            self.source_type = source_type
            self.source = source

            try:
                self._start_locked()
            except Exception as exc:
                logger.exception("Failed to switch camera source")
                self.source_type = previous_source_type
                self.source = previous_source

                if previous_was_running:
                    try:
                        self._start_locked()
                    except Exception:
                        logger.exception("Failed to restore previous camera source")

                return {
                    "ok": False,
                    "error": str(exc),
                    "status": self.get_status(),
                }

            return {
                "ok": True,
                "error": None,
                "status": self.get_status(),
            }

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._capture is None:
                time.sleep(0.1)
                continue

            ok, frame = self._capture.read()

            if not ok or frame is None:
                self._read_failures += 1
                self._last_error = "Failed to read frame from current source"
                time.sleep(0.03)
                continue

            self._frames_read += 1
            self._fps_counter += 1
            self._last_error = None

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
            "last_error": self._last_error,
        }

    def list_usb_cameras(self, max_index: int = 5) -> list[dict]:
        cameras: list[dict] = []
        active_usb_index = None

        if self._is_running and self.source_type == "usb":
            try:
                active_usb_index = int(self.source)
            except ValueError:
                active_usb_index = None

        for index in range(max_index + 1):
            if active_usb_index == index:
                status = self.get_status()
                is_opened = True
                width = status["frame_width"]
                height = status["frame_height"]
            else:
                capture, _ = build_video_capture(
                    source_type="usb",
                    source=str(index),
                    preferred_backend=self.capture_backend,
                )
                is_opened = capture.isOpened()
                width = None
                height = None

                if is_opened:
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        height, width = frame.shape[:2]

                capture.release()

            cameras.append(
                {
                    "index": index,
                    "available": is_opened,
                    "width": width,
                    "height": height,
                    "label": f"USB camera {index}",
                }
            )

        return cameras
