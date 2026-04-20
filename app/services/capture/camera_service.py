import logging
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)
STREAM_STALE_AFTER_MS = 2000

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


def build_camera_id(source_type: str, source: str) -> str:
    return f"{source_type}:{source}"


def parse_camera_id(camera_id: str) -> tuple[str, str]:
    source_type, separator, source = camera_id.partition(":")
    if not separator or not source_type or not source:
        raise ValueError("camera_id must be in format '<source_type>:<source>'")
    return source_type, source


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
        self._last_frame_at: float | None = None

        self._frames_read = 0
        self._read_failures = 0
        self._actual_fps = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_counter = 0
        self._is_running = False
        self._active_backend_name = "default"
        self._usb_camera_name_cache: dict[int, str] = {}
        self._usb_camera_name_cache_updated_at = 0.0
        self._cached_available_camera_sources: list[dict] = []

    def _reset_runtime_stats(self) -> None:
        self._frames_read = 0
        self._read_failures = 0
        self._actual_fps = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_counter = 0
        self._last_frame_at = None

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
        self._last_frame_at = None
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

    def refresh_available_camera_source_cache(self, max_index: int = 5) -> list[dict]:
        cameras = self.list_available_camera_sources(
            max_index=max_index,
            probe_usb=True,
        )
        with self._state_lock:
            self._cached_available_camera_sources = [dict(item) for item in cameras]
            return [dict(item) for item in self._cached_available_camera_sources]

    def get_cached_available_camera_sources(self) -> list[dict]:
        with self._state_lock:
            cached = [dict(item) for item in self._cached_available_camera_sources]
        return self._normalize_cached_camera_sources(cached)

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
            self._last_frame_at = time.time()

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

    def has_fresh_frame(self, stale_after_ms: int = STREAM_STALE_AFTER_MS) -> bool:
        with self._state_lock:
            if not self._is_running or self._last_frame_at is None:
                return False

            frame_age_ms = (time.time() - self._last_frame_at) * 1000
            return frame_age_ms <= stale_after_ms

    def get_stream_availability(self, stale_after_ms: int = STREAM_STALE_AFTER_MS) -> dict:
        frame_age_ms = None
        if self._last_frame_at is not None:
            frame_age_ms = round((time.time() - self._last_frame_at) * 1000, 2)

        status = self.get_status()
        with self._lock:
            has_frame = self._latest_frame is not None

        return {
            "stream_available": bool(
                status["is_running"]
                and has_frame
                and frame_age_ms is not None
                and frame_age_ms <= stale_after_ms
            ),
            "has_frame": has_frame,
            "frame_age_ms": frame_age_ms,
            "stale_after_ms": stale_after_ms,
            "active_camera_id": build_camera_id(status["source_type"], str(status["source"])),
            "active_camera": status,
            "last_error": status["last_error"],
        }

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
            "active_camera_id": build_camera_id(self.source_type, str(self.source)),
            "frame_width": actual_width,
            "frame_height": actual_height,
            "target_fps": target_fps,
            "actual_fps": round(self._actual_fps, 2),
            "frames_read": self._frames_read,
            "read_failures": self._read_failures,
            "last_error": self._last_error,
        }

    def list_usb_cameras(self, max_index: int = 5, available_only: bool = True) -> list[dict]:
        cameras: list[dict] = []
        active_usb_index = None
        usb_camera_names = self._list_usb_camera_names()

        if self._is_running and self.source_type == "usb":
            try:
                active_usb_index = int(self.source)
            except ValueError:
                active_usb_index = None

        for index in range(max_index + 1):
            is_available = False
            width = None
            height = None

            if active_usb_index == index:
                status = self.get_status()
                latest_frame = self.get_latest_frame()
                is_available = latest_frame is not None and self.has_fresh_frame()
                width = status["frame_width"]
                height = status["frame_height"]
                if latest_frame is not None and (width is None or height is None):
                    height, width = latest_frame.shape[:2]
            else:
                capture, _ = build_video_capture(
                    source_type="usb",
                    source=str(index),
                    preferred_backend=self.capture_backend,
                )
                if capture.isOpened():
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        is_available = True
                        height, width = frame.shape[:2]

                capture.release()

            if available_only and not is_available:
                continue

            camera_name = usb_camera_names.get(index)
            cameras.append(
                {
                    "index": index,
                    "available": is_available,
                    "width": width,
                    "height": height,
                    "label": self._build_usb_camera_label(index=index, name=camera_name),
                    "name": camera_name,
                }
            )

        return cameras

    def list_available_camera_sources(
        self,
        max_index: int = 5,
        probe_usb: bool = True,
    ) -> list[dict]:
        status = self.get_status()
        active_source_type = status["source_type"]
        active_source = str(status["source"])
        cameras: list[dict] = []

        if probe_usb:
            usb_items = self.list_usb_cameras(max_index=max_index, available_only=True)
        else:
            usb_items = self._list_usb_cameras_without_probe(max_index=max_index)

        for item in usb_items:
            cameras.append(
                {
                    "camera_id": build_camera_id("usb", str(item["index"])),
                    "source_type": "usb",
                    "source": str(item["index"]),
                    "label": item["label"],
                    "name": item.get("name"),
                    "is_active": active_source_type == "usb" and active_source == str(item["index"]),
                    "is_available": bool(item["available"]),
                    "frame_width": item["width"],
                    "frame_height": item["height"],
                }
            )

        if active_source_type in {"rtsp", "file"} and self.has_fresh_frame():
            cameras.append(
                {
                    "camera_id": build_camera_id(active_source_type, active_source),
                    "source_type": active_source_type,
                    "source": active_source,
                    "label": self._build_non_usb_source_label(
                        source_type=active_source_type,
                        source=active_source,
                    ),
                    "name": None,
                    "is_active": True,
                    "is_available": True,
                    "frame_width": status["frame_width"],
                    "frame_height": status["frame_height"],
                }
            )

        cameras.sort(key=lambda item: (not item["is_active"], item["source_type"], item["source"]))
        return cameras

    def _normalize_cached_camera_sources(self, cameras: list[dict]) -> list[dict]:
        status = self.get_status()
        active_source_type = status["source_type"]
        active_source = str(status["source"])
        active_camera_id = build_camera_id(active_source_type, active_source)
        active_is_available = bool(status["is_running"] and self.has_fresh_frame())

        normalized: list[dict] = []
        active_in_list = False

        for item in cameras:
            normalized_item = dict(item)
            is_active = normalized_item["camera_id"] == active_camera_id
            normalized_item["is_active"] = is_active

            if is_active:
                active_in_list = True
                normalized_item["is_available"] = active_is_available
                normalized_item["frame_width"] = status["frame_width"]
                normalized_item["frame_height"] = status["frame_height"]

            normalized.append(normalized_item)

        if not active_in_list:
            if active_source_type == "usb":
                normalized.append(
                    {
                        "camera_id": active_camera_id,
                        "source_type": "usb",
                        "source": active_source,
                        "label": self._build_usb_camera_label(index=int(active_source), name=None),
                        "name": None,
                        "is_active": True,
                        "is_available": active_is_available,
                        "frame_width": status["frame_width"],
                        "frame_height": status["frame_height"],
                    }
                )
            elif active_source_type in {"rtsp", "file"}:
                normalized.append(
                    {
                        "camera_id": active_camera_id,
                        "source_type": active_source_type,
                        "source": active_source,
                        "label": self._build_non_usb_source_label(
                            source_type=active_source_type,
                            source=active_source,
                        ),
                        "name": None,
                        "is_active": True,
                        "is_available": active_is_available,
                        "frame_width": status["frame_width"],
                        "frame_height": status["frame_height"],
                    }
                )

        normalized.sort(key=lambda item: (not item["is_active"], item["source_type"], item["source"]))
        return normalized

    def _list_usb_cameras_without_probe(self, max_index: int = 5) -> list[dict]:
        status = self.get_status()
        active_usb_index = None
        if status["source_type"] == "usb":
            try:
                active_usb_index = int(status["source"])
            except ValueError:
                active_usb_index = None

        cameras: list[dict] = []
        for index in range(max_index + 1):
            is_active = active_usb_index == index
            cameras.append(
                {
                    "index": index,
                    "available": True if is_active else False,
                    "width": status["frame_width"] if is_active else None,
                    "height": status["frame_height"] if is_active else None,
                    "label": self._build_usb_camera_label(index=index, name=None),
                    "name": None,
                }
            )

        return cameras

    def _list_usb_camera_names(self) -> dict[int, str]:
        return {}

    def _list_macos_usb_camera_names(self) -> dict[int, str]:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            return {}

        try:
            process = subprocess.run(
                [ffmpeg_path, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            logger.debug("Failed to enumerate macOS camera names via ffmpeg", exc_info=True)
            return {}

        output = "\n".join(part for part in (process.stdout, process.stderr) if part)
        names: dict[int, str] = {}
        in_video_section = False

        for line in output.splitlines():
            if "AVFoundation video devices" in line:
                in_video_section = True
                continue

            if "AVFoundation audio devices" in line:
                break

            if not in_video_section:
                continue

            match = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
            if match is None:
                continue

            index = int(match.group(1))
            names[index] = match.group(2).strip()

        return names

    def _build_usb_camera_label(self, index: int, name: str | None) -> str:
        return f"Camera {index + 1}"

    def _build_non_usb_source_label(self, source_type: str, source: str) -> str:
        if source_type == "file":
            return f"FILE - {Path(source).name or source}"
        return f"{source_type.upper()} - {source}"
