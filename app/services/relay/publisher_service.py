import logging
import shutil
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.services.capture.camera_service import CameraService
from app.services.vision.detector_service import DetectorService


logger = logging.getLogger(__name__)


class RelayPublisherService:
    def __init__(
        self,
        camera_service: CameraService,
        detector_service: DetectorService,
        enabled: bool,
        publish_url: str,
        output_variant: str,
        width: int,
        height: int,
        fps: int,
        video_bitrate_kbps: int,
        h264_preset: str,
        ffmpeg_bin: str,
    ) -> None:
        self.camera_service = camera_service
        self.detector_service = detector_service
        self.enabled = enabled
        self.publish_url = publish_url
        self.output_variant = output_variant
        self.width = width
        self.height = height
        self.fps = fps
        self.video_bitrate_kbps = video_bitrate_kbps
        self.h264_preset = h264_preset
        self.ffmpeg_bin = ffmpeg_bin

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._is_running = False
        self._frames_published = 0
        self._actual_fps = 0.0
        self._fps_counter = 0
        self._last_fps_calc_time = time.time()
        self._last_error: Optional[str] = None

    def start(self) -> None:
        if not self.enabled:
            logger.info("Relay publisher is disabled by config")
            return

        if self._is_running:
            logger.info("Relay publisher already running")
            return

        if not self.publish_url:
            self._last_error = "Relay publisher is enabled but RELAY_PUBLISH_URL is empty"
            logger.warning(self._last_error)
            return

        if not self.is_ffmpeg_available():
            self._last_error = f"ffmpeg binary not found: {self.ffmpeg_bin}"
            logger.warning(self._last_error)
            return

        self._last_error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info(
            "Relay publisher started | variant=%s | target=%sx%s@%sfps | url=%s",
            self.output_variant,
            self.width,
            self.height,
            self.fps,
            self.publish_url,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._stop_process()
        self._is_running = False
        logger.info("Relay publisher stopped")

    def is_ffmpeg_available(self) -> bool:
        return shutil.which(self.ffmpeg_bin) is not None

    def _publish_loop(self) -> None:
        target_interval = 1.0 / max(self.fps, 1)

        while not self._stop_event.is_set():
            try:
                frame = self._build_output_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue

                if self._process is None or self._process.poll() is not None:
                    self._start_process()

                process = self._process
                if process is None or process.stdin is None:
                    raise RuntimeError("ffmpeg process stdin is not available")

                process.stdin.write(frame.tobytes())
                process.stdin.flush()
                self._frames_published += 1
                self._fps_counter += 1
                self._update_fps()
            except BrokenPipeError as exc:
                self._last_error = f"ffmpeg pipe broken: {exc}"
                logger.exception("Relay publisher pipe broken")
                self._stop_process()
                time.sleep(1.0)
                continue
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("Relay publisher failed")
                self._stop_process()
                time.sleep(1.0)
                continue

            time.sleep(target_interval)

        self._stop_process()

    def _build_output_frame(self) -> np.ndarray | None:
        frame = self.camera_service.get_latest_frame()
        if frame is None:
            return None

        if self.output_variant == "annotated":
            frame = self.detector_service.compose_annotated_frame(frame)

        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(resized)

    def _start_process(self) -> None:
        self._stop_process()

        bitrate = f"{self.video_bitrate_kbps}k"
        # Keep keyframes aligned to a fixed cadence so SRS can cut stable HLS
        # fragments for live playback.
        gop = str(max(self.fps, 1))
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            self.h264_preset,
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            f"{self.video_bitrate_kbps * 2}k",
            "-g",
            gop,
            "-keyint_min",
            gop,
            "-sc_threshold",
            "0",
            "-force_key_frames",
            "expr:gte(t,n_forced*1)",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            "flv",
            self.publish_url,
        ]

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _stop_process(self) -> None:
        process = self._process
        self._process = None

        if process is None:
            return

        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

        stderr_bytes = b""
        if process.stderr is not None:
            try:
                stderr_bytes = process.stderr.read() or b""
            except OSError:
                stderr_bytes = b""

        stderr_text = stderr_bytes.decode("utf-8", errors="ignore").strip()
        if stderr_text:
            self._last_error = stderr_text

    def _update_fps(self) -> None:
        now = time.time()
        elapsed = now - self._last_fps_calc_time
        if elapsed >= 1.0:
            self._actual_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._last_fps_calc_time = now

    def get_status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "is_running": self._is_running,
                "publish_url": self.publish_url,
                "output_variant": self.output_variant,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "video_bitrate_kbps": self.video_bitrate_kbps,
                "ffmpeg_bin": self.ffmpeg_bin,
                "ffmpeg_available": self.is_ffmpeg_available(),
                "frames_published": self._frames_published,
                "actual_fps": round(self._actual_fps, 2),
                "last_error": self._last_error,
            }
