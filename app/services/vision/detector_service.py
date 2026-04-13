import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.services.capture.camera_service import CameraService


logger = logging.getLogger(__name__)


class DetectorService:
    def __init__(
        self,
        camera_service: CameraService,
        enabled: bool,
        model_path: str,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        inference_fps: int,
        jpeg_quality: int,
        log_interval_seconds: int = 5,
    ) -> None:
        self.camera_service = camera_service
        self.enabled = enabled
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.inference_fps = inference_fps
        self.jpeg_quality = jpeg_quality
        self.log_interval_seconds = log_interval_seconds

        self._model: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame_id = 0
        self._processed_frames = 0
        self._skipped_frames = 0
        self._actual_fps = 0.0
        self._fps_counter = 0
        self._last_fps_calc_time = time.time()
        self._last_inference_ms = 0.0
        self._last_error: Optional[str] = None
        self._last_logged_detection_at = 0.0
        self._detector_available = False
        self._is_running = False

        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_annotated_jpeg: Optional[bytes] = None
        self._latest_detection_payload: dict[str, Any] = {
            "frame_id": 0,
            "source_frame_size": None,
            "inference_ms": 0.0,
            "detections_count": 0,
            "detections": [],
        }

    def start(self) -> None:
        if not self.enabled:
            logger.info("Detector service is disabled by config")
            return

        if self._is_running:
            logger.info("Detector service already running")
            return

        self._last_error = None

        try:
            self._ensure_model_loaded()
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Detector service failed to initialize")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info("Detector service started | model=%s", self.model_path)

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

        self._thread = None
        self._is_running = False
        logger.info("Detector service stopped")

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        ultralytics_config_dir = Path(__file__).resolve().parents[3] / "data" / "ultralytics"
        ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config_dir))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._model = YOLO(self.model_path)
        self._detector_available = True

    def _inference_loop(self) -> None:
        target_interval = 1.0 / max(self.inference_fps, 1)

        while not self._stop_event.is_set():
            frame = self.camera_service.get_latest_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            started_at = time.perf_counter()

            try:
                annotated_frame, payload = self._infer_frame(frame)
                inference_ms = (time.perf_counter() - started_at) * 1000
                payload["inference_ms"] = round(inference_ms, 2)
                self._last_inference_ms = round(inference_ms, 2)
                self._save_latest_result(annotated_frame, payload)
                self._log_detections_if_needed(payload)
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("Detector inference failed")
                time.sleep(0.2)
                continue

            self._processed_frames += 1
            self._fps_counter += 1
            self._update_fps()

            elapsed = time.perf_counter() - started_at
            sleep_time = max(0.0, target_interval - elapsed)
            if sleep_time:
                time.sleep(sleep_time)

    def _infer_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("Detector model is not loaded")

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

        if not results:
            return frame.copy(), self._build_payload(frame, [])

        result = results[0]
        annotated_frame = result.plot()
        detections = self._extract_detections(result)
        return annotated_frame, self._build_payload(frame, detections)

    def _extract_detections(self, result: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}

        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []

        detections: list[dict[str, Any]] = []
        for coords, confidence, class_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = [int(value) for value in coords]
            class_id_int = int(class_id)
            detections.append(
                {
                    "class_id": class_id_int,
                    "class_name": str(names.get(class_id_int, f"class_{class_id_int}")),
                    "confidence": round(float(confidence), 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        return detections

    def _build_payload(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._latest_frame_id += 1
        height, width = frame.shape[:2]
        return {
            "frame_id": self._latest_frame_id,
            "source_frame_size": (width, height),
            "frame_timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_ms": 0.0,
            "detections_count": len(detections),
            "detections": detections,
        }

    def _save_latest_result(self, annotated_frame: np.ndarray, payload: dict[str, Any]) -> None:
        success, encoded = cv2.imencode(
            ".jpg",
            annotated_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )

        if not success:
            self._skipped_frames += 1
            return

        with self._lock:
            self._latest_annotated_frame = annotated_frame.copy()
            self._latest_annotated_jpeg = encoded.tobytes()
            self._latest_detection_payload = payload
            self._last_error = None

    def _log_detections_if_needed(self, payload: dict[str, Any]) -> None:
        if payload["detections_count"] == 0:
            return

        now = time.time()
        if now - self._last_logged_detection_at < self.log_interval_seconds:
            return

        summary = ", ".join(
            f"{item['class_name']}({item['confidence']:.2f})"
            for item in payload["detections"][:5]
        )
        logger.info(
            "Detections | frame_id=%s | count=%s | items=%s",
            payload["frame_id"],
            payload["detections_count"],
            summary,
        )
        self._last_logged_detection_at = now

    def _update_fps(self) -> None:
        now = time.time()
        elapsed = now - self._last_fps_calc_time
        if elapsed >= 1.0:
            self._actual_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._last_fps_calc_time = now

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_annotated_frame is None:
                return None
            return self._latest_annotated_frame.copy()

    def get_latest_annotated_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_annotated_jpeg

    def get_latest_detections(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frame_id": self._latest_detection_payload["frame_id"],
                "source_frame_size": self._latest_detection_payload["source_frame_size"],
                "frame_timestamp": self._latest_detection_payload.get("frame_timestamp"),
                "inference_ms": self._latest_detection_payload["inference_ms"],
                "detections_count": self._latest_detection_payload["detections_count"],
                "detections": [item.copy() for item in self._latest_detection_payload["detections"]],
            }

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_path": self.model_path,
            "is_running": self._is_running,
            "is_model_loaded": self._model is not None,
            "detector_available": self._detector_available,
            "latest_frame_id": self._latest_frame_id,
            "processed_frames": self._processed_frames,
            "skipped_frames": self._skipped_frames,
            "actual_fps": round(self._actual_fps, 2),
            "last_inference_ms": self._last_inference_ms,
            "last_error": self._last_error,
        }
