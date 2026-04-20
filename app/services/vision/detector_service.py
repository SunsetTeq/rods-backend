import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.capture.camera_service import CameraService


logger = logging.getLogger(__name__)
CLASS_NAME_TRANSLATIONS = {
    "knife": "нож",
    "fork": "вилка",
    "scissors": "ножницы",
}
DEFAULT_LABEL_FONT_CANDIDATES = (
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


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
        tracking_enabled: bool = True,
        tracking_persist: bool = True,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.camera_service = camera_service
        self.enabled = enabled
        self.model_path = self._resolve_model_path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.inference_fps = inference_fps
        self.jpeg_quality = jpeg_quality
        self.log_interval_seconds = log_interval_seconds
        self.tracking_enabled = tracking_enabled
        self.tracking_persist = tracking_persist
        self.tracker_config = tracker_config
        self._tracking_runtime_enabled = tracking_enabled
        self._tracking_fallback_reason: Optional[str] = None

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
        self._tracked_detections_count = 0

        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_annotated_jpeg: Optional[bytes] = None
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._label_font_path = self._resolve_label_font_path()
        self._latest_detection_payload: dict[str, Any] = {
            "frame_id": 0,
            "source_frame_size": None,
            "inference_ms": 0.0,
            "detections_count": 0,
            "detections": [],
        }

    def _resolve_model_path(self, model_path: str) -> str:
        normalized_path = model_path.strip().replace("\\", os.sep)
        if not normalized_path:
            return model_path

        repo_root = Path(__file__).resolve().parents[3]
        candidates: list[Path] = []

        direct_candidate = Path(normalized_path).expanduser()
        candidates.append(direct_candidate)

        if not direct_candidate.is_absolute():
            candidates.append(repo_root / direct_candidate)

        basename_candidate = repo_root / Path(normalized_path).name
        candidates.append(basename_candidate)

        seen_candidates: set[str] = set()
        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            if candidate.exists():
                return str(candidate.resolve())

        return normalized_path

    def _resolve_label_font_path(self) -> str | None:
        for candidate in DEFAULT_LABEL_FONT_CANDIDATES:
            if Path(candidate).exists():
                return candidate
        return None

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

        results = self._run_model(frame)

        if not results:
            self._tracked_detections_count = 0
            return frame.copy(), self._build_payload(frame, [])

        result = results[0]
        detections = self._extract_detections(result)
        annotated_frame = self._draw_detections(frame, detections)
        return annotated_frame, self._build_payload(frame, detections)

    def _run_model(self, frame: np.ndarray) -> list[Any]:
        if self._tracking_runtime_enabled:
            try:
                return self._model.track(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    tracker=self.tracker_config,
                    persist=self.tracking_persist,
                    verbose=False,
                )
            except ModuleNotFoundError as exc:
                if exc.name != "lap":
                    raise

                self._tracking_runtime_enabled = False
                self._tracking_fallback_reason = (
                    "Tracking disabled at runtime because dependency 'lap' is not installed. "
                    "Falling back to plain detection."
                )
                self._tracked_detections_count = 0
                logger.warning(self._tracking_fallback_reason)

        return self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

    def _extract_detections(self, result: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}

        if boxes is None:
            self._tracked_detections_count = 0
            return []

        xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []
        track_ids = boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None else []

        self._tracked_detections_count = len(track_ids)

        detections: list[dict[str, Any]] = []
        for index, (coords, confidence, class_id) in enumerate(zip(xyxy, confs, classes)):
            x1, y1, x2, y2 = [int(value) for value in coords]
            class_id_int = int(class_id)
            class_name_en = str(names.get(class_id_int, f"class_{class_id_int}"))
            class_name_ru = self._localize_class_name(class_name_en)
            detections.append(
                {
                    "class_id": class_id_int,
                    "class_name": class_name_ru,
                    "class_name_en": class_name_en,
                    "class_name_ru": class_name_ru,
                    "track_id": int(track_ids[index]) if index < len(track_ids) else None,
                    "confidence": round(float(confidence), 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        return detections

    def _localize_class_name(self, class_name: str) -> str:
        return CLASS_NAME_TRANSLATIONS.get(class_name.strip().lower(), class_name)

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
            (
                f"{item['class_name']}#{item['track_id']}({item['confidence']:.2f})"
                if item.get("track_id") is not None
                else f"{item['class_name']}({item['confidence']:.2f})"
            )
            for item in payload["detections"][:5]
        )
        logger.info(
            "Detections | frame_id=%s | count=%s | tracked=%s | items=%s",
            payload["frame_id"],
            payload["detections_count"],
            self._tracked_detections_count,
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

    def get_live_annotated_frame(self) -> Optional[np.ndarray]:
        frame = self.camera_service.get_latest_frame()
        if frame is None:
            return self.get_latest_annotated_frame()
        return self.compose_annotated_frame(frame)

    def get_live_annotated_jpeg(self) -> Optional[bytes]:
        frame = self.get_live_annotated_frame()
        if frame is None:
            return None

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            return None

        return encoded.tobytes()

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

    def compose_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.get_latest_detections()["detections"]
        return self._draw_detections(frame=frame, detections=detections)

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        annotated = frame.copy()
        if not detections:
            return annotated

        image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font = self._get_label_font(max(18, annotated.shape[1] // 55))

        for detection in detections:
            x1 = int(detection["x1"])
            y1 = int(detection["y1"])
            x2 = int(detection["x2"])
            y2 = int(detection["y2"])
            confidence = float(detection["confidence"])
            class_name = self._get_display_class_name(detection)
            track_id = detection.get("track_id")

            box_color = (36, 255, 12)
            text_color = (0, 0, 0)
            box_thickness = 2

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=box_color,
                width=box_thickness,
            )

            label = class_name
            if track_id is not None:
                label = f"{label} #{int(track_id)}"
            label = f"{label} {confidence:.2f}"

            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_padding_x = 8
            text_padding_y = 4
            text_top = max(0, y1 - text_height - (text_padding_y * 2) - 6)
            text_bottom = text_top + text_height + (text_padding_y * 2)
            text_right = min(image.width, x1 + text_width + (text_padding_x * 2))

            draw.rectangle(
                [(x1, text_top), (text_right, text_bottom)],
                fill=box_color,
            )
            draw.text(
                (x1 + text_padding_x, text_top + text_padding_y - 1),
                label,
                font=font,
                fill=text_color,
            )

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _get_label_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        cached_font = self._font_cache.get(size)
        if cached_font is not None:
            return cached_font

        if self._label_font_path is not None:
            try:
                font = ImageFont.truetype(self._label_font_path, size=size)
                self._font_cache[size] = font
                return font
            except OSError:
                logger.warning("Failed to load label font: %s", self._label_font_path)
                self._label_font_path = None

        font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def _get_display_class_name(self, detection: dict[str, Any]) -> str:
        class_name = str(detection.get("class_name") or "")
        class_name_en = str(detection.get("class_name_en") or class_name or "object")

        # If the runtime fell back to Pillow's default bitmap font, Cyrillic labels
        # are rendered as broken glyphs on the annotated stream. In that case we keep
        # the API payload localized, but draw the box label in English.
        if self._label_font_path is None and not class_name.isascii():
            return class_name_en

        return class_name

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_path": self.model_path,
            "tracking_enabled": self.tracking_enabled,
            "tracking_persist": self.tracking_persist,
            "tracker_config": self.tracker_config,
            "tracking_runtime_enabled": self._tracking_runtime_enabled,
            "is_running": self._is_running,
            "is_model_loaded": self._model is not None,
            "detector_available": self._detector_available,
            "latest_frame_id": self._latest_frame_id,
            "processed_frames": self._processed_frames,
            "skipped_frames": self._skipped_frames,
            "tracked_detections_count": self._tracked_detections_count,
            "actual_fps": round(self._actual_fps, 2),
            "last_inference_ms": self._last_inference_ms,
            "live_annotations_supported": True,
            "last_error": self._last_error or self._tracking_fallback_reason,
        }
