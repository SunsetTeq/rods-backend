import logging
import os
import threading
import time
from dataclasses import dataclass
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

Rect = tuple[int, int, int, int]


@dataclass
class ActiveTrack:
    track_id: int
    class_id: int
    class_name: str
    class_name_en: str
    class_name_ru: str
    bbox: Rect
    misses: int = 0


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
        focus_enabled: bool = True,
        focus_motion_threshold: int = 25,
        focus_min_motion_area: int = 1200,
        focus_padding: int = 64,
        focus_hold_frames: int = 4,
        focus_merge_gap: int = 32,
        focus_max_rois: int = 3,
        focus_full_frame_area_threshold: float = 0.45,
        focus_full_frame_refresh_interval: int = 20,
        focus_tracking_iou_threshold: float = 0.2,
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
        self.focus_enabled = focus_enabled
        self.focus_motion_threshold = max(1, focus_motion_threshold)
        self.focus_min_motion_area = max(1, focus_min_motion_area)
        self.focus_padding = max(0, focus_padding)
        self.focus_hold_frames = max(1, focus_hold_frames)
        self.focus_merge_gap = max(0, focus_merge_gap)
        self.focus_max_rois = max(1, focus_max_rois)
        self.focus_full_frame_area_threshold = min(
            max(focus_full_frame_area_threshold, 0.0),
            1.0,
        )
        self.focus_full_frame_refresh_interval = max(0, focus_full_frame_refresh_interval)
        self.focus_tracking_iou_threshold = min(
            max(focus_tracking_iou_threshold, 0.0),
            1.0,
        )
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
        self._next_track_id = 1
        self._active_tracks: dict[int, ActiveTrack] = {}
        self._previous_frame_gray: Optional[np.ndarray] = None
        self._frames_since_full_frame = 0
        self._last_focus_strategy = "full_frame"
        self._last_focus_regions: list[Rect] = []

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
        self._reset_runtime_focus_state()

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
        self._reset_runtime_focus_state()
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

        rois, strategy = self._plan_inference_rois(frame)
        detections: list[dict[str, Any]] = []

        for roi in rois:
            crop = self._crop_frame(frame, roi)
            if crop is None:
                continue
            detections.extend(self._infer_roi(crop=crop, roi=roi))

        detections = self._deduplicate_detections(detections)
        detections = self._assign_track_ids(detections)
        annotated_frame = self._draw_detections(frame, detections)
        self._last_focus_strategy = strategy
        self._last_focus_regions = [tuple(region) for region in rois]
        return annotated_frame, self._build_payload(frame, detections)

    def _infer_roi(self, crop: np.ndarray, roi: Rect) -> list[dict[str, Any]]:
        results = self._run_model(crop)
        if not results:
            return []

        return self._extract_detections(
            result=results[0],
            offset_x=roi[0],
            offset_y=roi[1],
        )

    def _run_model(self, frame: np.ndarray) -> list[Any]:
        return self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

    def _extract_detections(
        self,
        result: Any,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[dict[str, Any]]:
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
            class_name_en = str(names.get(class_id_int, f"class_{class_id_int}"))
            class_name_ru = self._localize_class_name(class_name_en)
            detections.append(
                {
                    "class_id": class_id_int,
                    "class_name": class_name_ru,
                    "class_name_en": class_name_en,
                    "class_name_ru": class_name_ru,
                    "track_id": None,
                    "confidence": round(float(confidence), 4),
                    "x1": x1 + offset_x,
                    "y1": y1 + offset_y,
                    "x2": x2 + offset_x,
                    "y2": y2 + offset_y,
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
            "Detections | frame_id=%s | count=%s | tracked=%s | strategy=%s | rois=%s | items=%s",
            payload["frame_id"],
            payload["detections_count"],
            self._tracked_detections_count,
            self._last_focus_strategy,
            len(self._last_focus_regions),
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

    def _plan_inference_rois(self, frame: np.ndarray) -> tuple[list[Rect], str]:
        frame_height, frame_width = frame.shape[:2]
        full_frame = [(0, 0, frame_width, frame_height)]

        if not self.focus_enabled:
            self._frames_since_full_frame = 0
            return full_frame, "full_frame"

        if self._latest_frame_id == 0:
            self._frames_since_full_frame = 0
            self._detect_motion_rois(frame)
            return full_frame, "full_frame_bootstrap"

        motion_rois = self._detect_motion_rois(frame)
        track_rois = self._get_active_track_rois(frame_width=frame_width, frame_height=frame_height)
        candidate_rois = self._merge_rects(motion_rois + track_rois)

        if not candidate_rois:
            if (
                self.focus_full_frame_refresh_interval > 0
                and self._frames_since_full_frame >= self.focus_full_frame_refresh_interval
            ):
                self._frames_since_full_frame = 0
                return full_frame, "full_frame_refresh"

            self._frames_since_full_frame += 1
            return [], "idle"

        merged_area_ratio = self._sum_rect_areas(candidate_rois) / max(frame_width * frame_height, 1)

        if merged_area_ratio > self.focus_full_frame_area_threshold:
            self._frames_since_full_frame = 0
            return full_frame, "full_frame_large_roi"

        if len(candidate_rois) > self.focus_max_rois:
            self._frames_since_full_frame = 0
            return full_frame, "full_frame_many_roi"

        self._frames_since_full_frame += 1
        return candidate_rois, "focus"

    def _detect_motion_rois(self, frame: np.ndarray) -> list[Rect]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        previous_gray = self._previous_frame_gray
        self._previous_frame_gray = gray

        if previous_gray is None:
            return []

        diff = cv2.absdiff(previous_gray, gray)
        _, threshold = cv2.threshold(
            diff,
            self.focus_motion_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        threshold = cv2.dilate(threshold, None, iterations=2)
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        frame_height, frame_width = frame.shape[:2]
        rois: list[Rect] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.focus_min_motion_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            rois.append(
                self._expand_rect(
                    rect=(x, y, x + width, y + height),
                    padding=self.focus_padding,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

        return rois

    def _get_active_track_rois(self, frame_width: int, frame_height: int) -> list[Rect]:
        if not self.tracking_enabled:
            return []

        return [
            self._expand_rect(
                rect=track.bbox,
                padding=self.focus_padding,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            for track in self._active_tracks.values()
        ]

    def _assign_track_ids(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.tracking_enabled:
            self._tracked_detections_count = 0
            self._active_tracks.clear()
            return detections

        existing_tracks = dict(self._active_tracks)
        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()

        matches: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            detection_bbox = self._detection_to_rect(detection)
            for track_id, track in existing_tracks.items():
                if detection["class_id"] != track.class_id:
                    continue

                iou = self._calculate_iou(detection_bbox, track.bbox)
                if iou < self.focus_tracking_iou_threshold:
                    continue
                matches.append((iou, detection_index, track_id))

        for _, detection_index, track_id in sorted(matches, reverse=True):
            if detection_index in matched_detection_indexes or track_id in matched_track_ids:
                continue

            detection = detections[detection_index]
            detection["track_id"] = track_id
            self._active_tracks[track_id] = ActiveTrack(
                track_id=track_id,
                class_id=int(detection["class_id"]),
                class_name=str(detection["class_name"]),
                class_name_en=str(detection.get("class_name_en") or detection["class_name"]),
                class_name_ru=str(detection.get("class_name_ru") or detection["class_name"]),
                bbox=self._detection_to_rect(detection),
                misses=0,
            )
            matched_track_ids.add(track_id)
            matched_detection_indexes.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue

            track_id = self._next_track_id
            self._next_track_id += 1
            detection["track_id"] = track_id
            self._active_tracks[track_id] = ActiveTrack(
                track_id=track_id,
                class_id=int(detection["class_id"]),
                class_name=str(detection["class_name"]),
                class_name_en=str(detection.get("class_name_en") or detection["class_name"]),
                class_name_ru=str(detection.get("class_name_ru") or detection["class_name"]),
                bbox=self._detection_to_rect(detection),
                misses=0,
            )
            matched_track_ids.add(track_id)

        expired_track_ids: list[int] = []
        for track_id, track in existing_tracks.items():
            if track_id in matched_track_ids:
                continue

            updated_track = self._active_tracks.get(track_id, track)
            updated_track.misses += 1
            if updated_track.misses >= self.focus_hold_frames:
                expired_track_ids.append(track_id)
            else:
                self._active_tracks[track_id] = updated_track

        for track_id in expired_track_ids:
            self._active_tracks.pop(track_id, None)

        self._tracked_detections_count = len(detections)
        return detections

    def _deduplicate_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduplicated: list[dict[str, Any]] = []

        for detection in sorted(
            detections,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        ):
            detection_rect = self._detection_to_rect(detection)
            is_duplicate = False

            for kept in deduplicated:
                if detection["class_id"] != kept["class_id"]:
                    continue

                kept_rect = self._detection_to_rect(kept)
                if self._calculate_iou(detection_rect, kept_rect) >= 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(detection)

        return deduplicated

    def _merge_rects(self, rects: list[Rect]) -> list[Rect]:
        merged = [self._normalize_rect(rect) for rect in rects if self._is_valid_rect(rect)]

        changed = True
        while changed:
            changed = False
            next_rects: list[Rect] = []

            while merged:
                current = merged.pop()
                was_merged = False

                for index, candidate in enumerate(merged):
                    if not self._rects_should_merge(current, candidate):
                        continue

                    merged[index] = self._union_rects(current, candidate)
                    changed = True
                    was_merged = True
                    break

                if not was_merged:
                    next_rects.append(current)

            merged = next_rects

        return sorted(merged)

    def _rects_should_merge(self, left: Rect, right: Rect) -> bool:
        lx1, ly1, lx2, ly2 = left
        rx1, ry1, rx2, ry2 = right

        horizontal_gap = max(0, max(lx1, rx1) - min(lx2, rx2))
        vertical_gap = max(0, max(ly1, ry1) - min(ly2, ry2))
        return horizontal_gap <= self.focus_merge_gap and vertical_gap <= self.focus_merge_gap

    def _union_rects(self, left: Rect, right: Rect) -> Rect:
        return (
            min(left[0], right[0]),
            min(left[1], right[1]),
            max(left[2], right[2]),
            max(left[3], right[3]),
        )

    def _expand_rect(
        self,
        rect: Rect,
        padding: int,
        frame_width: int,
        frame_height: int,
    ) -> Rect:
        x1, y1, x2, y2 = self._normalize_rect(rect)
        return (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(frame_width, x2 + padding),
            min(frame_height, y2 + padding),
        )

    def _normalize_rect(self, rect: Rect) -> Rect:
        x1, y1, x2, y2 = rect
        return (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )

    def _is_valid_rect(self, rect: Rect) -> bool:
        x1, y1, x2, y2 = self._normalize_rect(rect)
        return x2 > x1 and y2 > y1

    def _sum_rect_areas(self, rects: list[Rect]) -> int:
        return sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in rects)

    def _crop_frame(self, frame: np.ndarray, rect: Rect) -> np.ndarray | None:
        x1, y1, x2, y2 = self._normalize_rect(rect)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _detection_to_rect(self, detection: dict[str, Any]) -> Rect:
        return (
            int(detection["x1"]),
            int(detection["y1"]),
            int(detection["x2"]),
            int(detection["y2"]),
        )

    def _calculate_iou(self, left: Rect, right: Rect) -> float:
        left_x1, left_y1, left_x2, left_y2 = self._normalize_rect(left)
        right_x1, right_y1, right_x2, right_y2 = self._normalize_rect(right)

        intersection_x1 = max(left_x1, right_x1)
        intersection_y1 = max(left_y1, right_y1)
        intersection_x2 = min(left_x2, right_x2)
        intersection_y2 = min(left_y2, right_y2)

        if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
            return 0.0

        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        left_area = (left_x2 - left_x1) * (left_y2 - left_y1)
        right_area = (right_x2 - right_x1) * (right_y2 - right_y1)
        union_area = left_area + right_area - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def _reset_runtime_focus_state(self) -> None:
        self._next_track_id = 1
        self._active_tracks.clear()
        self._previous_frame_gray = None
        self._frames_since_full_frame = 0
        self._last_focus_strategy = "full_frame"
        self._last_focus_regions = []
