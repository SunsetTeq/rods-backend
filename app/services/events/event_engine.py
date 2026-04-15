import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db.repository import EventRepository
from app.services.capture.camera_service import CameraService
from app.services.storage.screenshot_service import ScreenshotService
from app.services.vision.detector_service import DetectorService


logger = logging.getLogger(__name__)


@dataclass
class TrackedObjectState:
    state_key: str
    class_name: str | None = None
    track_id: int | None = None
    tracked: bool = False
    state: str = "NOT_PRESENT"
    present_frames: int = 0
    absent_frames: int = 0
    cooldown_until: float | None = None
    first_seen_frame_id: int | None = None
    last_seen_frame_id: int | None = None
    latest_confidence: float | None = None
    class_id: int | None = None


class EventEngineService:
    def __init__(
        self,
        camera_service: CameraService,
        detector_service: DetectorService,
        repository: EventRepository,
        screenshot_service: ScreenshotService,
        stable_frames: int,
        absent_frames: int,
        cooldown_seconds: int,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.camera_service = camera_service
        self.detector_service = detector_service
        self.repository = repository
        self.screenshot_service = screenshot_service
        self.stable_frames = stable_frames
        self.absent_frames = absent_frames
        self.cooldown_seconds = cooldown_seconds
        self.poll_interval_seconds = poll_interval_seconds

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._states: dict[str, TrackedObjectState] = {}
        self._is_running = False
        self._processed_detection_frames = 0
        self._confirmed_events_total = 0
        self._latest_processed_frame_id = 0

    def start(self) -> None:
        if self._is_running:
            logger.info("Event engine already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info(
            "Event engine started | stable_frames=%s | absent_frames=%s | cooldown=%ss",
            self.stable_frames,
            self.absent_frames,
            self.cooldown_seconds,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._is_running = False
        logger.info("Event engine stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            payload = self.detector_service.get_latest_detections()
            frame_id = int(payload["frame_id"])

            if frame_id == 0 or frame_id == self._latest_processed_frame_id:
                time.sleep(self.poll_interval_seconds)
                continue

            self.process_detection_frame(payload)
            time.sleep(self.poll_interval_seconds)

    def process_detection_frame(self, payload: dict[str, Any]) -> None:
        frame_id = int(payload["frame_id"])
        frame_timestamp = payload.get("frame_timestamp") or datetime.now(timezone.utc).isoformat()
        detections = payload.get("detections", [])
        source_frame_size = payload.get("source_frame_size")
        width = source_frame_size[0] if source_frame_size else None
        height = source_frame_size[1] if source_frame_size else None

        detections_by_state_key: dict[str, dict[str, Any]] = {}
        for detection in detections:
            state_key = self._build_state_key(detection)
            existing = detections_by_state_key.get(state_key)
            if existing is None or float(detection["confidence"]) > float(existing["confidence"]):
                detections_by_state_key[state_key] = detection

        with self._lock:
            self._processed_detection_frames += 1
            self._latest_processed_frame_id = frame_id
            now = time.time()
            state_keys = set(self._states.keys()) | set(detections_by_state_key.keys())

            for state_key in state_keys:
                detection = detections_by_state_key.get(state_key)
                state = self._states.get(state_key)

                if state is None:
                    state = TrackedObjectState(
                        state_key=state_key,
                        class_name=detection["class_name"] if detection is not None else None,
                        track_id=(
                            int(detection["track_id"])
                            if detection is not None and detection.get("track_id") is not None
                            else None
                        ),
                        tracked=(
                            detection is not None and detection.get("track_id") is not None
                        ),
                    )
                    self._states[state_key] = state

                if state.state == "COOLDOWN" and state.cooldown_until is not None and now >= state.cooldown_until:
                    state.state = "NOT_PRESENT"
                    state.present_frames = 0
                    state.absent_frames = 0
                    state.cooldown_until = None
                    state.first_seen_frame_id = None

                if detection is not None:
                    self._handle_present_detection(
                        state_key=state_key,
                        state=state,
                        detection=detection,
                        frame_id=frame_id,
                        frame_timestamp=frame_timestamp,
                        source_frame_width=width,
                        source_frame_height=height,
                        now=now,
                    )
                else:
                    self._handle_absent_detection(state=state, now=now)

            self._states = {
                key: value
                for key, value in self._states.items()
                if not (
                    value.state == "NOT_PRESENT"
                    and value.present_frames == 0
                    and value.absent_frames == 0
                    and value.cooldown_until is None
                )
            }

    def _handle_present_detection(
        self,
        state_key: str,
        state: TrackedObjectState,
        detection: dict[str, Any],
        frame_id: int,
        frame_timestamp: str,
        source_frame_width: int | None,
        source_frame_height: int | None,
        now: float,
    ) -> None:
        state.class_name = detection["class_name"]
        state.track_id = (
            int(detection["track_id"]) if detection.get("track_id") is not None else None
        )
        state.tracked = state.track_id is not None
        state.last_seen_frame_id = frame_id
        state.latest_confidence = float(detection["confidence"])
        state.class_id = int(detection["class_id"])
        state.absent_frames = 0

        if state.state == "COOLDOWN" and state.cooldown_until is not None and now < state.cooldown_until:
            return

        if state.state in {"NOT_PRESENT", "COOLDOWN"}:
            state.state = "CANDIDATE"
            state.present_frames = 1
            state.first_seen_frame_id = frame_id
            return

        if state.state == "CANDIDATE":
            state.present_frames += 1
            if state.present_frames >= self.stable_frames:
                original_frame = self.camera_service.get_latest_frame()
                annotated_frame = self.detector_service.get_latest_annotated_frame()
                event_id = self.repository.create_event(
                    {
                        "event_type": "confirmed",
                        "class_name": detection["class_name"],
                        "class_id": int(detection["class_id"]),
                        "track_id": state.track_id,
                        "confidence": float(detection["confidence"]),
                        "state_key": state_key,
                        "first_seen_frame_id": state.first_seen_frame_id or frame_id,
                        "confirmed_frame_id": frame_id,
                        "last_seen_frame_id": frame_id,
                        "stable_frames_required": self.stable_frames,
                        "absent_frames_required": self.absent_frames,
                        "cooldown_seconds": self.cooldown_seconds,
                        "source_frame_width": source_frame_width,
                        "source_frame_height": source_frame_height,
                        "frame_timestamp": frame_timestamp,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "screenshot_original_path": None,
                        "screenshot_annotated_path": None,
                    }
                )
                screenshot_paths = self.screenshot_service.save_event_frames(
                    event_id=event_id,
                    original_frame=original_frame,
                    annotated_frame=annotated_frame,
                    frame_timestamp=frame_timestamp,
                )
                self.repository.update_event_screenshots(
                    event_id=event_id,
                    screenshot_original_path=screenshot_paths["screenshot_original_path"],
                    screenshot_annotated_path=screenshot_paths["screenshot_annotated_path"],
                )
                self._confirmed_events_total += 1
                state.state = "CONFIRMED"
                logger.info(
                    "Confirmed event | id=%s | class=%s | track_id=%s | confidence=%.2f | frame_id=%s | original=%s | annotated=%s",
                    event_id,
                    detection["class_name"],
                    state.track_id,
                    float(detection["confidence"]),
                    frame_id,
                    screenshot_paths["screenshot_original_path"],
                    screenshot_paths["screenshot_annotated_path"],
                )
            return

        if state.state == "CONFIRMED":
            state.present_frames += 1

    def _handle_absent_detection(self, state: TrackedObjectState, now: float) -> None:
        if state.state == "NOT_PRESENT":
            return

        if state.state == "COOLDOWN":
            if state.cooldown_until is not None and now >= state.cooldown_until:
                state.state = "NOT_PRESENT"
                state.present_frames = 0
                state.absent_frames = 0
                state.cooldown_until = None
                state.first_seen_frame_id = None
                state.last_seen_frame_id = None
                state.latest_confidence = None
            return

        state.absent_frames += 1
        if state.absent_frames < self.absent_frames:
            return

        if state.state == "CANDIDATE":
            state.state = "NOT_PRESENT"
            state.present_frames = 0
            state.absent_frames = 0
            state.first_seen_frame_id = None
            state.last_seen_frame_id = None
            state.latest_confidence = None
            return

        if state.state == "CONFIRMED":
            state.state = "COOLDOWN"
            state.present_frames = 0
            state.absent_frames = 0
            state.cooldown_until = now + self.cooldown_seconds
            return

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            active_states = []
            for key, state in sorted(self._states.items()):
                cooldown_until = None
                if state.cooldown_until is not None:
                    cooldown_until = datetime.fromtimestamp(
                        state.cooldown_until, tz=timezone.utc
                    ).isoformat()

                active_states.append(
                    {
                        "state_key": key,
                        "class_name": state.class_name,
                        "track_id": state.track_id,
                        "state": state.state,
                        "present_frames": state.present_frames,
                        "absent_frames": state.absent_frames,
                        "cooldown_until": cooldown_until,
                        "first_seen_frame_id": state.first_seen_frame_id,
                        "last_seen_frame_id": state.last_seen_frame_id,
                        "latest_confidence": state.latest_confidence,
                    }
                )

            return {
                "is_running": self._is_running,
                "stable_frames": self.stable_frames,
                "absent_frames": self.absent_frames,
                "cooldown_seconds": self.cooldown_seconds,
                "processed_detection_frames": self._processed_detection_frames,
                "confirmed_events_total": self._confirmed_events_total,
                "latest_processed_frame_id": self._latest_processed_frame_id,
                "active_states": active_states,
            }

    def _build_state_key(self, detection: dict[str, Any]) -> str:
        class_name = str(detection["class_name"])
        track_id = detection.get("track_id")
        if track_id is not None:
            return f"track:{int(track_id)}:{class_name}"
        return f"class:{class_name}"
