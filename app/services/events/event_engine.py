import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.db.repository import EventRepository
from app.services.capture.camera_service import CameraService
from app.services.events.serialization import serialize_event_row
from app.services.live_event_provider import live_event_service
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
    confirmed_event_id: int | None = None
    observed_classes: set[str] = field(default_factory=set)


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
        detections_by_state_key = self._group_detections_by_state_key(detections)

        with self._lock:
            self._processed_detection_frames += 1
            self._latest_processed_frame_id = frame_id
            now = time.time()
            state_keys = set(self._states.keys()) | set(detections_by_state_key.keys())

            for state_key in state_keys:
                detection_group = detections_by_state_key.get(state_key)
                state = self._states.get(state_key)

                if state is None:
                    state = self._create_state(state_key=state_key, detection_group=detection_group)
                    self._states[state_key] = state

                if (
                    state.state == "COOLDOWN"
                    and state.cooldown_until is not None
                    and now >= state.cooldown_until
                ):
                    self._reset_state(state)

                if detection_group is not None:
                    self._handle_present_detection(
                        state_key=state_key,
                        state=state,
                        detection_group=detection_group,
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

    def _create_state(
        self,
        state_key: str,
        detection_group: dict[str, Any] | None,
    ) -> TrackedObjectState:
        if detection_group is None:
            return TrackedObjectState(state_key=state_key)

        return TrackedObjectState(
            state_key=state_key,
            class_name=detection_group["class_name"],
            track_id=detection_group.get("track_id"),
            tracked=detection_group.get("track_id") is not None,
            observed_classes=set(detection_group["observed_classes"]),
        )

    def _handle_present_detection(
        self,
        state_key: str,
        state: TrackedObjectState,
        detection_group: dict[str, Any],
        frame_id: int,
        frame_timestamp: str,
        source_frame_width: int | None,
        source_frame_height: int | None,
        now: float,
    ) -> None:
        previous_classes = set(state.observed_classes)
        current_classes = set(detection_group["observed_classes"])

        if state.state != "CONFIRMED" or state.class_name is None:
            state.class_name = detection_group["class_name"]
            state.class_id = detection_group["class_id"]
        state.track_id = detection_group.get("track_id")
        state.tracked = state.track_id is not None
        state.last_seen_frame_id = frame_id
        state.latest_confidence = float(detection_group["confidence"])
        if state.observed_classes:
            state.observed_classes |= current_classes
        else:
            state.observed_classes = current_classes
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
                event_id = self._create_confirmed_event(
                    state_key=state_key,
                    state=state,
                    frame_id=frame_id,
                    frame_timestamp=frame_timestamp,
                    source_frame_width=source_frame_width,
                    source_frame_height=source_frame_height,
                )
                state.confirmed_event_id = event_id
                state.state = "CONFIRMED"
            return

        if state.state == "CONFIRMED":
            state.present_frames += 1
            if state.observed_classes != previous_classes:
                self._update_confirmed_event(
                    state=state,
                    frame_id=frame_id,
                    frame_timestamp=frame_timestamp,
                )

    def _create_confirmed_event(
        self,
        state_key: str,
        state: TrackedObjectState,
        frame_id: int,
        frame_timestamp: str,
        source_frame_width: int | None,
        source_frame_height: int | None,
    ) -> int:
        original_frame = self.camera_service.get_latest_frame()
        annotated_frame = self.detector_service.get_latest_annotated_frame()
        event_id = self.repository.create_event(
            {
                "event_type": "confirmed",
                "class_name": state.class_name or "unknown",
                "class_id": state.class_id,
                "track_id": state.track_id,
                "observed_classes": sorted(state.observed_classes),
                "confidence": float(state.latest_confidence or 0.0),
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
                "updated_at": None,
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

        row = self.repository.get_event_by_id(event_id)
        if row is not None:
            live_event_service.publish_from_thread(
                live_event_service.build_message(
                    message_type="event_confirmed",
                    event=serialize_event_row(row),
                )
            )

        logger.info(
            "Confirmed event | id=%s | class=%s | track_id=%s | classes=%s | confidence=%.2f | frame_id=%s",
            event_id,
            state.class_name,
            state.track_id,
            ",".join(sorted(state.observed_classes)),
            float(state.latest_confidence or 0.0),
            frame_id,
        )
        return event_id

    def _update_confirmed_event(
        self,
        state: TrackedObjectState,
        frame_id: int,
        frame_timestamp: str,
    ) -> None:
        if state.confirmed_event_id is None:
            return

        original_frame = self.camera_service.get_latest_frame()
        annotated_frame = self.detector_service.get_latest_annotated_frame()
        screenshot_paths = self.screenshot_service.save_event_frames(
            event_id=state.confirmed_event_id,
            original_frame=original_frame,
            annotated_frame=annotated_frame,
            frame_timestamp=frame_timestamp,
        )

        self.repository.update_event_observed_classes(
            event_id=state.confirmed_event_id,
            observed_classes=sorted(state.observed_classes),
            confidence=float(state.latest_confidence or 0.0),
            class_name=state.class_name or "unknown",
            class_id=state.class_id,
            last_seen_frame_id=frame_id,
            frame_timestamp=frame_timestamp,
            screenshot_original_path=screenshot_paths["screenshot_original_path"],
            screenshot_annotated_path=screenshot_paths["screenshot_annotated_path"],
        )

        row = self.repository.get_event_by_id(state.confirmed_event_id)
        if row is not None:
            live_event_service.publish_from_thread(
                live_event_service.build_message(
                    message_type="event_updated",
                    event=serialize_event_row(row),
                )
            )

        logger.info(
            "Updated confirmed event | id=%s | track_id=%s | classes=%s | frame_id=%s",
            state.confirmed_event_id,
            state.track_id,
            ",".join(sorted(state.observed_classes)),
            frame_id,
        )

    def _handle_absent_detection(self, state: TrackedObjectState, now: float) -> None:
        if state.state == "NOT_PRESENT":
            return

        if state.state == "COOLDOWN":
            if state.cooldown_until is not None and now >= state.cooldown_until:
                self._reset_state(state)
            return

        state.absent_frames += 1
        if state.absent_frames < self.absent_frames:
            return

        if state.state == "CANDIDATE":
            self._reset_state(state)
            return

        if state.state == "CONFIRMED":
            state.state = "COOLDOWN"
            state.present_frames = 0
            state.absent_frames = 0
            state.cooldown_until = now + self.cooldown_seconds

    def _reset_state(self, state: TrackedObjectState) -> None:
        state.state = "NOT_PRESENT"
        state.class_name = None
        state.class_id = None
        state.track_id = None
        state.tracked = False
        state.present_frames = 0
        state.absent_frames = 0
        state.cooldown_until = None
        state.first_seen_frame_id = None
        state.last_seen_frame_id = None
        state.latest_confidence = None
        state.confirmed_event_id = None
        state.observed_classes = set()

    def _group_detections_by_state_key(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}

        for detection in detections:
            state_key = self._build_state_key(detection)
            entry = grouped.get(state_key)
            if entry is None:
                entry = {
                    "class_name": detection["class_name"],
                    "class_id": int(detection["class_id"]),
                    "track_id": (
                        int(detection["track_id"])
                        if detection.get("track_id") is not None
                        else None
                    ),
                    "confidence": float(detection["confidence"]),
                    "observed_classes": {str(detection["class_name"])},
                }
                grouped[state_key] = entry
                continue

            entry["observed_classes"].add(str(detection["class_name"]))
            if float(detection["confidence"]) > float(entry["confidence"]):
                entry["class_name"] = detection["class_name"]
                entry["class_id"] = int(detection["class_id"])
                entry["confidence"] = float(detection["confidence"])

        normalized: dict[str, dict[str, Any]] = {}
        for state_key, entry in grouped.items():
            normalized[state_key] = {
                **entry,
                "observed_classes": sorted(entry["observed_classes"]),
            }

        return normalized

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            active_states = []
            for key, state in sorted(self._states.items()):
                cooldown_until = None
                if state.cooldown_until is not None:
                    cooldown_until = datetime.fromtimestamp(
                        state.cooldown_until,
                        tz=timezone.utc,
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
        track_id = detection.get("track_id")
        if track_id is not None:
            return f"track:{int(track_id)}"
        return f"class:{str(detection['class_name'])}"
