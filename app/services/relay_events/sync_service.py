import json
import logging
import queue
import ssl
import threading
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.db.repository import EventRepository
from app.services.storage.screenshot_service import ScreenshotService


logger = logging.getLogger(__name__)


class RelayEventSyncService:
    def __init__(
        self,
        repository: EventRepository,
        screenshot_service: ScreenshotService,
        enabled: bool,
        api_url: str,
        auth_token: str,
        source_id: str,
        timeout_seconds: float,
        ssl_context: ssl.SSLContext,
        sync_batch_size: int = 200,
    ) -> None:
        self.repository = repository
        self.screenshot_service = screenshot_service
        self.enabled = enabled
        self.api_url = api_url.strip().rstrip("/")
        self.auth_token = auth_token.strip()
        self.source_id = source_id.strip() or "rods-backend"
        self.timeout_seconds = timeout_seconds
        self.ssl_context = ssl_context
        self.sync_batch_size = sync_batch_size

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue[tuple[int, int]] = queue.Queue()
        self._pending_ids: set[int] = set()
        self._lock = threading.Lock()
        self._is_running = False
        self._synced_events = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if not self.enabled:
            logger.info("Relay event sync service is disabled by config")
            return

        if not self.api_url:
            self._last_error = "Relay event sync is enabled but RELAY_EVENTS_API_URL is empty"
            logger.warning(self._last_error)
            return

        if self._is_running:
            logger.info("Relay event sync service already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info(
            "Relay event sync service started | url=%s | source_id=%s",
            self.api_url,
            self.source_id,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._is_running = False
        logger.info("Relay event sync service stopped")

    def enqueue_event(self, event_id: int) -> None:
        if not self.enabled or event_id <= 0:
            return

        with self._lock:
            if event_id in self._pending_ids:
                return
            self._pending_ids.add(event_id)

        self._queue.put((event_id, 0))

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            pending = len(self._pending_ids)

        return {
            "enabled": self.enabled,
            "is_running": self._is_running,
            "api_url": self.api_url,
            "source_id": self.source_id,
            "pending_events": pending,
            "synced_events": self._synced_events,
            "last_error": self._last_error,
        }

    def _worker_loop(self) -> None:
        self._enqueue_existing_events()

        while not self._stop_event.is_set():
            try:
                event_id, attempt = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                synced = self._sync_event_by_id(event_id)
                if synced:
                    self._synced_events += 1
                    self._last_error = None
                    with self._lock:
                        self._pending_ids.discard(event_id)
                else:
                    with self._lock:
                        self._pending_ids.discard(event_id)
            except Exception as exc:
                self._last_error = str(exc)
                delay_seconds = min(30, 2 ** min(attempt, 4))
                logger.exception(
                    "Relay event sync failed | event_id=%s | retry_in=%ss",
                    event_id,
                    delay_seconds,
                )
                if self._stop_event.wait(delay_seconds):
                    break
                self._queue.put((event_id, attempt + 1))

    def _enqueue_existing_events(self) -> None:
        logger.info("Relay event sync backfill started")
        after_id = 0

        while not self._stop_event.is_set():
            rows, _ = self.repository.list_events_page(
                limit=self.sync_batch_size,
                after_id=after_id if after_id > 0 else None,
            )
            if not rows:
                break

            for row in rows:
                event_id = int(row["id"])
                self.enqueue_event(event_id)
                after_id = event_id

            if len(rows) < self.sync_batch_size:
                break

        logger.info("Relay event sync backfill queued")

    def _sync_event_by_id(self, event_id: int) -> bool:
        row = self.repository.get_event_by_id(event_id)
        if row is None:
            logger.warning("Relay event sync skipped missing event | event_id=%s", event_id)
            return False

        endpoint = f"{self.api_url}/api/v1/internal/events"
        payload = {
            "source_id": self.source_id,
            "event": self._build_event_payload(row),
        }
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        relay_event = self._request_json(
            method="POST",
            url=endpoint,
            json_body=payload,
            headers=headers or None,
        )

        annotated_path = self._resolve_optional_screenshot_path(row.get("screenshot_annotated_path"))
        if annotated_path is not None and annotated_path.exists():
            upload_headers = dict(headers)
            upload_headers["Content-Type"] = _guess_media_type(annotated_path)
            upload_headers["X-Frame-Timestamp"] = str(row["frame_timestamp"])
            self._request_json(
                method="PUT",
                url=f"{endpoint}/{relay_event['id']}/screenshots/annotated",
                body=annotated_path.read_bytes(),
                headers=upload_headers,
            )

        logger.info("Relay event synced | event_id=%s", event_id)
        return True

    def _resolve_optional_screenshot_path(self, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        return self.screenshot_service.get_absolute_path(relative_path)

    def _build_event_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "event_type": row["event_type"],
            "class_name": row["class_name"],
            "class_id": row.get("class_id"),
            "track_id": row.get("track_id"),
            "confidence": float(row["confidence"]),
            "state_key": row["state_key"],
            "first_seen_frame_id": int(row["first_seen_frame_id"]),
            "confirmed_frame_id": int(row["confirmed_frame_id"]),
            "last_seen_frame_id": int(row["last_seen_frame_id"]),
            "stable_frames_required": int(row["stable_frames_required"]),
            "absent_frames_required": int(row["absent_frames_required"]),
            "cooldown_seconds": int(row["cooldown_seconds"]),
            "source_frame_width": row.get("source_frame_width"),
            "source_frame_height": row.get("source_frame_height"),
            "frame_timestamp": row["frame_timestamp"],
            "created_at": row["created_at"],
            "updated_at": row.get("updated_at"),
        }

    def _request_json(
        self,
        method: str,
        url: str,
        json_body: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request_headers = dict(headers or {})
        request_body = body

        if json_body is not None:
            request_body = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        request = urllib_request.Request(
            url=url,
            data=request_body,
            headers=request_headers,
            method=method,
        )

        try:
            with urllib_request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self.ssl_context,
            ) as response:
                payload = response.read()
                status_code = int(response.status)
        except urllib_error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"Relay event sync request failed with status {exc.code}: {payload}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Relay event sync request failed: {exc}") from exc

        if status_code < 200 or status_code >= 300:
            raise RuntimeError(
                f"Relay event sync request returned unexpected status {status_code}"
            )

        if not payload:
            return {}

        return json.loads(payload.decode("utf-8"))


def _guess_media_type(path: Path) -> str:
    if path.suffix.lower() == ".png":
        return "image/png"
    return "image/jpeg"
