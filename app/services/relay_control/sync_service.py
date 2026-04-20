import json
import logging
import ssl
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from app.services.capture.camera_service import CameraService


logger = logging.getLogger(__name__)


class RelayCameraControlService:
    def __init__(
        self,
        camera_service: CameraService,
        enabled: bool,
        api_url: str,
        auth_token: str,
        source_id: str,
        timeout_seconds: float,
        poll_interval_seconds: float,
        state_sync_interval_seconds: float,
        camera_discovery_max_index: int,
        ssl_context: ssl.SSLContext,
    ) -> None:
        self.camera_service = camera_service
        self.enabled = enabled
        self.api_url = api_url.strip().rstrip("/")
        self.auth_token = auth_token.strip()
        self.source_id = source_id.strip() or "rods-backend"
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = max(poll_interval_seconds, 0.5)
        self.state_sync_interval_seconds = max(state_sync_interval_seconds, 1.0)
        self.camera_discovery_max_index = camera_discovery_max_index
        self.ssl_context = ssl_context

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._last_error: str | None = None
        self._last_state_signature: str | None = None
        self._state_syncs_total = 0
        self._commands_processed = 0

    def start(self) -> None:
        if not self.enabled:
            logger.info("Relay camera control service is disabled by config")
            return

        if not self.api_url:
            self._last_error = "Relay camera control is enabled but no relay API URL is configured"
            logger.warning(self._last_error)
            return

        if self._is_running:
            logger.info("Relay camera control service already running")
            return

        try:
            self.camera_service.refresh_available_camera_source_cache(
                max_index=self.camera_discovery_max_index,
            )
        except Exception:
            logger.exception("Initial relay camera discovery failed")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info(
            "Relay camera control service started | url=%s | source_id=%s",
            self.api_url,
            self.source_id,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._is_running = False
        logger.info("Relay camera control service stopped")

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "is_running": self._is_running,
            "api_url": self.api_url,
            "source_id": self.source_id,
            "state_syncs_total": self._state_syncs_total,
            "commands_processed": self._commands_processed,
            "last_error": self._last_error,
        }

    def _worker_loop(self) -> None:
        next_state_sync_at = 0.0

        while not self._stop_event.is_set():
            try:
                now = time.monotonic()
                if now >= next_state_sync_at:
                    self._sync_state(force=False)
                    next_state_sync_at = now + self.state_sync_interval_seconds

                command = self._fetch_next_command()
                if command is not None:
                    self._process_command(command=command)
                    next_state_sync_at = 0.0
                    self._commands_processed += 1
                    self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("Relay camera control loop failed")
                if self._stop_event.wait(self.poll_interval_seconds):
                    break
                continue

            if self._stop_event.wait(self.poll_interval_seconds):
                break

    def _sync_state(self, force: bool) -> None:
        payload = self._build_state_payload()
        signature = repr(payload)
        if not force and signature == self._last_state_signature:
            return

        self._request(
            method="PUT",
            path="/api/v1/internal/cameras/state",
            json_body=payload,
            expected_statuses={200},
        )
        self._last_state_signature = signature
        self._state_syncs_total += 1

    def _fetch_next_command(self) -> dict[str, Any] | None:
        query = urllib_parse.urlencode({"source_id": self.source_id})
        status_code, payload = self._request(
            method="GET",
            path=f"/api/v1/internal/cameras/commands/next?{query}",
            expected_statuses={200, 204},
        )
        if status_code == 204:
            return None
        return payload

    def _process_command(self, command: dict[str, Any]) -> None:
        payload = command.get("payload") or {}
        command_id = int(command["id"])
        command_type = str(command.get("command_type") or "")

        ok = False
        error: str | None = None

        try:
            if command_type != "select_camera":
                raise ValueError(f"Unsupported command type: {command_type}")

            camera_id = payload.get("camera_id")
            if not camera_id:
                raise ValueError("Camera command payload is missing camera_id")

            source_type, source = _parse_camera_id(str(camera_id))
            result = self.camera_service.switch_source(
                source_type=source_type,
                source=source,
            )
            ok = bool(result["ok"])
            error = result["error"]
            if ok:
                self.camera_service.refresh_available_camera_source_cache(
                    max_index=self.camera_discovery_max_index,
                )
        except Exception as exc:
            error = str(exc)

        ack_payload = {
            "ok": ok,
            "error": error,
            "state": self._build_state_payload(),
        }
        self._request(
            method="POST",
            path=f"/api/v1/internal/cameras/commands/{command_id}/complete",
            json_body=ack_payload,
            expected_statuses={200},
        )

        self._last_state_signature = None

    def _build_state_payload(self) -> dict[str, Any]:
        status = self.camera_service.get_status()
        return {
            "source_id": self.source_id,
            "active_camera_id": status["active_camera_id"],
            "active_camera": status,
            "cameras": self.camera_service.get_cached_available_camera_sources(),
        }

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        expected_statuses: set[int] | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        expected = expected_statuses or {200}
        headers = {"Accept": "application/json"}
        data = None

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib_request.Request(
            url=f"{self.api_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib_request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self.ssl_context,
            ) as response:
                status_code = int(response.status)
                raw_body = response.read()
        except urllib_error.HTTPError as exc:
            status_code = int(exc.code)
            raw_body = exc.read()
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Relay control request failed: {exc}") from exc

        if status_code not in expected:
            body_text = raw_body.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"Relay control request returned unexpected status {status_code}: {body_text}"
            )

        if not raw_body:
            return status_code, None

        return status_code, json.loads(raw_body.decode("utf-8"))


def _parse_camera_id(camera_id: str) -> tuple[str, str]:
    source_type, separator, source = camera_id.partition(":")
    if not separator or not source_type or not source:
        raise ValueError("camera_id must be in format '<source_type>:<source>'")
    return source_type, source
