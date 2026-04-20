import json
import logging
import ssl
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.services.vision.detector_service import DetectorService


logger = logging.getLogger(__name__)


class RelayVisionSyncService:
    def __init__(
        self,
        detector_service: DetectorService,
        enabled: bool,
        api_url: str,
        auth_token: str,
        source_id: str,
        timeout_seconds: float,
        poll_interval_seconds: float,
        ssl_context: ssl.SSLContext,
    ) -> None:
        self.detector_service = detector_service
        self.enabled = enabled
        self.api_url = api_url.strip().rstrip('/')
        self.auth_token = auth_token.strip()
        self.source_id = source_id.strip() or 'rods-backend'
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = max(poll_interval_seconds, 0.2)
        self.ssl_context = ssl_context

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._last_error: str | None = None
        self._last_frame_id = 0
        self._synced_frames = 0

    def start(self) -> None:
        if not self.enabled:
            logger.info('Relay vision sync service is disabled by config')
            return

        if not self.api_url:
            self._last_error = 'Relay vision sync is enabled but RELAY_EVENTS_API_URL is empty'
            logger.warning(self._last_error)
            return

        if self._is_running:
            logger.info('Relay vision sync service already running')
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info(
            'Relay vision sync service started | url=%s | source_id=%s',
            self.api_url,
            self.source_id,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._is_running = False
        logger.info('Relay vision sync service stopped')

    def get_status(self) -> dict[str, Any]:
        return {
            'enabled': self.enabled,
            'is_running': self._is_running,
            'api_url': self.api_url,
            'source_id': self.source_id,
            'last_frame_id': self._last_frame_id,
            'synced_frames': self._synced_frames,
            'last_error': self._last_error,
        }

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self.detector_service.get_latest_detections()
                frame_id = int(payload['frame_id'])

                if frame_id > 0 and frame_id != self._last_frame_id:
                    self._sync_detection_frame(payload)
                    self._last_frame_id = frame_id
                    self._synced_frames += 1
                    self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception('Relay vision sync failed')

            if self._stop_event.wait(self.poll_interval_seconds):
                break

    def _sync_detection_frame(self, payload: dict[str, Any]) -> None:
        self._request_json(
            method='PUT',
            path='/api/v1/internal/vision/detections/latest',
            json_body={
                'source_id': self.source_id,
                **payload,
            },
        )

    def _request_json(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any],
    ) -> dict[str, Any]:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'

        request = urllib_request.Request(
            url=f'{self.api_url}{path}',
            data=json.dumps(json_body).encode('utf-8'),
            headers=headers,
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
            payload = exc.read().decode('utf-8', errors='ignore').strip()
            raise RuntimeError(
                f'Relay vision sync request failed with status {exc.code}: {payload}'
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f'Relay vision sync request failed: {exc}') from exc

        if status_code != 200:
            body_text = payload.decode('utf-8', errors='ignore').strip()
            raise RuntimeError(
                f'Relay vision sync returned unexpected status {status_code}: {body_text}'
            )

        return json.loads(payload.decode('utf-8')) if payload else {}
