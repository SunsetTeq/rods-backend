import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger(__name__)


class LiveEventService:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop
        logger.info("Live event service started")

    def stop(self) -> None:
        with self._lock:
            self._queues.clear()
            self._loop = None
        logger.info("Live event service stopped")

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._queues.discard(queue)

    async def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            queues = list(self._queues)

        for queue in queues:
            await queue.put(payload)

    def publish_from_thread(self, payload: dict[str, Any]) -> None:
        with self._lock:
            loop = self._loop

        if loop is None:
            return

        asyncio.run_coroutine_threadsafe(self.publish(payload), loop)

    def build_message(
        self,
        message_type: str,
        event: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": message_type,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **(extra or {}),
        }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            subscribers = len(self._queues)
            is_running = self._loop is not None

        return {
            "is_running": is_running,
            "subscribers": subscribers,
        }
