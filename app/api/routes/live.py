import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.schemas.event import LiveEventStatusResponse
from app.services.live_event_provider import live_event_service


router = APIRouter(prefix="/api/v1/live", tags=["live"])


@router.get("/status", response_model=LiveEventStatusResponse)
def get_live_status() -> LiveEventStatusResponse:
    return LiveEventStatusResponse(**live_event_service.get_status())


@router.websocket("/ws")
async def live_events_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = live_event_service.subscribe()

    await websocket.send_json(
        {
            "type": "hello",
            "channel": "events",
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    try:
        while True:
            try:
                payload = await asyncio.wait_for(
                    queue.get(),
                    timeout=settings.live_ping_interval_seconds,
                )
                await websocket.send_json(payload)
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {
                        "type": "ping",
                        "channel": "events",
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
    except WebSocketDisconnect:
        pass
    finally:
        live_event_service.unsubscribe(queue)
