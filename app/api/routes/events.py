from fastapi import APIRouter, Query

from app.core.config import settings
from app.schemas.event import EventEngineStatusResponse, EventResponse
from app.services.events.provider import event_engine_service, event_repository


router = APIRouter(prefix="/api/v1/events", tags=["events"])


@router.get("/status", response_model=EventEngineStatusResponse)
def get_event_engine_status() -> EventEngineStatusResponse:
    return EventEngineStatusResponse(**event_engine_service.get_status())


@router.get("/recent", response_model=list[EventResponse])
def list_recent_events(
    limit: int = Query(default=settings.event_recent_limit, ge=1, le=500),
) -> list[EventResponse]:
    rows = event_repository.list_recent_events(limit=limit)
    return [EventResponse(**row) for row in rows]
