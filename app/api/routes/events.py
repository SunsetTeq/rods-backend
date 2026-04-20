from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.event import (
    EventEngineStatusResponse,
    EventListResponse,
    EventPaginationResponse,
    EventResponse,
)
from app.services.events.serialization import serialize_event_row
from app.services.events.provider import event_engine_service, event_repository
from app.services.storage.provider import screenshot_service


router = APIRouter(prefix="/api/v1/events", tags=["events"])

def _serialize_event(row: dict) -> EventResponse:
    return EventResponse(**serialize_event_row(row))


@router.get("/status", response_model=EventEngineStatusResponse)
def get_event_engine_status() -> EventEngineStatusResponse:
    return EventEngineStatusResponse(**event_engine_service.get_status())


@router.get("/recent", response_model=list[EventResponse])
def list_recent_events(
    limit: int = Query(default=settings.event_recent_limit, ge=1, le=500),
) -> list[EventResponse]:
    rows = event_repository.list_recent_events(limit=limit)
    return [_serialize_event(row) for row in rows]


@router.get("", response_model=EventListResponse)
def list_events(
    limit: int = Query(default=settings.event_recent_limit, ge=1, le=500),
    before_id: int | None = Query(default=None, ge=1),
    after_id: int | None = Query(default=None, ge=1),
) -> EventListResponse:
    if before_id is not None and after_id is not None:
        raise HTTPException(
            status_code=422,
            detail="Use either before_id or after_id, not both at the same time",
        )

    rows, has_more = event_repository.list_events_page(
        limit=limit,
        before_id=before_id,
        after_id=after_id,
    )
    items = [_serialize_event(row) for row in rows]
    item_ids = [item.id for item in items]
    order = "asc" if after_id is not None else "desc"

    return EventListResponse(
        items=items,
        pagination=EventPaginationResponse(
            limit=limit,
            before_id=before_id,
            after_id=after_id,
            order=order,
            count=len(items),
            has_more=has_more,
            next_before_id=(
                min(item_ids)
                if has_more and item_ids and order == "desc"
                else None
            ),
            next_after_id=(
                max(item_ids)
                if has_more and item_ids and order == "asc"
                else None
            ),
            oldest_id=min(item_ids) if item_ids else None,
            newest_id=max(item_ids) if item_ids else None,
        ),
    )


@router.get("/{event_id}", response_model=EventResponse)
def get_event(event_id: int) -> EventResponse:
    row = event_repository.get_event_by_id(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return _serialize_event(row)


@router.get("/{event_id}/screenshots/{variant}")
def get_event_screenshot(event_id: int, variant: str) -> FileResponse:
    row = event_repository.get_event_by_id(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")

    if variant == "original":
        relative_path = row.get("screenshot_original_path")
    elif variant == "annotated":
        relative_path = row.get("screenshot_annotated_path")
    else:
        raise HTTPException(status_code=404, detail="Unknown screenshot variant")

    if not relative_path:
        raise HTTPException(status_code=404, detail="Screenshot is not available for this event")

    absolute_path = screenshot_service.get_absolute_path(relative_path)
    if not absolute_path.exists() or not absolute_path.is_file():
        raise HTTPException(status_code=404, detail="Screenshot file not found")

    media_type = "image/jpeg"
    if absolute_path.suffix.lower() == ".png":
        media_type = "image/png"

    return FileResponse(Path(absolute_path), media_type=media_type)
