from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.camera import (
    CameraListResponse,
    CameraStatusResponse,
    CameraSwitchRequest,
    CameraSwitchResponse,
)
from app.services.capture.camera_service import build_camera_id, parse_camera_id
from app.services.capture.provider import camera_service


router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])


@router.get("", response_model=CameraListResponse)
def list_cameras() -> CameraListResponse:
    status = camera_service.get_status()
    cameras = camera_service.list_available_camera_sources(settings.camera_discovery_max_index)

    return CameraListResponse(
        active_camera=CameraStatusResponse(**status),
        active_camera_id=build_camera_id(status["source_type"], str(status["source"])),
        supported_source_types=["usb", "rtsp", "file"],
        cameras=cameras,
    )


@router.get("/available", response_model=CameraListResponse)
def list_available_cameras() -> CameraListResponse:
    return list_cameras()


@router.post("/select", response_model=CameraSwitchResponse)
@router.post("/activate", response_model=CameraSwitchResponse)
def select_camera(payload: CameraSwitchRequest) -> CameraSwitchResponse:
    source_type, source = _resolve_camera_selection(payload)
    result = camera_service.switch_source(
        source_type=source_type,
        source=source,
    )
    return CameraSwitchResponse(
        ok=result["ok"],
        error=result["error"],
        status=CameraStatusResponse(**result["status"]),
    )


def _resolve_camera_selection(payload: CameraSwitchRequest) -> tuple[str, str]:
    if payload.camera_id:
        try:
            return parse_camera_id(payload.camera_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if payload.source_type and payload.source:
        return payload.source_type, payload.source

    raise HTTPException(
        status_code=422,
        detail="Provide either camera_id or both source_type and source",
    )
