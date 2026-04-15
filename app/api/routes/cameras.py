from fastapi import APIRouter

from app.schemas.camera import (
    CameraListItemResponse,
    CameraListResponse,
    CameraStatusResponse,
    CameraSwitchRequest,
    CameraSwitchResponse,
)
from app.services.capture.provider import camera_service


router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])


@router.get("", response_model=CameraListResponse)
def list_cameras() -> CameraListResponse:
    status = camera_service.get_status()
    active_source_type = status["source_type"]
    active_source = str(status["source"])
    usb_cameras = camera_service.list_usb_cameras()

    cameras: list[CameraListItemResponse] = [
        CameraListItemResponse(
            source_type="usb",
            source=str(item["index"]),
            label=item["label"],
            is_active=active_source_type == "usb" and active_source == str(item["index"]),
            is_available=bool(item["available"]),
            frame_width=item["width"],
            frame_height=item["height"],
        )
        for item in usb_cameras
    ]

    if active_source_type in {"rtsp", "file"}:
        cameras.append(
            CameraListItemResponse(
                source_type=active_source_type,
                source=active_source,
                label=f"{active_source_type.upper()} source",
                is_active=True,
                is_available=True,
                frame_width=status["frame_width"],
                frame_height=status["frame_height"],
            )
        )

    return CameraListResponse(
        active_camera=CameraStatusResponse(**status),
        supported_source_types=["usb", "rtsp", "file"],
        cameras=cameras,
    )


@router.post("/select", response_model=CameraSwitchResponse)
def select_camera(payload: CameraSwitchRequest) -> CameraSwitchResponse:
    result = camera_service.switch_source(
        source_type=payload.source_type,
        source=payload.source,
    )
    return CameraSwitchResponse(
        ok=result["ok"],
        error=result["error"],
        status=CameraStatusResponse(**result["status"]),
    )
