import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.core.config import settings
from app.schemas.camera import (
    CameraStatusResponse,
    CameraSwitchRequest,
    CameraSwitchResponse,
    UsbCameraInfo,
)
from app.services.capture.provider import camera_service


router = APIRouter(prefix="/api/v1/stream", tags=["stream"])
DEBUG_PAGE_PATH = Path(__file__).resolve().parents[2] / "static" / "debug" / "stream.html"


@router.get("/status", response_model=CameraStatusResponse)
def get_camera_status() -> CameraStatusResponse:
    return CameraStatusResponse(**camera_service.get_status())


@router.get("/sources/usb", response_model=list[UsbCameraInfo])
def list_usb_sources(
    max_index: int = Query(default=settings.camera_discovery_max_index, ge=0, le=20),
) -> list[UsbCameraInfo]:
    return [UsbCameraInfo(**item) for item in camera_service.list_usb_cameras(max_index)]


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


@router.get("/frame.jpg")
def get_current_frame() -> Response:
    jpeg = camera_service.get_latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Frame is not ready yet")

    return Response(content=jpeg, media_type="image/jpeg")


async def mjpeg_generator(request: Request) -> AsyncIterator[bytes]:
    boundary = settings.stream_boundary

    while True:
        if await request.is_disconnected():
            break

        jpeg = camera_service.get_latest_jpeg()
        if jpeg is None:
            await asyncio.sleep(0.05)
            continue

        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        await asyncio.sleep(0.03)


@router.get("/mjpeg")
def get_mjpeg_stream(request: Request) -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator(request),
        media_type=f"multipart/x-mixed-replace; boundary={settings.stream_boundary}",
    )


@router.get("/view")
def get_stream_debug_page() -> FileResponse:
    return FileResponse(DEBUG_PAGE_PATH)
