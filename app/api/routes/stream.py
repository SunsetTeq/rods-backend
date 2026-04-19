import asyncio
import base64
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
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


@router.websocket("/ws")
async def stream_websocket(
    websocket: WebSocket,
    variant: str = Query(default="raw"),
) -> None:
    await websocket.accept()
    if variant not in {"raw", "annotated"}:
        await websocket.send_json(
            {
                "type": "error",
                "detail": "Unsupported stream variant. Use 'raw' or 'annotated'.",
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        await websocket.close(code=1008)
        return

    target_interval = 1.0 / max(settings.stream_ws_fps, 1)
    last_payload: str | None = None

    try:
        await websocket.send_json(
            {
                "type": "hello",
                "channel": "frames",
                "variant": variant,
                "fps": settings.stream_ws_fps,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        while True:
            jpeg = _get_stream_frame_bytes(variant)
            if jpeg is None:
                await asyncio.sleep(0.05)
                continue

            encoded = base64.b64encode(jpeg).decode("ascii")
            if encoded != last_payload:
                await websocket.send_json(
                    {
                        "type": "frame",
                        "channel": "frames",
                        "variant": variant,
                        "format": "image/jpeg",
                        "encoding": "base64",
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                        "data": encoded,
                    }
                )
                last_payload = encoded

            await asyncio.sleep(target_interval)
    except WebSocketDisconnect:
        return


def _get_stream_frame_bytes(variant: str) -> bytes | None:
    if variant == "annotated":
        from app.services.vision.provider import detector_service

        return detector_service.get_latest_annotated_jpeg()

    return camera_service.get_latest_jpeg()


@router.get("/view")
def get_stream_debug_page() -> FileResponse:
    return FileResponse(DEBUG_PAGE_PATH)
