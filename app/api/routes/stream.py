import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.schemas.camera import CameraStatusResponse
from app.services.capture.provider import camera_service


router = APIRouter(prefix="/api/v1/stream", tags=["stream"])


@router.get("/status", response_model=CameraStatusResponse)
def get_camera_status() -> CameraStatusResponse:
    return CameraStatusResponse(**camera_service.get_status())


@router.get("/frame.jpg")
def get_current_frame() -> Response:
    jpeg = camera_service.get_latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Frame is not ready yet")

    return Response(content=jpeg, media_type="image/jpeg")


async def mjpeg_generator(request: Request) -> AsyncIterator[bytes]:
    boundary = "frame"

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
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
