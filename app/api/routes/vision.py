import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.config import settings
from app.schemas.detection import (
    CurrentObjectItemResponse,
    CurrentObjectsResponse,
    DetectionFrameResponse,
    VisionStatusResponse,
)
from app.services.vision.provider import detector_service


router = APIRouter(prefix="/api/v1/vision", tags=["vision"])


@router.get("/status", response_model=VisionStatusResponse)
def get_detector_status() -> VisionStatusResponse:
    return VisionStatusResponse(**detector_service.get_status())


@router.get("/detections/latest", response_model=DetectionFrameResponse)
def get_latest_detections() -> DetectionFrameResponse:
    payload = detector_service.get_latest_detections()
    return DetectionFrameResponse(**payload)


@router.get("/objects/current", response_model=CurrentObjectsResponse)
def get_current_objects() -> CurrentObjectsResponse:
    payload = detector_service.get_latest_detections()
    objects = [
        CurrentObjectItemResponse(
            id=str(item["track_id"]) if item.get("track_id") is not None else f"det-{index + 1}",
            track_id=item.get("track_id"),
            class_id=int(item["class_id"]),
            class_name=str(item["class_name"]),
            confidence=float(item["confidence"]),
        )
        for index, item in enumerate(payload["detections"])
    ]
    return CurrentObjectsResponse(
        frame_id=int(payload["frame_id"]),
        frame_timestamp=payload.get("frame_timestamp"),
        objects_count=len(objects),
        objects=objects,
    )


@router.get("/frame.jpg")
def get_annotated_frame() -> Response:
    jpeg = detector_service.get_live_annotated_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Annotated frame is not ready yet")

    return Response(content=jpeg, media_type="image/jpeg")


async def annotated_mjpeg_generator(request: Request) -> AsyncIterator[bytes]:
    boundary = settings.stream_boundary

    while True:
        if await request.is_disconnected():
            break

        jpeg = detector_service.get_live_annotated_jpeg()
        if jpeg is None:
            await asyncio.sleep(0.05)
            continue

        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        await asyncio.sleep(0.03)


@router.get("/mjpeg")
def get_annotated_mjpeg_stream(request: Request) -> StreamingResponse:
    return StreamingResponse(
        annotated_mjpeg_generator(request),
        media_type=f"multipart/x-mixed-replace; boundary={settings.stream_boundary}",
    )
