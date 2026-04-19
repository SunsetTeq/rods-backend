from fastapi import APIRouter

from app.core.config import settings
from app.schemas.settings import (
    CameraSettingsResponse,
    EventSettingsResponse,
    RelaySettingsResponse,
    RuntimeSettingsResponse,
    StorageSettingsResponse,
    VisionSettingsResponse,
)


router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@router.get("", response_model=RuntimeSettingsResponse)
def get_runtime_settings() -> RuntimeSettingsResponse:
    return RuntimeSettingsResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        camera=CameraSettingsResponse(
            source_type=settings.camera_source_type,
            source=str(settings.camera_source),
            default_usb_index=settings.camera_default_usb_index,
            discovery_max_index=settings.camera_discovery_max_index,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
        ),
        vision=VisionSettingsResponse(
            enabled=settings.vision_enabled,
            model_path=settings.vision_model_path,
            confidence_threshold=settings.vision_confidence_threshold,
            iou_threshold=settings.vision_iou_threshold,
            max_detections=settings.vision_max_detections,
            inference_fps=settings.vision_inference_fps,
            log_interval_seconds=settings.vision_log_interval_seconds,
        ),
        events=EventSettingsResponse(
            stable_frames=settings.event_stable_frames,
            absent_frames=settings.event_absent_frames,
            cooldown_seconds=settings.event_cooldown_seconds,
            recent_limit=settings.event_recent_limit,
        ),
        storage=StorageSettingsResponse(
            database_path=settings.database_path,
            screenshots_dir=settings.screenshots_dir,
            frame_jpeg_quality=settings.frame_jpeg_quality,
            stream_boundary=settings.stream_boundary,
        ),
        relay=RelaySettingsResponse(
            enabled=settings.relay_enabled,
            publish_url=settings.relay_publish_url,
            output_variant=settings.relay_output_variant,
            width=settings.relay_width,
            height=settings.relay_height,
            fps=settings.relay_fps,
            video_bitrate_kbps=settings.relay_video_bitrate_kbps,
            h264_preset=settings.relay_h264_preset,
            ffmpeg_bin=settings.relay_ffmpeg_bin,
        ),
    )
