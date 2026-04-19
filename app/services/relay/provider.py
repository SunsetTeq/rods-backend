from app.core.config import settings
from app.services.capture.provider import camera_service
from app.services.relay.publisher_service import RelayPublisherService
from app.services.vision.provider import detector_service


relay_publisher_service = RelayPublisherService(
    camera_service=camera_service,
    detector_service=detector_service,
    enabled=settings.relay_enabled,
    publish_url=settings.relay_publish_url,
    output_variant=settings.relay_output_variant,
    width=settings.relay_width,
    height=settings.relay_height,
    fps=settings.relay_fps,
    video_bitrate_kbps=settings.relay_video_bitrate_kbps,
    h264_preset=settings.relay_h264_preset,
    ffmpeg_bin=settings.relay_ffmpeg_bin,
)
