from app.core.config import settings
from app.services.capture.camera_service import CameraService


camera_service = CameraService(
    source_type=settings.camera_source_type,
    source=str(settings.camera_source),
    capture_backend=settings.camera_capture_backend,
    width=settings.camera_width,
    height=settings.camera_height,
    fps=settings.camera_fps,
    jpeg_quality=settings.frame_jpeg_quality,
)
