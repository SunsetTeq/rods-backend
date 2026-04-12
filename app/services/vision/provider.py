from app.core.config import settings
from app.services.capture.provider import camera_service
from app.services.vision.detector_service import DetectorService


detector_service = DetectorService(
    camera_service=camera_service,
    enabled=settings.vision_enabled,
    model_path=settings.vision_model_path,
    confidence_threshold=settings.vision_confidence_threshold,
    iou_threshold=settings.vision_iou_threshold,
    max_detections=settings.vision_max_detections,
    inference_fps=settings.vision_inference_fps,
    jpeg_quality=settings.frame_jpeg_quality,
    log_interval_seconds=settings.vision_log_interval_seconds,
)
