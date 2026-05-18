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
    tracking_enabled=settings.vision_tracking_enabled,
    tracking_persist=settings.vision_tracking_persist,
    tracker_config=settings.vision_tracker_config,
    focus_enabled=settings.vision_focus_enabled,
    focus_motion_threshold=settings.vision_focus_motion_threshold,
    focus_min_motion_area=settings.vision_focus_min_motion_area,
    focus_padding=settings.vision_focus_padding,
    focus_hold_frames=settings.vision_focus_hold_frames,
    focus_merge_gap=settings.vision_focus_merge_gap,
    focus_max_rois=settings.vision_focus_max_rois,
    focus_full_frame_area_threshold=settings.vision_focus_full_frame_area_threshold,
    focus_full_frame_refresh_interval=settings.vision_focus_full_frame_refresh_interval,
    focus_tracking_iou_threshold=settings.vision_focus_tracking_iou_threshold,
)
