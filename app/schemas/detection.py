from pydantic import BaseModel


class DetectionItemResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionFrameResponse(BaseModel):
    frame_id: int
    source_frame_size: tuple[int, int] | None
    inference_ms: float
    detections_count: int
    detections: list[DetectionItemResponse]


class VisionStatusResponse(BaseModel):
    enabled: bool
    model_path: str
    is_running: bool
    is_model_loaded: bool
    detector_available: bool
    latest_frame_id: int
    processed_frames: int
    skipped_frames: int
    actual_fps: float
    last_inference_ms: float
    last_error: str | None
