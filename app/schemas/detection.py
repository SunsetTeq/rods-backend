from pydantic import BaseModel


class DetectionItemResponse(BaseModel):
    class_id: int
    class_name: str
    class_name_en: str | None = None
    class_name_ru: str | None = None
    track_id: int | None = None
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionFrameResponse(BaseModel):
    frame_id: int
    source_frame_size: tuple[int, int] | None
    frame_timestamp: str | None = None
    inference_ms: float
    detections_count: int
    detections: list[DetectionItemResponse]


class CurrentObjectItemResponse(BaseModel):
    id: str
    track_id: int | None = None
    class_id: int
    class_name: str
    confidence: float


class CurrentObjectsResponse(BaseModel):
    frame_id: int
    frame_timestamp: str | None = None
    objects_count: int
    objects: list[CurrentObjectItemResponse]


class VisionStatusResponse(BaseModel):
    enabled: bool
    model_path: str
    tracking_enabled: bool
    tracking_persist: bool
    tracker_config: str
    tracking_runtime_enabled: bool
    is_running: bool
    is_model_loaded: bool
    detector_available: bool
    latest_frame_id: int
    processed_frames: int
    skipped_frames: int
    tracked_detections_count: int
    actual_fps: float
    last_inference_ms: float
    live_annotations_supported: bool
    last_error: str | None
