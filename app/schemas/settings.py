from pydantic import BaseModel


class CameraSettingsResponse(BaseModel):
    source_type: str
    source: str
    default_usb_index: int
    discovery_max_index: int
    width: int
    height: int
    fps: int


class VisionSettingsResponse(BaseModel):
    enabled: bool
    model_path: str
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    inference_fps: int
    log_interval_seconds: int


class EventSettingsResponse(BaseModel):
    stable_frames: int
    absent_frames: int
    cooldown_seconds: int
    recent_limit: int


class StorageSettingsResponse(BaseModel):
    database_path: str
    screenshots_dir: str
    frame_jpeg_quality: int
    stream_boundary: str


class RelaySettingsResponse(BaseModel):
    enabled: bool
    publish_url: str
    output_variant: str
    width: int
    height: int
    fps: int
    video_bitrate_kbps: int
    h264_preset: str
    ffmpeg_bin: str


class RuntimeSettingsResponse(BaseModel):
    app_name: str
    app_version: str
    camera: CameraSettingsResponse
    vision: VisionSettingsResponse
    events: EventSettingsResponse
    storage: StorageSettingsResponse
    relay: RelaySettingsResponse
