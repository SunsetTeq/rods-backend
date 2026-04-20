from pydantic import BaseModel


class CameraStatusResponse(BaseModel):
    is_running: bool
    source_type: str
    source: str
    active_camera_id: str | None = None
    frame_width: int | None
    frame_height: int | None
    target_fps: int | None
    actual_fps: float
    frames_read: int
    read_failures: int
    last_error: str | None


class CameraSwitchRequest(BaseModel):
    camera_id: str | None = None
    source_type: str | None = None
    source: str | None = None


class CameraSwitchResponse(BaseModel):
    ok: bool
    error: str | None
    status: CameraStatusResponse


class UsbCameraInfo(BaseModel):
    index: int
    available: bool
    width: int | None
    height: int | None
    label: str
    name: str | None = None


class CameraListItemResponse(BaseModel):
    camera_id: str
    source_type: str
    source: str
    label: str
    name: str | None = None
    is_active: bool
    is_available: bool
    frame_width: int | None = None
    frame_height: int | None = None


class CameraListResponse(BaseModel):
    active_camera: CameraStatusResponse
    active_camera_id: str | None = None
    supported_source_types: list[str]
    cameras: list[CameraListItemResponse]


class StreamAvailabilityResponse(BaseModel):
    stream_available: bool
    has_frame: bool
    frame_age_ms: float | None
    stale_after_ms: int
    active_camera_id: str | None = None
    active_camera: CameraStatusResponse
    last_error: str | None
