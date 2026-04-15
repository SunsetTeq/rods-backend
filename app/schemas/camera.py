from pydantic import BaseModel


class CameraStatusResponse(BaseModel):
    is_running: bool
    source_type: str
    source: str
    frame_width: int | None
    frame_height: int | None
    target_fps: int | None
    actual_fps: float
    frames_read: int
    read_failures: int
    last_error: str | None


class CameraSwitchRequest(BaseModel):
    source_type: str
    source: str


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


class CameraListItemResponse(BaseModel):
    source_type: str
    source: str
    label: str
    is_active: bool
    is_available: bool
    frame_width: int | None = None
    frame_height: int | None = None


class CameraListResponse(BaseModel):
    active_camera: CameraStatusResponse
    supported_source_types: list[str]
    cameras: list[CameraListItemResponse]
