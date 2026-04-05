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