from pydantic import BaseModel


class RelayStatusResponse(BaseModel):
    enabled: bool
    is_running: bool
    publish_url: str
    output_variant: str
    width: int
    height: int
    fps: int
    video_bitrate_kbps: int
    ffmpeg_bin: str
    ffmpeg_available: bool
    frames_published: int
    actual_fps: float
    last_error: str | None
