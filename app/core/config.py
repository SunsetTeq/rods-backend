from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RODS Backend"
    app_version: str = "0.1.0"

    camera_source_type: str = "usb"   # usb | rtsp | file
    camera_source: str = "0"          # для usb: индекс камеры в виде строки
    camera_capture_backend: str = "auto"
    camera_default_usb_index: int = 0
    camera_discovery_max_index: int = 5
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    frame_jpeg_quality: int = 85
    stream_boundary: str = "frame"
    stream_ws_fps: int = 5

    vision_enabled: bool = True
    vision_model_path: str = "yolov8n.pt"
    vision_confidence_threshold: float = 0.5
    vision_iou_threshold: float = 0.45
    vision_max_detections: int = 100
    vision_inference_fps: int = 5
    vision_log_interval_seconds: int = 5
    vision_tracking_enabled: bool = True
    vision_tracking_persist: bool = True
    vision_tracker_config: str = "bytetrack.yaml"

    relay_enabled: bool = False
    relay_publish_url: str = ""
    relay_output_variant: str = "annotated"  # annotated | raw
    relay_width: int = 1280
    relay_height: int = 720
    relay_fps: int = 30
    relay_video_bitrate_kbps: int = 2500
    relay_h264_preset: str = "veryfast"
    relay_ffmpeg_bin: str = "ffmpeg"
    relay_events_enabled: bool = False
    relay_events_api_url: str = ""
    relay_events_token: str = ""
    relay_source_id: str = "rods-backend"
    relay_events_timeout_seconds: float = 10.0
    relay_control_enabled: bool = True
    relay_control_api_url: str = ""
    relay_control_token: str = ""
    relay_control_poll_interval_seconds: float = 2.0
    relay_control_state_sync_interval_seconds: float = 5.0
    relay_ssl_verify: bool = True
    relay_ssl_ca_file: str = ""

    database_path: str = "data/rods.db"
    screenshots_dir: str = "data/screenshots"
    event_stable_frames: int = 8
    event_absent_frames: int = 12
    event_cooldown_seconds: int = 30
    event_recent_limit: int = 100
    live_ping_interval_seconds: int = 15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
