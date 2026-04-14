from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RODS Backend"
    app_version: str = "0.1.0"

    camera_source_type: str = "usb"   # usb | rtsp | file
    camera_source: str = "0"          # для usb: индекс камеры в виде строки
    camera_default_usb_index: int = 0
    camera_discovery_max_index: int = 5
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    frame_jpeg_quality: int = 85
    stream_boundary: str = "frame"

    vision_enabled: bool = True
    vision_model_path: str = "yolov8n.pt"
    vision_confidence_threshold: float = 0.5
    vision_iou_threshold: float = 0.45
    vision_max_detections: int = 100
    vision_inference_fps: int = 5
    vision_log_interval_seconds: int = 5

    database_path: str = "data/rods.db"
    screenshots_dir: str = "data/screenshots"
    event_stable_frames: int = 8
    event_absent_frames: int = 12
    event_cooldown_seconds: int = 30
    event_recent_limit: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
