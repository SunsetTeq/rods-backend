from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RODS Backend"
    app_version: str = "0.1.0"

    camera_source_type: str = "usb"   # usb | rtsp | file
    camera_source: str = "0"          # для usb: индекс камеры в виде строки
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    frame_jpeg_quality: int = 85
    stream_boundary: str = "frame"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
