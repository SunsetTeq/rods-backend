from app.core.config import settings
from app.services.storage.screenshot_service import ScreenshotService


screenshot_service = ScreenshotService(
    base_dir=settings.screenshots_dir,
    jpeg_quality=settings.frame_jpeg_quality,
)
