from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.cameras import router as cameras_router
from app.api.routes.health import router as health_router
from app.api.routes.events import router as events_router
from app.api.routes.live import router as live_router
from app.api.routes.relay import router as relay_router
from app.api.routes.settings import router as settings_router
from app.api.routes.stream import router as stream_router
from app.api.routes.vision import router as vision_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.capture.provider import camera_service
from app.services.events.provider import event_engine_service
from app.services.live_event_provider import live_event_service
from app.services.relay_control.provider import relay_camera_control_service
from app.services.relay.provider import relay_publisher_service
from app.services.relay_events.provider import relay_event_sync_service
from app.services.vision.provider import detector_service


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    live_event_service.start(asyncio.get_running_loop())
    relay_event_sync_service.start()
    try:
        camera_service.start()
    except Exception:
        logging.getLogger(__name__).exception(
            "Camera service failed to start during application startup"
        )
    relay_camera_control_service.start()
    detector_service.start()
    event_engine_service.start()
    relay_publisher_service.start()
    yield
    relay_publisher_service.stop()
    event_engine_service.stop()
    detector_service.stop()
    relay_camera_control_service.stop()
    camera_service.stop()
    relay_event_sync_service.stop()
    live_event_service.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(health_router)
app.include_router(cameras_router)
app.include_router(events_router)
app.include_router(live_router)
app.include_router(relay_router)
app.include_router(settings_router)
app.include_router(stream_router)
app.include_router(vision_router)
