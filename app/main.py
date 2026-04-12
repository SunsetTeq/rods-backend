from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.routes.stream import router as stream_router
from app.api.routes.vision import router as vision_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.capture.provider import camera_service
from app.services.vision.provider import detector_service


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        camera_service.start()
    except Exception:
        logging.getLogger(__name__).exception(
            "Camera service failed to start during application startup"
        )
    detector_service.start()
    yield
    detector_service.stop()
    camera_service.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(health_router)
app.include_router(stream_router)
app.include_router(vision_router)
