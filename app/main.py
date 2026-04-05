from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.stream import router as stream_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.capture.provider import camera_service


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    camera_service.start()
    yield
    camera_service.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(stream_router)
