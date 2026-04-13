from app.core.config import settings
from app.db.repository import EventRepository
from app.services.events.event_engine import EventEngineService
from app.services.vision.provider import detector_service


event_repository = EventRepository(database_path=settings.database_path)

event_engine_service = EventEngineService(
    detector_service=detector_service,
    repository=event_repository,
    stable_frames=settings.event_stable_frames,
    absent_frames=settings.event_absent_frames,
    cooldown_seconds=settings.event_cooldown_seconds,
)
