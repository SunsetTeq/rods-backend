from app.core.config import settings
from app.db.repository import EventRepository
from app.services.capture.provider import camera_service
from app.services.events.event_engine import EventEngineService
from app.services.relay_events.provider import relay_event_sync_service
from app.services.storage.provider import screenshot_service
from app.services.vision.provider import detector_service


event_repository = EventRepository(database_path=settings.database_path)

event_engine_service = EventEngineService(
    camera_service=camera_service,
    detector_service=detector_service,
    repository=event_repository,
    screenshot_service=screenshot_service,
    relay_event_sync_service=relay_event_sync_service,
    stable_frames=settings.event_stable_frames,
    absent_frames=settings.event_absent_frames,
    cooldown_seconds=settings.event_cooldown_seconds,
)
