from app.core.config import settings
from app.db.repository import EventRepository
from app.services.relay_events.sync_service import RelayEventSyncService
from app.services.relay_ssl import build_ssl_context
from app.services.storage.provider import screenshot_service


relay_event_sync_service = RelayEventSyncService(
    repository=EventRepository(database_path=settings.database_path),
    screenshot_service=screenshot_service,
    enabled=settings.relay_events_enabled,
    api_url=settings.relay_events_api_url,
    auth_token=settings.relay_events_token,
    source_id=settings.relay_source_id,
    timeout_seconds=settings.relay_events_timeout_seconds,
    ssl_context=build_ssl_context(
        verify=settings.relay_ssl_verify,
        ca_file=settings.relay_ssl_ca_file,
    ),
)
