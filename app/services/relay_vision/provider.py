from app.core.config import settings
from app.services.relay_ssl import build_ssl_context
from app.services.relay_vision.sync_service import RelayVisionSyncService
from app.services.vision.provider import detector_service


relay_vision_sync_service = RelayVisionSyncService(
    detector_service=detector_service,
    enabled=settings.relay_events_enabled,
    api_url=settings.relay_events_api_url,
    auth_token=settings.relay_events_token,
    source_id=settings.relay_source_id,
    timeout_seconds=settings.relay_events_timeout_seconds,
    poll_interval_seconds=(1.0 / max(settings.vision_inference_fps, 1)),
    ssl_context=build_ssl_context(
        verify=settings.relay_ssl_verify,
        ca_file=settings.relay_ssl_ca_file,
    ),
)
