from app.core.config import settings
from app.services.capture.provider import camera_service
from app.services.relay_control.sync_service import RelayCameraControlService
from app.services.relay_ssl import build_ssl_context


relay_camera_control_service = RelayCameraControlService(
    camera_service=camera_service,
    enabled=settings.relay_control_enabled,
    api_url=(settings.relay_control_api_url or settings.relay_events_api_url),
    auth_token=(settings.relay_control_token or settings.relay_events_token),
    source_id=settings.relay_source_id,
    timeout_seconds=settings.relay_events_timeout_seconds,
    poll_interval_seconds=settings.relay_control_poll_interval_seconds,
    state_sync_interval_seconds=settings.relay_control_state_sync_interval_seconds,
    camera_discovery_max_index=settings.camera_discovery_max_index,
    ssl_context=build_ssl_context(
        verify=settings.relay_ssl_verify,
        ca_file=settings.relay_ssl_ca_file,
    ),
)
