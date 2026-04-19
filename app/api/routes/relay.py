from fastapi import APIRouter

from app.schemas.relay import RelayStatusResponse
from app.services.relay.provider import relay_publisher_service


router = APIRouter(prefix="/api/v1/relay", tags=["relay"])


@router.get("/status", response_model=RelayStatusResponse)
def get_relay_status() -> RelayStatusResponse:
    return RelayStatusResponse(**relay_publisher_service.get_status())
