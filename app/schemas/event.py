from pydantic import BaseModel


class EventResponse(BaseModel):
    id: int
    event_type: str
    class_name: str
    class_id: int | None
    confidence: float
    state_key: str
    first_seen_frame_id: int
    confirmed_frame_id: int
    last_seen_frame_id: int
    stable_frames_required: int
    absent_frames_required: int
    cooldown_seconds: int
    source_frame_width: int | None
    source_frame_height: int | None
    frame_timestamp: str
    created_at: str
    screenshot_original_path: str | None = None
    screenshot_annotated_path: str | None = None


class EventEngineStateItemResponse(BaseModel):
    state_key: str
    state: str
    present_frames: int
    absent_frames: int
    cooldown_until: str | None
    first_seen_frame_id: int | None
    last_seen_frame_id: int | None
    latest_confidence: float | None


class EventEngineStatusResponse(BaseModel):
    is_running: bool
    stable_frames: int
    absent_frames: int
    cooldown_seconds: int
    processed_detection_frames: int
    confirmed_events_total: int
    latest_processed_frame_id: int
    active_states: list[EventEngineStateItemResponse]
