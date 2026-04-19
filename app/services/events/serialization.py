import json
from typing import Any


def parse_observed_classes(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    return []


def serialize_event_row(row: dict[str, Any]) -> dict[str, Any]:
    event_id = int(row["id"])
    original_path = row.get("screenshot_original_path")
    annotated_path = row.get("screenshot_annotated_path")
    payload = dict(row)
    payload.pop("observed_classes_json", None)

    return {
        **payload,
        "observed_classes": parse_observed_classes(row.get("observed_classes_json")),
        "screenshot_original_url": (
            f"/api/v1/events/{event_id}/screenshots/original" if original_path else None
        ),
        "screenshot_annotated_url": (
            f"/api/v1/events/{event_id}/screenshots/annotated" if annotated_path else None
        ),
    }
