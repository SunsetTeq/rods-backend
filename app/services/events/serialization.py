from typing import Any


def serialize_event_row(row: dict[str, Any]) -> dict[str, Any]:
    event_id = int(row["id"])
    original_path = row.get("screenshot_original_path")
    annotated_path = row.get("screenshot_annotated_path")
    payload = dict(row)
    payload.pop("observed_classes_json", None)

    return {
        **payload,
        "screenshot_original_url": (
            f"/api/v1/events/{event_id}/screenshots/original" if original_path else None
        ),
        "screenshot_annotated_url": (
            f"/api/v1/events/{event_id}/screenshots/annotated" if annotated_path else None
        ),
    }
