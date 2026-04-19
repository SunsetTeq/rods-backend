import sqlite3
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class EventRepository:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def initialize(self) -> None:
        try:
            self._initialize_schema()
        except sqlite3.Error:
            self._recover_broken_database()
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    class_id INTEGER,
                    track_id INTEGER,
                    confidence REAL NOT NULL,
                    state_key TEXT NOT NULL,
                    first_seen_frame_id INTEGER NOT NULL,
                    confirmed_frame_id INTEGER NOT NULL,
                    last_seen_frame_id INTEGER NOT NULL,
                    stable_frames_required INTEGER NOT NULL,
                    absent_frames_required INTEGER NOT NULL,
                    cooldown_seconds INTEGER NOT NULL,
                    source_frame_width INTEGER,
                    source_frame_height INTEGER,
                    frame_timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    observed_classes_json TEXT,
                    screenshot_original_path TEXT,
                    screenshot_annotated_path TEXT
                )
                """
            )
            self._ensure_column(connection, "events", "track_id", "INTEGER")
            self._ensure_column(connection, "events", "updated_at", "TEXT")
            self._ensure_column(connection, "events", "observed_classes_json", "TEXT")
            self._ensure_column(connection, "events", "screenshot_original_path", "TEXT")
            self._ensure_column(connection, "events", "screenshot_annotated_path", "TEXT")
            connection.commit()

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        existing_columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in existing_columns:
            return

        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )

    def _recover_broken_database(self) -> None:
        journal_path = self.database_path.with_name(f"{self.database_path.name}-journal")
        if journal_path.exists():
            try:
                journal_path.unlink(missing_ok=True)
            except PermissionError:
                pass

        if self.database_path.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            backup_path = self.database_path.with_name(
                f"{self.database_path.stem}.corrupt-{timestamp}{self.database_path.suffix}"
            )
            try:
                self.database_path.replace(backup_path)
            except PermissionError:
                pass

    def create_event(self, payload: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (
                    event_type,
                    class_name,
                    class_id,
                    track_id,
                    confidence,
                    state_key,
                    first_seen_frame_id,
                    confirmed_frame_id,
                    last_seen_frame_id,
                    stable_frames_required,
                    absent_frames_required,
                    cooldown_seconds,
                    source_frame_width,
                    source_frame_height,
                    frame_timestamp,
                    created_at,
                    updated_at,
                    observed_classes_json,
                    screenshot_original_path,
                    screenshot_annotated_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["event_type"],
                    payload["class_name"],
                    payload["class_id"],
                    payload.get("track_id"),
                    payload["confidence"],
                    payload["state_key"],
                    payload["first_seen_frame_id"],
                    payload["confirmed_frame_id"],
                    payload["last_seen_frame_id"],
                    payload["stable_frames_required"],
                    payload["absent_frames_required"],
                    payload["cooldown_seconds"],
                    payload["source_frame_width"],
                    payload["source_frame_height"],
                    payload["frame_timestamp"],
                    payload["created_at"],
                    payload.get("updated_at"),
                    self._serialize_classes(payload.get("observed_classes")),
                    payload.get("screenshot_original_path"),
                    payload.get("screenshot_annotated_path"),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def update_event_screenshots(
        self,
        event_id: int,
        screenshot_original_path: str | None,
        screenshot_annotated_path: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE events
                SET screenshot_original_path = ?, screenshot_annotated_path = ?
                WHERE id = ?
                """,
                (
                    screenshot_original_path,
                    screenshot_annotated_path,
                    event_id,
                ),
            )
            connection.commit()

    def update_event_observed_classes(
        self,
        event_id: int,
        observed_classes: list[str],
        confidence: float,
        class_name: str,
        class_id: int | None,
        last_seen_frame_id: int,
        frame_timestamp: str,
        screenshot_original_path: str | None,
        screenshot_annotated_path: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE events
                SET
                    observed_classes_json = ?,
                    confidence = ?,
                    class_name = ?,
                    class_id = ?,
                    last_seen_frame_id = ?,
                    frame_timestamp = ?,
                    updated_at = ?,
                    screenshot_original_path = ?,
                    screenshot_annotated_path = ?
                WHERE id = ?
                """,
                (
                    self._serialize_classes(observed_classes),
                    confidence,
                    class_name,
                    class_id,
                    last_seen_frame_id,
                    frame_timestamp,
                    datetime.now(timezone.utc).isoformat(),
                    screenshot_original_path,
                    screenshot_annotated_path,
                    event_id,
                ),
            )
            connection.commit()

    def clear_events(self, reset_sequence: bool = True) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM events").fetchone()
            deleted_count = int(row["total"]) if row is not None else 0
            connection.execute("DELETE FROM events")
            if reset_sequence:
                connection.execute(
                    "DELETE FROM sqlite_sequence WHERE name = ?",
                    ("events",),
                )
            connection.commit()
            return deleted_count

    def get_event_by_id(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    event_type,
                    class_name,
                    class_id,
                    track_id,
                    confidence,
                    state_key,
                    first_seen_frame_id,
                    confirmed_frame_id,
                    last_seen_frame_id,
                    stable_frames_required,
                    absent_frames_required,
                    cooldown_seconds,
                    source_frame_width,
                    source_frame_height,
                    frame_timestamp,
                    created_at,
                    updated_at,
                    observed_classes_json,
                    screenshot_original_path,
                    screenshot_annotated_path
                FROM events
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_recent_events(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    event_type,
                    class_name,
                    class_id,
                    track_id,
                    confidence,
                    state_key,
                    first_seen_frame_id,
                    confirmed_frame_id,
                    last_seen_frame_id,
                    stable_frames_required,
                    absent_frames_required,
                    cooldown_seconds,
                    source_frame_width,
                    source_frame_height,
                    frame_timestamp,
                    created_at,
                    updated_at,
                    observed_classes_json,
                    screenshot_original_path,
                    screenshot_annotated_path
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _serialize_classes(self, observed_classes: list[str] | None) -> str:
        normalized = sorted({str(item) for item in (observed_classes or [])})
        return json.dumps(normalized, ensure_ascii=True)
