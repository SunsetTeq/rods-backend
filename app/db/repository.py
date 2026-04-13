import sqlite3
from datetime import datetime, timezone
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
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

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
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["event_type"],
                    payload["class_name"],
                    payload["class_id"],
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
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_recent_events(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    event_type,
                    class_name,
                    class_id,
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
                    created_at
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
