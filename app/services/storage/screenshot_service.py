from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ScreenshotService:
    def __init__(self, base_dir: str, jpeg_quality: int = 90) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        raw_base_dir = Path(base_dir)
        self.base_dir = (
            raw_base_dir.resolve()
            if raw_base_dir.is_absolute()
            else (self.project_root / raw_base_dir).resolve()
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = jpeg_quality

    def save_event_frames(
        self,
        event_id: int,
        original_frame: np.ndarray | None,
        annotated_frame: np.ndarray | None,
        frame_timestamp: str | None,
    ) -> dict[str, str | None]:
        timestamp = self._parse_timestamp(frame_timestamp)
        day_dir = self.base_dir / timestamp.strftime("%Y") / timestamp.strftime("%m") / timestamp.strftime("%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        original_path = self._save_frame(
            frame=original_frame,
            target_path=day_dir / f"event_{event_id}_original.jpg",
        )
        annotated_path = self._save_frame(
            frame=annotated_frame,
            target_path=day_dir / f"event_{event_id}_annotated.jpg",
        )

        return {
            "screenshot_original_path": self._to_posix_relative(original_path),
            "screenshot_annotated_path": self._to_posix_relative(annotated_path),
        }

    def get_absolute_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.base_dir / candidate).resolve()
            if not resolved.exists():
                legacy_resolved = (self.project_root / candidate).resolve()
                if legacy_resolved.exists():
                    resolved = legacy_resolved

        if self.base_dir not in resolved.parents and resolved != self.base_dir:
            raise ValueError("Requested screenshot path is outside of the storage directory")
        return resolved

    def _save_frame(self, frame: np.ndarray | None, target_path: Path) -> Path | None:
        if frame is None:
            return None

        success = cv2.imwrite(
            str(target_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            return None
        return target_path

    def _parse_timestamp(self, frame_timestamp: str | None) -> datetime:
        if not frame_timestamp:
            return datetime.utcnow()

        normalized = frame_timestamp.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.utcnow()

    def _to_posix_relative(self, path: Path | None) -> str | None:
        if path is None:
            return None

        try:
            relative_path = path.resolve().relative_to(self.base_dir)
        except ValueError:
            return path.resolve().as_posix()

        return relative_path.as_posix()
