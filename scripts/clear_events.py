import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.db.repository import EventRepository


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear all events from the configured SQLite database."
    )
    parser.add_argument(
        "--keep-sequence",
        action="store_true",
        help="Do not reset SQLite AUTOINCREMENT for the events table.",
    )
    args = parser.parse_args()

    repository = EventRepository(database_path=settings.database_path)
    deleted_count = repository.clear_events(reset_sequence=not args.keep_sequence)

    print(f"Database: {settings.database_path}")
    print(f"Deleted events: {deleted_count}")
    if args.keep_sequence:
        print("Event ID sequence preserved.")
    else:
        print("Event ID sequence reset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
