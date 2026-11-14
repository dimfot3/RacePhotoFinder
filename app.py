"""Race Photo Finder command-line entry point and web application.

Examples:
    python app.py photos/ --event athens-2026 --participants-csv bibs.csv
    python app.py --show-unpublished          # review locally before publishing
    python app.py --event athens-2026 --delete-event
"""

import argparse
import os
from pathlib import Path

from database_core.core import (
    import_csv_only,
    import_photo_folder,
    initialize_database,
    rebuild_thumbnails,
    refresh_face_suggestions,
)
from database_core.events import Event
from ui import create_app


app = create_app()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Recursively import pictures from this folder instead of starting the UI",
    )
    parser.add_argument(
        "--event",
        help="Event name used in URLs and storage, e.g. athens-marathon-2026",
    )
    parser.add_argument(
        "--participants-csv",
        type=Path,
        help="Optional CSV with bib, name, email, and finish_time columns",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the search UI after importing",
    )
    parser.add_argument(
        "--rebuild-thumbnails",
        action="store_true",
        help="Create small previews for pictures already in the event database",
    )
    parser.add_argument(
        "--refresh-face-suggestions",
        action="store_true",
        help="Recompute stored face suggestions without re-running detection "
        "(e.g. after changing search_similarity in face_matching/config.yaml)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Clear this event before importing the supplied folder",
    )
    face_group = parser.add_mutually_exclusive_group()
    face_group.add_argument(
        "--no-face-matching",
        dest="face_matching",
        action="store_false",
        default=None,
        help="Disable face matching for this event: no face data is stored "
        "and search shows bib matches only (saved in event.yaml)",
    )
    face_group.add_argument(
        "--face-matching",
        dest="face_matching",
        action="store_true",
        help="Enable face matching for this event (saved in event.yaml)",
    )
    parser.add_argument(
        "--delete-event",
        action="store_true",
        help="Permanently delete this event's database, thumbnails, and settings",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask for confirmation when deleting an event",
    )
    parser.add_argument(
        "--show-unpublished",
        action="store_true",
        help="Serve unpublished events too (local review only)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    changes_settings = args.face_matching is not None
    needs_event = (
        args.folder or args.participants_csv or args.rebuild_thumbnails
        or args.rerun or args.delete_event or changes_settings
        or args.refresh_face_suggestions
    )
    if needs_event and not args.event:
        parser.error("--event is required to import, rebuild, or delete an event")
    if args.rerun and not args.folder:
        parser.error("--rerun requires a photo folder")

    try:
        event = Event(args.event) if args.event else None
    except ValueError as error:
        parser.error(str(error))

    if args.delete_event:
        delete_event(event, confirm=not args.yes)
        return

    if changes_settings:
        had_data = event.exists()
        event.update_settings(face_matching=args.face_matching)
        if had_data and not args.folder:
            args.refresh_face_suggestions = True
    if event:
        initialize_database(event)
    if args.folder:
        import_photo_folder(event, args.folder, args.participants_csv, rerun=args.rerun)
    elif args.participants_csv:
        import_csv_only(event, args.participants_csv)
    if args.rebuild_thumbnails:
        rebuild_thumbnails(event)
    if args.refresh_face_suggestions and not args.folder:
        refresh_face_suggestions(event)

    if event and needs_event:
        settings = event.settings()
        state = "published" if settings["published"] else "NOT published"
        print(f"Event '{event.slug}' is {state}. Settings: {event.settings_path}")

    if not needs_event or args.serve:
        if args.show_unpublished:
            os.environ["RPF_SHOW_UNPUBLISHED"] = "1"
        app.run(host=args.host, port=args.port, debug=False)


def delete_event(event, confirm=True):
    if not event.directory.exists():
        print(f"Event '{event.slug}' does not exist.")
        return
    if confirm:
        answer = input(
            f"Permanently delete event '{event.slug}' ({event.directory})? "
            "Original photos are not touched. Type the event name to confirm: "
        )
        if answer.strip() != event.slug:
            print("Cancelled.")
            return
    event.delete()
    print(f"Event '{event.slug}' deleted.")


if __name__ == "__main__":
    main()
