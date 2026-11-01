"""Race Photo Finder command-line entry point and web application."""

import argparse
from pathlib import Path

from database_core.core import (
    import_csv_only,
    import_photo_folder,
    initialize_database,
    rebuild_thumbnails,
)
from ui import create_app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Recursively import pictures from this folder instead of starting the UI",
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
        help="Create small previews for pictures already in the database",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Clear the previous event before importing the supplied folder",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    initialize_database()

    if args.rerun and not args.folder:
        parser.error("--rerun requires a photo folder")

    if args.folder:
        import_photo_folder(args.folder, args.participants_csv, rerun=args.rerun)
    elif args.participants_csv:
        import_csv_only(args.participants_csv)
    if args.rebuild_thumbnails:
        rebuild_thumbnails()

    did_import = args.folder or args.participants_csv or args.rebuild_thumbnails
    if not did_import or args.serve:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
