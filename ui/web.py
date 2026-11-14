"""Flask routes for the bib-search user interface.

Every event lives under ``/e/<event>/`` and is served from its own database.
Unknown and unpublished events return 404, unless ``RPF_SHOW_UNPUBLISHED=1``
is set for local review before publishing.
"""

import os
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    jsonify,
    send_file,
    url_for,
)

from database_core.core import (
    connect_database,
    create_thumbnail,
    find_participant_search_results,
    find_picture,
    local_path,
)
from database_core.events import SLUG_PATTERN, Event, list_events


UI_ROOT = Path(__file__).resolve().parent


def show_unpublished():
    return os.environ.get("RPF_SHOW_UNPUBLISHED") == "1"


def create_app():
    app = Flask(
        __name__,
        template_folder=str(UI_ROOT / "templates"),
        static_folder=str(UI_ROOT / "static"),
    )

    @app.after_request
    def no_indexing(response):
        # Runners' photos should not end up in search engines.
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    @app.get("/")
    def home():
        visible = []
        for event in list_events():
            settings = event.settings()
            if settings["published"] or show_unpublished():
                visible.append({"slug": event.slug, **settings})
        visible.sort(key=lambda item: item["date"] or "", reverse=True)
        return render_template("home.html", events=visible)

    @app.get("/e/<event_slug>/")
    def index(event_slug):
        event, settings = _event_or_abort(event_slug)
        bib = request.args.get("bib", "").strip()
        participant = None
        pictures = []
        searched = bool(bib)
        error = None

        if searched:
            try:
                participant, confirmed_pictures, possible_pictures = (
                    find_participant_search_results(
                        event, bib, include_faces=settings["face_matching"]
                    )
                )
                # Bib-confirmed photos first, then face suggestions ordered by
                # descending similarity from the search service.
                pictures = confirmed_pictures + possible_pictures
            except sqlite3.Error:
                app.logger.exception("Database search failed")
                error = "Could not connect to the photo database."

        return render_template(
            "index.html",
            event_slug=event.slug,
            event=settings,
            bib=bib,
            participant=participant,
            pictures=pictures,
            searched=searched,
            error=error,
        )

    @app.get("/e/<event_slug>/pictures/<int:picture_id>")
    def picture(event_slug, picture_id):
        event, _ = _event_or_abort(event_slug)
        record = _find_picture_or_abort(app, event, picture_id, "Picture lookup failed")
        full_path = record["full_path"]
        parsed = urlparse(full_path)
        if parsed.scheme in {"http", "https"}:
            return redirect(full_path)
        if parsed.scheme == "s3":
            abort(501, "S3 picture delivery requires a presigned URL integration")
        if parsed.scheme:
            abort(400, "Unsupported picture path")

        path = local_path(full_path)
        if not path.is_file():
            abort(404)
        return send_file(path, conditional=True)

    @app.get("/e/<event_slug>/thumbnails/<int:picture_id>")
    def thumbnail(event_slug, picture_id):
        event, _ = _event_or_abort(event_slug)
        record = _find_picture_or_abort(app, event, picture_id, "Thumbnail lookup failed")
        thumbnail_path = event.thumbnail_file(record["thumbnail_path"])
        if thumbnail_path and thumbnail_path.is_file():
            return send_file(thumbnail_path, conditional=True, max_age=86400)

        full_path = record["full_path"]
        parsed = urlparse(full_path)
        if parsed.scheme in {"http", "https"}:
            return redirect(full_path)
        if parsed.scheme:
            abort(400, "Unsupported thumbnail source")

        original_path = local_path(full_path)
        if not original_path.is_file():
            abort(404)
        try:
            thumbnail_name = create_thumbnail(event, original_path)
            with connect_database(event) as connection:
                connection.execute(
                    "UPDATE pictures SET thumbnail_path = ? WHERE id = ?",
                    (thumbnail_name, picture_id),
                )
        except (OSError, ValueError):
            app.logger.exception("Thumbnail creation failed")
            abort(500)
        return send_file(
            event.thumbnail_dir / thumbnail_name, conditional=True, max_age=86400
        )

    @app.get("/e/<event_slug>/pictures/<int:picture_id>/download")
    def download_picture(event_slug, picture_id):
        event, _ = _event_or_abort(event_slug)
        record = _find_picture_or_abort(
            app, event, picture_id, "Picture download lookup failed"
        )
        full_path = record["full_path"]
        parsed = urlparse(full_path)
        if parsed.scheme in {"http", "https"}:
            return redirect(full_path)
        if parsed.scheme:
            abort(400, "Unsupported picture path")
        path = local_path(full_path)
        if not path.is_file():
            abort(404)
        return send_file(
            path,
            as_attachment=True,
            download_name=path.name,
            conditional=True,
        )

    @app.get("/e/<event_slug>/photos.json")
    def photo_manifest(event_slug):
        """List a bib's photos for the "Download all" button.

        The browser fetches every photo itself (zipping it where supported,
        otherwise downloading the photos one by one), so photos never stream
        through this server once they are served from R2.
        """
        event, archive_name, records = _bib_photos_or_abort(app, event_slug)
        names = _unique_names(Path(record["full_path"]).name for record in records)
        return jsonify(
            archive_name=archive_name,
            files=[
                {
                    "name": name,
                    "url": url_for(
                        "picture",
                        event_slug=event.slug,
                        picture_id=record["picture_id"],
                    ),
                    "download_url": url_for(
                        "download_picture",
                        event_slug=event.slug,
                        picture_id=record["picture_id"],
                    ),
                }
                for name, record in zip(names, records)
            ],
        )

    return app


def _unique_names(names):
    """Make file names unique inside an archive: a.jpg, a-2.jpg, ..."""
    used = set()
    result = []
    for name in names:
        stem, dot, suffix = name.rpartition(".")
        if not dot:
            stem, suffix = name, ""
        candidate, counter = name, 2
        while candidate in used:
            candidate = f"{stem}-{counter}{dot}{suffix}"
            counter += 1
        used.add(candidate)
        result.append(candidate)
    return result


def _bib_photos_or_abort(app, event_slug):
    """Return (event, archive name, photo records) shown for ?bib=..."""
    event, settings = _event_or_abort(event_slug)
    bib = request.args.get("bib", "").strip()
    if not bib or len(bib) > 30:
        abort(400)
    try:
        participant, confirmed, possible = find_participant_search_results(
            event, bib, include_faces=settings["face_matching"]
        )
    except sqlite3.Error:
        app.logger.exception("Bib photo lookup failed")
        abort(503)
    if participant is None or not (confirmed or possible):
        abort(404)
    safe_bib = "".join(ch for ch in bib if ch.isalnum() or ch in "-_") or "photos"
    return event, f"{event.slug}-bib-{safe_bib}.zip", confirmed + possible


def _event_or_abort(event_slug):
    if not SLUG_PATTERN.fullmatch(event_slug):
        abort(404)
    event = Event(event_slug)
    if not event.exists():
        abort(404)
    settings = event.settings()
    if not settings["published"] and not show_unpublished():
        abort(404)
    return event, settings


def _find_picture_or_abort(app, event, picture_id, log_message):
    try:
        record = find_picture(event, picture_id)
    except sqlite3.Error:
        app.logger.exception(log_message)
        abort(503)
    if not record:
        abort(404)
    return record
