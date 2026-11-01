"""Flask routes for the bib-search user interface."""

from pathlib import Path
import sqlite3
from urllib.parse import urlparse

from flask import Flask, abort, redirect, render_template, request, send_file

from database_core.core import (
    connect_database,
    create_thumbnail,
    find_participant_search_results,
    find_picture,
    local_path,
)


UI_ROOT = Path(__file__).resolve().parent


def create_app():
    app = Flask(
        __name__,
        template_folder=str(UI_ROOT / "templates"),
        static_folder=str(UI_ROOT / "static"),
    )

    @app.get("/")
    def index():
        bib = request.args.get("bib", "").strip()
        participant = None
        pictures = []
        searched = bool(bib)
        error = None

        if searched:
            try:
                participant, confirmed_pictures, possible_pictures = (
                    find_participant_search_results(bib)
                )
                # Bib-confirmed photos first, then face suggestions ordered by
                # descending similarity from the search service.
                pictures = confirmed_pictures + possible_pictures
            except sqlite3.Error:
                app.logger.exception("Database search failed")
                error = "Could not connect to the photo database."

        return render_template(
            "index.html",
            bib=bib,
            participant=participant,
            pictures=pictures,
            searched=searched,
            error=error,
        )

    @app.get("/pictures/<int:picture_id>")
    def picture(picture_id):
        record = _find_picture_or_abort(app, picture_id, "Picture lookup failed")
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

    @app.get("/thumbnails/<int:picture_id>")
    def thumbnail(picture_id):
        record = _find_picture_or_abort(app, picture_id, "Thumbnail lookup failed")
        full_path = record["full_path"]
        parsed = urlparse(full_path)
        if parsed.scheme in {"http", "https"}:
            return redirect(full_path)
        if parsed.scheme:
            abort(400, "Unsupported thumbnail source")

        original_path = local_path(full_path)
        if not original_path.is_file():
            abort(404)
        thumbnail_path = (
            Path(record["thumbnail_path"]) if record["thumbnail_path"] else None
        )
        if not thumbnail_path or not thumbnail_path.is_file():
            try:
                thumbnail_path = create_thumbnail(original_path)
                with connect_database() as connection:
                    connection.execute(
                        "UPDATE pictures SET thumbnail_path = ? WHERE id = ?",
                        (str(thumbnail_path), picture_id),
                    )
            except (OSError, ValueError):
                app.logger.exception("Thumbnail creation failed")
                abort(500)
        return send_file(thumbnail_path, conditional=True, max_age=86400)

    @app.get("/pictures/<int:picture_id>/download")
    def download_picture(picture_id):
        record = _find_picture_or_abort(app, picture_id, "Picture download lookup failed")
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

    return app


def _find_picture_or_abort(app, picture_id, log_message):
    try:
        record = find_picture(picture_id)
    except sqlite3.Error:
        app.logger.exception(log_message)
        abort(503)
    if not record:
        abort(404)
    return record

