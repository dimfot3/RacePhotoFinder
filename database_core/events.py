"""Per-event storage: each event is one self-contained folder.

    <data root>/events/<slug>/
        event.yaml      name, date, published, face_matching
        event.db        SQLite index for this event only
        thumbnails/     generated previews

Keeping events separate means publishing an event is uploading one folder and
deleting an event is removing one folder, with no risk to other events.
"""

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import re
import shutil

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLUG_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
DEFAULT_SETTINGS = {
    "name": None,
    "date": None,
    "published": False,
    "face_matching": True,
}


def data_root():
    configured = os.environ.get("RPF_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "data"


def events_root():
    return data_root() / "events"


def validate_slug(slug):
    if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
        raise ValueError(
            f"Invalid event name {slug!r}: use lowercase letters, digits, and "
            "hyphens, e.g. athens-marathon-2026"
        )
    return slug


@dataclass(frozen=True)
class Event:
    slug: str

    def __post_init__(self):
        validate_slug(self.slug)

    @property
    def directory(self):
        return events_root() / self.slug

    @property
    def database_path(self):
        return self.directory / "event.db"

    @property
    def thumbnail_dir(self):
        return self.directory / "thumbnails"

    @property
    def settings_path(self):
        return self.directory / "event.yaml"

    def exists(self):
        return self.database_path.is_file()

    def settings(self):
        settings = dict(DEFAULT_SETTINGS)
        if self.settings_path.is_file():
            loaded = yaml.safe_load(self.settings_path.read_text()) or {}
            settings.update({k: v for k, v in loaded.items() if k in DEFAULT_SETTINGS})
        if not settings["name"]:
            settings["name"] = self.slug.replace("-", " ").title()
        if isinstance(settings["date"], date):
            settings["date"] = settings["date"].isoformat()
        settings["published"] = bool(settings["published"])
        settings["face_matching"] = bool(settings["face_matching"])
        return settings

    def ensure_settings(self):
        """Create event.yaml with safe defaults (unpublished) if missing."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.settings_path.is_file():
            content = dict(DEFAULT_SETTINGS)
            content["name"] = self.slug.replace("-", " ").title()
            self.settings_path.write_text(
                "# Event settings. Search only shows this event once published.\n"
                + yaml.safe_dump(content, sort_keys=False)
            )
        return self.settings()

    def update_settings(self, **changes):
        """Change settings in event.yaml (creating it if needed)."""
        unknown = set(changes) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"Unknown event setting(s): {', '.join(sorted(unknown))}")
        self.ensure_settings()
        content = yaml.safe_load(self.settings_path.read_text()) or {}
        content.update(changes)
        self.settings_path.write_text(
            "# Event settings. Search only shows this event once published.\n"
            + yaml.safe_dump(content, sort_keys=False)
        )
        return self.settings()

    def thumbnail_file(self, stored_path):
        """Resolve a stored thumbnail path, confined to this event's folder."""
        if not stored_path:
            return None
        path = Path(stored_path)
        if not path.is_absolute():
            path = self.thumbnail_dir / path
        path = path.resolve()
        if not path.is_relative_to(self.thumbnail_dir.resolve()):
            return None
        return path

    def delete(self):
        if self.directory.is_dir():
            shutil.rmtree(self.directory)


def list_events():
    root = events_root()
    if not root.is_dir():
        return []
    events = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and SLUG_PATTERN.fullmatch(child.name):
            event = Event(child.name)
            if event.exists():
                events.append(event)
    return events
