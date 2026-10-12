"""Shared Qdrant connection selection and friendly local lock failures."""
from pathlib import Path

from portalocker.exceptions import AlreadyLocked
from qdrant_client import QdrantClient

from . import config

LOCK_MESSAGE = (
    'Another GitContext process (usually the server) is using the database. '
    'Press Ctrl + C in its server terminal, or run: taskkill /F /T /IM python.exe '
    '(stops all Python processes). Then try again. '
    'For concurrent processes, configure QDRANT_URL to use a Qdrant server.'
)


class DatabaseLocked(ValueError):
    """The local database is owned by another process."""


def open_client():
    """Connect remotely when configured; otherwise open the local database."""
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    Path(config.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=config.QDRANT_PATH)
    except AlreadyLocked:
        raise DatabaseLocked(LOCK_MESSAGE) from None
    except RuntimeError as error:
        if 'already accessed by another instance' in str(error).lower():
            raise DatabaseLocked(LOCK_MESSAGE) from None
        raise


def run_cli(command):
    """Suppress tracebacks only for the known local database lock failure."""
    try:
        command()
    except DatabaseLocked as error:
        raise SystemExit(str(error)) from None
