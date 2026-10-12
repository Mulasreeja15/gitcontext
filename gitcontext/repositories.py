"""Stable repository identities and safe collection/state names."""
import re
from pathlib import Path
from urllib.parse import urlsplit

from . import config


def repo_id(value: str) -> str:
    """Validate owner/name and normalize GitHub-style identities to lowercase."""
    parts = value.lower().split('/')
    if len(parts) != 2 or any(not re.fullmatch(r'[a-z0-9][a-z0-9._-]*', p)
                              or '__' in p or p in {'.', '..'} for p in parts):
        raise ValueError('Repository must be owner/name using letters, numbers, dots, underscores, or hyphens.')
    return '/'.join(parts)


def collection_name(value: str) -> str:
    """Use a collision-free separator, with path traversal rejected."""
    return 'repo__' + repo_id(value).replace('/', '__')


def state_path(value: str | None = None) -> Path:
    """Keep each repository's state in its own file."""
    return config.INDEX_STATE_DIR / (repo_id(value or config.DEFAULT_REPO).replace('/', '__') + '.json')


def infer_repo(source: str, root: Path, remote: str = '') -> str:
    """Infer owner/name from URL/SCP remotes; use local/name for local folders."""
    candidate = remote or source
    if candidate.startswith('git@') and ':' in candidate:
        path = candidate.split(':', 1)[1]
    elif candidate.startswith(('http://', 'https://', 'ssh://')):
        path = urlsplit(candidate).path.lstrip('/')
    else:
        return repo_id('local/' + root.name)
    return repo_id(path.rstrip('/').removesuffix('.git'))


def list_repos(client=None):
    """List saved repositories using one local Qdrant connection."""
    from .database import open_client
    from .indexing import load_state, index_status
    from .store import VectorStore
    owned = client is None
    client = client if client is not None else open_client()
    try:
        rows = []
        for path in sorted(config.INDEX_STATE_DIR.glob('*.json')):
            identity = path.stem.replace('__', '/')
            try:
                state = load_state(identity)
            except ValueError:
                continue
            if state:
                count = VectorStore(identity, client=client).count()
                rows.append({'repo': identity, 'chunk_count': count,
                             'indexed_commit': state['indexed_commit'],
                             'stale': index_status(identity)['stale'] or count != state['chunk_count']})
        return rows
    finally:
        if owned:
            client.close()


def remove_repo(value, client=None):
    """Remove one collection/state while preserving source checkouts."""
    from .database import open_client
    identity = repo_id(value)
    owned = client is None
    client = client if client is not None else open_client()
    try:
        collection = collection_name(identity)
        exists = client.collection_exists(collection)
        if not exists and not state_path(identity).exists():
            raise ValueError(f'Repository {identity} is not indexed.')
        if exists:
            client.delete_collection(collection)
        state_path(identity).unlink(missing_ok=True)
    finally:
        if owned:
            client.close()
