"""Background service indexing reuses the server's storage and embedding model."""
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from . import config
from .indexing import full_index, git, _write_state
from .repositories import repo_id


def github_repo(url: str) -> str:
    """Accept only canonical HTTPS GitHub repository URLs, without URL credentials."""
    match = re.fullmatch(r'https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?', url)
    if not match:
        raise ValueError('Use https://github.com/<owner>/<name> without query parameters or credentials.')
    return repo_id(match[1] + '/' + match[2].removesuffix('.git'))


def language_profile(root: Path) -> dict:
    """Estimate the main language from filenames without reading file contents."""
    counts = Counter()
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in config.SKIP_DIRS and not name.startswith('.')]
        for name in files:
            language = config.LANGUAGE_EXTENSIONS.get(Path(name).suffix.lower())
            if language:
                counts[language] += 1
    python = counts.get('Python', 0)
    dominant = counts.most_common(1)[0][0] if counts else 'unknown language'
    warning = (f'GitContext currently understands Python code best; this repository is mostly {dominant}.'
               if counts and python / sum(counts.values()) < config.PYTHON_FILE_RATIO else None)
    return {'python_files': python, 'language': dominant, 'language_warning': warning}


class ProgressEmbedder:
    """Serialize model batches with searches and publish completed chunk counts."""
    def __init__(self, embedder, lock, progress):
        self.embedder, self.lock, self.progress = embedder, lock, progress
        self.dim = embedder.dim

    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), config.API_EMBED_BATCH_SIZE):
            with self.lock:
                vectors.extend(self.embedder.encode(texts[start:start + config.API_EMBED_BATCH_SIZE]))
            self.progress('embedding', f'{len(vectors)}/{len(texts)} chunks')
        return vectors


def index_repository(url, identity, retriever, lock, progress):
    """Shallow-clone a validated public URL and atomically publish its collection."""
    root = (config.REPOS_DIR / identity.replace('/', '__')).resolve()
    progress('cloning', '')
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/' + identity, str(root)],
                       check=True, capture_output=True, timeout=config.API_CLONE_TIMEOUT)
    elif github_repo(git(root, 'remote', 'get-url', 'origin')) != identity:
        raise ValueError('Existing checkout belongs to a different repository.')
    result = full_index(str(root), repo=identity, store=retriever.store,
                        embedder=ProgressEmbedder(retriever.embedder, lock, progress),
                        progress=progress, write_lock=lock)
    state = {**result['state'], **language_profile(root)}
    with lock:
        _write_state(state)
        retriever.invalidate_index()
    return state
