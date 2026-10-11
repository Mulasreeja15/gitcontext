"""Full/incremental indexing and Git-based index freshness without LLM calls."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from contextlib import nullcontext

from . import config
from .chunker import chunk_file
from .ingest import collect_chunks, resolve_repo, supported_path
from .repositories import infer_repo, repo_id, state_path


def file_metadata(root: Path, commit: str | None) -> dict[str, dict]:
    """Read history once, recording first occurrence of each file; names, never emails."""
    if not commit:
        return {}
    history = git(root, 'log', '--date-order', '--no-renames', '-z', '--name-only',
                  '--format=%x1e%H%x1f%an%x1f%cI', commit, '--')
    metadata, current = {}, None
    for field in history.split('\0'):
        field = field.lstrip('\n')
        if field.startswith('\x1e'):
            sha, author, modified = field[1:].split('\x1f', 2)
            current = dict(last_commit_sha=sha, last_author=author, last_modified=modified)
        elif field and current is not None:
            metadata.setdefault(field, current)
    return metadata


def annotate_chunks(chunks, metadata):
    """Attach provenance without changing chunk IDs or embedding text."""
    return [replace(chunk, **metadata.get(chunk.file_path, {})) for chunk in chunks]


def git(root: Path, *args: str, optional: bool = False) -> str:
    """Run Git with explicit argument boundaries; never inspect source credentials."""
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                            encoding='utf-8', errors='replace')
    if result.returncode and not optional:
        raise ValueError(f'Git {args[0]} failed. Check repository history, permissions, or working tree.')
    return result.stdout.rstrip('\n') if result.returncode == 0 else ''


def load_state(repo: str | None = None) -> dict | None:
    """Treat absent or damaged state as requiring a full index."""
    try:
        state = json.loads(state_path(repo).read_text(encoding='utf-8'))
        required = {'repo_path', 'remote_url', 'indexed_commit', 'repo_name', 'timestamp', 'chunk_count', 'embed_model'}
        if not isinstance(state, dict) or not required <= state.keys():
            return None
        if state.get('repo_id') != repo_id(repo or config.DEFAULT_REPO):
            return None
        if not isinstance(state['repo_path'], str) or not isinstance(state['repo_name'], str):
            return None
        if not isinstance(state['embed_model'], str) or (
                state['indexed_commit'] is not None and not isinstance(state['indexed_commit'], str)):
            return None
        return state
    except (OSError, ValueError):
        return None


def save_state(root: Path, store, commit: str | None) -> dict:
    """Atomically publish state only after all database writes have succeeded."""
    remotes = git(root, 'remote', optional=True).splitlines()
    remote = 'origin' if 'origin' in remotes else (remotes[0] if remotes else None)
    previous = load_state(store.repo_id) or {}
    retained = {key: value for key, value in previous.items() if key.startswith('history_')
                and previous['embed_model'] == config.EMBED_MODEL}
    state = {**retained, 'repo_id': store.repo_id, 'repo_path': str(root.resolve()), 'remote_url':
             git(root, 'remote', 'get-url', remote, optional=True) if remote else None,
             'indexed_commit': commit, 'repo_name': store.repo_id.split('/')[1],
             'timestamp': datetime.now(timezone.utc).isoformat(),
             'chunk_count': store.count(), 'embed_model': config.EMBED_MODEL,
             'incomplete': False, 'payload_version': config.INDEX_PAYLOAD_VERSION}
    _write_state(state)
    return state


def _write_state(state: dict) -> None:
    """Replace state atomically, also when marking an interrupted update unsafe."""
    destination = state_path(state['repo_id'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)


def _mark_incomplete(repo: str) -> None:
    """Ensure a crash during database mutation cannot leave a false fresh status."""
    state = load_state(repo)
    if state:
        _write_state({**state, 'incomplete': True})


def changed_files(root: Path, old: str, new: str) -> list[tuple[str, str, str | None]]:
    """Parse NUL-delimited Git name/status, including renames and spaced paths."""
    fields = git(root, 'diff', '--name-status', '-z', '-M', old, new, '--').split('\0')
    changes, i = [], 0
    while i < len(fields) and fields[i]:
        kind = fields[i][0]
        old_path = fields[i + 1]
        if kind in {'R', 'C'}:
            changes.append((kind, old_path, fields[i + 2]))
            i += 3
        else:
            changes.append((kind, old_path, None))
            i += 2
    return changes


def index_status(repo: str | None = None) -> dict:
    """Compare saved commit with local HEAD; health/status never fetch the remote."""
    state = load_state(repo)
    result = {'stale': True, 'indexed_commit': None, 'current_head': None,
              'commits_behind': 0, 'changed_files': 0, 'repo_name': None}
    if not state:
        return result
    result.update(indexed_commit=state['indexed_commit'], repo_name=state['repo_name'])
    root = Path(state['repo_path'])
    head = git(root, 'rev-parse', 'HEAD', optional=True)
    result['current_head'] = head or None
    old = state['indexed_commit']
    if not head or not old:
        return result
    try:
        result['commits_behind'] = int(git(root, 'rev-list', '--count', f'{old}..{head}'))
        result['changed_files'] = len(changed_files(root, old, head))
        dirty = git(root, 'diff', '--name-only', '-z', 'HEAD', optional=True).split('\0')
        untracked = git(root, 'ls-files', '--others', '--exclude-standard', '-z', optional=True).split('\0')
        pending = {path for path in dirty + untracked if path and supported_path(root, path)}
        result['changed_files'] += len(pending)
        result['stale'] = (head != old or bool(pending) or state['embed_model'] != config.EMBED_MODEL
                           or state.get('incomplete', False) or state.get('payload_version') != config.INDEX_PAYLOAD_VERSION)
    except ValueError:
        pass  # old commit no longer exists, so a full index is needed
    return result


def _clean_tree(root: Path) -> None:
    """Reject uncommitted supported changes to keep commit-based state accurate."""
    dirty = git(root, 'diff', '--name-only', '-z', 'HEAD', optional=True).split('\0')
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '-z', optional=True).split('\0')
    if any(supported_path(root, path) for path in dirty + untracked):
        raise ValueError('Commit supported-file changes before indexing/updating so the index matches HEAD.')


def _verify_snapshot(root: Path, head: str | None) -> None:
    """Do not certify a commit if the checkout changed while embedding ran."""
    if head:
        if git(root, 'rev-parse', 'HEAD', optional=True) != head:
            raise ValueError('Repository HEAD changed during indexing. Run update again.')
        _clean_tree(root)


def repository_identity(source: str, root: Path) -> str:
    """Prefer the origin remote; local-only repositories use local/<directory>."""
    remotes = git(root, 'remote', optional=True).splitlines()
    remote = 'origin' if 'origin' in remotes else (remotes[0] if remotes else None)
    url = git(root, 'remote', 'get-url', remote, optional=True) if remote else ''
    return infer_repo(source, root, url)


def check_repo_size(root: Path) -> None:
    """Count disk bytes, including Git history, without reading file contents."""
    size = 0
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                size += path.stat().st_size
            if size > config.MAX_REPO_BYTES:
                raise ValueError(f'Repository exceeds disk size limit ({config.MAX_REPO_BYTES / 1024**2:g} MB).')


def check_chunk_count(count: int) -> None:
    """Reject oversized indexes before embedding or mutating existing collections."""
    if count > config.MAX_REPO_CHUNKS:
        raise ValueError(f'Repository has {count:,} chunks; maximum is {config.MAX_REPO_CHUNKS:,}.')


def full_index(source: str, *, store=None, embedder=None, repo=None, progress=None, write_lock=None) -> dict:
    """Replace the collection and record repository identity and embedding model."""
    started = time.perf_counter()
    root = resolve_repo(source)
    selected = repo_id(repo) if repo else repository_identity(source, root)
    if store is None:
        from .store import VectorStore
        store = VectorStore(selected)
        try:
            return full_index(str(root), store=store, embedder=embedder, repo=selected,
                              progress=progress, write_lock=write_lock)
        finally:
            store.client.close()
    store.repo_id = selected
    check_repo_size(root)
    head = git(root, 'rev-parse', 'HEAD', optional=True) or None
    if head:
        _clean_tree(root)
    if progress:
        progress('chunking', '')
    chunks = annotate_chunks(collect_chunks(root), file_metadata(root, head))
    previous = load_state(selected)
    preserve_history = bool(previous and previous['embed_model'] == config.EMBED_MODEL and store.count())
    history_count = len(store.all_payloads(sources='history')) if preserve_history else 0
    check_chunk_count(len(chunks) + history_count)
    if embedder is None:
        from .store import Embedder
        embedder = Embedder()
    removed = store.count() - history_count
    if progress:
        progress('embedding', f'0/{len(chunks)} chunks')
    vectors = embedder.encode([chunk.embedding_text() for chunk in chunks]) if chunks else []
    _verify_snapshot(root, head)
    with write_lock if write_lock is not None else nullcontext():
        _mark_incomplete(selected)
        if preserve_history and store.count():
            store.delete_code()
        else:
            store.reset(embedder.dim)
        store.upsert(chunks, vectors)
        state = save_state(root, store, head)
        if selected == 'psf/requests' and store.client.collection_exists('gitcontext'):
            store.client.delete_collection('gitcontext')
    return {'mode': 'full', 'commits_behind': 0, 'files_added': len({c.file_path for c in chunks}),
            'files_modified': 0, 'files_deleted': 0, 'files_renamed': 0,
            'chunks_removed': removed, 'chunks_added': len(chunks),
            'seconds': time.perf_counter() - started, 'state': state}


def update_index(source: str | None = None, *, store=None, embedder=None, repo=None) -> dict:
    """Pull managed clones, then replace only chunks touched by the commit diff."""
    started = time.perf_counter()
    if source is None:
        state = load_state(repo)
        if not state:
            raise ValueError(f'Repository {repo or config.DEFAULT_REPO} is not indexed. Run index first.')
        source = state['repo_path']
    root = resolve_repo(source)
    selected = repo_id(repo) if repo else repository_identity(source, root)
    if store is None:
        from .store import VectorStore
        store = VectorStore(selected)
        try:
            return update_index(str(root), store=store, embedder=embedder, repo=selected)
        finally:
            store.client.close()
    store.repo_id = selected
    _clean_tree(root)
    remotes = git(root, 'remote', optional=True)
    if root.parent == config.REPOS_DIR.resolve() and remotes:
        git(root, 'pull', '--ff-only')
    _clean_tree(root)
    check_repo_size(root)
    state = load_state(selected)
    head = git(root, 'rev-parse', 'HEAD', optional=True)
    if (not state or Path(state['repo_path']).resolve() != root or
            state['embed_model'] != config.EMBED_MODEL or state.get('incomplete') or not head or not state['indexed_commit']
            or state.get('payload_version') != config.INDEX_PAYLOAD_VERSION):
        return full_index(str(root), store=store, embedder=embedder, repo=selected)
    old = state['indexed_commit']
    ancestor = subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', old, head],
                              capture_output=True).returncode == 0
    if not ancestor:
        return full_index(str(root), store=store, embedder=embedder, repo=selected)
    changes = changed_files(root, old, head)
    metadata = file_metadata(root, head) if changes else {}
    behind = int(git(root, 'rev-list', '--count', f'{old}..{head}'))
    if not store.count():
        return full_index(str(root), store=store, embedder=embedder, repo=selected)
    summary = {'mode': 'incremental', 'commits_behind': behind, 'files_added': 0,
               'files_modified': 0, 'files_deleted': 0, 'files_renamed': 0,
               'chunks_removed': 0, 'chunks_added': 0}
    plans = []
    for kind, old_path, new_path in changes:
        delete_old = kind != 'C' and supported_path(root, old_path)
        target = new_path if kind in {'R', 'C'} else old_path
        add_new = kind != 'D' and supported_path(root, target)
        if not delete_old and not add_new:
            continue
        counter = {'A': 'files_added', 'C': 'files_added', 'D': 'files_deleted', 'R': 'files_renamed'}.get(kind, 'files_modified')
        summary[counter] += 1
        path = root / target
        chunks = []
        if add_new and path.is_file() and path.stat().st_size <= config.MAX_FILE_BYTES:
            try:
                chunks = annotate_chunks(chunk_file(target, path.read_text(encoding='utf-8')), metadata)
            except UnicodeDecodeError:
                pass
        plans.append((delete_old, old_path, chunks))
    counts = {}
    for _, payload in store.all_payloads():
        path = payload['file_path']
        counts[path] = counts.get(path, 0) + 1
    planned_count = store.count() + sum(len(chunks) - (counts.get(path, 0) if delete else 0)
                                      for delete, path, chunks in plans)
    check_chunk_count(planned_count)
    store.ensure_file_index()
    if plans:
        _mark_incomplete(selected)
    for delete_old, old_path, chunks in plans:
        if chunks and embedder is None:
            from .store import Embedder
            embedder = Embedder()
        # Compute embeddings before deletion so transient model errors preserve old chunks.
        vectors = embedder.encode([c.embedding_text() for c in chunks]) if chunks else []
        if delete_old:
            summary['chunks_removed'] += store.delete_by_file(old_path)
        store.upsert(chunks, vectors)
        summary['chunks_added'] += len(chunks)
    _verify_snapshot(root, head)
    summary['state'] = save_state(root, store, head)
    summary['seconds'] = time.perf_counter() - started
    return summary


def print_summary(summary: dict) -> None:
    """Print comparable CLI summaries for full and incremental runs."""
    print(f"Mode: {summary['mode']}; commits behind: {summary['commits_behind']}")
    print(f"Files added/modified/deleted/renamed: {summary['files_added']}/{summary['files_modified']}/"
          f"{summary['files_deleted']}/{summary['files_renamed']}")
    print(f"Chunks removed/added: {summary['chunks_removed']}/{summary['chunks_added']}")
    print(f"Indexed {summary['state']['repo_name']}: {summary['state']['chunk_count']} chunks")
    print(f"Time taken: {summary['seconds']:.2f}s")
