"""Local commit history and optional cached GitHub discussions as cited sources."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from . import config
from .chunker import HistoryChunk, _split
from .indexing import git, load_state, _write_state, check_repo_size, check_chunk_count
from .ingest import supported_path
from .repositories import repo_id


def is_shallow(root: Path) -> bool:
    """A shallow clone cannot provide reliable earlier commit history."""
    return git(root, 'rev-parse', '--is-shallow-repository').strip() == 'true'


def parse_commits(output: str, root: Path, identity: str) -> list[HistoryChunk]:
    """Parse one NUL-delimited numstat pass; author names never include emails."""
    chunks = []
    for record in output.split('\x1e'):
        if not record.strip():
            continue
        header, _, stats = record.partition('\0')
        sha, author, date, message = header.split('\x1f', 3)
        files, stat_lines = [], []
        for field in stats.split('\0'):
            parts = field.lstrip('\n').split('\t', 2)
            if len(parts) == 3:
                added, removed, path = parts
                files.append(path)
                stat_lines.append(f'{path}: +{added} -{removed}')
        if not any(supported_path(root, path) for path in files):
            continue
        # Preserve attribution names in trailers without retaining email addresses.
        message = re.sub(r'<[^<>\s]+@[^<>\s]+>', '', message).strip()
        title = message.splitlines()[0]
        url = f'https://github.com/{identity}/commit/{sha}' if not identity.startswith('local/') else None
        text = f'Commit {sha}\nAuthor: {author}\nDate: {date}\n\n{message}\n\nChanged files (diff stat):\n' + '\n'.join(stat_lines)
        chunk = HistoryChunk(file_path=f'commit::{sha}', start_line=1, end_line=len(text.splitlines()),
            kind='commit', name=title, language='text', text=text, sha=sha, title=title,
            author=author, date=date, url=url, changed_files=tuple(files),
            last_commit_sha=sha, last_author=author, last_modified=date)
        chunks.extend(_split(chunk))
    return chunks


def commit_chunks(root: Path, identity: str) -> list[HistoryChunk]:
    """Only the latest N commits touching supported paths become sources."""
    output = git(root, 'log', f'-{config.HISTORY_MAX_COMMITS}', '--no-renames',
        '--diff-merges=first-parent', '-z', '--numstat', '--format=%x1e%H%x1f%an%x1f%aI%x1f%B%x00', 'HEAD', '--')
    return parse_commits(output, root, identity)


def pr_chunks(pr: dict, files: list[dict], comments: list[dict], identity: str) -> list[HistoryChunk]:
    """Split a merged PR discussion with stable PR metadata on each excerpt."""
    number, title = pr['number'], pr['title']
    author = (pr.get('user') or {}).get('login', 'unknown')
    merged = pr['merged_at']
    paths = tuple(file['filename'] for file in files)
    labels = tuple(label['name'] for label in pr.get('labels', []))
    url = f'https://github.com/{identity}/pull/{number}'
    lines = [f'PR #{number}: {title}', f'Author: {author}', f'Merged: {merged}',
             'Labels: ' + ', '.join(labels), '', pr.get('body') or '', '',
             'Changed files: ' + ', '.join(paths), '', 'Discussion:']
    for comment in comments:
        body = comment.get('body') or ''
        if body.strip():
            login = (comment.get('user') or {}).get('login', 'unknown')
            location = f" [{comment['path']}]" if comment.get('path') else ''
            lines.append(f"{login} ({comment.get('created_at') or comment.get('submitted_at') or ''}){location}: {body}")
    text = '\n'.join(lines)
    chunk = HistoryChunk(file_path=f'pr::{number}', start_line=1, end_line=len(text.splitlines()),
        kind='pr', name=title, language='text', text=text, number=number, title=title,
        author=author, date=merged, merged_at=merged, url=url, changed_files=paths,
        labels=labels, last_author=author, last_modified=merged)
    return _split(chunk)


class GitHubClient:
    """Paginated REST data cached per repository; credentials never enter cache/logs."""
    def __init__(self, identity: str, token: str):
        self.identity, self.token = repo_id(identity), token
        self.directory = config.GITHUB_CACHE_DIR / self.identity.replace('/', '__')
        self.wait_until = 0.0

    def _wait(self, seconds: float):
        if seconds > config.GITHUB_MAX_RATE_WAIT:
            raise ValueError('GitHub rate limit exceeded. Retry later; completed responses are cached.')
        if seconds > 0:
            time.sleep(seconds)

    def get(self, endpoint: str, page: int) -> list[dict]:
        """Cache response JSON only, and honor rate/reset headers before requests."""
        key = hashlib.sha256(f'{endpoint}:{page}:{config.GITHUB_PAGE_SIZE}'.encode()).hexdigest()
        path = self.directory / (key + '.json')
        if path.exists() and time.time() - path.stat().st_mtime < config.GITHUB_CACHE_SECONDS:
            try:
                value = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(value, list):
                    return value
            except (OSError, ValueError):
                pass
        params = {'page': page, 'per_page': config.GITHUB_PAGE_SIZE}
        if endpoint == 'pulls':
            params.update(state='closed', sort='updated', direction='desc')
        for _ in range(config.GITHUB_MAX_RETRIES):
            self._wait(max(0, self.wait_until - time.time()))
            response = requests.get(f'{config.GITHUB_API_URL}/repos/{self.identity}/{endpoint}',
                params=params, headers={'Authorization': f'Bearer {self.token}',
                'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'},
                timeout=config.GITHUB_TIMEOUT, allow_redirects=False)
            try:
                remaining = response.headers.get('X-RateLimit-Remaining')
                retry = response.headers.get('Retry-After')
                if remaining == '0' or retry or response.status_code == 429:
                    if retry:
                        try:
                            wait = float(retry)
                        except ValueError:
                            wait = parsedate_to_datetime(retry).timestamp() - time.time()
                    else:
                        wait = max(1, float(response.headers.get('X-RateLimit-Reset', time.time() + 1)) - time.time())
                    self.wait_until = time.time() + max(0, wait)
                    if response.status_code in {403, 429}:
                        continue
                if response.status_code != 200:
                    raise ValueError(f'GitHub returned HTTP {response.status_code}; PR indexing skipped. Check access and retry.')
                value = response.json()
                if not isinstance(value, list):
                    raise ValueError('Unexpected GitHub response; PR indexing skipped.')
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
                temporary.replace(path)
                return value
            finally:
                response.close()
        raise ValueError('GitHub remains rate limited; retry later. Cached responses were preserved.')

    def pages(self, endpoint: str):
        """Request successive pages until a short page terminates pagination."""
        page = 1
        while True:
            values = self.get(endpoint, page)
            yield from values
            if len(values) < config.GITHUB_PAGE_SIZE:
                return
            page += 1

    def discussions(self):
        """Only merged PRs, including review bodies, inline and issue comments."""
        chunks, count = [], 0
        for pr in self.pages('pulls'):
            if not pr.get('merged_at'):
                continue
            number = pr['number']
            files = list(self.pages(f'pulls/{number}/files'))
            comments = (list(self.pages(f'issues/{number}/comments')) +
                        list(self.pages(f'pulls/{number}/comments')) +
                        list(self.pages(f'pulls/{number}/reviews')))
            chunks.extend(pr_chunks(pr, files, comments, self.identity))
            count += 1
            if count >= config.PR_MAX:
                break
        return chunks


def index_history(identity: str, *, unshallow=False, commits_only=False, store=None, embedder=None):
    """Replace history sources, leaving code and other repositories unchanged."""
    identity = repo_id(identity)
    state = load_state(identity)
    if not state:
        raise ValueError(f'Repository {identity} is not indexed. Index its code first.')
    root = Path(state['repo_path'])
    if state['embed_model'] != config.EMBED_MODEL or state.get('incomplete'):
        raise ValueError('Rebuild the code index before indexing history: model changed or index is incomplete.')
    check_repo_size(root)
    messages = []
    if is_shallow(root):
        if unshallow:
            subprocess.run(['git', '-C', str(root), 'fetch', '--unshallow'],
                           check=True, capture_output=True, timeout=config.HISTORY_FETCH_TIMEOUT)
            check_repo_size(root)
        else:
            messages.append('Shallow clone: commit history skipped. Run index-history --unshallow to fetch full history.')
    commits = None if is_shallow(root) else commit_chunks(root, identity)
    prs = None
    token = os.getenv(config.GITHUB_KEY_NAME)
    if commits_only:
        messages.append('PR discussions skipped (--commits-only).')
    elif not token:
        messages.append('PR discussions skipped: GITHUB_TOKEN is not set. Export it yourself and rerun index-history.')
    elif identity.startswith('local/'):
        messages.append('PR discussions skipped: no GitHub repository identity.')
    else:
        try:
            prs = GitHubClient(identity, token).discussions()
        except (ValueError, requests.RequestException) as error:
            messages.append(str(error) if isinstance(error, ValueError) else 'GitHub request failed; PR discussions skipped. Cached data was preserved.')
    if store is None:
        from .store import VectorStore
        store = VectorStore(identity)
        try:
            return _publish_history(state, commits, prs, messages, store, embedder)
        finally:
            store.client.close()
    return _publish_history(state, commits, prs, messages, store, embedder)


def _publish_history(state, commits, prs, messages, store, embedder):
    """Preserve previous PR data when fetching is unavailable, then publish state."""
    chunks = (commits or []) + (prs or [])
    old = store.all_payloads()
    removed = [(pid, p) for pid, p in old if (commits is not None and p['kind'] == 'commit') or (prs is not None and p['kind'] == 'pr')]
    check_chunk_count(len(old) - len(removed) + len(chunks))
    if chunks and embedder is None:
        from .store import Embedder
        embedder = Embedder()
    vectors = embedder.encode([chunk.embedding_text() for chunk in chunks]) if chunks else []
    if removed or chunks:
        _write_state({**state, 'incomplete': True})
        for path in {p['file_path'] for _, p in removed}:
            store.delete_by_file(path)
        store.upsert(chunks, vectors)
    history_count = len(store.all_payloads(sources='history'))
    updated = {**state, 'chunk_count': store.count(), 'history_chunks': history_count, 'history_indexed_at': datetime.now(timezone.utc).isoformat(),
               'history_commit': git(Path(state['repo_path']), 'rev-parse', 'HEAD'), 'incomplete': state.get('incomplete', False)}
    _write_state(updated)
    return {'commits': len({chunk.sha for chunk in commits or []}), 'prs': len({chunk.number for chunk in prs or []}),
            'chunks_added': len(chunks), 'chunks_removed': len(removed), 'messages': messages}
