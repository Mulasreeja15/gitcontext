"""FastAPI service with a single long-lived retriever and grounded answer streams."""
from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from contextlib import asynccontextmanager
from threading import Lock, RLock, Event, Thread

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.responses import JSONResponse, HTMLResponse
from collections import deque
from pydantic import BaseModel, Field, field_validator

from . import answer, config
from .indexing import index_status, load_state, update_index
from datetime import date
from typing import Literal
from .repositories import repo_id, list_repos, remove_repo
from .service_indexing import github_repo, index_repository, language_profile
from .history import is_shallow


class SearchRequest(BaseModel):
    """Shared retrieval parameters for search and both answer routes."""
    query: str
    repo: str | None = None
    k: int = Field(default=6, ge=1, le=config.API_MAX_K)
    bm25: bool = False
    rerank: bool = False
    rewrite: bool = False
    include_tests: bool = False
    since: str | None = None
    author: str | None = Field(default=None, min_length=1)
    sources: Literal['code', 'history', 'all'] = 'code'

    @field_validator('repo')
    @classmethod
    def repository(cls, value):
        return repo_id(value) if value is not None else None

    @field_validator('since')
    @classmethod
    def since_date(cls, value):
        """Require the same canonical date format as the CLI."""
        if value is not None and date.fromisoformat(value).isoformat() != value:
            raise ValueError('since must be YYYY-MM-DD')
        return value

    @field_validator('query')
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        """Reject empty/whitespace queries before loading or querying services."""
        if not value.strip():
            raise ValueError('query must not be empty')
        if config.PUBLIC_DEMO and len(value) > config.PUBLIC_MAX_QUERY_CHARS:
            raise ValueError(f'Questions must be at most {config.PUBLIC_MAX_QUERY_CHARS} characters in the public demo.')
        return value.strip()


class HistoryMetadata(BaseModel):
    sha: str | None = None
    number: int | None = None
    title: str
    author: str
    date: str
    url: str | None = None
    merged_at: str | None = None
    changed_files: list[str] = []
    labels: list[str] = []


class Source(BaseModel):
    """Source order corresponds to one-based inline answer citations."""
    file_path: str
    start_line: int
    end_line: int
    kind: str
    name: str
    score: float
    is_test: bool
    text: str
    last_commit_sha: str | None = None
    last_author: str | None = None
    last_modified: str | None = None
    history: HistoryMetadata | None = None


class HealthResponse(BaseModel):
    status: str = 'ok'
    chunks_indexed: int
    repo_name: str | None = None
    stale: bool
    indexed_commit: str | None = None


class ReindexResponse(BaseModel):
    mode: str
    commits_behind: int
    files_added: int
    files_modified: int
    files_deleted: int
    files_renamed: int
    chunks_removed: int
    chunks_added: int
    seconds: float
    state: dict


class SearchResponse(BaseModel):
    hits: list[Source]
    timings: dict[str, float]


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    timings: dict[str, float]


class SourcesEvent(BaseModel):
    sources: list[Source]
    timings: dict[str, float]


class TokenEvent(BaseModel):
    token: str


class DoneEvent(BaseModel):
    timings: dict[str, float]


class ErrorEvent(BaseModel):
    detail: str


class StatusEvent(BaseModel):
    message: str


def create_retriever():
    """Import expensive dependencies only during application startup."""
    from .retriever import Retriever
    return Retriever()


def create_repo_retriever(identity, base):
    """Create repo-local caches without opening another client or loading models."""
    from .retriever import Retriever
    retriever = Retriever(identity, client=base.store.client, embedder=base.embedder)
    retriever._reranker = base._reranker
    return retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once per process and close Qdrant cleanly on service shutdown."""
    app.state.retriever = create_retriever()
    app.state.retrievers = {config.DEFAULT_REPO: app.state.retriever}
    app.state.retrieval_lock = RLock()
    app.state.job_lock = Lock()
    app.state.status_lock = Lock()
    app.state.index_job = dict(state='idle', repo=None, progress='', error=None)
    app.state.index_thread = None
    app.state.rate_limiter = PublicRateLimiter()
    try:
        yield
    finally:
        if app.state.index_thread is not None:
            import asyncio
            await asyncio.to_thread(app.state.index_thread.join)
        app.state.retriever.store.client.close()


app = FastAPI(title='GitContext', lifespan=lifespan)


class PublicRateLimiter:
    """Thread-safe sliding window shared by search and answer endpoints.

    Expired clients are discarded to bound memory by recent active clients.
    Client addresses come from ASGI, never arbitrary forwarded headers here.
    """
    def __init__(self):
        self.clients = {}
        self.lock = Lock()

    def allow(self, client, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            cutoff = now - config.PUBLIC_RATE_SECONDS
            for address in list(self.clients):
                queue = self.clients[address]
                while queue and queue[0] <= cutoff:
                    queue.popleft()
                if not queue:
                    del self.clients[address]
            queue = self.clients.setdefault(client, deque())
            if len(queue) >= config.PUBLIC_RATE_REQUESTS:
                return False
            queue.append(now)
            return True


@app.middleware('http')
async def public_demo_guard(request: Request, call_next):
    """Reject public mutations before validation, threads, or model work."""
    if config.PUBLIC_DEMO:
        path, method = request.url.path, request.method
        if ((method == 'POST' and path in {'/index', '/reindex'}) or
                (method == 'DELETE' and path.startswith('/repos/'))):
            return JSONResponse({'detail': 'Repository changes are disabled in the public demo.'}, status_code=403)
        if method == 'POST' and path in {'/search', '/ask', '/ask/stream'}:
            client = request.client.host if request.client else 'unknown'
            if not request.app.state.rate_limiter.allow(client):
                return JSONResponse({'detail': 'Too many requests, try again in a minute.'}, status_code=429,
                                    headers={'Retry-After': str(config.PUBLIC_RATE_SECONDS)})
    return await call_next(request)


@app.get('/settings')
def public_settings():
    """Expose only safe UI capabilities, never environment variables or keys."""
    return {'public_demo': config.PUBLIC_DEMO,
            'max_query_chars': config.PUBLIC_MAX_QUERY_CHARS if config.PUBLIC_DEMO else None}


class IndexRequest(BaseModel):
    repo_url: str

    @field_validator('repo_url')
    @classmethod
    def valid_url(cls, value):
        github_repo(value)
        return value


class IndexStatus(BaseModel):
    state: str
    repo: str | None
    progress: str
    error: str | None


class RepoResponse(BaseModel):
    repo: str
    chunk_count: int
    indexed_commit: str | None
    stale: bool
    language_warning: str | None = None
    history_notice: str | None = None


def _job_status(app):
    """Read job status independently of long-running model or database work."""
    with app.state.status_lock:
        return dict(app.state.index_job)


def _repo_retriever(app, identity):
    """Caller holds retrieval_lock while accessing the repository cache registry."""
    if identity not in app.state.retrievers:
        app.state.retrievers[identity] = create_repo_retriever(identity, app.state.retriever)
    return app.state.retrievers[identity]


def _index_job(app, url, identity):
    """Run one job, publishing status without leaking upstream error details."""
    def progress(state, value):
        with app.state.status_lock:
            app.state.index_job.update(state=state, progress=value)
    retriever = None
    try:
        with app.state.retrieval_lock:
            retriever = _repo_retriever(app, identity)
        index_repository(url, identity, retriever, app.state.retrieval_lock, progress)
        progress('done', _job_status(app)['progress'])
    except (Exception, SystemExit) as error:
        message = str(error) if isinstance(error, ValueError) else 'Indexing failed. Check GitHub access, disk space, and the repository URL.'
        with app.state.status_lock:
            app.state.index_job.update(state='error', error=message)
    finally:
        if retriever is not None:
            # Publication may have failed after mutating the collection.
            with app.state.retrieval_lock:
                retriever.invalidate_index()
        app.state.job_lock.release()


@app.get('/repos', response_model=list[RepoResponse])
def repos(request: Request):
    """List repository collections through the one shared storage connection."""
    with request.app.state.retrieval_lock:
        rows = list_repos(client=request.app.state.retriever.store.client)
        for row in rows:
            state = load_state(row['repo']) or {}
            profile = state
            if 'language_warning' not in state and state.get('repo_path'):
                profile = language_profile(Path(state['repo_path']))
            row['language_warning'] = profile.get('language_warning')
            row['history_notice'] = None
            if state.get('repo_path'):
                try:
                    shallow = is_shallow(Path(state['repo_path']))
                    if shallow or not state.get('history_chunks'):
                        option = ' --unshallow' if shallow else ''
                        prefix = 'Shallow clone: earlier commit history is unavailable. ' if shallow else 'History is not indexed. '
                        row['history_notice'] = (prefix + 'Stop the server and run python main.py index-history --repo '
                                                 + row['repo'] + option + '.')
                except ValueError:
                    row['history_notice'] = 'History unavailable: repository checkout cannot be read.'
        return rows


@app.post('/index', status_code=202, response_model=IndexStatus)
def start_index(body: IndexRequest, request: Request):
    """Reserve a single background job before returning accepted status."""
    app = request.app
    identity = github_repo(body.repo_url)
    if not app.state.job_lock.acquire(blocking=False):
        raise HTTPException(409, 'An indexing job is already running.')
    with app.state.status_lock:
        app.state.index_job = dict(state='cloning', repo=identity, progress='', error=None)
    snapshot = _job_status(app)
    app.state.index_thread = Thread(target=_index_job, args=(app, body.repo_url, identity), daemon=True)
    try:
        app.state.index_thread.start()
    except Exception:
        app.state.index_thread = None
        app.state.job_lock.release()
        with app.state.status_lock:
            app.state.index_job.update(state='error', error='Could not start indexing job.')
        raise HTTPException(503, 'Could not start indexing job.') from None
    return snapshot


@app.get('/index/status', response_model=IndexStatus)
def indexing_status(request: Request):
    return _job_status(request.app)


@app.delete('/repos/{owner}/{name}')
def delete_repo(owner: str, name: str, request: Request):
    """Remove one collection and caches, never the checkout or shared client."""
    try:
        identity = repo_id(owner + '/' + name)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    app = request.app
    if not app.state.job_lock.acquire(blocking=False):
        raise HTTPException(409, 'Wait for the indexing job to finish before removing repositories.')
    try:
        with app.state.retrieval_lock:
            try:
                remove_repo(identity, client=app.state.retriever.store.client)
            except ValueError as error:
                raise HTTPException(404, str(error)) from None
            removed = app.state.retrievers.pop(identity, None)
            if removed is not None:
                removed.invalidate_index()
            if identity == config.DEFAULT_REPO:
                app.state.retrievers[identity] = app.state.retriever
        return {'removed': identity}
    finally:
        app.state.job_lock.release()


@app.get('/', response_class=FileResponse)
def home():
    """Serve the self-contained UI without a static-file server or build step."""
    if config.PUBLIC_DEMO:
        # Hide management before JavaScript fetches capabilities (no initial flash).
        html = config.UI_PATH.read_text(encoding='utf-8')
        for identity in ('open-repository', 'index-button'):
            html = html.replace(f'id="{identity}"', f'id="{identity}" hidden disabled')
        return HTMLResponse(html)
    return FileResponse(config.UI_PATH, media_type='text/html')


def _retrieve(body: SearchRequest, request: Request):
    """Serialize retrieval to protect lazy model initialization and local storage."""
    from .retriever import RetrievalOptions
    opts = RetrievalOptions(bm25=body.bm25, rerank=body.rerank,
                            penalize_tests=not body.include_tests, rewrite=body.rewrite,
                            since=body.since, author=body.author)
    opts.sources = body.sources
    identity = body.repo or config.DEFAULT_REPO
    job = _job_status(request.app)
    if job['repo'] == identity and job['state'] in {'cloning', 'chunking', 'embedding'}:
        raise HTTPException(409, 'This repository is currently indexing.')
    with request.app.state.retrieval_lock:
        retriever = _repo_retriever(request.app, identity)
        if retriever.store.count() == 0:
            raise HTTPException(503, 'Index is empty. Run main.py index before searching.')
        try:
            retriever.prepare(opts)
            if opts.rerank:
                # Reuse the loaded cross-encoder across repo-local caches too.
                for other in request.app.state.retrievers.values():
                    other._reranker = retriever._reranker
            return retriever.retrieve(body.query, body.k, opts)
        except (SystemExit, requests.RequestException):
            raise HTTPException(503, 'Query rewriting is unavailable. Check Groq configuration or retry later.') from None


def _sources(hits) -> list[Source]:
    """Keep source order stable while exposing only public source metadata."""
    from .retriever import is_test
    return [Source(**{field: payload[field] for field in
                    ('file_path', 'start_line', 'end_line', 'kind', 'name', 'text')},
                   score=score, is_test=is_test(payload),
                   history=HistoryMetadata(**payload) if payload['kind'] in {'commit', 'pr'} else None,
                   **{field: payload.get(field) for field in ('last_commit_sha', 'last_author', 'last_modified')}) for score, payload in hits]


def _answer_timings(timings: dict, started: float) -> dict[str, float]:
    """Include answer generation in total latency while retaining retrieval timing."""
    elapsed = (time.perf_counter() - started) * 1000
    return {**timings, 'retrieval_ms': timings.get('total_ms', 0.0),
            'answer_ms': elapsed, 'total_ms': timings.get('total_ms', 0.0) + elapsed}


@app.get('/health', response_model=HealthResponse)
def health(request: Request, repo: str | None = None):
    """Report readiness and count without scanning every payload."""
    startup = getattr(request.app.state, 'startup', None)
    if startup and startup['state'] != 'ready':
        return HealthResponse(status=startup['state'], chunks_indexed=0, stale=False)
    with request.app.state.retrieval_lock:
        try:
            identity = repo_id(repo or config.DEFAULT_REPO)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        count = _repo_retriever(request.app, identity).store.count()
        status = index_status(identity)
    return HealthResponse(chunks_indexed=count, repo_name=status['repo_name'],
                          stale=status['stale'], indexed_commit=status['indexed_commit'])


@app.post('/reindex', response_model=ReindexResponse)
def reindex(request: Request):
    """Update in-process using the existing Qdrant connection and loaded embedder."""
    if not request.app.state.job_lock.acquire(blocking=False):
        raise HTTPException(409, 'An indexing job is already running.')
    try:
        with request.app.state.retrieval_lock:
            state = load_state()
            if not state:
                raise HTTPException(409, 'No index state. Run main.py index <repo> first.')
            retriever = request.app.state.retriever
            try:
                return update_index(state['repo_path'], store=retriever.store, embedder=retriever.embedder)
            except (ValueError, SystemExit, OSError):
                raise HTTPException(409, 'Update failed. Check repository history, working tree, and remote access.') from None
            finally:
                retriever.invalidate_index()
    finally:
        request.app.state.job_lock.release()


@app.post('/search', response_model=SearchResponse)
def search(body: SearchRequest, request: Request):
    """Retrieve ranked chunks from the shared retriever."""
    result = _retrieve(body, request)
    return SearchResponse(hits=_sources(result.hits), timings=result.timings)


@app.post('/ask', response_model=AskResponse)
def ask(body: SearchRequest, request: Request):
    """Generate a complete cited answer, mapping unavailable Groq to HTTP 503."""
    result = _retrieve(body, request)
    started = time.perf_counter()
    try:
        generated = answer.generate_answer(body.query, result.hits)
    except (SystemExit, requests.RequestException):
        raise HTTPException(503, 'Answer generation is unavailable. Check Groq configuration or retry later.') from None
    return AskResponse(answer=generated, sources=_sources(result.hits),
                       timings=_answer_timings(result.timings, started))


def _event(name: str, value: BaseModel) -> str:
    """Encode safe, single-line SSE JSON, including tokens containing newlines."""
    return f'event: {name}\ndata: {value.model_dump_json()}\n\n'


@app.post('/ask/stream', response_class=StreamingResponse)
def ask_stream(body: SearchRequest, request: Request):
    """Send sources first, then content deltas and final timing or a stream error."""
    result = _retrieve(body, request)
    started = time.perf_counter()
    def events():
        messages = Queue()
        stopped = Event()

        def generate():
            """Keep the SSE connection responsive while Groq backs off on 429."""
            tokens = None
            try:
                tokens = answer.stream_answer(body.query, result.hits,
                    on_retry=lambda wait: messages.put(('status', StatusEvent(
                        message='Groq is rate limiting, retrying...'))))
                for token in tokens:
                    if stopped.is_set():
                        break
                    messages.put(('token', TokenEvent(token=token)))
                messages.put(('done', DoneEvent(timings=_answer_timings(result.timings, started))))
            except (SystemExit, Exception):
                messages.put(('error', ErrorEvent(detail='Answer generation is unavailable or interrupted. Try Search only or retry.')))
            finally:
                close = getattr(tokens, 'close', None)
                if close is not None:
                    close()

        try:
            yield _event('sources', SourcesEvent(sources=_sources(result.hits), timings=result.timings))
            Thread(target=generate, daemon=True).start()
            while True:
                name, value = messages.get()
                yield _event(name, value)
                if name in {'done', 'error'}:
                    break
        finally:
            stopped.set()

    return StreamingResponse(events(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
