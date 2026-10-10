"""FastAPI service with a single long-lived retriever and grounded answer streams."""
from __future__ import annotations

import time
from queue import Queue
from contextlib import asynccontextmanager
from threading import Lock, Event, Thread

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

from . import answer, config
from .indexing import index_status, load_state, update_index
from datetime import date


class SearchRequest(BaseModel):
    """Shared retrieval parameters for search and both answer routes."""
    query: str
    k: int = Field(default=6, ge=1, le=config.API_MAX_K)
    bm25: bool = False
    rerank: bool = False
    rewrite: bool = False
    include_tests: bool = False
    since: str | None = None
    author: str | None = Field(default=None, min_length=1)

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
        return value.strip()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once per process and close Qdrant cleanly on service shutdown."""
    app.state.retriever = create_retriever()
    app.state.retrieval_lock = Lock()
    try:
        yield
    finally:
        app.state.retriever.store.client.close()


app = FastAPI(title='GitContext', lifespan=lifespan)


@app.get('/', response_class=FileResponse)
def home():
    """Serve the self-contained UI without a static-file server or build step."""
    return FileResponse(config.UI_PATH, media_type='text/html')


def _retrieve(body: SearchRequest, request: Request):
    """Serialize retrieval to protect lazy model initialization and local storage."""
    from .retriever import RetrievalOptions
    opts = RetrievalOptions(bm25=body.bm25, rerank=body.rerank,
                            penalize_tests=not body.include_tests, rewrite=body.rewrite,
                            since=body.since, author=body.author)
    retriever = request.app.state.retriever
    with request.app.state.retrieval_lock:
        if retriever.store.count() == 0:
            raise HTTPException(503, 'Index is empty. Run main.py index before searching.')
        try:
            retriever.prepare(opts)
            return retriever.retrieve(body.query, body.k, opts)
        except (SystemExit, requests.RequestException):
            raise HTTPException(503, 'Query rewriting is unavailable. Check Groq configuration or retry later.') from None


def _sources(hits) -> list[Source]:
    """Keep source order stable while exposing only public source metadata."""
    from .retriever import is_test
    return [Source(**{field: payload[field] for field in
                    ('file_path', 'start_line', 'end_line', 'kind', 'name', 'text')},
                   score=score, is_test=is_test(payload),
                   **{field: payload.get(field) for field in ('last_commit_sha', 'last_author', 'last_modified')}) for score, payload in hits]


def _answer_timings(timings: dict, started: float) -> dict[str, float]:
    """Include answer generation in total latency while retaining retrieval timing."""
    elapsed = (time.perf_counter() - started) * 1000
    return {**timings, 'retrieval_ms': timings.get('total_ms', 0.0),
            'answer_ms': elapsed, 'total_ms': timings.get('total_ms', 0.0) + elapsed}


@app.get('/health', response_model=HealthResponse)
def health(request: Request):
    """Report readiness and count without scanning every payload."""
    with request.app.state.retrieval_lock:
        count = request.app.state.retriever.store.count()
        status = index_status()
    return HealthResponse(chunks_indexed=count, repo_name=status['repo_name'],
                          stale=status['stale'], indexed_commit=status['indexed_commit'])


@app.post('/reindex', response_model=ReindexResponse)
def reindex(request: Request):
    """Update in-process using the existing Qdrant connection and loaded embedder."""
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
