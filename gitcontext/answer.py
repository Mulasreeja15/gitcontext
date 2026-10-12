"""LLM calls via Groq: cited answers and query rewriting."""
from __future__ import annotations

import os
import json
import time
from collections.abc import Iterator
from functools import lru_cache

import requests

from . import config

SYSTEM_PROMPT = """You are GitContext, an assistant that answers questions about a code repository.
Rules:
- Use ONLY the numbered sources provided. Do not invent functions, files or behavior.
- Cite sources inline like [1] or [2][3] after each claim.
- Mention file paths when you refer to functions or classes.
- Prefer implementation code over tests when explaining how something works.
- For why questions, use commit messages and PR discussions when provided; distinguish documented intent from inference.
- If the sources don't contain the answer, say so plainly and suggest what to search for."""

REWRITE_PROMPT = """You turn a developer's question about a codebase into a search query for a code search engine.
Write the identifiers, function names, class names and technical terms most likely to appear
in the source code that answers the question.
Return ONLY the query, on one line, with no explanation and no quotes."""

STRICT_SYSTEM_PROMPT = SYSTEM_PROMPT + """
Strict grounding requirements:
- Only state facts that appear in the numbered sources.
- Every sentence with a factual claim must end with a citation such as [1].
- If the sources do not show something, explicitly say "the provided sources don't show X", replacing X with the missing information.
- Never fill gaps with general knowledge or knowledge from training.
- Do not describe code that is not in the sources, including urllib3 internals.
- Do not assume an implementation from a function name, a call site, or general library behavior."""


def system_prompt(variant: str | None = None) -> str:
    """Select a validated answer prompt without changing global settings."""
    variant = config.ANSWER_PROMPT if variant is None else variant
    if variant not in {'default', 'strict'}:
        raise ValueError('Answer prompt must be default or strict')
    return STRICT_SYSTEM_PROMPT if variant == 'strict' else SYSTEM_PROMPT

MAX_RETRIES = 6
SESSION = requests.Session()


def _network_retry(attempt: int) -> None:
    """Wait before retrying a transport failure or temporary server error."""
    wait = min(config.GROQ_RETRY_BASE_SECONDS * 2 ** attempt, config.GROQ_RETRY_MAX_SECONDS)
    print(f'Network error talking to Groq, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})')
    time.sleep(wait)


class RequestTooLarge(Exception):
    """Groq rejected the request size; callers may retry with smaller context."""


def _chat(messages: list[dict], temperature: float = 0.1, timeout: int = 120,
          model: str | None = None, stream: bool = False, on_retry=None,
          max_tokens: int | None = None, _start_attempt: int = 0) -> str | Iterator[str]:
    """Call Groq with shared rate-limit retries and an optional model override."""
    key = os.getenv(config.LLM_KEY_NAME)
    if not key:
        raise SystemExit(f"Missing {config.LLM_KEY_NAME}. Add it to your .env file.")

    for attempt in range(_start_attempt, MAX_RETRIES):
        body = {"model": model or config.LLM_MODEL, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            body['max_tokens'] = max_tokens
        request_options = {}
        if stream:
            body['stream'] = True
            request_options['stream'] = True
        try:
            r = SESSION.post(
                config.LLM_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body, timeout=timeout, **request_options,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_RETRIES - 1:
                raise requests.ConnectionError('Could not reach Groq after several retries. Check your connection and try again.') from None
            _network_retry(attempt)
            continue
        if r.status_code in {500, 502, 503, 504}:
            r.close()
            if attempt == MAX_RETRIES - 1:
                raise requests.ConnectionError('Groq is temporarily unavailable after several retries. Try again later.')
            _network_retry(attempt)
            continue
        if r.status_code == 429:
            # Rate limited: wait as long as the server asks, or back off exponentially.
            retry_after = r.headers.get("retry-after")
            wait = float(retry_after) if retry_after else 5 * 2 ** attempt
            wait = min(wait, 60)
            r.close()
            if on_retry is not None:
                on_retry(wait)
            print(f"  Rate limited by Groq, waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES}) ...")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            r.close()
            raise SystemExit("API key rejected (401). Check the key in your .env file.")
        if r.status_code == 413:
            r.close()
            raise RequestTooLarge("Request too large (413). Try fewer or shorter sources.")
        try:
            r.raise_for_status()
        except requests.RequestException:
            r.close()
            raise
        if stream:
            def reconnect():
                if attempt == MAX_RETRIES - 1:
                    raise requests.ConnectionError('Groq stream failed after several retries.')
                _network_retry(attempt)
                return _chat(messages, temperature, timeout, model, stream=True,
                             on_retry=on_retry, max_tokens=max_tokens, _start_attempt=attempt + 1)
            return _ResponseStream(r, retry=reconnect)
        try:
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        finally:
            r.close()

    raise SystemExit("Still rate limited after several retries. Wait a few minutes and try again.")


class _ResponseStream:
    """Close HTTP even if a connected token iterator is never consumed."""

    def __init__(self, response, retry=None):
        self.response = response
        self.tokens = _stream_tokens(response)
        self.started = False
        self.closed = False
        self.retry = retry
        self.emitted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        self.started = True
        try:
            try:
                token = next(self.tokens)
            except (requests.ConnectionError, requests.Timeout):
                if self.emitted or self.retry is None:
                    raise  # never replay an answer after the user has seen tokens
                self.tokens = self.retry()
                self.retry = None  # the replacement stream owns subsequent retries
                token = next(self.tokens)
            self.emitted = True
            return token
        except BaseException:
            self.closed = True  # token generator's finally already closed HTTP
            raise

    def close(self):
        """Release an open response on cancellation, including before iteration."""
        if not self.closed:
            self.closed = True
            if self.started:
                self.tokens.close()
            else:
                self.response.close()


def _stream_tokens(response) -> Iterator[str]:
    """Parse OpenAI SSE data frames, preserving content whitespace and closing HTTP."""
    try:
        for line in response.iter_lines():
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                return
            if not data:
                continue
            event = json.loads(data)
            if 'error' in event:
                raise ValueError('Upstream streaming error')
            choices = event.get('choices', [])
            if choices:
                token = choices[0].get('delta', {}).get('content')
                if token:
                    yield token
    finally:
        response.close()


def format_sources(hits: list[tuple[float, dict]]) -> str:
    blocks = []
    for i, (_, p) in enumerate(hits, 1):
        lang = "python" if p["language"] == "python" else ""
        blocks.append(
            f"[{i}] {p['file_path']} lines {p['start_line']}-{p['end_line']} "
            f"({p['kind']}: {p['name']})\n"
            + (f"Date: {p.get('date', '')}; author: {p.get('author', '')}; link: {p.get('url') or 'local history'}\n"
               if p['kind'] in {'commit', 'pr'} else '')
            + f"```{lang}\n{p['text']}\n```"
        )
    return "\n\n".join(blocks)


def _answer_messages(question: str, hits: list[tuple[float, dict]],
                     prompt_variant: str | None = None) -> list[dict]:
    """Use identical grounding prompts for buffered and streaming answers."""
    prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt(prompt_variant)},
        {"role": "user", "content": prompt},
    ]


def generate_answer(question: str, hits: list[tuple[float, dict]],
                    prompt_variant: str | None = None) -> str:
    """Generate a complete answer through the shared retrying Groq client."""
    return _chat(_answer_messages(question, hits, prompt_variant))


def stream_answer(question: str, hits: list[tuple[float, dict]], on_retry=None) -> Iterator[str]:
    """Open a retrying Groq stream before returning its content iterator."""
    options = {'stream': True}
    if on_retry is not None:
        options['on_retry'] = on_retry
    return _chat(_answer_messages(question, hits), **options)


@lru_cache(maxsize=1024)
def rewrite_query(question: str, repo: str = config.DEFAULT_REPO) -> str:
    out = _chat(
        [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": question}],
        temperature=0.0,
        timeout=60,
    )
    first_line = out.splitlines()[0] if out else ""
    return first_line.strip().strip('"`') or question
