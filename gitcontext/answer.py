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
- If the sources don't contain the answer, say so plainly and suggest what to search for."""

REWRITE_PROMPT = """You turn a developer's question about a codebase into a search query for a code search engine.
Write the identifiers, function names, class names and technical terms most likely to appear
in the source code that answers the question.
Return ONLY the query, on one line, with no explanation and no quotes."""

MAX_RETRIES = 6


def _chat(messages: list[dict], temperature: float = 0.1, timeout: int = 120,
          model: str | None = None, stream: bool = False, on_retry=None) -> str | Iterator[str]:
    """Call Groq with shared rate-limit retries and an optional model override."""
    key = os.getenv(config.LLM_KEY_NAME)
    if not key:
        raise SystemExit(f"Missing {config.LLM_KEY_NAME}. Add it to your .env file.")

    for attempt in range(MAX_RETRIES):
        body = {"model": model or config.LLM_MODEL, "messages": messages, "temperature": temperature}
        request_options = {}
        if stream:
            body['stream'] = True
            request_options['stream'] = True
        r = requests.post(
            config.LLM_URL,
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=timeout,
            **request_options,
        )
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
            raise SystemExit("Request too large (413). Try fewer sources: add  -k 4  to your command.")
        try:
            r.raise_for_status()
        except requests.RequestException:
            r.close()
            raise
        if stream:
            return _ResponseStream(r)
        try:
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        finally:
            r.close()

    raise SystemExit("Still rate limited after several retries. Wait a few minutes and try again.")


class _ResponseStream:
    """Close HTTP even if a connected token iterator is never consumed."""

    def __init__(self, response):
        self.response = response
        self.tokens = _stream_tokens(response)
        self.started = False
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        self.started = True
        try:
            return next(self.tokens)
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
            f"({p['kind']}: {p['name']})\n```{lang}\n{p['text']}\n```"
        )
    return "\n\n".join(blocks)


def _answer_messages(question: str, hits: list[tuple[float, dict]]) -> list[dict]:
    """Use identical grounding prompts for buffered and streaming answers."""
    prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def generate_answer(question: str, hits: list[tuple[float, dict]]) -> str:
    """Generate a complete answer through the shared retrying Groq client."""
    return _chat(_answer_messages(question, hits))


def stream_answer(question: str, hits: list[tuple[float, dict]], on_retry=None) -> Iterator[str]:
    """Open a retrying Groq stream before returning its content iterator."""
    options = {'stream': True}
    if on_retry is not None:
        options['on_retry'] = on_retry
    return _chat(_answer_messages(question, hits), **options)


@lru_cache(maxsize=1024)
def rewrite_query(question: str) -> str:
    out = _chat(
        [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": question}],
        temperature=0.0,
        timeout=60,
    )
    first_line = out.splitlines()[0] if out else ""
    return first_line.strip().strip('"`') or question
