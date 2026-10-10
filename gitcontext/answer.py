"""LLM calls via Groq: cited answers and query rewriting."""
from __future__ import annotations

import os
import time
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


def _chat(messages: list[dict], temperature: float = 0.1, timeout: int = 120) -> str:
    key = os.getenv(config.LLM_KEY_NAME)
    if not key:
        raise SystemExit(f"Missing {config.LLM_KEY_NAME}. Add it to your .env file.")

    for attempt in range(MAX_RETRIES):
        r = requests.post(
            config.LLM_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": config.LLM_MODEL, "messages": messages, "temperature": temperature},
            timeout=timeout,
        )
        if r.status_code == 429:
            # Rate limited: wait as long as the server asks, or back off exponentially.
            retry_after = r.headers.get("retry-after")
            wait = float(retry_after) if retry_after else 5 * 2 ** attempt
            wait = min(wait, 60)
            print(f"  Rate limited by Groq, waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES}) ...")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            raise SystemExit("API key rejected (401). Check the key in your .env file.")
        if r.status_code == 413:
            raise SystemExit("Request too large (413). Try fewer sources: add  -k 4  to your command.")
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

    raise SystemExit("Still rate limited after several retries. Wait a few minutes and try again.")


def format_sources(hits: list[tuple[float, dict]]) -> str:
    blocks = []
    for i, (_, p) in enumerate(hits, 1):
        lang = "python" if p["language"] == "python" else ""
        blocks.append(
            f"[{i}] {p['file_path']} lines {p['start_line']}-{p['end_line']} "
            f"({p['kind']}: {p['name']})\n```{lang}\n{p['text']}\n```"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, hits: list[tuple[float, dict]]) -> str:
    prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return _chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])


@lru_cache(maxsize=1024)
def rewrite_query(question: str) -> str:
    out = _chat(
        [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": question}],
        temperature=0.0,
        timeout=60,
    )
    first_line = out.splitlines()[0] if out else ""
    return first_line.strip().strip('"`') or question