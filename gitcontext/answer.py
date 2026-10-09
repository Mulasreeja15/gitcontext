"""Turn retrieved chunks into a cited answer using Groq."""
from __future__ import annotations

import os

import requests

from . import config

SYSTEM_PROMPT = """You are GitContext, an assistant that answers questions about a code repository.
Rules:
- Use ONLY the numbered sources provided. Do not invent functions, files or behavior.
- Cite sources inline like [1] or [2][3] after each claim.
- Mention file paths when you refer to functions or classes.
- If the sources don't contain the answer, say so plainly and suggest what to search for."""


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
    key = os.getenv(config.LLM_KEY_NAME)
    if not key:
        raise SystemExit(f"Missing {config.LLM_KEY_NAME}. Add it to your .env file.")

    prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    r = requests.post(
        config.LLM_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        },
        timeout=120,
    )
    if r.status_code == 401:
        raise SystemExit("API key rejected (401). Check the key in your .env file.")
    if r.status_code == 413:
        raise SystemExit("Request too large (413). Try fewer sources: add  -k 4  to your command.")
    if r.status_code == 429:
        raise SystemExit("Rate limit hit (429). Wait a minute, or try fewer sources with  -k 4")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]