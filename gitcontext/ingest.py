from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import config
from .chunker import Chunk, chunk_file


def resolve_repo(source: str) -> Path:
    """Clone a Git URL into data/repos (full history, needed in later phases)
    or return a local directory."""
    if source.startswith(("http://", "https://", "git@")):
        name = source.rstrip("/").split("/")[-1].removesuffix(".git")
        dest = config.REPOS_DIR / name
        if not dest.exists():
            config.REPOS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Cloning {source} ...")
            subprocess.run(["git", "clone", source, str(dest)], check=True)
        return dest.resolve()

    path = Path(source).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"Not a directory: {path}")
    return path


def iter_files(root: Path):
    """Yield (absolute_path, repo_relative_posix_path) for supported files."""
    supported = set(config.CODE_EXTENSIONS) | config.DOC_EXTENSIONS
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skipped and hidden dirs in place so os.walk never enters them
        dirnames[:] = sorted(
            d for d in dirnames if d not in config.SKIP_DIRS and not d.startswith(".")
        )
        for fname in sorted(filenames):
            path = Path(dirpath) / fname
            if path.suffix.lower() not in supported:
                continue
            if path.stat().st_size > config.MAX_FILE_BYTES:
                continue
            yield path, path.relative_to(root).as_posix()


def collect_chunks(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, rel in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        chunks.extend(chunk_file(rel, text))
    return chunks