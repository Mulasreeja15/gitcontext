from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import config
from .chunker import Chunk, chunk_file
from .repositories import infer_repo


def resolve_repo(source: str) -> Path:
    """Clone a Git URL into data/repos (full history, needed in later phases)
    or return a local directory."""
    if source.startswith(("http://", "https://", "git@")):
        identity = infer_repo(source, Path('.'))
        dest = config.REPOS_DIR / identity.replace('/', '__')
        # Reuse older checkouts only when their remote identifies the same repo.
        legacy = config.REPOS_DIR / identity.split('/')[1]
        if not dest.exists() and legacy.is_dir():
            result = subprocess.run(['git', '-C', str(legacy), 'remote', 'get-url', 'origin'],
                                    capture_output=True, text=True)
            if result.returncode == 0 and infer_repo(result.stdout.strip(), legacy) == identity:
                dest = legacy
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
    root = root.resolve()
    supported = set(config.CODE_EXTENSIONS) | config.DOC_EXTENSIONS
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skipped and hidden dirs in place so os.walk never enters them
        dirnames[:] = sorted(
            d for d in dirnames if d not in config.SKIP_DIRS and not d.startswith(".")
            and not (root == config.PROJECT_ROOT and Path(dirpath) == root
                     and d in config.SELF_SKIP_DIRS)
        )
        for fname in sorted(filenames):
            path = Path(dirpath) / fname
            if path.suffix.lower() not in supported:
                continue
            if path.stat().st_size > config.MAX_FILE_BYTES:
                continue
            yield path, path.relative_to(root).as_posix()


def supported_path(root: Path, relative: str) -> bool:
    """Apply full-index extension/directory exclusions to a Git diff path."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        return False
    if path.suffix.lower() not in set(config.CODE_EXTENSIONS) | config.DOC_EXTENSIONS:
        return False
    if any(part in config.SKIP_DIRS or part.startswith('.') for part in path.parts[:-1]):
        return False
    return not (root.resolve() == config.PROJECT_ROOT and path.parts[0] in config.SELF_SKIP_DIRS)


def collect_chunks(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, rel in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        chunks.extend(chunk_file(rel, text))
    return chunks
