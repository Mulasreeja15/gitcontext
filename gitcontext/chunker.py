from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import PurePosixPath

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from . import config

_PARSER = Parser(Language(tspython.language()))
_DEF_TYPES = {"function_definition", "class_definition"}


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    kind: str       # function | class | method | module | doc
    name: str
    language: str
    text: str
    parent: str = ""
    last_commit_sha: str | None = None
    last_author: str | None = None
    last_modified: str | None = None

    @property
    def id(self) -> str:
        key = f"{self.file_path}:{self.start_line}:{self.end_line}:{self.name}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def embedding_text(self) -> str:
        """Text that gets embedded: a small context header + the code.
        The header helps queries like 'login in auth.py' match."""
        header = [f"File: {self.file_path}", f"{self.kind.title()}: {self.name}"]
        if self.parent:
            header.append(f"Inside class: {self.parent}")
        return "\n".join(header) + "\n\n" + self.text

    def payload(self) -> dict:
        return asdict(self)


# ---------- helpers ----------

def _text(src: bytes, start: int, end: int) -> str:
    return src[start:end].decode("utf-8", errors="replace")


def _unwrap(node):
    """Decorated defs wrap the real def. Return (definition, outer_node)."""
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition"), node
    return node, node


def _name(src: bytes, node) -> str:
    n = node.child_by_field_name("name")
    return _text(src, n.start_byte, n.end_byte) if n else "<anonymous>"


def _make(path, src, node, kind, name, parent="") -> Chunk:
    return Chunk(
        file_path=path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        kind=kind,
        name=name,
        language="python",
        text=_text(src, node.start_byte, node.end_byte),
        parent=parent,
    )


def _split(chunk: Chunk) -> list[Chunk]:
    """Split an oversized chunk into line windows, keeping line numbers right."""
    if len(chunk.text) <= config.MAX_CHUNK_CHARS:
        return [chunk]
    lines = chunk.text.split("\n")
    parts, buf, buf_start, size = [], [], 0, 0
    for i, line in enumerate(lines):
        if buf and size + len(line) + 1 > config.MAX_CHUNK_CHARS:
            parts.append((buf_start, buf))
            buf, size, buf_start = [], 0, i
        buf.append(line)
        size += len(line) + 1
    if buf:
        parts.append((buf_start, buf))
    return [
        replace(
            chunk,
            start_line=chunk.start_line + s,
            end_line=chunk.start_line + s + len(b) - 1,
            name=f"{chunk.name} (part {n + 1}/{len(parts)})",
            text="\n".join(b),
        )
        for n, (s, b) in enumerate(parts)
    ]


# ---------- python ----------

def _chunk_class(path: str, src: bytes, cls, outer) -> list[Chunk]:
    name = _name(src, cls)
    whole = _make(path, src, outer, "class", name)
    if len(whole.text) <= config.MAX_CHUNK_CHARS:
        return [whole]

    # Big class: one summary chunk (signature, docstring, attributes,
    # method list) plus one chunk per method.
    body = cls.child_by_field_name("body")
    methods, others = [], []
    for child in (body.children if body else []):
        definition, child_outer = _unwrap(child)
        if definition is not None and definition.type == "function_definition":
            m_name = _name(src, definition)
            methods.append(_make(path, src, child_outer, "method", f"{name}.{m_name}", parent=name))
        else:
            others.append(child)

    signature = _text(src, outer.start_byte, body.start_byte if body else outer.end_byte)
    summary = signature.rstrip() + "\n"
    summary += "\n".join("    " + _text(src, n.start_byte, n.end_byte) for n in others)
    if methods:
        summary += "\n    # methods: " + ", ".join(m.name.split(".", 1)[1] for m in methods)
    end_node = others[-1] if others else outer
    summary_chunk = Chunk(
        file_path=path,
        start_line=outer.start_point[0] + 1,
        end_line=(end_node.end_point[0] + 1) if others else outer.start_point[0] + 1,
        kind="class",
        name=name,
        language="python",
        text=summary,
    )
    out = _split(summary_chunk)
    for m in methods:
        out.extend(_split(m))
    return out


def chunk_python(path: str, source: str) -> list[Chunk]:
    src = source.encode("utf-8")
    tree = _PARSER.parse(src)
    chunks: list[Chunk] = []
    pending = []  # consecutive top-level statements that aren't defs

    def flush():
        if not pending:
            return
        text = _text(src, pending[0].start_byte, pending[-1].end_byte)
        if text.strip():
            chunks.extend(_split(Chunk(
                file_path=path,
                start_line=pending[0].start_point[0] + 1,
                end_line=pending[-1].end_point[0] + 1,
                kind="module",
                name=f"{PurePosixPath(path).stem} (module-level code)",
                language="python",
                text=text,
            )))
        pending.clear()

    for node in tree.root_node.children:
        definition, outer = _unwrap(node)
        if definition is None or definition.type not in _DEF_TYPES:
            pending.append(node)
            continue
        flush()
        if definition.type == "function_definition":
            chunks.extend(_split(_make(path, src, outer, "function", _name(src, definition))))
        else:
            chunks.extend(_chunk_class(path, src, definition, outer))
    flush()
    return chunks


# ---------- docs ----------

def chunk_doc(path: str, text: str) -> list[Chunk]:
    lines = text.split("\n")
    is_md = path.lower().endswith(".md")
    sections, cur_start, cur_title, in_fence = [], 0, PurePosixPath(path).name, False

    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if is_md and not in_fence and line.startswith("#"):
            if i > cur_start:
                sections.append((cur_start, i, cur_title))
                cur_start = i
            cur_title = line.lstrip("#").strip() or cur_title
    sections.append((cur_start, len(lines), cur_title))

    chunks = []
    for start, end, title in sections:
        body = "\n".join(lines[start:end])
        if body.strip():
            chunks.extend(_split(Chunk(path, start + 1, end, "doc", title, "markdown", body)))
    return chunks


def chunk_file(path: str, text: str) -> list[Chunk]:
    suffix = PurePosixPath(path).suffix.lower()
    if config.CODE_EXTENSIONS.get(suffix) == "python":
        return chunk_python(path, text)
    if suffix in config.DOC_EXTENSIONS:
        return chunk_doc(path, text)
    return []
