"""Syntax boundaries, source locations, and Markdown sections."""
from gitcontext import config
from gitcontext.chunker import Chunk, _split, chunk_doc, chunk_python


def test_python_chunk_kinds_and_source_lines():
    source = ('VALUE = 1\n@decorate\ndef decorated():\n    return VALUE\n'
              'class SmallClass:\n    def run(self):\n        return 1\n')
    big_start = len(source.splitlines()) + 1
    methods = []
    for i in range(40):
        methods.append(f'    def method_{i}(self):\n        """' + 'x' * 100
                       + f'"""\n        return {i}\n')
    big_source = 'class BigClass:\n    """Big documentation."""\n' + ''.join(methods)
    assert len(big_source) > config.MAX_CHUNK_CHARS
    source += big_source + 'AFTER = 2\n'
    chunks = chunk_python('example.py', source)
    by_name = {c.name: c for c in chunks if c.kind != 'module'}
    function = by_name['decorated']
    assert (function.kind, function.parent, function.start_line, function.end_line) == (
        'function', '', 2, 4)
    small = by_name['SmallClass']
    assert (small.kind, small.parent, small.start_line, small.end_line) == ('class', '', 5, 7)
    summary = by_name['BigClass']
    assert (summary.kind, summary.parent, summary.start_line, summary.end_line) == (
        'class', '', big_start, big_start + 1)
    assert 'method_3' in summary.text
    method = by_name['BigClass.method_3']
    assert (method.kind, method.parent) == ('method', 'BigClass')
    assert (method.start_line, method.end_line) == (big_start + 11, big_start + 13)
    modules = [c for c in chunks if c.kind == 'module']
    assert [(c.start_line, c.end_line) for c in modules] == [(1, 1),
        (len(source.splitlines()), len(source.splitlines()))]
    assert all(c.parent == '' and c.name == 'example (module-level code)' for c in modules)
    lines = source.splitlines()
    for chunk in chunks:
        if chunk is not summary:  # summaries intentionally synthesize a method list
            expected_lines = lines[chunk.start_line - 1:chunk.end_line]
            if chunk.kind == 'method':  # tree-sitter starts at def, after indentation
                expected_lines[0] = expected_lines[0].lstrip()
            assert chunk.text == '\n'.join(expected_lines)


def test_split_preserves_line_windows_and_metadata(monkeypatch):
    monkeypatch.setattr(config, 'MAX_CHUNK_CHARS', 12)
    chunk = Chunk('x.py', 10, 13, 'method', 'Big.run', 'python',
                  'aaaa\nbbbb\ncccc\ndddd', parent='Big')
    parts = _split(chunk)
    assert [(p.start_line, p.end_line, p.text) for p in parts] == [
        (10, 11, 'aaaa\nbbbb'), (12, 13, 'cccc\ndddd')]
    assert [p.name for p in parts] == ['Big.run (part 1/2)', 'Big.run (part 2/2)']
    assert all((p.parent, p.kind, p.file_path) == ('Big', 'method', 'x.py') for p in parts)
    assert _split(parts[0]) == [parts[0]]


def test_markdown_headings_ignore_fenced_hashes():
    source = 'Intro\n# First\nText\n```python\n# not a heading\n```\n## Second\nEnd'
    chunks = chunk_doc('guide.md', source)
    assert [(c.name, c.start_line, c.end_line) for c in chunks] == [
        ('guide.md', 1, 1), ('First', 2, 6), ('Second', 7, 8)]
    assert all(c.kind == 'doc' and c.parent == '' for c in chunks)
    for chunk in chunks:
        assert chunk.text == '\n'.join(source.splitlines()[chunk.start_line - 1:chunk.end_line])
