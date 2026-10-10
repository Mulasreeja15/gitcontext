"""Our own unit tests are excluded; external repositories retain their tests."""
from gitcontext import config
from gitcontext.ingest import collect_chunks, iter_files


def test_own_tests_excluded_but_external_tests_retained(tmp_path, monkeypatch):
    own = tmp_path / 'gitcontext'
    external = tmp_path / 'external'
    for root in (own, external):
        (root / 'tests').mkdir(parents=True)
        (root / 'tests' / 'test_example.py').write_text('def test_example():\n    pass\n', encoding='utf-8')
        (root / 'main.py').write_text('def main():\n    pass\n', encoding='utf-8')
    monkeypatch.setattr(config, 'PROJECT_ROOT', own)
    assert [rel for _, rel in iter_files(own)] == ['main.py']
    assert {c.file_path for c in collect_chunks(own)} == {'main.py'}
    assert {rel for _, rel in iter_files(external)} == {'main.py', 'tests/test_example.py'}
