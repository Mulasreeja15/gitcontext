"""Real Git histories and in-memory Qdrant, with all embeddings mocked."""
import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from gitcontext import config, indexing


def git(root, *args):
    """Run test-only Git commands with an explicit local identity."""
    return subprocess.run(['git', '-C', str(root), '-c', 'user.name=Unit Test',
                           '-c', 'user.email=unit@example.invalid', *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def commit(root, message):
    """Stage only the synthetic fixture repository and commit its history."""
    git(root, 'add', '--all')
    git(root, 'commit', '-m', message)
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    root.mkdir()
    git(root, 'init')
    for name in ['a.py', 'unchanged.py', 'old name.py', 'deleted.py']:
        (root / name).write_text('def original():\n    return 1\n', encoding='utf-8')
    commit(root, 'initial')
    monkeypatch.setattr(config, 'INDEX_STATE_PATH', tmp_path / 'index_state.json')
    return root


@pytest.fixture
def resources():
    """Load the real store under an alias, avoiding helper-module storage mocks."""
    path = Path(__file__).resolve().parents[1] / 'gitcontext' / 'store.py'
    spec = importlib.util.spec_from_file_location('gitcontext._memory_store', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = module.VectorStore.__new__(module.VectorStore)
    store.client = QdrantClient(':memory:')
    embedder = Mock(dim=3)
    embedder.encode.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    yield store, embedder
    store.client.close()


def test_full_state_and_incremental_changes(repo, resources):
    store, embedder = resources
    full = indexing.full_index(str(repo), store=store, embedder=embedder)
    state = indexing.load_state()
    assert state['indexed_commit'] == git(repo, 'rev-parse', 'HEAD')
    assert state['repo_path'] == str(repo.resolve()) and state['repo_name'] == repo.name
    assert state['remote_url'] is None and state['embed_model'] == config.EMBED_MODEL
    assert state['timestamp'] and state['chunk_count'] == 4
    assert full['chunks_added'] == 4 and not indexing.index_status()['stale']
    unchanged = {pid: p for pid, p in store.all_payloads() if p['file_path'] == 'unchanged.py'}
    embedder.reset_mock()
    (repo / 'a.py').write_text('def modified():\n    return 2\n', encoding='utf-8')
    (repo / 'added.py').write_text('def added():\n    return 3\n', encoding='utf-8')
    (repo / 'deleted.py').unlink()
    (repo / 'old name.py').rename(repo / 'new name.py')
    (repo / 'build').mkdir()
    (repo / 'build' / 'skip.py').write_text('def skip(): pass', encoding='utf-8')
    (repo / 'unsupported.js').write_text('let ignored = 1;', encoding='utf-8')
    commit(repo, 'change files')
    before = indexing.index_status()
    assert before['stale'] and before['commits_behind'] == 1
    summary = indexing.update_index(str(repo), store=store, embedder=embedder)
    assert summary['mode'] == 'incremental' and summary['commits_behind'] == 1
    assert (summary['files_added'], summary['files_modified'], summary['files_deleted'], summary['files_renamed']) == (1, 1, 1, 1)
    assert summary['chunks_removed'] == 3 and summary['chunks_added'] == 3
    assert sum(len(call.args[0]) for call in embedder.encode.call_args_list) == 3
    payloads = store.all_payloads()
    assert {p['file_path'] for _, p in payloads} == {'a.py', 'added.py', 'new name.py', 'unchanged.py'}
    assert {pid: p for pid, p in payloads if p['file_path'] == 'unchanged.py'} == unchanged
    assert not indexing.index_status()['stale']
    embedder.reset_mock()
    summary = indexing.update_index(str(repo), store=store, embedder=embedder)
    assert summary['chunks_removed'] == summary['chunks_added'] == 0
    embedder.encode.assert_not_called()


def test_delete_filter_is_exact(repo, resources):
    store, embedder = resources
    indexing.full_index(str(repo), store=store, embedder=embedder)
    assert store.delete_by_file('a.py') == 1
    assert store.delete_by_file('a.py') == 0
    assert store.count() == 3
    assert 'unchanged.py' in {p['file_path'] for _, p in store.all_payloads()}


@pytest.mark.parametrize('reason', ['missing-state', 'changed-model', 'missing-collection'])
def test_fallback_full(repo, resources, monkeypatch, reason):
    store, embedder = resources
    if reason != 'missing-state':
        indexing.full_index(str(repo), store=store, embedder=embedder)
    if reason == 'changed-model':
        monkeypatch.setattr(config, 'EMBED_MODEL', 'another-embedding-model')
    if reason == 'missing-collection':
        store.client.delete_collection(config.COLLECTION)
    embedder.reset_mock()
    summary = indexing.update_index(str(repo), store=store, embedder=embedder)
    assert summary['mode'] == 'full' and summary['chunks_added'] == 4
    embedder.encode.assert_called_once()


def test_dirty_working_tree_is_stale_and_requires_commit(repo, resources):
    store, embedder = resources
    indexing.full_index(str(repo), store=store, embedder=embedder)
    (repo / 'a.py').write_text('def changed(): pass', encoding='utf-8')
    assert indexing.index_status()['stale']
    with pytest.raises(ValueError, match='Commit supported-file changes'):
        indexing.update_index(str(repo), store=store, embedder=embedder)


def test_interrupted_full_index_is_stale_and_recovers(repo, resources, monkeypatch):
    store, embedder = resources
    indexing.full_index(str(repo), store=store, embedder=embedder)
    with monkeypatch.context() as scoped:
        scoped.setattr(store, 'upsert', Mock(side_effect=RuntimeError('interrupted')))
        with pytest.raises(RuntimeError):
            indexing.full_index(str(repo), store=store, embedder=embedder)
    assert indexing.index_status()['stale']
    assert indexing.load_state()['incomplete']
    summary = indexing.update_index(str(repo), store=store, embedder=embedder)
    assert summary['mode'] == 'full' and not indexing.index_status()['stale']


def test_deleted_last_file_leaves_valid_empty_index(repo, resources):
    store, embedder = resources
    indexing.full_index(str(repo), store=store, embedder=embedder)
    for path in repo.glob('*.py'):
        path.unlink()
    commit(repo, 'delete all')
    summary = indexing.update_index(str(repo), store=store, embedder=embedder)
    assert summary['chunks_removed'] == 4 and store.count() == 0
    assert indexing.load_state()['chunk_count'] == 0


def test_pull_only_managed_clone(repo, resources, tmp_path, monkeypatch):
    store, embedder = resources
    remote = tmp_path / 'remote.git'
    remote.mkdir()
    git(remote, 'init', '--bare')
    git(repo, 'remote', 'add', 'origin', str(remote))
    git(repo, 'push', '-u', 'origin', 'HEAD')
    managed = tmp_path / 'repos'
    managed.mkdir()
    clone = managed / 'clone'
    git(managed, 'clone', str(remote), str(clone))
    monkeypatch.setattr(config, 'REPOS_DIR', managed)
    indexing.full_index(str(clone), store=store, embedder=embedder)
    (repo / 'added.py').write_text('def remote_change(): pass', encoding='utf-8')
    commit(repo, 'remote changes')
    git(repo, 'push')
    summary = indexing.update_index(str(clone), store=store, embedder=embedder)
    assert summary['commits_behind'] == 1 and summary['chunks_added'] == 1
    assert git(clone, 'rev-parse', 'HEAD') == git(repo, 'rev-parse', 'HEAD')
    # Local repos with remotes must not be pulled: remote is now unavailable.
    git(repo, 'remote', 'set-url', 'origin', str(tmp_path / 'nonexistent.git'))
    indexing.full_index(str(repo), store=store, embedder=embedder)
    assert indexing.update_index(str(repo), store=store, embedder=embedder)['chunks_added'] == 0
