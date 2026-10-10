"""Service pipeline progress, shallow cloning, and language detection stay offline."""
from threading import RLock
from unittest.mock import Mock, patch

from gitcontext import config, service_indexing


def test_shallow_clone_and_shared_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'REPOS_DIR', tmp_path / 'repos')
    retriever, lock, progress = Mock(), RLock(), Mock()
    result = {'state': {'repo_id':'owner/project', 'chunk_count':2}}
    with patch.object(service_indexing.subprocess, 'run') as run, \
            patch.object(service_indexing, 'full_index', return_value=result) as full, \
            patch.object(service_indexing, '_write_state') as write:
        state = service_indexing.index_repository('https://github.com/owner/project',
            'owner/project', retriever, lock, progress)
    assert run.call_args.args[0] == ['git', 'clone', '--depth', '1',
        'https://github.com/owner/project', str((config.REPOS_DIR / 'owner__project').resolve())]
    assert full.call_args.kwargs['store'] is retriever.store
    assert full.call_args.kwargs['write_lock'] is lock
    assert full.call_args.kwargs['embedder'].embedder is retriever.embedder
    write.assert_called_once_with(state)
    retriever.invalidate_index.assert_called_once()
    progress.assert_called_once_with('cloning', '')


def test_embedding_batch_progress(monkeypatch):
    monkeypatch.setattr(config, 'API_EMBED_BATCH_SIZE', 2)
    embedder = Mock(dim=3)
    embedder.encode.side_effect = lambda texts: [[1.,0.,0.] for _ in texts]
    progress = Mock()
    adapter = service_indexing.ProgressEmbedder(embedder, RLock(), progress)
    assert len(adapter.encode(['a','b','c'])) == 3
    assert [call.args for call in progress.call_args_list] == [
        ('embedding','2/3 chunks'), ('embedding','3/3 chunks')]
    assert [call.args[0] for call in embedder.encode.call_args_list] == [['a','b'], ['c']]


def test_language_profile_ignores_skipped_content(tmp_path):
    for index in range(6):
        (tmp_path / f'app{index}.ts').write_text('ignored contents', encoding='utf-8')
    (tmp_path / 'helper.py').write_text('def helper(): pass', encoding='utf-8')
    ignored = tmp_path / 'node_modules'
    ignored.mkdir()
    (ignored / 'ignored.py').write_text('', encoding='utf-8')
    result = service_indexing.language_profile(tmp_path)
    assert result['python_files'] == 1 and result['language'] == 'TypeScript'
    assert result['language_warning'] == ('GitContext currently understands Python code best; '
                                           'this repository is mostly TypeScript.')
    (tmp_path / 'another.py').write_text('', encoding='utf-8')
    assert service_indexing.language_profile(tmp_path)['language_warning'] is None
