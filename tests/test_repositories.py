"""Repository isolation uses synthetic sources, memory Qdrant and fake vectors."""
import argparse
import ast
import importlib.util
import importlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient

from gitcontext import config, indexing, repositories


@pytest.fixture
def multi(tmp_path, monkeypatch):
    """Share a single memory client across two independent collections."""
    path = config.PROJECT_ROOT / 'gitcontext' / 'store.py'
    spec = importlib.util.spec_from_file_location('gitcontext._repo_store', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(config, 'INDEX_STATE_DIR', tmp_path / 'state')
    monkeypatch.setattr(importlib.import_module('gitcontext.store'), 'VectorStore', module.VectorStore)
    client = QdrantClient(':memory:')
    embedder = Mock(dim=3)
    embedder.encode.side_effect = lambda texts: [[1., 0., 0.] for _ in texts]
    stores = {}
    for name in ['a', 'b']:
        root = tmp_path / name
        root.mkdir()
        (root / 'same.py').write_text(f'def shared():\n    return "repository {name}"\n', encoding='utf-8')
        store = module.VectorStore('owner/' + name, client=client)
        indexing.full_index(str(root), store=store, embedder=embedder, repo='owner/' + name)
        stores[name] = store
    yield stores, client, embedder
    client.close()


def test_collection_names():
    assert repositories.collection_name('PSF/Requests') == 'repo__psf__requests'
    assert repositories.collection_name('owner-name/repo_name.py') == 'repo__owner-name__repo_name.py'
    assert repositories.collection_name('a/b-c') != repositories.collection_name('a-b/c')
    for bad in ['../requests', 'owner/../../other', 'owner__x/name', 'name', 'a/b/c', 'a/with space']:
        with pytest.raises(ValueError):
            repositories.collection_name(bad)


def test_isolation_vector_and_bm25(multi, helper_modules):
    stores, _, embedder = multi
    module, _ = helper_modules
    retrievers = []
    for name in ['a', 'b']:
        retriever = module.Retriever.__new__(module.Retriever)
        retriever.store, retriever.embedder = stores[name], embedder
        retriever._payloads, retriever._bm25_ids = {}, []
        retriever._bm25, retriever._reranker = None, None
        for hybrid in [False, True]:
            result = retriever.retrieve('shared repository', opts=module.RetrievalOptions(bm25=hybrid))
            assert result.hits
            assert all(f'repository {name}' in payload['text'] for _, payload in result.hits)
        retrievers.append(retriever)
    assert retrievers[0]._bm25 is not retrievers[1]._bm25
    assert repositories.state_path('owner/a') != repositories.state_path('owner/b')
    # Rebuilding A preserves B, even when point IDs and relative filenames match.
    stores['a'].reset(3)
    assert stores['b'].count() == 1


def test_list_and_remove(multi):
    stores, client, _ = multi
    rows = repositories.list_repos(client=client)
    assert [row['repo'] for row in rows] == ['owner/a', 'owner/b']
    assert all(row['chunk_count'] == 1 for row in rows)
    repositories.remove_repo('owner/a', client=client)
    assert not repositories.state_path('owner/a').exists()
    assert stores['a'].count() == 0 and stores['b'].count() == 1
    assert [row['repo'] for row in repositories.list_repos(client=client)] == ['owner/b']
    with pytest.raises(ValueError, match='not indexed'):
        repositories.remove_repo('owner/a', client=client)


def cli_namespace():
    """Exercise actual argument parsing and repo commands without importing ML."""
    source = config.PROJECT_ROOT / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'main', 'add_retrieval_flags', 'add_metadata_flags', 'cmd_repos', 'cmd_remove'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'argparse': argparse, 'config': config}
    namespace.update({name: Mock() for name in ['cmd_index', 'cmd_update', 'cmd_status',
                                               'cmd_search', 'cmd_ask', 'cmd_serve', 'cmd_index_history']})
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def test_repos_remove_cli(multi, capsys):
    stores, client, _ = multi
    namespace = cli_namespace()
    actual_list, actual_remove = repositories.list_repos, repositories.remove_repo
    with patch.object(repositories, 'list_repos', side_effect=lambda: actual_list(client=client)), \
            patch('sys.argv', ['main.py', 'repos']):
        namespace['main']()
    assert 'owner/a | 1 | unknown | yes' in capsys.readouterr().out
    with patch.object(repositories, 'remove_repo', side_effect=lambda repo: actual_remove(repo, client=client)), \
            patch('sys.argv', ['main.py', 'remove', 'owner/a']):
        namespace['main']()
    assert 'Removed index: owner/a' in capsys.readouterr().out
    assert stores['b'].count() == 1


@pytest.mark.parametrize('command,args', [('search', ['query']), ('ask', ['question']),
                                         ('status', []), ('update', [])])
def test_cli_repo_selection(command, args):
    namespace = cli_namespace()
    with patch('sys.argv', ['main.py', command, *args, '--repo', 'pallets/click']):
        namespace['main']()
    assert namespace['cmd_' + command].call_args.args[0].repo == 'pallets/click'
    with patch('sys.argv', ['main.py', command, *args]):
        namespace['main']()
    assert namespace['cmd_' + command].call_args.args[0].repo == 'psf/requests'


@pytest.mark.parametrize('limit', ['disk', 'chunks'])
def test_limits_preserve_existing_index(multi, monkeypatch, limit):
    stores, _, embedder = multi
    store = stores['a']
    state = indexing.load_state('owner/a')
    embedder.reset_mock()
    monkeypatch.setattr(config, 'MAX_REPO_BYTES' if limit == 'disk' else 'MAX_REPO_CHUNKS', 0)
    with pytest.raises(ValueError, match='limit|maximum'):
        indexing.full_index(state['repo_path'], repo='owner/a', store=store, embedder=embedder)
    embedder.encode.assert_not_called()
    assert store.count() == stores['b'].count() == 1
    assert indexing.load_state('owner/a') == state


def test_legacy_migration_keeps_other_repos(multi):
    stores, client, embedder = multi
    from qdrant_client.models import Distance, VectorParams
    client.create_collection('gitcontext', vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    state = indexing.load_state('owner/a')
    indexing.full_index(state['repo_path'], repo='psf/requests', store=stores['a'], embedder=embedder)
    assert not client.collection_exists('gitcontext')
    assert client.collection_exists('repo__psf__requests')
    assert stores['b'].count() == 1


def test_rewrite_and_judge_caches_are_per_repo(tmp_path):
    from gitcontext import answer, judge
    answer.rewrite_query.cache_clear()
    try:
        with patch.object(answer, '_chat', return_value='rewritten') as chat:
            for repo in ['owner/a', 'owner/b', 'owner/a']:
                assert answer.rewrite_query('same question', repo=repo) == 'rewritten'
            assert chat.call_count == 2
    finally:
        answer.rewrite_query.cache_clear()
    path = tmp_path / 'judge.jsonl'
    cache = judge.JudgeCache(path, repo='owner/a')
    record = {'repo': 'owner/a', 'question': 'q', 'config': 'vector',
              'model': config.JUDGE_MODEL, 'answer_model': config.LLM_MODEL}
    cache.save(record)
    assert judge.JudgeCache(path, repo='owner/a').get('q', 'vector') == record
    assert judge.JudgeCache(path, repo='owner/b').get('q', 'vector') is None
