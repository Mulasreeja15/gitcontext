"""Multi-repo HTTP contracts with indexing, storage and LLMs mocked."""
import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch
from threading import Event

import pytest

from test_api import service
from gitcontext import answer


@pytest.mark.parametrize('url', ['http://github.com/pallets/click', 'https://example.com/a/b',
    'https://github.com/a/b/tree/main', 'https://github.com/a/b?token=x',
    'https://user:secret@github.com/a/b', 'https://github.com/a/..',
    'https://github.com/a/b#main', 'https://github.com/a/b.git/extra'])
def test_index_url_validation(service, url):
    client, _, _ = service
    api = importlib.import_module('gitcontext.api')
    with patch.object(api, 'index_repository') as pipeline:
        assert client.post('/index', json={'repo_url':url}).status_code == 422
        pipeline.assert_not_called()


def test_index_job_transitions_and_busy(service):
    client, base, _ = service
    api = importlib.import_module('gitcontext.api')
    repo_retriever = Mock()
    stages = []

    def pipeline(url, identity, retriever, lock, progress):
        assert identity == 'pallets/click' and retriever is repo_retriever
        stages.append(client.get('/index/status').json()['state'])
        for stage, value in [('chunking', ''), ('embedding', '0/2400 chunks'),
                             ('embedding', '812/2400 chunks')]:
            progress(stage, value)
            status = client.get('/index/status').json()
            stages.append(status['state'])
            assert status['progress'] == value
        return {}

    with patch.object(api, 'Thread') as thread, \
            patch.object(api, 'create_repo_retriever', return_value=repo_retriever), \
            patch.object(api, 'index_repository', side_effect=pipeline) as index:
        assert client.get('/index/status').json()['state'] == 'idle'
        response = client.post('/index', json={'repo_url':'https://github.com/pallets/click'})
        assert response.status_code == 202 and response.json()['state'] == 'cloning'
        assert client.post('/index', json={'repo_url':'https://github.com/psf/requests'}).status_code == 409
        assert client.delete('/repos/psf/requests').status_code == 409
        assert client.post('/reindex').status_code == 409
        assert client.post('/search', json={'query':'q', 'repo':'pallets/click'}).status_code == 409
        assert client.post('/search', json={'query':'q'}).status_code == 200
        thread.call_args.kwargs['target'](*thread.call_args.kwargs['args'])
        assert stages == ['cloning', 'chunking', 'embedding', 'embedding']
        assert client.get('/index/status').json() == dict(state='done', repo='pallets/click',
            progress='812/2400 chunks', error=None)
        index.assert_called_once()
        assert client.post('/index', json={'repo_url':'https://github.com/pallets/click.git/'}).status_code == 202
        thread.call_args.kwargs['target'](*thread.call_args.kwargs['args'])


def test_index_job_failure_releases_busy(service):
    client, _, _ = service
    api = importlib.import_module('gitcontext.api')
    with patch.object(api, 'Thread') as thread, \
            patch.object(api, 'create_repo_retriever', return_value=Mock()), \
            patch.object(api, 'index_repository', side_effect=OSError('private upstream details')):
        client.post('/index', json={'repo_url':'https://github.com/pallets/click'})
        thread.call_args.kwargs['target'](*thread.call_args.kwargs['args'])
        result = client.get('/index/status').json()
        assert result['state'] == 'error' and 'private upstream' not in result['error']
        assert client.post('/index', json={'repo_url':'https://github.com/pallets/click'}).status_code == 202
        thread.call_args.kwargs['target'](*thread.call_args.kwargs['args'])


@pytest.mark.parametrize('route', ['/search', '/ask', '/ask/stream'])
def test_repo_field_routes_to_distinct_retriever(service, route):
    client, base, _ = service
    api = importlib.import_module('gitcontext.api')
    other = Mock()
    other.store.count.return_value = 1
    payload = {**base.retrieve.return_value.hits[0][1], 'file_path':'src/click/core.py'}
    other.retrieve.return_value = SimpleNamespace(hits=[(0.5, payload)], timings={'total_ms':1.0})
    with patch.object(api, 'create_repo_retriever', return_value=other) as create, \
            patch.object(answer, 'generate_answer', return_value='Answer [1]'), \
            patch.object(answer, 'stream_answer', return_value=iter(['Answer [1]'])):
        response = client.post(route, json={'query':'q', 'repo':'pallets/click'})
        assert response.status_code == 200 and 'src/click/core.py' in response.text
        assert client.post('/search', json={'query':'q', 'repo':'pallets/click'}).status_code == 200
        create.assert_called_once_with('pallets/click', base)
        base.retrieve.assert_not_called()
        assert client.post(route, json={'query':'q', 'repo':'../invalid'}).status_code == 422


def test_repo_factory_reuses_client_and_embedder(service, helper_modules):
    _, base, _ = service
    api = importlib.import_module('gitcontext.api')
    module, _ = helper_modules
    with patch.object(module, 'VectorStore', return_value=Mock()) as store, \
            patch.object(module, 'Embedder', side_effect=AssertionError('must reuse model')):
        other = api.create_repo_retriever('pallets/click', base)
    assert other.embedder is base.embedder
    store.assert_called_once_with('pallets/click', client=base.store.client)


def test_list_remove_use_shared_client(service):
    client, base, _ = service
    api = importlib.import_module('gitcontext.api')
    row = dict(repo='pallets/click', chunk_count=4, indexed_commit='abc', stale=False)
    with patch.object(api, 'list_repos', return_value=[row]) as listing, \
            patch.object(api, 'remove_repo') as remove:
        assert client.get('/repos').json() == [{**row, 'language_warning':None, 'history_notice':None}]
        listing.assert_called_once_with(client=base.store.client)
        assert client.delete('/repos/pallets/click').json() == {'removed':'pallets/click'}
        remove.assert_called_once_with('pallets/click', client=base.store.client)
        base.store.client.close.assert_not_called()


def test_index_runs_in_background_and_status_stays_available(service):
    client, _, _ = service
    api = importlib.import_module('gitcontext.api')
    started, finish = Event(), Event()
    retriever = Mock()

    def pipeline(url, identity, selected, lock, progress):
        progress('embedding', '16/32 chunks')
        started.set()
        assert finish.wait(5)

    with patch.object(api, 'create_repo_retriever', return_value=retriever), \
            patch.object(api, 'index_repository', side_effect=pipeline):
        try:
            response = client.post('/index', json={'repo_url':'https://github.com/pallets/click'})
            assert response.status_code == 202
            assert started.wait(2)
            assert client.get('/index/status').json()['progress'] == '16/32 chunks'
            assert client.post('/index', json={'repo_url':'https://github.com/psf/requests'}).status_code == 409
        finally:
            finish.set()
            api.app.state.index_thread.join(timeout=5)
    assert client.get('/index/status').json()['state'] == 'done'
    retriever.invalidate_index.assert_called_once()


def test_thread_start_failure_returns_error_and_releases_lock(service):
    client, _, _ = service
    api = importlib.import_module('gitcontext.api')
    with patch.object(api, 'Thread') as thread:
        thread.return_value.start.side_effect = RuntimeError('cannot start thread')
        for _ in range(2):
            assert client.post('/index', json={'repo_url':'https://github.com/pallets/click'}).status_code == 503
        assert client.get('/index/status').json()['state'] == 'error'
        assert api.app.state.index_thread is None


def test_history_option_and_source_metadata(service):
    client, retriever, _ = service
    payload = retriever.retrieve.return_value.hits[0][1]
    payload.update(kind='commit', sha='a' * 40, title='Fix timeout', author='Contributor',
                   date='2024-03-02T00:00:00Z', url='https://github.com/owner/repo/commit/' + 'a' * 40,
                   changed_files=['src/code.py'])
    response = client.post('/search',json={'query':'why', 'sources':'history'})
    assert response.status_code == 200
    assert response.json()['hits'][0]['history']['sha'] == 'a' * 40
    assert retriever.retrieve.call_args.args[2].sources == 'history'
    assert client.post('/search',json={'query':'why', 'sources':'invalid'}).status_code == 422
