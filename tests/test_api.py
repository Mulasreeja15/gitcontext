"""Service contracts and lifecycle with all retrieval and LLM calls mocked."""
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from gitcontext import answer


@pytest.fixture
def service(helper_modules, tmp_path, monkeypatch):
    """Use real HTTP routing with a single mocked persistent retriever."""
    api = importlib.import_module('gitcontext.api')
    monkeypatch.setattr(api.config, 'INDEX_STATE_DIR', tmp_path / 'index_state')
    payload = {'file_path': 'tests/test_x.py', 'start_line': 2, 'end_line': 5,
               'kind': 'function', 'name': 'test_x', 'language': 'python', 'text': 'def test_x(): pass'}
    retriever = Mock()
    retriever.store.count.return_value = 10
    retriever.retrieve.return_value = SimpleNamespace(hits=[(0.75, payload)],
                                                       timings={'search_ms': 2.0, 'total_ms': 3.0})
    with patch.object(api, 'create_retriever', return_value=retriever) as factory, \
            patch.object(answer.requests, 'post', side_effect=AssertionError('Network forbidden')):
        with TestClient(api.app) as client:
            yield client, retriever, factory
    retriever.store.client.close.assert_called_once()


def test_health_and_retriever_loaded_once(service):
    client, _, factory = service
    assert client.get('/health').json() == {'status': 'ok', 'chunks_indexed': 10,
                                          'repo_name': None, 'stale': True, 'indexed_commit': None}
    client.post('/search', json={'query': 'question'})
    client.post('/search', json={'query': 'another'})
    factory.assert_called_once()


@pytest.mark.parametrize('method,path', [('post', '/index'), ('post', '/reindex'),
                                        ('delete', '/repos/psf/requests')])
def test_public_demo_blocks_mutations_before_validation(service, monkeypatch, method, path):
    client, retriever, _ = service
    from gitcontext import config
    monkeypatch.setattr(config, 'PUBLIC_DEMO', True)
    response = getattr(client, method)(path)
    assert response.status_code == 403
    assert 'disabled in the public demo' in response.json()['detail']
    retriever.retrieve.assert_not_called()
    assert client.get('/settings').json() == {'public_demo': True, 'max_query_chars': 500}
    html = client.get('/').text
    assert 'id="open-repository" hidden disabled' in html
    assert 'id="index-button" hidden disabled' in html


@pytest.mark.parametrize('path', ['/search', '/ask', '/ask/stream'])
def test_public_demo_query_limit(service, monkeypatch, path):
    client, retriever, _ = service
    from gitcontext import config
    monkeypatch.setattr(config, 'PUBLIC_DEMO', True)
    assert client.post(path, json={'query': 'x' * 501}).status_code == 422
    retriever.retrieve.assert_not_called()


def test_public_rate_limit_shared_across_all_routes(service, monkeypatch):
    client, retriever, _ = service
    from gitcontext import config
    monkeypatch.setattr(config, 'PUBLIC_DEMO', True)
    for _ in range(5):
        assert client.post('/search', json={'query': 'x' * 500}).status_code == 200
    for path in ['/search', '/ask', '/ask/stream']:
        response = client.post(path, json={'query': 'q'})
        assert response.status_code == 429
        assert response.headers['retry-after'] == '60'
        assert response.json()['detail'] == 'Too many requests, try again in a minute.'
    assert retriever.retrieve.call_count == 5


def test_rate_limiter_clients_expiry_and_spoofed_headers(service, monkeypatch):
    client, _, _ = service
    from gitcontext import config
    from gitcontext.api import PublicRateLimiter
    monkeypatch.setattr(config, 'PUBLIC_DEMO', True)
    monkeypatch.setattr(config, 'PUBLIC_RATE_REQUESTS', 1)
    limiter = PublicRateLimiter()
    assert limiter.allow('a', now=0)
    assert not limiter.allow('a', now=59)
    assert limiter.allow('b', now=59)
    assert limiter.allow('a', now=60)
    assert limiter.allow('c', now=120)
    assert list(limiter.clients) == ['c']
    assert client.post('/search', json={'query': 'q'}).status_code == 200
    assert client.post('/search', json={'query': 'q'},
                       headers={'X-Forwarded-For': 'fake-address'}).status_code == 429


def test_local_mode_is_not_limited(service, monkeypatch):
    client, _, _ = service
    from gitcontext import config
    monkeypatch.setattr(config, 'PUBLIC_DEMO', False)
    monkeypatch.setattr(config, 'PUBLIC_RATE_REQUESTS', 1)
    for _ in range(2):
        assert client.post('/search', json={'query': 'x' * 501}).status_code == 200


def test_search_defaults_and_metadata(service):
    client, retriever, _ = service
    response = client.post('/search', json={'query': ' question '})
    assert response.status_code == 200
    body = response.json()
    assert body['hits'] == [{'file_path': 'tests/test_x.py', 'start_line': 2, 'end_line': 5,
                            'kind': 'function', 'name': 'test_x', 'score': 0.75, 'is_test': True,
                            'text': 'def test_x(): pass', 'last_commit_sha': None,
                            'last_author': None, 'last_modified': None, 'history': None}]
    assert body['timings'] == {'search_ms': 2.0, 'total_ms': 3.0}
    query, k, options = retriever.retrieve.call_args.args
    assert query == 'question' and k == 6
    assert not options.bm25 and not options.rerank and not options.rewrite and options.penalize_tests


def test_search_option_mapping(service):
    client, retriever, _ = service
    assert client.post('/search', json={'query': 'q', 'k': 20, 'bm25': True,
                                        'rerank': True, 'rewrite': True, 'include_tests': True}).status_code == 200
    _, k, options = retriever.retrieve.call_args.args
    assert k == 20 and options.bm25 and options.rerank and options.rewrite and not options.penalize_tests


def test_metadata_options_and_response(service):
    client, retriever, _ = service
    payload = retriever.retrieve.return_value.hits[0][1]
    payload.update(last_author='Author', last_commit_sha='abc1234', last_modified='2024-03-02T00:00:00Z')
    response = client.post('/search', json={'query':'q', 'since':'2024-03-02', 'author':'Author'})
    assert response.json()['hits'][0]['last_author'] == 'Author'
    options = retriever.retrieve.call_args.args[2]
    assert options.since == '2024-03-02' and options.author == 'Author'
    assert client.post('/search', json={'query':'q', 'since':'2024-02-30'}).status_code == 422


@pytest.mark.parametrize('path', ['/search', '/ask', '/ask/stream'])
@pytest.mark.parametrize('body', [{'query': ''}, {'query': '   '}, {'query': 'q', 'k': 0},
                                 {'query': 'q', 'k': 21}, {}])
def test_validation(service, path, body):
    client, retriever, _ = service
    response = client.post(path, json=body)
    assert response.status_code == 422
    assert response.json()['detail']
    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize('path', ['/search', '/ask', '/ask/stream'])
def test_empty_index_is_503(service, path):
    client, retriever, _ = service
    retriever.store.count.return_value = 0
    assert client.get('/health').json()['chunks_indexed'] == 0
    response = client.post(path, json={'query': 'q'})
    assert response.status_code == 503
    assert 'Index is empty' in response.json()['detail']
    retriever.retrieve.assert_not_called()


def test_ask(service):
    client, retriever, _ = service
    with patch.object(answer, 'generate_answer', return_value='Answer [1]') as generate:
        response = client.post('/ask', json={'query': 'question'})
    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == 'Answer [1]'
    assert body['sources'][0]['name'] == 'test_x'
    assert body['timings']['retrieval_ms'] == 3.0
    assert body['timings']['total_ms'] >= 3.0
    generate.assert_called_once_with('question', retriever.retrieve.return_value.hits)


def test_stream_sources_first_and_tokens(service):
    client, _, _ = service
    with patch.object(answer, 'stream_answer', return_value=iter(['Hello', ' world\n', '[1]'])):
        response = client.post('/ask/stream', json={'query': 'q'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    events = response.text.strip().split('\n\n')
    assert [event.splitlines()[0] for event in events] == [
        'event: sources', 'event: token', 'event: token', 'event: token', 'event: done']
    values = [json.loads(event.splitlines()[1][6:]) for event in events]
    assert values[0]['sources'][0]['score'] == 0.75
    assert ''.join(value['token'] for value in values[1:-1]) == 'Hello world\n[1]'
    assert 'answer_ms' in values[-1]['timings']


@pytest.mark.parametrize('path,function', [('/ask', 'generate_answer')])
def test_llm_unavailable_returns_503(service, path, function):
    client, _, _ = service
    with patch.object(answer, function, side_effect=SystemExit('missing credential')):
        response = client.post(path, json={'query': 'q'})
    assert response.status_code == 503
    assert 'unavailable' in response.json()['detail']


def test_stream_unavailable_keeps_sources_and_sends_error(service):
    client, _, _ = service
    with patch.object(answer, 'stream_answer', side_effect=SystemExit('missing credential')):
        response = client.post('/ask/stream', json={'query': 'q'})
    assert response.status_code == 200
    assert response.text.startswith('event: sources')
    assert 'event: error' in response.text and 'unavailable' in response.text


def test_home_serves_self_contained_html(service):
    client, retriever, _ = service
    response = client.get('/')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/html')
    assert '<title>GitContext' in response.text
    assert 'Search only' in response.text and '/ask/stream' in response.text
    assert 'id="repo-picker"' in response.text and 'id="index-form"' in response.text
    assert 'id="open-repository"' in response.text
    dialog = response.text.split('<dialog ', 1)[1].split('</dialog>', 1)[0]
    assert 'id="index-form"' in dialog and 'aria-labelledby="add-repository-title"' in dialog
    assert 'id="index-form"' not in response.text.split('<main>', 1)[1].split('</main>', 1)[0]
    assert '/index/status' in response.text and 'repo:selectedRepo' in response.text
    assert 'innerHTML' not in response.text
    assert '<script src=' not in response.text
    retriever.retrieve.assert_not_called()


def test_health_repo_name_from_index_metadata(service):
    client, _, _ = service
    from gitcontext import config
    from gitcontext.repositories import state_path
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({'repo_id': config.DEFAULT_REPO, 'repo_name': 'requests', 'repo_path': 'missing-repo',
        'remote_url': None, 'indexed_commit': None, 'timestamp': 'now',
        'chunk_count': 10, 'embed_model': config.EMBED_MODEL}), encoding='utf-8')
    assert client.get('/health').json()['repo_name'] == 'requests'


def test_reindex_uses_existing_client_and_invalidates_caches(service):
    client, retriever, _ = service
    api = importlib.import_module('gitcontext.api')
    summary = dict(mode='incremental', commits_behind=1, files_added=1,
        files_modified=0, files_deleted=0, files_renamed=0, chunks_removed=0,
        chunks_added=1, seconds=0.01, state={})
    with patch.object(api, 'load_state', return_value={'repo_path': 'repo'}), \
            patch.object(api, 'update_index', return_value=summary) as update:
        response = client.post('/reindex')
    assert response.status_code == 200 and response.json() == summary
    update.assert_called_once_with('repo', store=retriever.store, embedder=retriever.embedder)
    retriever.invalidate_index.assert_called_once()


def test_reindex_requires_state(service):
    client, _, _ = service
    assert client.post('/reindex').status_code == 409


def test_retry_status_event_arrives_before_tokens(service):
    client, _, _ = service

    def stream(query, hits, on_retry=None):
        on_retry(5)
        return iter(['answer'])

    with patch.object(answer, 'stream_answer', side_effect=stream):
        response = client.post('/ask/stream', json={'query': 'q'})
    assert response.text.startswith('event: sources')
    assert 'Groq is rate limiting, retrying...' in response.text
    assert response.text.index('event: status') < response.text.index('event: token')


def test_stream_error_event_and_cleanup(service):
    client, _, _ = service
    closed = Mock()

    def broken_stream():
        try:
            yield 'partial'
            raise requests.ConnectionError('sensitive upstream details')
        finally:
            closed()

    with patch.object(answer, 'stream_answer', return_value=broken_stream()):
        response = client.post('/ask/stream', json={'query': 'q'})
    assert 'event: error' in response.text
    assert 'sensitive upstream details' not in response.text
    closed.assert_called_once()


def test_rewrite_failure_is_503(service):
    client, retriever, _ = service
    retriever.retrieve.side_effect = SystemExit('missing credential')
    assert client.post('/search', json={'query': 'q', 'rewrite': True}).status_code == 503
