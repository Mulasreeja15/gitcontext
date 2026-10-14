"""Container initialization never needs network, storage, or models in tests."""
import importlib
from unittest.mock import Mock


def test_bootstrap_resumes_and_reuses_client(helper_modules, monkeypatch):
    deployment = importlib.import_module('gitcontext.deployment')
    client, embedder = Mock(), Mock()
    stores = {repo: Mock() for repo in deployment.config.DEMO_REPOS}
    for store in stores.values():
        store.count.return_value = 0
    states = {}
    monkeypatch.setattr(deployment, 'open_client', lambda: client)
    factory = Mock(return_value=embedder)
    monkeypatch.setattr(deployment, 'Embedder', factory)
    monkeypatch.setattr(deployment, 'VectorStore', lambda repo, client: stores[repo])
    monkeypatch.setattr(deployment, 'load_state', lambda repo: states.get(repo))

    def index(url, *, repo, store, embedder, progress):
        assert url == 'https://github.com/' + repo
        progress('embedding', '1/1 chunks')
        states[repo] = {'history_chunks': 0}
        store.count.return_value = 1

    full = Mock(side_effect=index)
    monkeypatch.setattr(deployment, 'full_index', full)

    def history(repo, **kwargs):
        assert kwargs['commits_only'] and kwargs['unshallow']
        assert kwargs['store'] is stores[repo] and kwargs['embedder'] is embedder
        states[repo]['history_chunks'] = 5
        return {'chunks_added': 5}

    history_mock = Mock(side_effect=history)
    monkeypatch.setattr(deployment, 'index_history', history_mock)
    deployment.bootstrap()
    assert full.call_count == 2
    factory.assert_called_once()
    history_mock.assert_called_once()
    client.close.assert_called_once()
    deployment.bootstrap()
    assert full.call_count == 2 and history_mock.call_count == 1


def test_container_listener_uses_config(helper_modules, monkeypatch):
    deployment = importlib.import_module('gitcontext.deployment')
    bootstrap, run = Mock(), Mock()
    monkeypatch.setattr(deployment, 'bootstrap', bootstrap)
    monkeypatch.setattr(deployment.uvicorn, 'run', run)
    monkeypatch.setattr(deployment.config, 'API_PORT', 9876)
    deployment.main()
    bootstrap.assert_called_once()
    run.assert_called_once_with('gitcontext.api:app', host='0.0.0.0', port=9876, workers=1)
