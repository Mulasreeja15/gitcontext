"""Use real Gradio routing with indexing and models disabled."""
import importlib
from threading import Event
from unittest.mock import Mock

from fastapi.testclient import TestClient


def test_space_builds_without_indexing_and_keeps_routes(helper_modules, monkeypatch):
    space = importlib.import_module('app')
    monkeypatch.setattr(space.config, 'PUBLIC_DEMO', True)
    initializer = Mock(side_effect=AssertionError('Indexing forbidden'))
    web, demo = space.build_app(start_indexing=False, initializer=initializer)
    assert demo is not None
    with TestClient(web) as client:
        assert client.get('/').status_code == 200
        assert 'GitContext' in client.get('/').text
        assert client.get('/gradio/').status_code == 200
        assert client.get('/health').json()['status'] == 'indexing'
        assert client.get('/repos').json() == []
        assert client.post('/index', json={'repo_url':'https://github.com/x/y'}).status_code == 403
        response = client.post('/search', json={'query':'q'})
        assert response.status_code == 503
        assert response.json()['detail'] == 'Indexing, ready in a few minutes'
    initializer.assert_not_called()


def test_background_initialization_does_not_block_server(helper_modules):
    space = importlib.import_module('app')
    started, release, finished = Event(), Event(), Event()
    retriever = Mock()

    def initializer(application):
        started.set()
        release.wait(timeout=10)
        application.state.retriever = retriever
        finished.set()

    web, _ = space.build_app(initializer=initializer)
    with TestClient(web) as client:
        try:
            assert started.wait(timeout=5)
            assert client.get('/').status_code == 200
            assert client.get('/health').json()['status'] == 'indexing'
        finally:
            release.set()
        assert finished.wait(timeout=5)
    retriever.store.client.close.assert_called_once()


def test_gradio_launch_uses_same_routes_and_lifespan(helper_modules, monkeypatch):
    space = importlib.import_module('app')
    web, demo = space.build_app(start_indexing=False)
    monkeypatch.setattr(space, 'build_app', lambda: (web, demo))

    def launch(**kwargs):
        assert kwargs['server_name'] == '0.0.0.0'
        assert kwargs['server_port'] == space.config.API_PORT
        assert kwargs['share'] is False
        # This is the same factory Blocks.launch uses for its underlying server.
        from gradio.routes import App
        launched = App.create_app(demo, app_kwargs=kwargs['app_kwargs'])
        with TestClient(launched) as client:
            assert client.get('/').status_code == 200
            assert 'Ask your codebase' in client.get('/').text
            assert client.get('/health').json()['status'] == 'indexing'
            assert client.get('/gradio/').status_code == 200
            assert client.get('/gradio_api/startup-events').status_code == 200

    mock_launch = Mock(side_effect=launch)
    monkeypatch.setattr(demo, 'launch', mock_launch)
    space.main()
    mock_launch.assert_called_once()
