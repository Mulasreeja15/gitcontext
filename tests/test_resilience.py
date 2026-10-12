"""Database and Groq failure handling stays offline and never takes real locks."""
import ast
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from portalocker.exceptions import AlreadyLocked

from gitcontext import answer, config, database, doctor


def test_dependency_lock_has_only_pinned_packages():
    lines = (config.PROJECT_ROOT / 'requirements-lock.txt').read_text(encoding='utf-8').splitlines()
    assert lines
    for line in lines:
        assert re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s;/]+(?:; sys_platform == "win32")?', line)
    assert any(line.startswith('qdrant-client==') for line in lines)
    assert any(line.startswith('pytest==') for line in lines)


@pytest.mark.parametrize('error', [AlreadyLocked(), RuntimeError('Storage folder is already accessed by another instance')])
def test_local_lock_has_actionable_message(monkeypatch, tmp_path, error):
    monkeypatch.setattr(config, 'QDRANT_URL', None)
    monkeypatch.setattr(config, 'QDRANT_PATH', str(tmp_path))
    with patch.object(database, 'QdrantClient', side_effect=error):
        with pytest.raises(database.DatabaseLocked, match='Ctrl \\+ C') as caught:
            database.open_client()
    assert 'taskkill /F /T /IM python.exe' in str(caught.value)
    with pytest.raises(SystemExit, match='Another GitContext process'):
        database.run_cli(Mock(side_effect=caught.value))


def test_unrelated_storage_errors_propagate(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'QDRANT_URL', None)
    monkeypatch.setattr(config, 'QDRANT_PATH', str(tmp_path))
    with patch.object(database, 'QdrantClient', side_effect=RuntimeError('corrupt database')):
        with pytest.raises(RuntimeError, match='corrupt'):
            database.open_client()


def test_remote_client_does_not_touch_local_storage(monkeypatch):
    monkeypatch.setattr(config, 'QDRANT_URL', 'http://qdrant:6333')
    with patch.object(database, 'QdrantClient') as client, patch.object(database.Path, 'mkdir') as mkdir:
        assert database.open_client() is client.return_value
        client.assert_called_once_with(url='http://qdrant:6333')
        mkdir.assert_not_called()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('failure', [requests.ConnectionError, requests.Timeout, 500, 502, 503, 504])
def test_chat_retries_transport_and_server_errors(monkeypatch, stream, failure):
    monkeypatch.setenv(config.LLM_KEY_NAME, 'synthetic-test-key')
    success = Mock(status_code=200)
    success.json.return_value = {'choices': [{'message': {'content': 'OK'}}]}
    success.iter_lines.return_value = iter([b'data: [DONE]'])
    error = failure() if isinstance(failure, type) else Mock(status_code=failure)
    with patch.object(answer.SESSION, 'post', side_effect=[error, error, success]) as post, \
            patch.object(answer.time, 'sleep') as sleep:
        result = answer._chat([], stream=stream)
        if stream:
            assert list(result) == []
        else:
            assert result == 'OK'
        assert post.call_count == 3
        assert [call.args[0] for call in sleep.call_args_list] == [5, 10]
    if not isinstance(failure, type):
        assert error.close.call_count == 2


@pytest.mark.parametrize('failure', [requests.ConnectionError(), requests.Timeout(), Mock(status_code=503)])
def test_network_retry_exhaustion_is_bounded(monkeypatch, failure):
    monkeypatch.setenv(config.LLM_KEY_NAME, 'synthetic-test-key')
    effects = [failure] * answer.MAX_RETRIES
    with patch.object(answer.SESSION, 'post', side_effect=effects) as post, \
            patch.object(answer.time, 'sleep') as sleep:
        with pytest.raises(requests.ConnectionError):
            answer._chat([])
        assert post.call_count == answer.MAX_RETRIES
        assert sleep.call_count == answer.MAX_RETRIES - 1
        assert max(call.args[0] for call in sleep.call_args_list) <= config.GROQ_RETRY_MAX_SECONDS


@pytest.fixture
def diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'INDEX_STATE_DIR', tmp_path)
    monkeypatch.setenv(config.LLM_KEY_NAME, 'synthetic-test-key')
    client = Mock()
    with patch.object(doctor.importlib, 'import_module'), patch.object(doctor, 'open_client', return_value=client):
        yield client, tmp_path


def test_doctor_offline_and_repository_fixes(diagnostic, capsys):
    client, directory = diagnostic
    (directory / 'psf__requests.json').write_text('{}')
    client.collection_exists.return_value = False
    with patch.object(doctor, 'load_state', return_value={'chunk_count': 1}), \
            patch.object(doctor.answer, '_chat', side_effect=AssertionError('offline')):
        assert not doctor.run(offline=True)
    output = capsys.readouterr().out
    assert 'GROQ_API_KEY: set' in output and 'synthetic-test-key' not in output
    assert 'collection missing' in output and 'update --repo psf/requests' in output
    assert 'skipped (--offline)' in output
    client.close.assert_called_once()


def test_doctor_fresh_and_stale_repos(diagnostic):
    client, directory = diagnostic
    for name in ['owner__fresh', 'owner__stale']:
        (directory / (name + '.json')).write_text('{}')
    client.count.return_value.count = 3
    with patch.object(doctor, 'load_state', return_value={'chunk_count': 3}), \
            patch.object(doctor, 'index_status', side_effect=[{'stale': False}, {'stale': True}]):
        rows = doctor.checks(offline=True)
    repo_rows = [row for row in rows if row[0].startswith('Repository')]
    assert [row[1] for row in repo_rows] == [True, False]


def test_doctor_lock_missing_key_import_and_version(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'INDEX_STATE_DIR', tmp_path)
    monkeypatch.delenv(config.LLM_KEY_NAME, raising=False)
    class Version:
        major, minor = 3, 13
        def __getitem__(self, item):
            return (3, 13)[item]
    with patch.object(doctor.sys, 'version_info', Version()), \
            patch.object(doctor.importlib, 'import_module', side_effect=ImportError()), \
            patch.object(doctor, 'open_client', side_effect=database.DatabaseLocked(database.LOCK_MESSAGE)), \
            patch.object(answer, '_chat', side_effect=AssertionError('no key')):
        rows = doctor.checks()
    assert not rows[0][1]
    assert any('locked' in label and not ok for label, ok, _ in rows)
    assert any('missing' in label and not ok for label, ok, _ in rows)
    assert any('Package' in label and 'requirements-lock' in fix for label, ok, fix in rows)


def test_doctor_tiny_reachability_request_is_mocked(diagnostic):
    with patch.object(answer, '_chat', return_value='OK') as chat:
        rows = doctor.checks()
    assert ('Groq reachability', True, '') in rows
    assert chat.call_args.kwargs['max_tokens'] == 1
    with patch.object(answer, '_chat', side_effect=SystemExit('rejected')):
        assert any(label == 'Groq reachability' and not ok for label, ok, _ in doctor.checks())


@pytest.mark.parametrize('entry', ['evaluate.py', 'judge.py', 'check_labels.py'])
def test_evaluation_entrypoints_use_friendly_lock_guard(entry):
    tree = ast.parse((config.PROJECT_ROOT / entry).read_text(encoding='utf-8'))
    guard = tree.body[-1]
    assert isinstance(guard, ast.If)
    namespace = {'__name__': '__main__', 'main': Mock(side_effect=database.DatabaseLocked(database.LOCK_MESSAGE))}
    with pytest.raises(SystemExit, match='Ctrl \\+ C'):
        exec(compile(ast.Module(body=[guard], type_ignores=[]), entry, 'exec'), namespace)



def test_stream_reconnects_before_first_token(monkeypatch):
    monkeypatch.setenv(config.LLM_KEY_NAME, 'synthetic-test-key')
    first, second = Mock(status_code=200), Mock(status_code=200)
    def broken():
        raise requests.Timeout()
        yield
    first.iter_lines.return_value = broken()
    second.iter_lines.return_value = iter([b'data: {"choices":[{"delta":{"content":"OK"}}]}', b'data: [DONE]'])
    with patch.object(answer.SESSION, 'post', side_effect=[first, second]) as post, \
            patch.object(answer.time, 'sleep') as sleep:
        assert list(answer._chat([], stream=True)) == ['OK']
        assert post.call_count == 2
        sleep.assert_called_once_with(5)
    first.close.assert_called_once()
    second.close.assert_called_once()


def test_stream_does_not_replay_emitted_text(monkeypatch):
    monkeypatch.setenv(config.LLM_KEY_NAME, 'synthetic-test-key')
    reply = Mock(status_code=200)
    def broken():
        yield b'data: {"choices":[{"delta":{"content":"First"}}]}'
        raise requests.ConnectionError()
    reply.iter_lines.return_value = broken()
    with patch.object(answer.SESSION, 'post', return_value=reply) as post:
        stream = answer._chat([], stream=True)
        assert next(stream) == 'First'
        with pytest.raises(requests.ConnectionError):
            next(stream)
        post.assert_called_once()
    reply.close.assert_called_once()


def test_doctor_cli_dispatch(helper_modules):
    import main
    with patch('sys.argv', ['main.py', 'doctor', '--offline']), patch.object(doctor, 'run') as run:
        main.main()
    run.assert_called_once_with(offline=True)


def test_cli_lock_message_has_no_traceback(helper_modules, capsys):
    import main
    with patch('sys.argv', ['main.py', 'repos']), \
            patch.object(main, 'cmd_repos', side_effect=database.DatabaseLocked(database.LOCK_MESSAGE)):
        with pytest.raises(SystemExit) as error:
            main.main()
    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert 'Ctrl + C' in stderr and 'Traceback' not in stderr
