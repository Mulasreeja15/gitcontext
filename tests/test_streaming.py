"""Shared Groq retries and OpenAI streaming parsing, entirely offline."""
import json
from unittest.mock import Mock, patch

import pytest

from gitcontext import answer, config


def frame(content):
    """Construct a content delta exactly as the OpenAI SSE wire format."""
    return ('data: ' + json.dumps({'choices': [{'delta': {'content': content}}]})).encode()


def test_stream_retries_before_first_token_and_preserves_whitespace(monkeypatch):
    monkeypatch.setenv(config.LLM_KEY_NAME, 'unit-test-placeholder')
    limited = Mock(status_code=429, headers={'retry-after': '1'})
    success = Mock(status_code=200)
    success.iter_lines.return_value = iter([
        b': keepalive', b'', b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        frame('Hello'), frame(' world\n'), b'data: {"choices":[]}', b'data: [DONE]', frame('ignored')])
    with patch.object(answer.SESSION, 'post', side_effect=[limited, success]) as post, \
            patch.object(answer.time, 'sleep') as sleep:
        tokens = answer.stream_answer('q', [])
        assert post.call_count == 2  # connect/retry happens before iteration
        assert post.call_args.kwargs['stream'] is True
        assert post.call_args.kwargs['json']['stream'] is True
        assert list(tokens) == ['Hello', ' world\n']
        sleep.assert_called_once_with(1.0)
    limited.close.assert_called_once()
    success.close.assert_called_once()


def test_stream_early_close_releases_response():
    response = Mock()
    response.iter_lines.return_value = iter([frame('one'), frame('two')])
    tokens = answer._stream_tokens(response)
    assert next(tokens) == 'one'
    tokens.close()
    response.close.assert_called_once()


def test_connected_stream_can_close_before_consumption():
    response = Mock()
    tokens = answer._ResponseStream(response)
    tokens.close()
    tokens.close()
    response.close.assert_called_once()
    response.iter_lines.assert_not_called()
    assert list(tokens) == []


@pytest.mark.parametrize('line', [b'data: not-json', b'data: {"error":{"message":"failed"}}'])
def test_stream_parse_errors_release_response(line):
    response = Mock()
    response.iter_lines.return_value = iter([line])
    with pytest.raises(ValueError):
        list(answer._stream_tokens(response))
    response.close.assert_called_once()


def test_stream_retry_exhaustion(monkeypatch):
    monkeypatch.setenv(config.LLM_KEY_NAME, 'unit-test-placeholder')
    limited = Mock(status_code=429, headers={})
    with patch.object(answer.SESSION, 'post', return_value=limited) as post, \
            patch.object(answer.time, 'sleep'):
        with pytest.raises(SystemExit, match='Still rate limited'):
            answer.stream_answer('q', [])
    assert post.call_count == answer.MAX_RETRIES


def test_stream_and_buffered_answers_share_prompts():
    with patch.object(answer, '_chat', return_value='mock') as chat:
        answer.generate_answer('q', [])
        buffered = chat.call_args.args[0]
        answer.stream_answer('q', [])
        assert chat.call_args.args[0] == buffered
        assert chat.call_args.kwargs == {'stream': True}


def test_rate_limit_callback_precedes_sleep(monkeypatch):
    monkeypatch.setenv(config.LLM_KEY_NAME, 'unit-test-placeholder')
    limited = Mock(status_code=429, headers={'retry-after': '2'})
    success = Mock(status_code=200)
    success.iter_lines.return_value = iter([b'data: [DONE]'])
    events = []
    with patch.object(answer.SESSION, 'post', side_effect=[limited, success]), \
            patch.object(answer.time, 'sleep', side_effect=lambda wait: events.append(('sleep', wait))):
        stream = answer.stream_answer('q', [], on_retry=lambda wait: events.append(('retry', wait)))
        assert list(stream) == []
    assert events == [('retry', 2.0), ('sleep', 2.0)]
