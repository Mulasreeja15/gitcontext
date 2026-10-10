"""Mocked HTTP replies exercise rewriting, caching, and instant retries."""
from unittest.mock import Mock, patch

import pytest

from gitcontext import answer, config


@pytest.fixture(autouse=True)
def isolated_rewrites(monkeypatch):
    """Use a synthetic credential and clear cached queries on both sides."""
    monkeypatch.setenv(config.LLM_KEY_NAME, 'unit-test-placeholder')
    answer.rewrite_query.cache_clear()
    yield
    answer.rewrite_query.cache_clear()


def response(status=200, content=' result ', headers=None):
    """Build the small HTTP response surface consumed by _chat."""
    reply = Mock(status_code=status, headers=headers or {})
    reply.json.return_value = {'choices': [{'message': {'content': content}}]}
    return reply


def test_chat_success():
    messages = [{'role': 'user', 'content': 'question'}]
    reply = response()
    with patch.object(answer.requests, 'post', return_value=reply) as post:
        assert answer._chat(messages) == 'result'
    kwargs = post.call_args.kwargs
    assert post.call_args.args == (config.LLM_URL,)
    assert kwargs['json'] == {'model': config.LLM_MODEL, 'messages': messages, 'temperature': 0.1}
    assert kwargs['timeout'] == 120
    reply.raise_for_status.assert_called_once()


def test_chat_model_override():
    with patch.object(answer.requests, 'post', return_value=response()) as post:
        answer._chat([], temperature=0.0, model='different-judge')
    assert post.call_args.kwargs['json']['model'] == 'different-judge'
    assert post.call_args.kwargs['json']['temperature'] == 0.0


@pytest.mark.parametrize('headers,wait', [({'retry-after': '2'}, 2), ({}, 5), ({'retry-after': '120'}, 60)])
def test_chat_retries_429_then_succeeds(headers, wait):
    limited, success = response(429, headers=headers), response(content='ok')
    with patch.object(answer.requests, 'post', side_effect=[limited, success]) as post, \
            patch.object(answer.time, 'sleep') as sleep:
        assert answer._chat([]) == 'ok'
        assert post.call_count == 2
        sleep.assert_called_once_with(wait)
    limited.raise_for_status.assert_not_called()
    success.raise_for_status.assert_called_once()


@pytest.mark.parametrize('content,expected', [('"resolve_redirects"\nexplanation', 'resolve_redirects'),
                                             ('', 'question')])
def test_rewrite_and_cache(content, expected):
    with patch.object(answer.requests, 'post', return_value=response(content=content)) as post:
        assert answer.rewrite_query('question') == expected
        assert answer.rewrite_query('question') == expected
        post.assert_called_once()
        assert post.call_args.kwargs['json']['temperature'] == 0.0
        assert post.call_args.kwargs['timeout'] == 60


def test_chat_retry_exhaustion():
    with patch.object(answer.requests, 'post', return_value=response(429)) as post, \
            patch.object(answer.time, 'sleep') as sleep:
        with pytest.raises(SystemExit, match='Still rate limited'):
            answer._chat([])
        assert post.call_count == answer.MAX_RETRIES
        assert sleep.call_count == answer.MAX_RETRIES
