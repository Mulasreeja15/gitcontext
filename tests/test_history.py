"""Commit parsing, GitHub import, filters and labels are tested without network/ML."""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from gitcontext import config, history, indexing
from test_indexing import repo, resources


def log_record(sha='a' * 40, paths=('src/code.py',)):
    return ('\x1e' + '\x1f'.join([sha, 'Author Name', '2024-03-02T12:00:00Z',
        'Fix timeout\n\nAvoid waiting forever.\nCo-authored-by: Other <private@example.invalid>'])
        + '\0' + '\0'.join('2\t1\t' + path for path in paths) + '\0')


def test_commit_parsing_and_supported_files(tmp_path):
    chunks = history.parse_commits(log_record() + log_record('b' * 40, ('build/skip.py',)), tmp_path, 'owner/repo')
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.kind == 'commit' and chunk.sha == 'a' * 40
    assert chunk.author == 'Author Name' and chunk.date == '2024-03-02T12:00:00Z'
    assert chunk.changed_files == ('src/code.py',)
    assert 'src/code.py: +2 -1' in chunk.text and 'private@example.invalid' not in chunk.text
    assert chunk.url.endswith('/commit/' + 'a' * 40)
    with patch.object(history, 'git', return_value=log_record()) as git:
        assert history.commit_chunks(tmp_path, 'owner/repo')
    args = git.call_args.args
    assert '-500' in args and '--numstat' in args and '-p' not in args


@pytest.mark.parametrize('output,result', [('true',True), ('false',False)])
def test_shallow_detection(tmp_path, output, result):
    with patch.object(history, 'git', return_value=output) as git:
        assert history.is_shallow(tmp_path) is result
    git.assert_called_once_with(tmp_path, 'rev-parse', '--is-shallow-repository')


def merged_pr():
    return dict(number=42, title='Explain timeout change', body='Avoid repeated waits.',
                merged_at='2024-03-02T00:00:00Z', user={'login':'maintainer'}, labels=[{'name':'bug'}])


def test_pr_chunk_creation_and_splitting(monkeypatch):
    monkeypatch.setattr(config, 'MAX_CHUNK_CHARS', 220)
    comments = [dict(user={'login':'reviewer'}, created_at='now', path='src/code.py', body='Reason: ' + 'long review ' * 100)]
    chunks = history.pr_chunks(merged_pr(), [{'filename':'src/code.py'}], comments, 'owner/repo')
    assert len(chunks) > 1
    assert all(chunk.kind == 'pr' and chunk.number == 42 and chunk.author == 'maintainer' for chunk in chunks)
    assert all(chunk.changed_files == ('src/code.py',) and chunk.labels == ('bug',) for chunk in chunks)
    assert all(chunk.url == 'https://github.com/owner/repo/pull/42' for chunk in chunks)
    assert 'reviewer' in '\n'.join(chunk.text for chunk in chunks)


def response(value, status=200, headers=None):
    return Mock(status_code=status, headers=headers or {}, json=Mock(return_value=value))


def test_github_pages_reviews_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'GITHUB_CACHE_DIR', tmp_path)
    monkeypatch.setattr(config, 'GITHUB_PAGE_SIZE', 2)
    monkeypatch.setattr(config, 'PR_MAX', 1)
    replies = [response([{'merged_at':None}, merged_pr()]),
        response([{'filename':'a.py'}, {'filename':'b.py'}]), response([]),
        response([{'body':'Issue discussion'}]), response([{'body':'Inline review'}]),
        response([{'body':'Review summary'}])]
    with patch.object(history.requests, 'get', side_effect=replies) as get:
        chunks = history.GitHubClient('owner/repo', 'secret-test-token').discussions()
        assert get.call_count == 6
        assert len(history.GitHubClient('owner/repo', 'secret-test-token').discussions()) == len(chunks)
        assert get.call_count == 6
    text = '\n'.join(chunk.text for chunk in chunks)
    assert all(value in text for value in ['Issue discussion','Inline review','Review summary'])
    assert len(list((tmp_path / 'owner__repo').glob('*.json'))) == 6
    assert all('secret-test-token' not in path.read_text(encoding='utf-8') for path in tmp_path.rglob('*.json'))


def test_github_rate_limit_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'GITHUB_CACHE_DIR', tmp_path)
    with patch.object(history.requests, 'get', side_effect=[response([],429,{'Retry-After':'2'}), response([])]) as get, \
            patch.object(history.time, 'time', return_value=100), patch.object(history.time, 'sleep') as sleep:
        assert history.GitHubClient('owner/repo','token').get('pulls',1) == []
        assert get.call_count == 2
        sleep.assert_called_once_with(2)


def test_success_exhausted_quota_waits_before_next_request(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'GITHUB_CACHE_DIR', tmp_path)
    with patch.object(history.requests, 'get', side_effect=[response([],200,{'X-RateLimit-Remaining':'0',
                'X-RateLimit-Reset':'105'}), response([])]), \
            patch.object(history.time, 'time', return_value=100), patch.object(history.time, 'sleep') as sleep:
        client = history.GitHubClient('owner/repo','token')
        client.get('pulls',1)
        client.get('pulls',2)
        sleep.assert_called_once_with(5)


def test_history_publication_preserves_code_and_missing_token(repo, resources, monkeypatch):
    store, embedder = resources
    indexing.full_index(str(repo), store=store, embedder=embedder)
    monkeypatch.delenv(config.GITHUB_KEY_NAME, raising=False)
    chunks = history.parse_commits(log_record(), repo, 'local/repository')
    with patch.object(history, 'commit_chunks', return_value=chunks), \
            patch.object(history.requests, 'get', side_effect=AssertionError('No token: no network')):
        result = history.index_history('local/repository',store=store,embedder=embedder)
        assert result['commits'] == 1 and 'GITHUB_TOKEN is not set' in result['messages'][0]
        assert store.count() == 5
        assert len(store.all_payloads(sources='code')) == 4
        assert len(store.all_payloads(sources='history')) == 1
        assert all(p['kind'] == 'commit' for _,_,p in store.search([1.,0.,0.],sources='history'))
        assert all(p['kind'] != 'commit' for _,_,p in store.search([1.,0.,0.]))
        assert len(store.search([1.,0.,0.],sources='all')) == 5
        history.index_history('local/repository',store=store,embedder=embedder)
        assert store.count() == 5
    indexing.full_index(str(repo),store=store,embedder=embedder)
    assert store.count() == 5 and len(store.all_payloads(sources='history')) == 1


def test_shallow_message_and_unshallow_option(repo, resources):
    store, embedder = resources
    indexing.full_index(str(repo),store=store,embedder=embedder)
    with patch.object(history, 'is_shallow', return_value=True), \
            patch.object(history, 'commit_chunks') as commits:
        result = history.index_history('local/repository',commits_only=True,store=store,embedder=embedder)
        assert 'Shallow clone' in result['messages'][0] and '--unshallow' in result['messages'][0]
        commits.assert_not_called()
    with patch.object(history, 'is_shallow', side_effect=[True,False]), \
            patch.object(history.subprocess, 'run') as run, patch.object(history,'commit_chunks',return_value=[]), \
            patch.object(history, 'git', return_value='a' * 40):
        history.index_history('local/repository',unshallow=True,commits_only=True,store=store,embedder=embedder)
        assert run.call_args.args[0][-2:] == ['fetch','--unshallow']


def test_history_labels(helper_modules):
    _, evaluate = helper_modules
    assert evaluate.is_match({'kind':'commit','sha':'abcdef1234'},'commit::abcdef')
    assert not evaluate.is_match({'kind':'pr','number':42},'commit::abcdef')
    assert evaluate.is_match({'kind':'pr','number':42},'pr::42')
    assert not evaluate.is_match({'kind':'pr','number':43},'pr::42')
    questions = evaluate.load_questions(config.PROJECT_ROOT / 'eval' / 'history_questions.jsonl')
    assert len(questions) == 8


def test_bm25_sources_do_not_leak(repo, resources, helper_modules, monkeypatch):
    store, embedder = resources
    indexing.full_index(str(repo),store=store,embedder=embedder)
    monkeypatch.delenv(config.GITHUB_KEY_NAME,raising=False)
    with patch.object(history,'commit_chunks',return_value=history.parse_commits(log_record(),repo,'local/repository')):
        history.index_history('local/repository',store=store,embedder=embedder)
    module, _ = helper_modules
    retriever = module.Retriever.__new__(module.Retriever)
    retriever.store, retriever.embedder = store, embedder
    retriever._payloads, retriever._bm25_ids = {}, []
    retriever._bm25, retriever._reranker = None, None
    for source in ['history','code','all','history']:
        hits = retriever.retrieve('timeout',opts=module.RetrievalOptions(bm25=True,sources=source)).hits
        assert hits
        if source == 'history':
            assert all(p['kind'] == 'commit' for _,p in hits)
        if source == 'code':
            assert all(p['kind'] != 'commit' for _,p in hits)
