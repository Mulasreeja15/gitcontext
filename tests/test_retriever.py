"""Code vocabulary and test-path detection without storage or models."""
import pytest


def test_tokenize_identifiers_and_stopwords(helper_modules):
    retriever, _ = helper_modules
    tokens = retriever.tokenize('how does the resolve_redirects HTTPAdapter and self work')
    assert {'resolve_redirects', 'resolve', 'redirects', 'httpadapter', 'http', 'adapter', 'work'} <= set(tokens)
    assert not {'how', 'does', 'the', 'and', 'self'} & set(tokens)
    assert retriever.tokenize('the and self') == []


@pytest.mark.parametrize('path,expected', [
    ('tests/x.py', True), ('test_x.py', True), ('x_test.py', True),
    ('conftest.py', True), ('src/requests/models.py', False),
])
def test_is_test(helper_modules, path, expected):
    retriever, _ = helper_modules
    assert retriever.is_test({'file_path': path}) is expected


def test_index_invalidation_retains_models(helper_modules):
    module, _ = helper_modules
    retriever = module.Retriever.__new__(module.Retriever)
    retriever.embedder, retriever._reranker = object(), object()
    embedder, reranker = retriever.embedder, retriever._reranker
    retriever._payloads = {'old': {'file_path': 'old.py'}}
    retriever._bm25 = object()
    retriever._bm25_ids = ['old']
    retriever.invalidate_index()
    assert retriever._payloads == {} and retriever._bm25 is None and retriever._bm25_ids == []
    assert retriever.embedder is embedder and retriever._reranker is reranker
