"""Answer evaluation stays offline with mocked retrieval and LLM responses."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gitcontext import answer, config, judge

SCORE = {'correctness': 4, 'faithfulness': 0.75,
         'unsupported_claims': ['An extra claim.'], 'reasoning': 'Mostly correct.'}


@pytest.mark.parametrize('wrapper', ['{}', '```json\n{}\n```', '```\n{}\n```'])
def test_parse_json_and_fences(wrapper):
    assert judge.parse_judgment(wrapper.format(json.dumps(SCORE))) == SCORE


@pytest.mark.parametrize('changes', [
    {'correctness': 0}, {'correctness': 6}, {'correctness': True},
    {'faithfulness': -0.1}, {'faithfulness': 1.1}, {'faithfulness': float('nan')},
    {'faithfulness': True}, {'unsupported_claims': 'claim'},
    {'unsupported_claims': [1]}, {'reasoning': ''}, {'extra': 'field'},
])
def test_invalid_schema(changes):
    with pytest.raises(ValueError):
        judge.parse_judgment(json.dumps({**SCORE, **changes}))


def test_json_retry_once():
    with patch.object(answer, '_chat', side_effect=['not JSON', json.dumps(SCORE)]) as chat:
        assert judge.judge_answer('question', 'answer', [], 'gold') == SCORE
        assert chat.call_count == 2
        assert chat.call_args.kwargs == {'temperature': 0.0, 'model': config.JUDGE_MODEL}
        assert 'Return only JSON' in chat.call_args.args[0][-1]['content']
    with patch.object(answer, '_chat', return_value='invalid') as chat:
        with pytest.raises(ValueError, match='after one retry'):
            judge.judge_answer('question', 'answer', [], 'gold')
        assert chat.call_count == 2


@pytest.fixture
def example(helper_modules):
    """Supply two sources, only the second matching the expected label."""
    payload = {'file_path': 'src/code.py', 'name': 'work', 'kind': 'function',
               'start_line': 1, 'end_line': 2, 'language': 'python',
               'text': 'def work():\n    return 1'}
    other = {**payload, 'file_path': 'other.py'}
    hits = [(0.5, other), (0.4, payload)]
    questions = [{'question': 'What does work do?', 'expected': ['src/code.py::work']}]
    retriever = Mock()
    retriever.retrieve.return_value = SimpleNamespace(hits=hits)
    retriever.store.all_payloads.return_value = [('other', other), ('gold', payload)]
    return questions, hits, retriever


@pytest.mark.parametrize('text,expected', [
    ('Supported [2]', True), ('Wrong [1]', False), ('Both [1][2]', True),
    ('No citation', False), ('Invalid [0][-1][3][999]', False),
    ('Supported 【2】', True), ('Supported 【2†L1-L22】', True),
    ('Supported 【2†lines\u202f106‑107】', True),
    ('Both 【1†L1-L2】【2†L1-L2】', True),
    ('Wrong 【1】', False), ('Wrong 【1†L1-L2】', False),
    ('Invalid 【0】【3】【999】【-1】', False),
    ('Not citations ["Authorization"] 【2foo】 【2†L1\nL2】', False),
])
def test_citations(example, text, expected):
    questions, hits, _ = example
    assert judge.cites_correct_source(text, hits, questions[0]['expected']) is expected


def test_cached_six_source_answer_keeps_original_numbering(example):
    """Older cached answers must not be sliced to today's JUDGE_K=4."""
    questions, hits, _ = example
    old_hits = [hits[0]] * 5 + [hits[1]]
    assert config.JUDGE_K == 4
    assert judge.cites_correct_source('Supported 【6†L1-L2】', old_hits,
                                      questions[0]['expected'])
    assert not judge.cites_correct_source('Different source 【1†L1-L2】', old_hits,
                                          questions[0]['expected'])


def test_gold_context_and_prompt(example):
    questions, hits, retriever = example
    payloads = [p for _, p in retriever.store.all_payloads()]
    gold = judge.gold_context(payloads, questions[0]['expected'])
    assert 'src/code.py:1-2' in gold and 'def work()' in gold
    assert 'other.py' not in gold
    with patch.object(answer, '_chat', return_value=json.dumps(SCORE)) as chat:
        judge.judge_answer('question', 'answer [2]', hits, gold)
        prompt = json.loads(chat.call_args.args[0][1]['content'])
        assert prompt['gold_context'] == gold
        assert '[1] other.py' in prompt['numbered_sources']
        assert '[2] src/code.py' in prompt['numbered_sources']
    with pytest.raises(ValueError, match='No gold context'):
        judge.gold_context(payloads, ['missing.py'])


def test_cache_skips_generation_judging_and_retrieval(example, tmp_path):
    questions, _, retriever = example
    cache = tmp_path / 'cache.jsonl'
    with patch.object(answer, 'generate_answer', return_value='It returns 1 [2].') as generate, \
            patch.object(answer, '_chat', return_value=json.dumps(SCORE)) as chat:
        first = judge.evaluate_answers(questions, cache_path=cache, retriever=retriever)
        second = judge.evaluate_answers(questions, cache_path=cache, retriever=retriever)
        assert first == second
        generate.assert_called_once()
        chat.assert_called_once()
        retriever.retrieve.assert_called_once()
        opts = retriever.retrieve.call_args.args[2]
        assert not opts.bm25 and not opts.rerank and opts.penalize_tests and not opts.rewrite
        assert retriever.retrieve.call_args.args[1] == config.JUDGE_K == 4
        assert first[0]['cites_correct_source']
    with patch.object(answer, 'generate_answer', side_effect=AssertionError('must be cached')), \
            patch.object(answer, '_chat', side_effect=AssertionError('must be cached')), \
            patch('gitcontext.retriever.Retriever', side_effect=AssertionError('must be cached')):
        assert judge.evaluate_answers(questions, cache_path=cache) == first


def test_partial_cache_resumes_judgment(example, tmp_path):
    questions, _, retriever = example
    cache = tmp_path / 'cache.jsonl'
    with patch.object(answer, 'generate_answer', return_value='Answer [2]') as generate, \
            patch.object(answer, '_chat', side_effect=['bad', 'bad', json.dumps(SCORE)]) as chat:
        with pytest.raises(ValueError):
            judge.evaluate_answers(questions, cache_path=cache, retriever=retriever)
        rows = judge.evaluate_answers(questions, cache_path=cache, retriever=retriever)
        generate.assert_called_once()
        assert chat.call_count == 3
        assert rows[0]['judgment'] == SCORE


def test_cache_config_and_model_separation(example, tmp_path, monkeypatch):
    questions, _, retriever = example
    cache = tmp_path / 'cache.jsonl'
    with patch.object(answer, 'generate_answer', return_value='Answer [2]') as generate, \
            patch.object(answer, '_chat', return_value=json.dumps(SCORE)):
        rows = judge.evaluate_answers(questions, rewrite=True, cache_path=cache, retriever=retriever)
        assert len(rows) == 2
        assert retriever.retrieve.call_args.args[2].rewrite
        monkeypatch.setattr(config, 'JUDGE_MODEL', 'different-judge')
        judge.evaluate_answers(questions, cache_path=cache, retriever=retriever)
        monkeypatch.setattr(config, 'LLM_MODEL', 'different-answer')
        judge.evaluate_answers(questions, cache_path=cache, retriever=retriever)
        assert generate.call_count == 4


def test_no_cache_neither_reads_nor_writes(example, tmp_path):
    questions, _, retriever = example
    cache = tmp_path / 'cache.jsonl'
    with patch.object(answer, 'generate_answer', return_value='Answer [2]') as generate, \
            patch.object(answer, '_chat', return_value=json.dumps(SCORE)):
        for _ in range(2):
            judge.evaluate_answers(questions, no_cache=True, cache_path=cache, retriever=retriever)
        assert generate.call_count == 2
        assert not cache.exists()


def test_report_aggregates_and_orders_low_scores():
    rows = [{'config': 'vector', 'question': f'Question {i}|extra',
             'expected': ['src/code.py'],
             'hits': [(1.0, {'file_path': 'src/code.py', 'name': 'work', 'text': 'def work(): pass'})],
             'answer': 'Claim [1]' if i % 2 == 1 else 'No citation',
             'judgment': {**SCORE, 'correctness': i, 'faithfulness': 1.0,
                          'unsupported_claims': []}, 'cites_correct_source': i % 2 == 1}
            for i in range(1, 6)]
    report = judge.render_report(rows)
    assert '| vector | 3.00 | 1.000 | 60.0% | 100.0% |' in report
    assert 'Question 1\\|extra' in report
    worst = report.split('## Five lowest-scoring questions')[1]
    assert worst.index('Question 1') < worst.index('Question 5')
    assert worst.count('Mostly correct.') == 5



def test_prompt_truncation_preserves_sources_and_gold(example):
    _, hits, _ = example
    long_hits = [(score, {**payload, 'text': 'x' * 4000}) for score, payload in hits]
    with patch.object(answer, '_chat', return_value=json.dumps(SCORE)) as chat:
        judge.judge_answer('question', 'answer', long_hits, 'g' * 5000)
    data = json.loads(chat.call_args.args[0][1]['content'])
    assert len(data['gold_context']) == config.JUDGE_MAX_GOLD_CHARS
    assert data['gold_context'].endswith('... [truncated]')
    assert data['numbered_sources'].count('... [truncated]') == 2
    assert 'x' * config.JUDGE_MAX_SOURCE_CHARS not in data['numbered_sources']
    assert '[1] other.py' in data['numbered_sources']
    assert '[2] src/code.py' in data['numbered_sources']
    assert long_hits[0][1]['text'] == 'x' * 4000
    assert judge.truncate_context('short', 30) == 'short'


def test_judge_413_retries_half_context_and_reuses_answer(example, tmp_path):
    questions, _, retriever = example
    payload = {**retriever.retrieve.return_value.hits[1][1], 'text': 'x' * 5000}
    retriever.retrieve.return_value.hits = [(0.5, payload)]
    retriever.store.all_payloads.return_value = [('gold', payload)]
    with patch.object(answer, 'generate_answer', return_value='Answer [1]') as generate, \
            patch.object(answer, '_chat', side_effect=[answer.RequestTooLarge(), json.dumps(SCORE)]) as chat:
        rows = judge.evaluate_answers(questions, cache_path=tmp_path / 'cache', retriever=retriever)
        generate.assert_called_once()
        first, second = [json.loads(call.args[0][1]['content']) for call in chat.call_args_list]
        assert len(first['gold_context']) == 3000
        assert len(second['gold_context']) == 1500
        assert 'x' * 750 not in second['numbered_sources']
        assert rows[0]['judgment'] == SCORE


def test_413_skips_continues_and_caches_terminal_results(example, tmp_path):
    questions, _, retriever = example
    questions = questions + [{**questions[0], 'question': 'Next question'}]
    cache = tmp_path / 'cache'
    with patch.object(answer, 'generate_answer', return_value='Answer [2]'), \
            patch.object(answer, '_chat', side_effect=[answer.RequestTooLarge(), answer.RequestTooLarge(), json.dumps(SCORE)]) as chat:
        rows = judge.evaluate_answers(questions, cache_path=cache, retriever=retriever)
        assert chat.call_count == 3
    assert rows[0]['status'] == 'skipped: too large'
    assert 'judgment' not in rows[0]
    assert rows[1]['judgment'] == SCORE
    report = judge.render_report(rows)
    assert 'Skipped questions: 1' in report
    assert 'skipped: too large' in report
    assert '| 4.00 | 0.750 | 100.0% | 0.0% | 1 |' in report
    assert 'Next question' in report.split('## Five lowest-scoring questions')[1]
    assert 'What does work do?' not in report.split('## Five lowest-scoring questions')[1]
    with patch.object(answer, '_chat', side_effect=AssertionError('cached')), \
            patch.object(answer, 'generate_answer', side_effect=AssertionError('cached')):
        assert judge.evaluate_answers(questions, cache_path=cache, retriever=retriever) == rows
    assert 'N/A' in judge.render_report(rows[:1])


@pytest.mark.parametrize('succeeds', [True, False])
def test_answer_generation_413_uses_smaller_sources(example, tmp_path, succeeds):
    questions, _, retriever = example
    payload = {**retriever.retrieve.return_value.hits[1][1], 'text': 'x' * 4000}
    retriever.retrieve.return_value.hits = [(0.5, payload)]
    effects = [answer.RequestTooLarge(), 'Answer [1]' if succeeds else answer.RequestTooLarge()]
    with patch.object(answer, 'generate_answer', side_effect=effects) as generate, \
            patch.object(answer, '_chat', return_value=json.dumps(SCORE)) as chat:
        rows = judge.evaluate_answers(questions, cache_path=tmp_path / 'cache', retriever=retriever)
        assert [len(call.args[1][0][1]['text']) for call in generate.call_args_list] == [1500, 750]
        assert chat.call_count == int(succeeds)
    if not succeeds:
        assert rows[0]['status'] == 'skipped: too large'



def test_completed_judgment_is_saved_before_next_question_crashes(example, tmp_path):
    questions, _, retriever = example
    questions = questions + [{**questions[0], 'question': 'Later question'}]
    path = tmp_path / 'cache.jsonl'
    with patch.object(answer, 'generate_answer', return_value='Answer [2]'), \
            patch.object(answer, '_chat', side_effect=[json.dumps(SCORE), RuntimeError('interrupted')]):
        with pytest.raises(RuntimeError, match='interrupted'):
            judge.evaluate_answers(questions, cache_path=path, retriever=retriever)
    cache = judge.JudgeCache(path)
    from gitcontext.retriever import RetrievalOptions
    first = cache.get(questions[0]['question'], RetrievalOptions().describe())
    assert first['judgment'] == SCORE
    with patch.object(answer, 'generate_answer', side_effect=AssertionError('already cached')), \
            patch.object(answer, '_chat', return_value=json.dumps(SCORE)) as chat:
        rows = judge.evaluate_answers(questions, cache_path=path, retriever=retriever)
    chat.assert_called_once()
    assert len(rows) == 2



def test_prompt_variants_have_distinct_cache_keys_and_reports(example, tmp_path):
    questions, _, retriever = example
    path = tmp_path / 'cache.jsonl'
    with patch.object(answer, '_chat', side_effect=['Default [2]', json.dumps(SCORE),
                                                 'Strict [2]', json.dumps(SCORE)]) as chat:
        default = judge.evaluate_answers(questions, cache_path=path, retriever=retriever, prompt='default')
        strict = judge.evaluate_answers(questions, cache_path=path, retriever=retriever, prompt='strict')
        assert chat.call_args_list[0].args[0][0]['content'] == answer.SYSTEM_PROMPT
        assert chat.call_args_list[2].args[0][0]['content'] == answer.STRICT_SYSTEM_PROMPT
        assert chat.call_count == 4
    assert default[0]['config'].endswith('(prompt: default)')
    assert strict[0]['config'].endswith('(prompt: strict)')
    assert judge.JudgeCache.key(default[0]) != judge.JudgeCache.key({**default[0], 'prompt': 'strict'})
    with patch.object(answer, '_chat', side_effect=AssertionError('already cached')), \
            patch.object(answer, 'generate_answer', side_effect=AssertionError('already cached')):
        assert judge.evaluate_answers(questions, cache_path=path, retriever=retriever, prompt='strict') == strict
        combined = judge.comparison_rows(questions, strict, cache_path=path)
    assert len(combined) == 2
    report = judge.render_report(combined)
    assert '(prompt: default)' in report and '(prompt: strict)' in report
    assert judge.comparison_rows(questions, strict, cache_path=path, no_cache=True) == strict


def test_legacy_cache_is_default_only_and_needs_no_llm(example, tmp_path):
    from gitcontext.retriever import RetrievalOptions
    questions, hits, _ = example
    path = tmp_path / 'cache.jsonl'
    legacy = {'question': questions[0]['question'], 'expected': questions[0]['expected'],
              'config': RetrievalOptions().describe(), 'model': config.JUDGE_MODEL,
              'answer_model': config.LLM_MODEL, 'hits': hits, 'answer': 'Answer [2]',
              'gold': 'gold', 'judgment': SCORE}
    cache = judge.JudgeCache(path)
    cache.save(legacy)
    assert cache.get(legacy['question'], legacy['config'], prompt='strict') is None
    assert cache.get(legacy['question'], legacy['config'], prompt='default') == legacy
    with patch.object(answer, '_chat', side_effect=AssertionError('legacy cached')), \
            patch('gitcontext.retriever.Retriever', side_effect=AssertionError('legacy cached')):
        rows = judge.evaluate_answers(questions, cache_path=path, prompt='default')
    assert rows[0]['config'].endswith('(prompt: default)')
    assert rows[0]['prompt'] == 'default' and rows[0]['judgment'] == SCORE


def test_cached_comparison_excludes_changed_labels_and_partial_results(example, tmp_path):
    questions, _, retriever = example
    path = tmp_path / 'cache.jsonl'
    with patch.object(answer, 'generate_answer', return_value='Answer [2]'), \
            patch.object(answer, '_chat', return_value=json.dumps(SCORE)):
        strict = judge.evaluate_answers(questions, cache_path=path, retriever=retriever, prompt='strict')
    cache = judge.JudgeCache(path)
    record = {**strict[0], 'prompt': 'default', 'config': strict[0]['config'].replace('strict', 'default')}
    record.pop('judgment')
    cache.save(record)
    assert judge.comparison_rows(questions, strict, cache_path=path) == strict
    cache.save({**record, 'judgment': SCORE, 'expected': ['changed.py']})
    assert judge.comparison_rows(questions, strict, cache_path=path) == strict


def test_judge_cli_forwards_prompt_without_llm(example, tmp_path, monkeypatch):
    import ast
    import argparse
    from unittest.mock import Mock
    source = config.PROJECT_ROOT / 'judge.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    evaluate = Mock(return_value=[])
    compare = Mock(return_value=[])
    namespace = {'argparse': argparse, 'config': config, '__doc__': 'judge',
                 'evaluate_answers': evaluate, 'comparison_rows': compare,
                 'render_report': Mock(return_value='mock report')}
    monkeypatch.setattr(config, 'JUDGE_RESULTS_PATH', tmp_path / 'report.md')
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), namespace)
    with patch('sys.argv', ['judge.py', '--prompt', 'strict', '--limit', '1']), \
            patch('evaluate.load_questions', return_value=example[0]):
        namespace['main']()
    assert evaluate.call_args.kwargs['prompt'] == 'strict'
    assert compare.call_args.kwargs == {'no_cache': False}



@pytest.mark.parametrize('text,expected', [
    ("The provided sources don't show how this works.", True),
    ('The supplied source excerpts do not include the implementation.', True),
    ('The code you’ve shown does not contain the password lookup.', True),
    ('The implementation is not shown in the provided sources.', True),
    ('The reason cannot be determined from the given files.', True),
    ('The sources show a working implementation [1].', False),
    ('I do not know the answer.', False),
])
def test_explicit_abstention_detection(text, expected):
    assert judge.explicitly_abstains(text) is expected


def test_gold_retrieval_and_honest_abstention_are_separate(example):
    questions, hits, _ = example
    record = {'question': questions[0]['question'], 'expected': questions[0]['expected'],
              'hits': hits, 'answer': "The provided sources don't show the implementation.",
              'judgment': SCORE, 'config': 'vector'}
    found = judge.classify_answer(record)
    assert found['gold_retrieved']
    assert found['category'] != 'honest_abstain'
    missing = judge.classify_answer({**record, 'hits': [hits[0]]})
    assert not missing['gold_retrieved']
    assert missing['category'] == 'honest_abstain'
    assert missing['judgment'] is SCORE
    assert record['hits'] == hits and 'category' not in record
    alternative = judge.classify_answer({**record, 'hits': [hits[0]], 'answer': 'Alternate implementation [1]'})
    assert alternative['category'] == 'no_gold_retrieved'


def test_report_separates_gold_and_excludes_abstentions_from_generation_worst(example):
    questions, hits, _ = example
    base = {'expected': questions[0]['expected'], 'hits': hits, 'config': 'vector',
            'answer': 'Correct [2]', 'judgment': {**SCORE, 'correctness': 5}}
    abstain = {**base, 'hits': [hits[0]], 'question': 'Missing implementation',
               'answer': "The provided sources don't show the answer.",
               'judgment': {**SCORE, 'correctness': 1}}
    found = {**base, 'question': 'Retrieved implementation'}
    report = judge.render_report([found, abstain])
    assert '| vector | 1 | 5.00 | 1 | 1.00 | 1 |' in report
    assert '| Missing implementation | no | honest_abstain | 1 |' in report
    assert 'gold_retrieved' in report
    worst = report.split('## Five lowest-scoring questions')[1]
    assert 'Retrieved implementation' in worst and 'Missing implementation' not in worst
    assert abstain['judgment']['correctness'] == 1
