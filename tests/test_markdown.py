"""Render the actual UI JavaScript with a DOM harness, without network access."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def render(sample):
    html = (ROOT / 'gitcontext/static/index.html').read_text(encoding='utf-8')
    element = html[html.index('function element('):html.index('function status(')]
    renderer = html[html.index('// Build a small Markdown subset'):html.index('function renderAnswer(')]
    node = shutil.which('node')
    assert node, 'Node.js is required for the Markdown rendering tests'
    result = subprocess.run(
        [node, str(ROOT / 'tests/markdown_dom.cjs')],
        input=json.dumps({'renderer': element + renderer, 'sample': sample}),
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def descendants(node, tag):
    return ([node] if node['tag'] == tag else []) + [
        match for child in node['children'] for match in descendants(child, tag)
    ]


def test_sample_answer_nested_emphasis_headings_code_links_and_lists():
    tree = render('''# Proxy handling
## Steps
1. **Gather the *no-proxy* list** and **use `resolve_proxy`** [1].
   - Read [the **documentation**](https://example.com/docs).

   Keep this paragraph with the first step.

2. **Check *each* entry**.
   - Preserve the nested list.
3. Return the result.

A separate paragraph ends the list.

3. Resume at three.
4. Finish.
### Details
**Use `x**y` safely** and **bold *italic***.
''')
    assert [node['text'] for node in descendants(tree, 'h1')] == ['Proxy handling']
    assert [node['text'] for node in descendants(tree, 'h2')] == ['Steps']
    assert [node['text'] for node in descendants(tree, 'h3')] == ['Details']
    strong = descendants(tree, 'strong')
    assert strong[0]['text'] == 'Gather the no-proxy list'
    assert descendants(strong[0], 'em')[0]['text'] == 'no-proxy'
    assert descendants(strong[1], 'code')[0]['text'] == 'resolve_proxy'
    assert descendants(strong[-2], 'code')[0]['text'] == 'x**y'
    assert descendants(strong[-1], 'em')[0]['text'] == 'italic'
    lists = descendants(tree, 'ol')
    assert [node['attrs']['start'] for node in lists] == ['1', '3']
    assert [len(node['children']) for node in lists] == [3, 2]
    assert len(descendants(lists[0]['children'][0], 'ul')) == 1
    assert descendants(lists[0]['children'][0], 'p')[0]['text'].startswith('Keep this paragraph')
    links = descendants(tree, 'a')
    assert links[0]['href'] == '#source-1'
    assert links[1]['href'] == 'https://example.com/docs'
    assert links[1]['rel'] == 'noopener noreferrer'
    assert descendants(links[1], 'strong')[0]['text'] == 'documentation'


def test_explicit_numbers_and_repeated_one_markers():
    tree = render('1. First\n1. Second\n5. Fifth\n1. Sixth')
    items = descendants(tree, 'ol')[0]['children']
    assert [item['attrs'].get('value') for item in items] == [None, None, '5', None]


def test_untrusted_html_and_unsafe_links_remain_text():
    sample = '<img src=x onerror=alert(1)> [bad](javascript:alert(1)) [99]\n\n```html\n<script>alert(1)</script>\n```'
    tree = render(sample)
    assert not descendants(tree, 'img')
    assert not descendants(tree, 'script')
    assert not descendants(tree, 'a')
    assert '[bad](javascript:alert(1))' in tree['text']
    assert descendants(tree, 'code')[0]['text'] == '<script>alert(1)</script>'
