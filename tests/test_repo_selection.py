"""Execute the UI picker logic with mocked storage and repository responses."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('saved,repos,preferred,blocked,expected', [
    ('pallets/click', ['psf/requests', 'pallets/click'], None, False, 'pallets/click'),
    (None, ['pallets/click', 'psf/requests'], None, False, 'psf/requests'),
    (None, ['pallets/click'], None, False, 'pallets/click'),
    ('removed/repo', ['pallets/click', 'psf/requests'], None, False, 'psf/requests'),
    ('psf/requests', ['psf/requests', 'pallets/click'], 'pallets/click', False, 'pallets/click'),
    (None, ['psf/requests'], None, True, 'psf/requests'),
    ('removed/repo', [], None, False, ''),
])
def test_repository_restoration_and_fallback(saved, repos, preferred, blocked, expected):
    html = (ROOT / 'gitcontext/static/index.html').read_text(encoding='utf-8')
    initialization = html[html.index('function savedRepository('):html.index('const indexing =')]
    loading = html[html.index('async function loadRepos('):html.index('async function pollIndex(')]
    change = html[html.index("$('repo-picker').addEventListener('change'"):html.index("$('source-kind').addEventListener")]
    script = '''
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
let stored = input.saved;
const localStorage = {
  getItem() {if (input.blocked) throw Error('blocked'); return stored;},
  setItem(key, value) {if (input.blocked) throw Error('blocked'); stored = value;},
  removeItem() {if (input.blocked) throw Error('blocked'); stored = null;}
};
const nodes = {}, callbacks = {};
const $ = id => nodes[id] ||= {replaceChildren(){}, append(){}, addEventListener(event, fn){callbacks[id] = fn;}};
const element = () => ({}), selectedMetadata = () => {}, status = () => {};
const checked = async response => response;
const fetch = async () => ({json: async () => input.repos.map(repo => ({repo, chunk_count:1}))});
(async () => {
''' + initialization + loading + change + '''
await loadRepos(input.preferred);
const initial = selectedRepo, persisted = stored;
if (input.repos.length) {
  $('repo-picker').value = input.repos[0]; callbacks['repo-picker']();
}
process.stdout.write(JSON.stringify({initial, persisted, changed:selectedRepo, stored}));
})();
'''
    node = shutil.which('node')
    assert node, 'Node.js is required for UI tests'
    result = subprocess.run([node, '-e', script], input=json.dumps({
        'saved': saved, 'repos': repos, 'preferred': preferred, 'blocked': blocked,
    }), text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    assert output['initial'] == expected
    if not blocked:
        assert output['persisted'] == (expected or None)
        assert output['stored'] == (repos[0] if repos else None)
    if repos:
        assert output['changed'] == repos[0]
