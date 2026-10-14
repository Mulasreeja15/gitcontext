"""Run real capability and error UI logic with a tiny offline DOM stub."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('public', [True, False])
def test_public_controls_and_rate_limit_message(public):
    html = (Path(__file__).resolve().parents[1] / 'gitcontext/static/index.html').read_text(encoding='utf-8')
    health = html[html.index('async function health()'):html.index('async function consumeStream(')]
    buttons = html[html.index('function updateButtons()'):html.index('function selectedMetadata()')]
    script = '''
const nodes = {}, $ = id => nodes[id] ||= {};
let publicDemo = false, busy = false, selectedRepo = 'psf/requests';
const indexJob = {}, indexing = () => false, emptyMessage = '', status = () => {};
const loadRepos = async () => {};
const fetch = async () => ({ok:true, json:async () => ({public_demo:PUBLIC, max_query_chars:PUBLIC ? 500 : null})});
''' .replace('PUBLIC', json.dumps(public)) + health + buttons + '''
(async () => {
await health(); updateButtons();
let error;
try {await checked({ok:false, status:429, json:async () => ({detail:'Too many requests, try again in a minute.'})});}
catch (value) {error = value.message;}
process.stdout.write(JSON.stringify({nodes, error}));
})();
'''
    result = subprocess.run([shutil.which('node'), '-e', script], capture_output=True,
                            text=True, encoding='utf-8', check=True)
    output = json.loads(result.stdout)
    assert output['nodes']['open-repository']['hidden'] == public
    assert output['nodes']['open-repository']['disabled'] == public
    assert output['nodes']['index-button']['disabled'] == public
    if public:
        assert output['nodes']['query']['maxLength'] == 500
    assert output['error'] == 'Too many requests, try again in a minute.'
