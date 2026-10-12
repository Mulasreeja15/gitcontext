"""Verify CLI service dispatch without importing models or launching a server."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gitcontext import config


@pytest.mark.parametrize('flags,host,port', [
    ([], config.API_HOST, config.API_PORT),
    (['--port', '9000'], config.API_HOST, 9000),
    (['--host', '0.0.0.0', '--port', '8000'], '0.0.0.0', 8000),
])
def test_serve_dispatch(flags, host, port):
    source = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'main', 'cmd_serve', 'add_retrieval_flags', 'add_metadata_flags'}]
    namespace = {'argparse': argparse, 'config': config,
                 'cmd_index': Mock(), 'cmd_update': Mock(), 'cmd_status': Mock(),
                 'cmd_search': Mock(), 'cmd_ask': Mock(), 'cmd_repos': Mock(), 'cmd_remove': Mock(), 'cmd_index_history': Mock(), 'cmd_doctor': Mock()}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    runner = Mock()
    with patch('gitcontext.database.open_client', return_value=Mock()), patch.dict('sys.modules', {'uvicorn': SimpleNamespace(run=runner)}), \
            patch('sys.argv', ['main.py', 'serve', *flags]):
        namespace['main']()
    runner.assert_called_once_with('gitcontext.api:app', host=host, port=port)
