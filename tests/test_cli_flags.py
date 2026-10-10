"""Check CLI parsing without loading models, credentials, or network clients."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def cli():
    """Load just the flag functions so tests do not import application services."""
    source = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"add_retrieval_flags", "options_from"}]
    namespace = {"RetrievalOptions": SimpleNamespace}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    parser = argparse.ArgumentParser(add_help=False)
    namespace["add_retrieval_flags"](parser)
    return parser, namespace["options_from"]


def test_defaults_and_exact_flags(cli):
    parser, options_from = cli
    assert set(parser._option_string_actions) == {
        "-k", "--bm25", "--rerank", "--include-tests", "--rewrite"
    }
    args = parser.parse_args([])
    assert args.k == 6
    assert vars(options_from(args)) == dict(
        bm25=False, rerank=False, penalize_tests=True, rewrite=False
    )


@pytest.mark.parametrize("flag,field,value", [
    ("--bm25", "bm25", True),
    ("--rerank", "rerank", True),
    ("--include-tests", "penalize_tests", False),
    ("--rewrite", "rewrite", True),
])
def test_flags_map_independently(cli, flag, field, value):
    parser, options_from = cli
    expected = dict(bm25=False, rerank=False, penalize_tests=True, rewrite=False)
    expected[field] = value
    assert vars(options_from(parser.parse_args([flag]))) == expected


def test_result_count(cli):
    parser, _ = cli
    assert parser.parse_args(["-k", "4"]).k == 4
    with pytest.raises(SystemExit):
        parser.parse_args(["-k", "invalid"])


@pytest.mark.parametrize("flag", ["--no-rerank", "--no-bm25"])
def test_obsolete_flags_rejected(cli, flag):
    parser, _ = cli
    with pytest.raises(SystemExit):
        parser.parse_args([flag])
