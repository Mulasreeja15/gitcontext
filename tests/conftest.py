"""Unit-test isolation: disable dotenv and replace storage before helper imports."""
import importlib
import os
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

os.environ["PYTHON_DOTENV_DISABLED"] = "1"


@pytest.fixture(scope="session")
def helper_modules():
    """Import real helpers with storage guarded against any instantiation."""
    store = ModuleType("gitcontext.store")
    store.Embedder = Mock(side_effect=AssertionError("ML models are forbidden"))
    store.VectorStore = Mock(side_effect=AssertionError("Qdrant is forbidden"))
    with patch.dict(sys.modules, {"gitcontext.store": store}):
        retriever = importlib.import_module("gitcontext.retriever")
        evaluate = importlib.import_module("evaluate")
        yield retriever, evaluate
