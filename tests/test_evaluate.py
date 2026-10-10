"""Expected-label matching for files, symbols, and split methods."""
import pytest


@pytest.mark.parametrize('name,text,target,expected', [
    ('anything', '', 'src/adapters.py', True),
    ('HTTPAdapter', '', 'src/adapters.py::HTTPAdapter', True),
    ('HTTPAdapter.send (part 2/2)', '', 'src/adapters.py::HTTPAdapter.send', True),
    ('HTTPAdapter.send (part 2/2)', '', 'src/adapters.py::send', True),
    ('HTTPAdapter', '    def send(self):\n        pass', 'src/adapters.py::HTTPAdapter.send', True),
    ('HTTPAdapter', '    def send(self):\n        pass', 'src/adapters.py::send', True),
    ('HTTPAdapter', '', 'other.py::HTTPAdapter', False),
    ('HTTPAdapter', 'def other(): pass', 'src/adapters.py::send', False),
])
def test_is_match(helper_modules, name, text, target, expected):
    _, evaluate = helper_modules
    assert evaluate.is_match({'file_path': 'src/adapters.py', 'name': name, 'text': text}, target) is expected
