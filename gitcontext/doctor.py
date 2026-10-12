"""Environment diagnostics without loading models or displaying credentials."""
import importlib
import os
import sys

from . import answer, config
from .database import DatabaseLocked, open_client
from .indexing import index_status, load_state
from .repositories import collection_name


def checks(offline: bool = False) -> list[tuple[str, bool, str]]:
    """Collect actionable diagnostics; only the optional reachability check calls Groq."""
    results = []
    supported = (3, 11) <= sys.version_info[:2] <= (3, 12)
    results.append((f'Python {sys.version_info.major}.{sys.version_info.minor}', supported,
                    'Use Python 3.12 (CI tested); supported range is 3.11–3.12.'))
    for package in config.DOCTOR_PACKAGES:
        try:
            importlib.import_module(package)
            results.append((f'Package {package}', True, ''))
        except Exception:
            results.append((f'Package {package}', False,
                            'Install dependencies: .venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt'))
    key_set = bool(os.getenv(config.LLM_KEY_NAME))
    results.append((f'{config.LLM_KEY_NAME}: {"set" if key_set else "missing"}', key_set,
                    f'Set {config.LLM_KEY_NAME} in your environment to enable answers.'))
    try:
        client = open_client()
    except DatabaseLocked as error:
        results.append(('Database: locked by another process', False, str(error)))
    except Exception:
        results.append(('Database: unavailable', False,
                        'Check QDRANT_URL/server availability or local data directory permissions.'))
    else:
        try:
            client.get_collections()
            results.append(('Database: reachable; no local lock conflict', True, ''))
            paths = sorted(config.INDEX_STATE_DIR.glob('*.json'))
            if not paths:
                results.append(('Indexed repositories: none', False,
                                'Run python main.py index https://github.com/psf/requests'))
            for path in paths:
                identity = path.stem.replace('__', '/')
                try:
                    state = load_state(identity)
                    exists = client.collection_exists(collection_name(identity))
                    count = client.count(collection_name(identity), exact=True).count if exists else 0
                    fresh = bool(state) and exists and count == state['chunk_count'] and not index_status(identity)['stale']
                    results.append((f'Repository {identity}: collection {"exists" if exists else "missing"}, '
                                    f'{"current" if fresh else "stale or invalid"}', fresh,
                                    f'Stop a local-mode server, then run python main.py update --repo {identity}'))
                except Exception:
                    results.append((f'Repository {identity}: unable to inspect', False,
                                    'Check its index state and rebuild with python main.py index <repo>.'))
        except Exception:
            results.append(('Database: inspection failed', False, 'Check the Qdrant server and data permissions.'))
        finally:
            client.close()
    if offline:
        results.append(('Groq reachability: skipped (--offline)', True, ''))
    elif not key_set:
        results.append(('Groq reachability: skipped (key missing)', False, f'Set {config.LLM_KEY_NAME}, or use --offline.'))
    else:
        try:
            answer._chat([{'role': 'user', 'content': 'Reply OK.'}], temperature=0.0,
                         timeout=config.GROQ_DOCTOR_TIMEOUT, max_tokens=1)
            results.append(('Groq reachability', True, ''))
        except (Exception, SystemExit):
            results.append(('Groq reachability', False,
                            'Check connectivity, key validity and Groq limits; retry later or use --offline.'))
    return results


def run(offline: bool = False) -> bool:
    """Print a checklist with fixes and return whether all checks passed."""
    results = checks(offline)
    for label, ok, fix in results:
        print(f'[{"OK" if ok else "WARN"}] {label}')
        if not ok:
            print(f'  Fix: {fix}')
    return all(ok for _, ok, _ in results)
