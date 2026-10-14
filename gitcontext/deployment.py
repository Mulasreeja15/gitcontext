"""Container entry point: provision demo indexes, then serve one process."""
import uvicorn

from . import config
from .database import open_client
from .history import index_history
from .indexing import full_index, load_state
from .store import Embedder, VectorStore


def bootstrap():
    """Resume missing code/history indexes with one client and one embedder.

    URL indexing clones full Git history. Never import PRs or call an LLM here.
    State plus collection counts make restarts skip completed work.
    """
    client = open_client()
    embedder = None
    try:
        for identity in config.DEMO_REPOS:
            store = VectorStore(identity, client=client)
            state = load_state(identity)
            if not state or state.get('incomplete') or store.count() == 0:
                embedder = embedder or Embedder()
                print(f'[startup] Indexing {identity}', flush=True)
                full_index('https://github.com/' + identity, repo=identity, store=store,
                           embedder=embedder,
                           progress=lambda stage, value: print(f'[startup] {stage}: {value}', flush=True))
                state = load_state(identity)
            if identity == config.DEFAULT_REPO and not state.get('history_chunks'):
                embedder = embedder or Embedder()
                print(f'[startup] Indexing full-history commits for {identity}', flush=True)
                result = index_history(identity, unshallow=True, commits_only=True,
                                       store=store, embedder=embedder)
                print(f'[startup] History: {result}', flush=True)
            print(f'[startup] {identity} ready: {store.count()} chunks', flush=True)
    finally:
        client.close()


def main():
    """Finish initialization before opening the public listener."""
    bootstrap()
    uvicorn.run('gitcontext.api:app', host=config.API_HOST, port=config.API_PORT, workers=1)


if __name__ == '__main__':
    main()
