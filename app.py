"""Gradio-SDK entry point retaining GitContext's original FastAPI routes."""
import os

# Set these before importing config (and therefore before any dotenv/model code).
os.environ.setdefault('PUBLIC_DEMO', '1')
os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')
os.environ.setdefault('GITCONTEXT_DEVICE', 'cpu')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import asyncio
from contextlib import asynccontextmanager
from threading import Lock, RLock, Thread

import gradio as gr
import spaces  # noqa: F401 -- SDK integration; this app has no GPU work to decorate
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from gitcontext import api, config


def initialize(application):
    """Provision indexes before giving the server its sole persistent client."""
    from gitcontext.deployment import bootstrap
    bootstrap()  # Owns/closes one client; imports commits only, never calls Groq.
    retriever = api.create_retriever()
    application.state.retriever = retriever
    application.state.retrievers = {config.DEFAULT_REPO: retriever}


def build_app(*, start_indexing=True, initializer=None):
    """Build without side effects; background work starts only during lifespan.

    Return the mounted FastAPI app and Blocks so both uvicorn/TestClient and
    Gradio's supported launch(app_kwargs=...) path use identical routes/lifespan.
    """
    initializer = initializer or initialize

    @asynccontextmanager
    async def lifespan(application):
        application.state.retrieval_lock = RLock()
        application.state.job_lock = Lock()
        application.state.status_lock = Lock()
        application.state.index_job = dict(state='idle', repo=None, progress='', error=None)
        application.state.index_thread = None
        application.state.rate_limiter = api.PublicRateLimiter()
        application.state.startup = {'state': 'indexing', 'error': None}
        application.state.retrievers = {}

        def prepare():
            try:
                initializer(application)
                application.state.startup = {'state': 'ready', 'error': None}
            except (Exception, SystemExit):
                print('[startup] Initialization failed. Check repository access, model downloads and disk space.', flush=True)
                application.state.startup = {'state': 'error', 'error': 'Initialization failed. Check the Space logs.'}

        thread = Thread(target=prepare, daemon=True) if start_indexing else None
        if thread:
            thread.start()
        try:
            yield
        finally:
            if thread:
                await asyncio.to_thread(thread.join)
            retriever = getattr(application.state, 'retriever', None)
            if retriever is not None:
                retriever.store.client.close()

    web = FastAPI(title='GitContext', lifespan=lifespan)
    # Copy routes without merging the original blocking model-loading lifespan.
    web.router.routes.extend(api.app.router.routes)
    web.middleware('http')(api.public_demo_guard)

    @web.middleware('http')
    async def readiness(request, call_next):
        state = request.app.state.startup
        if request.method == 'POST' and request.url.path in {'/ask', '/ask/stream', '/search'} and state['state'] != 'ready':
            return JSONResponse({'detail': state['error'] or 'Indexing, ready in a few minutes'}, status_code=503)
        if request.url.path == '/repos' and state['state'] != 'ready':
            return JSONResponse([])
        return await call_next(request)

    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown('# GitContext\n[Open the code search and cited-answer UI](/). '
                    'Models run on CPU; startup indexing runs in the background.')
    mounted = gr.mount_gradio_app(web, demo, path='/gradio')
    return mounted, demo


def main():
    """Launch through Gradio, preserving its own startup-events endpoints."""
    web, demo = build_app()
    demo.launch(server_name=config.API_HOST, server_port=config.API_PORT, share=False,
                ssr_mode=False, app_kwargs={'routes': list(web.router.routes),
                'lifespan': web.router.lifespan_context, 'middleware': web.user_middleware})


if __name__ == '__main__':
    main()
