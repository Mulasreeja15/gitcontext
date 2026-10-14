---
title: GitContext
emoji: 🔎
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
python_version: "3.12"
short_description: Cited answers about Python code and commit history
---

# GitContext

Ask questions about Python code and get answers with clickable code citations.
Choose **psf/requests** or **pallets/click**. History mode includes requests commit
messages. The public demo allows 500-character questions and five combined
search/answer requests per client IP per minute. Repository changes are disabled.

Initial startup builds the indexes; watch the container logs until both are ready.
The original UI is at `/`; a small Gradio panel is at `/gradio`. Indexing and
embeddings run on CPU, with no GPU-decorated functions. If ZeroGPU rejects a
non-GPU app, the fallback is a static showcase on GitHub Pages (no live answers).
Free ephemeral storage means a restart can require rebuilding them.

[Source code and evaluation](https://github.com/Mulasreeja15/gitcontext)
