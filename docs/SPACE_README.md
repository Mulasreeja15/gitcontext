---
title: GitContext
emoji: 🔎
sdk: docker
app_port: 7860
---

# GitContext

Ask questions about Python code and get answers with clickable code citations.
Choose **psf/requests** or **pallets/click**. History mode includes requests commit
messages. The public demo allows 500-character questions and five combined
search/answer requests per client IP per minute. Repository changes are disabled.

Initial startup builds the indexes; watch the container logs until both are ready.
Free ephemeral storage means a restart can require rebuilding them.

[Source code and evaluation](https://github.com/Mulasreeja15/gitcontext)
