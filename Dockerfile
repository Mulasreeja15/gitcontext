FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_DOTENV_DISABLED=1 \
    PUBLIC_DEMO=1 \
    PORT=7860 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
RUN useradd -m -u 1000 user && chown user:user /app
USER user
COPY --chown=user:user . .
RUN mkdir -p /app/data
RUN python -c "from gitcontext import config; from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer(config.EMBED_MODEL); CrossEncoder(config.RERANK_MODEL)"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 7860
CMD ["python", "-m", "gitcontext.deployment"]
