FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_DOTENV_DISABLED=1 \
    HF_HOME=/opt/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from gitcontext import config; from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer(config.EMBED_MODEL); CrossEncoder(config.RERANK_MODEL)"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
