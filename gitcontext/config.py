"""Central settings. Change models, paths and limits here."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from the .env file

DATA_DIR = Path("./data")
REPOS_DIR = DATA_DIR / "repos"          # cloned repos land here
QDRANT_PATH = str(DATA_DIR / "qdrant")  # local Qdrant, no Docker needed
COLLECTION = "gitcontext"

# Free, local embedding model (max 512 tokens)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MAX_TOKENS = 512

# LLM that writes the answers (Groq, free tier)
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_KEY_NAME = "GROQ_API_KEY"

MAX_CHUNK_CHARS = 4000      # oversized functions/sections get split
MAX_FILE_BYTES = 500_000    # skip huge/generated files
CODE_EXTENSIONS = {".py": "python"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__",
    "dist", "build", "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
}