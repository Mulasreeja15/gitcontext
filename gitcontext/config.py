"""Central settings. Change models, paths and limits here."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from the .env file

DATA_DIR = Path("./data")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELF_SKIP_DIRS = {"tests"}  # exclude our unit tests only when indexing this project
REPOS_DIR = DATA_DIR / "repos"          # cloned repos land here
QDRANT_PATH = str(DATA_DIR / "qdrant")  # local Qdrant, no Docker needed
DEFAULT_REPO = "psf/requests"
COLLECTION = "repo__psf__requests"
MAX_REPO_BYTES = 200 * 1024 * 1024
MAX_REPO_CHUNKS = 20_000

# Embeddings (local, free)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MAX_TOKENS = 512

# Reranker (local, free): a cross-encoder that reads query + code together
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MAX_TOKENS = 512

# Retrieval tuning
CANDIDATES = 30              # results each retriever contributes before fusion/reranking
RRF_K = 60                   # standard Reciprocal Rank Fusion constant
TEST_RRF_WEIGHT = 0.5        # test files count half during fusion
TEST_RERANK_PENALTY = 3.0    # subtracted from the reranker score for test files

# LLM that writes answers and rewrites queries (Groq, free tier)
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = LLM_MODEL  # a different judge model reduces self-preference bias
JUDGE_CACHE_PATH = Path("eval/judge_cache.jsonl")
JUDGE_RESULTS_PATH = Path("eval/judge_results.md")
JUDGE_K = 6
API_HOST = "127.0.0.1"
API_PORT = 8000
API_MAX_K = 20
INDEX_STATE_DIR = DATA_DIR / "index_state"
INDEX_PAYLOAD_VERSION = 2  # file-level Git provenance and metadata filtering
UI_PATH = PROJECT_ROOT / "gitcontext" / "static" / "index.html"
LLM_KEY_NAME = "GROQ_API_KEY"

MAX_CHUNK_CHARS = 4000      # oversized functions/sections get split
MAX_FILE_BYTES = 500_000    # skip huge/generated files
CODE_EXTENSIONS = {".py": "python"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__",
    "dist", "build", "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
}
