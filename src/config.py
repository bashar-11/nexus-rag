import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------
# Resolution order for every setting: local .env file, then
# Streamlit Cloud Secrets (Settings -> Secrets). This lets the
# exact same code run locally and on Streamlit Community Cloud.
# ---------------------------------------------------------
def _get_setting(key: str, default=None):
    value = os.getenv(key, default)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(key, default)
    except Exception:  # noqa: BLE001 - Streamlit secrets are unavailable outside a Streamlit runtime
        return default


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_system")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------
# Groq - language model for answer generation
# ---------------------------------------------------------
GROQ_API_KEY = _get_setting("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY is not set. The app will start, but answering a "
        "question requires it. Add it to your .env file."
    )

# ---------------------------------------------------------
# Qdrant Cloud - vector store for dense semantic search.
# A local Qdrant instance works too (it is the preferred option
# on Streamlit Community Cloud, which has an ephemeral filesystem).
# ---------------------------------------------------------
QDRANT_URL = _get_setting("QDRANT_URL")
QDRANT_API_KEY = _get_setting("QDRANT_API_KEY")
COLLECTION_NAME = _get_setting("COLLECTION_NAME", "nexus_rag_docs")

if not QDRANT_URL or not QDRANT_API_KEY:
    logger.warning(
        "QDRANT_URL or QDRANT_API_KEY is not set. Semantic search will fail "
        "until they are provided. Add them to your .env file, or to "
        "Settings -> Secrets when running on Streamlit Cloud."
    )

# ---------------------------------------------------------
# Embeddings
# Multilingual model: the output dimension must match the size
# configured on the Qdrant collection.
# ---------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # Must match VectorParams(size=...) on the Qdrant collection
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------
TOP_K = 4
FETCH_K = 15
RRF_K = 60
SIMILARITY_SCORE_THRESHOLD = 0.25
USE_MMR = True
MMR_LAMBDA = 0.5

# ---------------------------------------------------------
# Groq - model identifier
# ---------------------------------------------------------
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
