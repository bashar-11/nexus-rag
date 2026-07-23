import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_system")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Keys & Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

try:
    import streamlit as st
    if not GROQ_API_KEY and "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    if not QDRANT_URL and "QDRANT_URL" in st.secrets:
        QDRANT_URL = st.secrets["QDRANT_URL"]
    if not QDRANT_API_KEY and "QDRANT_API_KEY" in st.secrets:
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
except Exception:
    pass

COLLECTION_NAME = "nexus_rag_docs"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 32

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TOP_K = 4
FETCH_K = 15
RRF_K = 60
SIMILARITY_SCORE_THRESHOLD = 0.25
USE_MMR = True
MMR_LAMBDA = 0.5

GROQ_LLM_MODEL = "llama-3.3-70b-versatile"