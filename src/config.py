import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# إعداد الـ Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_system")

# ---------------------------------------------------------
# المسارات
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"

# ---------------------------------------------------------
# قراءة مفتاح Groq (يدعم البيئة المحلية و Streamlit Cloud Secrets)
# ---------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY غير موجود في ملف .env أو Streamlit Secrets — لن يعمل توليد الإجابات حتى يتم ضبطه."
    )

# ---------------------------------------------------------
# إعدادات الـ Embeddings
# ---------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------
# إعدادات التقطيع (Chunking)
# ---------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ---------------------------------------------------------
# إعدادات الاسترجاع (Retrieval)
# ---------------------------------------------------------
TOP_K = 4
FETCH_K = 15
RRF_K = 60
SIMILARITY_SCORE_THRESHOLD = 0.25
USE_MMR = True
MMR_LAMBDA = 0.5

# ---------------------------------------------------------
# نموذج Groq
# ---------------------------------------------------------
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"