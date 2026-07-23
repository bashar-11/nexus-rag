import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# دعم st.secrets في Streamlit Cloud بجانب متغيرات البيئة المحلية
# (على Streamlit Cloud، القيم بتتحط في Settings -> Secrets وليس .env)
# ---------------------------------------------------------
def _get_setting(key: str, default=None):
    value = os.getenv(key, default)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


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

# ---------------------------------------------------------
# مفتاح Groq
# ---------------------------------------------------------
GROQ_API_KEY = _get_setting("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY غير موجود — لن يعمل توليد الإجابات حتى يتم ضبطه.")

# ---------------------------------------------------------
# إعدادات Qdrant Cloud (بديل Chroma المحلي لضمان الاستمرارية على Streamlit Cloud)
# ---------------------------------------------------------
QDRANT_URL = _get_setting("QDRANT_URL")
QDRANT_API_KEY = _get_setting("QDRANT_API_KEY")
COLLECTION_NAME = _get_setting("COLLECTION_NAME", "nexus_rag_docs")

if not QDRANT_URL or not QDRANT_API_KEY:
    logger.warning(
        "QDRANT_URL أو QDRANT_API_KEY غير مضبوطين — لازم تتحطوا في "
        "Streamlit Cloud Secrets أو ملف .env محليًا قبل تشغيل الفهرسة."
    )

# ---------------------------------------------------------
# إعدادات الـ Embeddings (يجب أن يطابق حجم المتجه VectorParams(size=...) في Qdrant)
# ---------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # ناتج هذا النموذج تحديدًا؛ غيّره لو غيّرت النموذج
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
