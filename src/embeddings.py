import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


@st.cache_resource(show_spinner=False)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    تحميل نموذج الـ Embeddings وتخزينه في الذاكرة المؤقتة (Cache)
    لتجنب استهلاك الـ RAM وإعادة تحميل النموذج عند كل استفسار.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
    )
    return embeddings