import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME


@st.cache_resource(show_spinner=False)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Return the shared embedding model.

    The model is loaded once per server process and reused across
    sessions and reruns. Caching matters here: without it, every
    Streamlit rerun would reload the weights into RAM, which costs
    seconds and significant memory each time.
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
