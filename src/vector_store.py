from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    logger,
)
from src.embeddings import get_embedding_function


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("لم يتم ضبط QDRANT_URL أو QDRANT_API_KEY في البيئة أو Secrets.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def init_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    embeddings = get_embedding_function()

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logger.info(f"تم إنشاء مجموعة Qdrant جديدة: {COLLECTION_NAME}")

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def add_documents_to_vector_store(documents: List[Document]) -> QdrantVectorStore:
    vector_store = init_vector_store()
    vector_store.add_documents(documents)
    logger.info(f"تمت إضافة {len(documents)} مستند إلى Qdrant Cloud بنجاح.")
    return vector_store


def clear_vector_store() -> None:
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"تم مسح مجموعة Qdrant: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"خطأ أثناء مسح قاعدة البيانات: {e}")