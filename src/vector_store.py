from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    TOP_K,
    FETCH_K,
    RRF_K,
    SIMILARITY_SCORE_THRESHOLD,
    USE_MMR,
    MMR_LAMBDA,
    logger,
)
from src.embeddings import get_embedding_function


# ============================================================
# Qdrant client and collection
# ============================================================
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY are required. Set them in .env "
            "or in Streamlit Cloud Secrets."
        )
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _ensure_collection(client: QdrantClient) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")


def get_vector_store() -> QdrantVectorStore:
    """Open the Qdrant-backed vector store, creating the collection if needed."""
    client = get_qdrant_client()
    _ensure_collection(client)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embedding_function(),
    )


def clear_vector_store() -> None:
    """
    Drop the collection entirely. This is the cleanest reset: it removes
    the Qdrant index and forces a full rebuild on the next upload.
    """
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Deleted Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Could not clear the vector store: {e}")


def add_documents_to_vector_store(documents: List[Document], reset: bool = True) -> QdrantVectorStore:
    """
    Index documents in Qdrant Cloud.

    With reset=True the collection is dropped first, so repeated uploads
    never accumulate duplicates. With reset=False, new chunks are appended
    to whatever is already indexed (useful when incrementally adding files).
    """
    if reset:
        clear_vector_store()

    vector_store = get_vector_store()
    if documents:
        vector_store.add_documents(documents)
        logger.info(f"Indexed {len(documents)} chunks in Qdrant (reset={reset})")
    return vector_store


# ============================================================
# Hybrid retrieval: dense vectors (Qdrant) + lexical BM25, fused with RRF
# ============================================================
def _doc_key(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    chunk_idx = doc.metadata.get("chunk_index", "")
    return f"{source}::{chunk_idx}::{doc.page_content.strip()[:200]}"


def _semantic_search(query: str, fetch_k: int) -> List[Document]:
    try:
        vector_store = get_vector_store()
        if USE_MMR:
            return vector_store.max_marginal_relevance_search(
                query, k=fetch_k, fetch_k=max(fetch_k * 3, 20), lambda_mult=MMR_LAMBDA
            )
        scored_docs = vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)
        return [doc for doc, score in scored_docs if score >= SIMILARITY_SCORE_THRESHOLD]
    except Exception as e:
        logger.error(f"Semantic search failed (Qdrant): {e}")
        return []


def _bm25_search(query: str, documents: List[Document], fetch_k: int) -> List[Document]:
    try:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = fetch_k
        return bm25_retriever.invoke(query)
    except Exception as e:
        logger.error(f"Keyword search failed (BM25): {e}")
        return []


def reciprocal_rank_fusion(
    result_lists: List[List[Document]], k: int = RRF_K, top_k: int = TOP_K
) -> List[Document]:
    """
    Merge ranked lists with Reciprocal Rank Fusion.

    Each document scores 1 / (k + rank) in every list it appears in, and the
    scores are summed. RRF fuses on rank position only, so sparse and dense
    retrievers do not need their scores placed on a common scale.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            content = doc.page_content.strip()
            if not content:
                continue
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_map[key] for key in ranked_keys[:top_k]]


def get_hybrid_documents(
    query: str, documents: List[Document], top_k: int = TOP_K, fetch_k: int = FETCH_K
) -> List[Document]:
    if not documents:
        return []

    vector_docs = _semantic_search(query, fetch_k)
    bm25_docs = _bm25_search(query, documents, fetch_k)

    if not vector_docs and not bm25_docs:
        logger.warning(
            f"No results for query: {query!r}. Both retrievers (Qdrant and BM25) "
            "returned nothing."
        )
        return []

    return reciprocal_rank_fusion([vector_docs, bm25_docs], top_k=top_k)