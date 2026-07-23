import os
import shutil
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.config import (
    CHROMA_PATH,
    TOP_K,
    FETCH_K,
    RRF_K,
    SIMILARITY_SCORE_THRESHOLD,
    USE_MMR,
    MMR_LAMBDA,
    logger,
)
from src.embeddings import get_embedding_function


def clear_vector_store():
    """تفريغ قاعدة البيانات المحلية لتفادي تداخل المتجهات القديمة"""
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            logger.info("تم تفريغ قاعدة بيانات Chroma بنجاح")
        except Exception as e:
            logger.error(f"فشل حذف قاعدة بيانات Chroma القديمة: {e}")


def get_vector_store() -> Chroma:
    """جلب كائن قاعدة بيانات المتجهات Chroma"""
    embedding_fn = get_embedding_function()
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_fn,
    )


def add_documents_to_vector_store(documents: List[Document], reset: bool = False) -> Chroma:
    """
    إضافة مستندات إلى قاعدة المتجهات المحلية.
    إذا كانت reset=True يتم تفريغ قاعدة البيانات أولاً.
    """
    if reset:
        clear_vector_store()

    if not documents:
        return None

    embedding_fn = get_embedding_function()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=str(CHROMA_PATH),
    )
    logger.info(f"تمت إضافة {len(documents)} chunk إلى الـ vector store (reset={reset})")
    return vector_store


def rebuild_vector_store_from_chunks(documents: List[Document]) -> Chroma:
    """
    إعادة بناء قاعدة البيانات بالكامل من قائمة Chunks المتبقية بعد حذف مصدر معين.
    """
    clear_vector_store()
    if not documents:
        return None
    return add_documents_to_vector_store(documents, reset=False)


def _doc_key(doc: Document) -> str:
    """مفتاح فريد للمستند لأغراض إزالة التكرار والدمج"""
    source = doc.metadata.get("source", "")
    chunk_idx = doc.metadata.get("chunk_index", "")
    return f"{source}::{chunk_idx}::{doc.page_content.strip()[:200]}"



def _semantic_search(query: str, fetch_k: int) -> List[Document]:
    """استرجاع دلالي (Semantic) مع MMR أو Similarity Score Threshold"""
    try:
        vector_store = get_vector_store()

        if USE_MMR:
            docs = vector_store.max_marginal_relevance_search(
                query, k=fetch_k, fetch_k=max(fetch_k * 3, 20), lambda_mult=MMR_LAMBDA
            )
            return docs

        scored_docs = vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)
        return [doc for doc, score in scored_docs if score >= SIMILARITY_SCORE_THRESHOLD]

    except Exception as e:
        logger.error(f"فشل البحث الدلالي: {e}")
        return []


def _bm25_search(query: str, documents: List[Document], fetch_k: int) -> List[Document]:
    """استرجاع نصي عبر BM25"""
    try:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = fetch_k
        return bm25_retriever.invoke(query)
    except Exception as e:
        logger.error(f"فشل البحث النصي BM25: {e}")
        return []


def reciprocal_rank_fusion(
    result_lists: List[List[Document]], k: int = RRF_K, top_k: int = TOP_K
) -> List[Document]:
    """دمج نتائج البحث الدلالي والنصي بأسلوب RRF مع إزالة التكرار"""
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
    """استرجاع هجين (Semantic + BM25) دقيق وموثوق"""
    if not documents:
        return []

    vector_docs = _semantic_search(query, fetch_k)
    bm25_docs = _bm25_search(query, documents, fetch_k)

    if not vector_docs and not bm25_docs:
        logger.warning("لم يُرجع أي من محركي الاسترجاع نتائج لهذه الجملة")
        return []

    return reciprocal_rank_fusion([vector_docs, bm25_docs], top_k=top_k)