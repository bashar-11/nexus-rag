import re
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.config import GROQ_API_KEY, GROQ_LLM_MODEL, TOP_K, FETCH_K, logger
from src.vector_store import get_hybrid_documents


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY مفقود. تأكد من ضبطه في ملف .env قبل استخدام النظام."
        )

    return ChatGroq(
        model_name=GROQ_LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
    )


def build_prompt_template() -> PromptTemplate:
    template = """You are an intelligent, highly accurate AI document research assistant.
Analyze the provided Context carefully to answer the user's question.

Guidelines:
1. Language: Answer in the SAME language as the user's question (Arabic for Arabic questions, English for English questions).
2. Quality: Provide a well-structured, detailed response using bullet points and clear headings where relevant.
3. Accuracy: Rely strictly on the information in the provided Context. Do not invent facts or mix information from unrelated sections.
4. Fallback: If the answer cannot be inferred from the context, state clearly that the document does not contain this specific information.

Context:
---------------------
{context}
---------------------

User Question:
{question}

Answer:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])


# كلمات مفتاحية تدل على أن السؤال عام (طلب ملخص/نظرة عامة) وليس سؤالًا محددًا
_OVERVIEW_KEYWORDS = [
    "summary", "summarize", "overview", "about", "main topic", "what is this",
    "ملخص", "تلخيص", "عن ماذا", "موضوع", "فكرة", "محتوى", "شرح",
]
# نبني regex بحدود كلمة (\b) لتجنب مطابقة جزئية خاطئة داخل كلمات أخرى
_OVERVIEW_PATTERN = re.compile(
    r"(" + "|".join(re.escape(kw) for kw in _OVERVIEW_KEYWORDS) + r")",
    flags=re.IGNORECASE,
)


def is_overview_query(query: str) -> bool:
    """التحقق مما إذا كان السؤال يتطلب نظرة عامة أو ملخصًا للمستند"""
    return bool(_OVERVIEW_PATTERN.search(query))


def _format_context(documents: List[Document]) -> str:
    """
    تنسيق المقاطع المسترجعة كسياق واحد، مع ذكر المصدر ورقم الصفحة
    (إن وُجد) لكل مقطع — يساعد الـ LLM على تمييز المصادر ويقلل من
    خلط المعلومات بين مستندات مختلفة.
    """
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "مستند")
        page = doc.metadata.get("page")
        label = f"{source} - صفحة {page}" if page else source
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_rag_response(query: str, documents: List[Document], top_k: int = TOP_K) -> Dict[str, Any]:
    if not documents:
        return {
            "answer": "⚠️ No documents uploaded to search from.",
            "source_documents": [],
        }

    # 1. الاسترجاع الهجين (Semantic + BM25 مدموجة عبر RRF)
    relevant_docs = get_hybrid_documents(query, documents, top_k=top_k, fetch_k=FETCH_K)

    # 2. لو السؤال عام (Summary / Overview)، نضيف أول مقاطع المستند لضمان
    #    توفر سياق عام حتى لو الاسترجاع الهجين ركّز على تفاصيل جزئية
    if is_overview_query(query):
        existing_keys = {(d.metadata.get("source"), d.metadata.get("chunk_index")) for d in relevant_docs}
        for doc in documents[:2]:
            doc_key = (doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if doc_key not in existing_keys:
                relevant_docs.insert(0, doc)
                existing_keys.add(doc_key)

    if not relevant_docs:
        logger.warning(f"لم يتم العثور على أي مقاطع ذات صلة للسؤال: {query!r}")
        return {
            "answer": "Could not find relevant sections in the document.",
            "source_documents": [],
        }

    # 3. تجميع السياق مع معلومات المصدر/الصفحة
    context_text = _format_context(relevant_docs)

    # 4. التوليد عبر Groq
    prompt_template = build_prompt_template()
    formatted_prompt = prompt_template.format(context=context_text, question=query)

    llm = get_llm()
    response = llm.invoke(formatted_prompt)

    return {
        "answer": response.content,
        "source_documents": relevant_docs,
    }
