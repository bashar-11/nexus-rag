import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.config import FETCH_K, GROQ_API_KEY, GROQ_LLM_MODEL, TOP_K, logger
from src.vector_store import get_hybrid_documents


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Set it in your .env file or in "
            "Streamlit Cloud Secrets."
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


# Keywords that signal the user wants a summary of the whole document
# rather than a specific fact. Matched with lookaround guards, case-insensitive,
# against both English and Arabic so the bilingual UI behaves the same way.
#
# Note on Arabic: the bare interrogatives "ما هو" / "ما هي" are deliberately
# absent. They open almost every Arabic question, including plain factual ones
# such as "ما هي النتائج في التقرير؟" (what are the results in the report?).
# Only the self-referential forms are listed, which unambiguously mean "tell me
# about this document".
_OVERVIEW_KEYWORDS = [
    # English
    "summary",
    "summarize",
    "summarise",
    "overview",
    "about",
    "main topic",
    "what is this",
    "key points",
    "highlights",
    "brief",
    # Arabic
    "ملخص",
    "لخص",
    "تلخيص",
    "نظرة عامة",
    "اشرح",
    "اختصر",
    "باختصار",
    "أهم",
    "أبرز",
    "ماذا",
    "ما هو هذا",
    "ما هي هذه",
    "ما هذا",
]

# Compile once so the pattern is not rebuilt on every query.
# Lookaround guards rather than \b: they behave correctly for Arabic, where a
# keyword such as "نتائج" must not match inside the longer word "النتائج", and
# where \b interacts badly with non-ASCII word characters.
_OVERVIEW_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(kw) for kw in _OVERVIEW_KEYWORDS) + r")(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)


def is_overview_query(query: str) -> bool:
    """Return True when the user is asking for a summary rather than a fact."""
    return bool(_OVERVIEW_PATTERN.search(query))


def _format_context(documents: list[Document]) -> str:
    """
    Build the context block handed to the model.

    Each passage is prefixed with its source file (and page, for PDFs) so the
    answer can be traced back to an exact location, and passages are separated
    by a clear delimiter to keep the model from treating them as continuous
    prose.
    """
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        label = f"{source} — page {page}" if page else source
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_rag_response(
    query: str, documents: list[Document], top_k: int = TOP_K
) -> dict[str, Any]:
    if not documents:
        return {
            "answer": "No documents have been uploaded yet. Upload a PDF or TXT file to get started.",
            "source_documents": [],
        }

    # 1. Retrieve with hybrid search (semantic + BM25, fused with RRF).
    relevant_docs = get_hybrid_documents(query, documents, top_k=top_k, fetch_k=FETCH_K)

    # 2. Broad questions ("summarise this") do not match any single chunk well,
    #    because the answer is spread across the whole document. Inject the
    #    opening chunks so the model has the framing it needs.
    if is_overview_query(query):
        existing_keys = {
            (d.metadata.get("source"), d.metadata.get("chunk_index"))
            for d in relevant_docs
        }
        for doc in documents[:2]:
            doc_key = (doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if doc_key not in existing_keys:
                relevant_docs.insert(0, doc)
                existing_keys.add(doc_key)

    if not relevant_docs:
        logger.warning(f"No relevant sections found for query: {query!r}")
        return {
            "answer": "Could not find relevant sections in the document.",
            "source_documents": [],
        }

    # 3. Format the retrieved passages into a single context block.
    context_text = _format_context(relevant_docs)

    # 4. Generate the answer with Groq.
    prompt_template = build_prompt_template()
    formatted_prompt = prompt_template.format(context=context_text, question=query)

    llm = get_llm()
    response = llm.invoke(formatted_prompt)

    return {
        "answer": response.content,
        "source_documents": relevant_docs,
    }
