import os
import tempfile
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.document_loader import load_document, split_documents
from src.vector_store import (
    add_documents_to_vector_store,
    clear_vector_store,
    rebuild_vector_store_from_chunks
)
from src.rag_engine import generate_rag_response
from src.config import TOP_K, GROQ_LLM_MODEL, EMBEDDING_MODEL_NAME

st.set_page_config(
    page_title="NexusRAG | AI Knowledge Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Cairo', sans-serif !important;
    }

    /* Main background */
    .stApp {
        background: #090d16 !important;
        color: #f1f5f9 !important;
    }

    /* Direction handling for mixed text */
    .stMarkdown, p, div, .source-box {
        direction: auto !important;
        text-align: start !important;
    }

    /* Header banner styling */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .hero-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Metric cards styling */
    .metric-badge {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 18px;
        text-align: center;
        min-width: 120px;
    }

    .metric-value {
        color: #38bdf8;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Source file cards in Sidebar */
    .source-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease-in-out;
    }

    .source-card:hover {
        border-color: #38bdf8;
        background: #24334a;
    }

    .source-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e2e8f0;
        word-break: break-all;
    }

    .source-meta {
        font-size: 0.75rem;
        color: #94a3b8;
    }

    /* Source citation boxes in chat */
    .source-box {
        background-color: rgba(30, 41, 59, 0.85);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
        font-size: 0.95rem;
        color: #f1f5f9 !important;
        line-height: 1.6;
    }

    /* Sidebar container & text forced high-contrast styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
    }

    /* File uploader dark theme override */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #1e293b !important;
        border: 1.5px dashed #38bdf8 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #e2e8f0 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #334155 !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
    }

    /* Custom Chat Input - High Contrast Text & Full BaseWeb Overrides */
    [data-testid="stChatInput"] {
        background-color: #1e293b !important;
        border-radius: 16px !important;
        border: 1.5px solid #38bdf8 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        padding: 4px 8px !important;
    }

    /* Force background and text color on inner BaseWeb input wrappers */
    [data-testid="stChatInput"] div,
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="input"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: #1e293b !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        caret-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
    }

    /* Chat Messages - High Contrast & High Legibility */
    [data-testid="stChatMessage"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        margin-bottom: 14px !important;
        color: #f8fafc !important;
    }

    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li {
        color: #f8fafc !important;
        font-size: 1.05rem !important;
        line-height: 1.75 !important;
    }

    /* Headings inside answers */
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }

    [data-testid="stChatMessage"] code {
        color: #38bdf8 !important;
        background-color: #1e293b !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
    }

    /* Styled buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }

    div.stButton > button[kind="secondary"] {
        background-color: rgba(239, 68, 68, 0.1) !important;
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background-color: rgba(239, 68, 68, 0.2) !important;
        color: #ef4444 !important;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_documents" not in st.session_state:
    st.session_state.all_documents = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}  # {filename: {"chunks": int, "pages": int}}

if "top_k" not in st.session_state:
    st.session_state.top_k = TOP_K

def delete_source_file(filename_to_delete: str):
    """حذف مصدر محدد وإعادة بناء الفهرس بقية المستندات"""
    # 1. إزالة المستندات الخاصة بهذا الملف من الـ Session State
    st.session_state.all_documents = [
        doc for doc in st.session_state.all_documents
        if doc.metadata.get("source") != filename_to_delete
    ]

    # 2. إزالة الملف من سجل الملفات المعالجة
    if filename_to_delete in st.session_state.processed_files:
        del st.session_state.processed_files[filename_to_delete]

    # 3. إعادة بناء الـ Vector Store
    if st.session_state.all_documents:
        rebuild_vector_store_from_chunks(st.session_state.all_documents)
    else:
        clear_vector_store()

    st.toast(f"🗑️ Deleted document: {filename_to_delete}", icon="✨")


def clear_all_data():
    """حذف جميع المستندات والمحادثات"""
    st.session_state.messages = []
    st.session_state.all_documents = []
    st.session_state.processed_files = {}
    clear_vector_store()
    st.toast("🧹 Reset complete. All data cleared.", icon="🚀")

with st.sidebar:
    st.markdown("## ⚙️ Knowledge Base Manager")
    st.markdown("Upload, inspect, and manage your source documents.")
    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF / TXT):",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # التصفية لجلب الملفات الجديدة فقط
        new_uploads = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

        if new_uploads:
            if st.button("📥 Process & Index New Files", use_container_width=True, type="primary"):
                with st.spinner("Processing & indexing vectors..."):
                    new_chunks = []
                    for uploaded_file in new_uploads:
                        suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        try:
                            raw_docs = load_document(tmp_path)
                            for doc in raw_docs:
                                doc.metadata["source"] = uploaded_file.name

                            chunks = split_documents(raw_docs)
                            new_chunks.extend(chunks)

                            # حفظ معلومات الملف
                            pages_count = len(raw_docs) if uploaded_file.name.endswith(".pdf") else 1
                            st.session_state.processed_files[uploaded_file.name] = {
                                "chunks": len(chunks),
                                "pages": pages_count
                            }
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                    if new_chunks:
                        # الإضافة المتراكمة بقاعدة المتجهات
                        add_documents_to_vector_store(new_chunks, reset=False)
                        st.session_state.all_documents.extend(new_chunks)
                        st.success(f"Added {len(new_chunks)} chunks from {len(new_uploads)} file(s)!")
                        st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Active Sources")

    if st.session_state.processed_files:
        for fname, meta in list(st.session_state.processed_files.items()):
            col_info, col_del = st.columns([0.8, 0.2])
            with col_info:
                st.markdown(f"""
                <div class="source-name">📄 {fname}</div>
                <div class="source-meta">{meta['chunks']} Chunks • {meta['pages']} Page(s)</div>
                """, unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}", type="secondary"):
                    delete_source_file(fname)
                    st.rerun()
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    else:
        st.info("No documents indexed yet. Upload files to start chatting.")

    st.markdown("---")
    st.markdown("### 🎛️ Retrieval Settings")

    st.session_state.top_k = st.slider(
        "Context Chunks (Top-K):",
        min_value=2,
        max_value=10,
        value=st.session_state.top_k,
        help="Number of relevant document snippets passed to LLM."
    )

    st.markdown("---")
    col_reset, col_export = st.columns(2)
    with col_reset:
        if st.button("🧹 Reset All", use_container_width=True, type="secondary"):
            clear_all_data()
            st.rerun()
    
    with col_export:
        if st.session_state.messages:
            chat_json = json.dumps(st.session_state.messages, default=str, ensure_ascii=False, indent=2)
            st.download_button(
                "💾 Export Chat",
                data=chat_json,
                file_name="rag_chat_history.json",
                mime="application/json",
                use_container_width=True
            )

st.markdown(f"""
<div class="hero-banner">
    <div>
        <h1 class="hero-title">NexusRAG Intelligence Hub</h1>
        <div class="hero-subtitle">Hybrid Search (BM25 + Semantic RRF) powered by Groq {GROQ_LLM_MODEL}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics display
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-value">{len(st.session_state.processed_files)}</div>
        <div class="metric-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-value">{len(st.session_state.all_documents)}</div>
        <div class="metric-label">Total Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-value">{st.session_state.top_k}</div>
        <div class="metric-label">Top-K Context</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="metric-value">Llama-3.3</div>
        <div class="metric-label">LLM Engine</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Verified Context & Cited Sources:"):
                for idx, doc in enumerate(message["sources"], 1):
                    source_name = doc.metadata.get("source", "Document")
                    page_num = doc.metadata.get("page", None)
                    page_label = f" (Page {page_num})" if page_num else ""
                    st.markdown(f"**Snippet {idx}:** `{source_name}`{page_label}")
                    st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)

user_query = st.chat_input("Ask a question about your uploaded documents or ask for a summary...")

if user_query:
    # Display user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    if not st.session_state.all_documents:
        warning_msg = "⚠️ No active documents found. Please upload at least one PDF or TXT document from the sidebar."
        with st.chat_message("assistant"):
            st.warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching hybrid vector index & synthesizing response..."):
                try:
                    rag_result = generate_rag_response(
                        query=user_query,
                        documents=st.session_state.all_documents,
                        top_k=st.session_state.top_k
                    )

                    answer = rag_result["answer"]
                    sources = rag_result["source_documents"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("🔍 Verified Context & Cited Sources:"):
                            for idx, doc in enumerate(sources, 1):
                                source_name = doc.metadata.get("source", "Document")
                                page_num = doc.metadata.get("page", None)
                                page_label = f" (Page {page_num})" if page_num else ""
                                st.markdown(f"**Snippet {idx}:** `{source_name}`{page_label}")
                                st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"❌ An error occurred during response generation: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})