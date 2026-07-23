"""
واجهة مستخدم لنظام RAG — "غرفة القراءة" (Reading Room)
تصميم مستوحى من فكرة فهرس المكتبة الليلي: خلفية داكنة كأنك في مكتبة
بعد إغلاق الأبواب، وضوء عنبري دافئ (كضوء لمبة القراءة) يبرز كل ملف
كأنه بطاقة فهرسة (index card) قابلة للسحب والإزالة.
"""

import streamlit as st
import os
import time
from pathlib import Path

from src.config import DATA_DIR, GROQ_API_KEY, logger
from src.document_loader import load_document, split_documents
from src.vector_store import add_documents_to_vector_store, clear_vector_store
from src.rag_engine import generate_rag_response

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="غرفة القراءة — مساعد الوثائق الذكي",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# CSS: نظام تصميم "غرفة القراءة الليلية"
# ألوان: خلفية نيلية داكنة + ضوء عنبري دافئ (لمبة القراءة)
# خطوط: Lora (عناوين/بطاقات) + Inter (نص عام) + JetBrains Mono (بيانات وصفية)
# ============================================================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #12141C;
            --surface: #1A1E29;
            --surface-alt: #202538;
            --border: #2E3446;
            --text: #EDEEF3;
            --text-muted: #8B92A8;
            --accent: #E8A33D;
            --accent-soft: rgba(232, 163, 61, 0.14);
            --user-bubble: #26314A;
            --assistant-bubble: #1F2430;
            --danger: #E5626A;
        }

        html, body, [class*="css"]  { direction: rtl; }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(232,163,61,0.07), transparent 40%),
                var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }

        section[data-testid="stSidebar"] {
            background: var(--surface);
            border-left: 1px solid var(--border);
        }

        /* ===== الترويسة ===== */
        .app-header {
            display: flex;
            align-items: baseline;
            gap: 14px;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 22px;
        }
        .app-header h1 {
            font-family: 'Lora', serif;
            font-weight: 600;
            font-size: 30px;
            color: var(--text);
            margin: 0;
        }
        .app-header span.tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--accent);
            background: var(--accent-soft);
            padding: 3px 9px;
            border-radius: 20px;
            letter-spacing: 0.03em;
        }
        .app-sub {
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 6px;
        }

        /* ===== بطاقات الملفات (فهرس المكتبة) ===== */
        .file-card {
            background: var(--surface-alt);
            border: 1px solid var(--border);
            border-right: 3px solid var(--accent);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        .file-card .fname {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12.5px;
            color: var(--text);
            word-break: break-all;
        }
        .file-card .fmeta {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .sidebar-label {
            font-family: 'Lora', serif;
            font-size: 15px;
            font-weight: 600;
            color: var(--text);
            margin: 18px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* ===== أزرار ===== */
        .stButton>button {
            border-radius: 7px;
            border: 1px solid var(--border);
            background: var(--surface-alt);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            transition: all 0.15s ease;
        }
        .stButton>button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* ===== فقاعات المحادثة ===== */
        div[data-testid="stChatMessage"] {
            background: var(--assistant-bubble);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        /* ===== رقاقات المصادر (Citations) ===== */
        .source-chip {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid rgba(232,163,61,0.35);
            border-radius: 20px;
            padding: 3px 10px;
            margin: 3px 4px 0 0;
        }

        .empty-state {
            text-align: center;
            color: var(--text-muted);
            font-size: 13.5px;
            padding: 26px 10px;
            border: 1px dashed var(--border);
            border-radius: 10px;
            background: var(--surface-alt);
        }

        .status-dot {
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            margin-left: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# الحالة (Session State)
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_registry" not in st.session_state:
    # اسم الملف -> مسار الملف على القرص
    st.session_state.file_registry = {}
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "processed_upload_ids" not in st.session_state:
    st.session_state.processed_upload_ids = set()


# ============================================================
# منطق إدارة الملفات والفهرسة
# ============================================================
def rebuild_index():
    """
    إعادة بناء الفهرس (Vector Store) بالكامل من كل الملفات الحالية.
    تُستدعى بعد كل إضافة أو إزالة ملف حتى يبقى الاسترجاع متسقًا مع
    مكتبة الملفات الفعلية الظاهرة في الواجهة.
    """
    all_docs = []
    for filename, filepath in st.session_state.file_registry.items():
        try:
            raw_docs = load_document(filepath)
            all_docs.extend(raw_docs)
        except Exception as e:
            logger.error(f"فشل تحميل الملف {filename}: {e}")
            st.error(f"⚠️ تعذّرت معالجة الملف: {filename}")

    if all_docs:
        chunks = split_documents(all_docs)
        st.session_state.all_chunks = chunks
        add_documents_to_vector_store(chunks, reset=True)
    else:
        st.session_state.all_chunks = []
        clear_vector_store()


def add_file(uploaded_file) -> None:
    """حفظ ملف مرفوع جديد على القرص وإضافته لسجل الملفات."""
    dest_path = DATA_DIR / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.file_registry[uploaded_file.name] = str(dest_path)


def remove_file(filename: str) -> None:
    """إزالة ملف من القرص ومن سجل الملفات، ثم إعادة بناء الفهرس."""
    filepath = st.session_state.file_registry.pop(filename, None)
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            logger.error(f"فشل حذف الملف {filename} من القرص: {e}")
    rebuild_index()


# ============================================================
# الشريط الجانبي: فهرس المكتبة (إضافة / إزالة الملفات)
# ============================================================
with st.sidebar:
    st.markdown(
        """<div class="sidebar-label">📚 مكتبة المستندات</div>""",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "إضافة ملف (PDF أو TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_file_added = False
        for uf in uploaded_files:
            upload_id = f"{uf.name}-{uf.size}"
            if upload_id not in st.session_state.processed_upload_ids:
                add_file(uf)
                st.session_state.processed_upload_ids.add(upload_id)
                new_file_added = True

        if new_file_added:
            with st.spinner("جاري فهرسة الملفات..."):
                rebuild_index()
            st.rerun()

    st.markdown(
        """<div class="sidebar-label">🗂️ الملفات الحالية</div>""",
        unsafe_allow_html=True,
    )

    if not st.session_state.file_registry:
        st.markdown(
            '<div class="empty-state">لا توجد ملفات بعد.<br>ارفع PDF أو TXT للبدء.</div>',
            unsafe_allow_html=True,
        )
    else:
        for filename in list(st.session_state.file_registry.keys()):
            n_chunks = sum(
                1 for c in st.session_state.all_chunks
                if c.metadata.get("source") == filename
            )
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"""<div class="file-card">
                            <div class="fname">📄 {filename}</div>
                            <div class="fmeta">{n_chunks} مقطع مفهرس</div>
                        </div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑️", key=f"remove_{filename}", help=f"إزالة {filename}"):
                    with st.spinner("جاري تحديث الفهرس..."):
                        remove_file(filename)
                    st.rerun()

    st.markdown("---")
    total_chunks = len(st.session_state.all_chunks)
    dot_color = "#4CD964" if GROQ_API_KEY else "#E5626A"
    st.markdown(
        f"""
        <div style="font-size:12px; color:var(--text-muted); line-height:1.9;">
            <span class="status-dot" style="background:{dot_color};"></span>
            حالة GROQ_API_KEY: {"متصل" if GROQ_API_KEY else "غير مضبوط"}<br>
            عدد الملفات: {len(st.session_state.file_registry)}<br>
            عدد المقاطع المفهرسة: {total_chunks}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history and st.button("🧹 مسح المحادثة"):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================
# الواجهة الرئيسية: المحادثة
# ============================================================
st.markdown(
    """
    <div class="app-header">
        <h1>غرفة القراءة</h1>
        <span class="tag">RAG · Hybrid Retrieval</span>
    </div>
    <div class="app-sub">اسأل عن محتوى مستنداتك، وستُجاب استنادًا إلى النصوص المرفوعة فقط.</div>
    """,
    unsafe_allow_html=True,
)

# عرض سجل المحادثة
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="📖" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{s}</span>' for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

# صندوق الإدخال
query = st.chat_input("اكتب سؤالك هنا...")

if query:
    if not st.session_state.file_registry:
        st.warning("⚠️ يرجى رفع ملف واحد على الأقل قبل طرح الأسئلة.")
    elif not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY غير مضبوط في ملف .env — لا يمكن توليد إجابة.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="📖"):
            with st.spinner("جاري البحث والتحليل..."):
                result = generate_rag_response(query, st.session_state.all_chunks)
                answer = result["answer"]
                sources = result.get("source_documents", [])

                source_labels = []
                for doc in sources:
                    src = doc.metadata.get("source", "مستند")
                    page = doc.metadata.get("page")
                    source_labels.append(f"{src}" + (f" · ص{page}" if page else ""))
                # إزالة التكرار مع الحفاظ على الترتيب
                source_labels = list(dict.fromkeys(source_labels))

            st.markdown(answer)
            if source_labels:
                chips = "".join(f'<span class="source-chip">{s}</span>' for s in source_labels)
                st.markdown(chips, unsafe_allow_html=True)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": source_labels}
        )
