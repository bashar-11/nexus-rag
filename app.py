import os
import streamlit as st
from pathlib import Path

# الحفاظ على كافة مسارات الاستدعاءات كما هي
from src.config import DATA_DIR, GROQ_API_KEY, logger
from src.document_loader import load_document, split_documents
from src.vector_store import add_documents_to_vector_store, clear_vector_store
from src.rag_engine import generate_rag_response

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="Nexus RAG — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# الحالة (Session State) — الأسماء محفوظة كما هي
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_registry" not in st.session_state:
    st.session_state.file_registry = {}
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "processed_upload_ids" not in st.session_state:
    st.session_state.processed_upload_ids = set()
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ============================================================
# نظام الألوان (Design Tokens)
# ============================================================
THEMES = {
    "dark": {
        "bg": "#09090B",
        "surface": "#18181B",
        "card": "#202127",
        "card-hover": "#26272E",
        "border": "rgba(255,255,255,0.06)",
        "border-strong": "rgba(255,255,255,0.12)",
        "text": "#FAFAFA",
        "muted": "#A1A1AA",
        "dim": "#71717A",
        "primary": "#6366F1",
        "primary-hover": "#4F46E5",
        "primary-soft": "rgba(99,102,241,0.12)",
        "primary-ring": "rgba(99,102,241,0.35)",
        "success": "#22C55E",
        "danger": "#EF4444",
        "user-bubble": "linear-gradient(135deg, #4F46E5 0%, #6366F1 100%)",
        "user-bubble-text": "#FAFAFA",
        "assistant-bubble": "#18181B",
        "scheme": "dark",
    },
    "light": {
        "bg": "#FAFAFA",
        "surface": "#FFFFFF",
        "card": "#FFFFFF",
        "card-hover": "#F4F4F5",
        "border": "rgba(0,0,0,0.06)",
        "border-strong": "rgba(0,0,0,0.12)",
        "text": "#111827",
        "muted": "#6B7280",
        "dim": "#9CA3AF",
        "primary": "#4F46E5",
        "primary-hover": "#4338CA",
        "primary-soft": "rgba(79,70,229,0.10)",
        "primary-ring": "rgba(79,70,229,0.25)",
        "success": "#16A34A",
        "danger": "#DC2626",
        "user-bubble": "linear-gradient(135deg, #4F46E5 0%, #6366F1 100%)",
        "user-bubble-text": "#FFFFFF",
        "assistant-bubble": "#FFFFFF",
        "scheme": "light",
    },
}

_theme = THEMES[st.session_state.theme]
_root_vars = "\n".join(
    [f"            --{k}: {v};" for k, v in _theme.items() if k != "scheme"]
)

# ============================================================
# اللوجو — علامة Nexus مبسّطة (شبكة معرفة)
# ============================================================
def get_nexus_logo(size: int = 32) -> str:
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="lg-nx" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#818CF8"/>
      <stop offset="100%" stop-color="#4F46E5"/>
    </linearGradient>
  </defs>
  <rect x="1" y="1" width="38" height="38" rx="10" fill="url(#lg-nx)"/>
  <circle cx="20" cy="20" r="4" fill="#FFFFFF"/>
  <circle cx="10" cy="10" r="2.2" fill="#FFFFFF" opacity="0.95"/>
  <circle cx="30" cy="10" r="2.2" fill="#FFFFFF" opacity="0.95"/>
  <circle cx="10" cy="30" r="2.2" fill="#FFFFFF" opacity="0.95"/>
  <circle cx="30" cy="30" r="2.2" fill="#FFFFFF" opacity="0.95"/>
  <line x1="20" y1="20" x2="10" y2="10" stroke="#FFFFFF" stroke-width="1.4" opacity="0.8"/>
  <line x1="20" y1="20" x2="30" y2="10" stroke="#FFFFFF" stroke-width="1.4" opacity="0.8"/>
  <line x1="20" y1="20" x2="10" y2="30" stroke="#FFFFFF" stroke-width="1.4" opacity="0.8"/>
  <line x1="20" y1="20" x2="30" y2="30" stroke="#FFFFFF" stroke-width="1.4" opacity="0.8"/>
</svg>
"""

NEXUS_LOGO_SVG = get_nexus_logo(32)

# ============================================================
# CSS — تصميم SaaS بسيط وأنيق
# ============================================================
CUSTOM_CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {{
{_root_vars}
        color-scheme: {_theme["scheme"]};
    }}

    /* ============ التهيئة العامة ============ */
    html, body, [class*="css"], [class*="st-"] {{
        font-family: 'Inter', 'Cairo', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    * {{
        unicode-bidi: plaintext;
    }}
    .stApp {{
        background: var(--bg) !important;
        color: var(--text) !important;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 0 !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 900px !important;
    }}

    /* ============ الشريط الجانبي ============ */
    section[data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 1rem;
    }}

    /* Brand */
    .brand {{
        display: flex; align-items: center; gap: 12px;
        padding: 4px 4px 20px 4px;
        margin-bottom: 8px;
    }}
    .brand-text {{ display: flex; flex-direction: column; line-height: 1; }}
    .brand-name {{
        font-family: 'Inter', sans-serif !important;
        font-size: 17px; font-weight: 700; color: var(--text);
        letter-spacing: -0.02em;
    }}
    .brand-tag {{
        font-size: 10.5px; font-weight: 500; color: var(--muted);
        letter-spacing: 0.08em; text-transform: uppercase;
        margin-top: 4px;
    }}

    /* Section labels */
    .side-label {{
        font-size: 11px; font-weight: 600; color: var(--dim);
        letter-spacing: 0.08em; text-transform: uppercase;
        margin: 18px 4px 10px 4px;
    }}

    /* File card */
    .file-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 12px;
        display: flex; align-items: center; gap: 10px;
        transition: all .18s ease;
    }}
    .file-card:hover {{
        background: var(--card-hover);
        border-color: var(--border-strong);
        transform: translateY(-1px);
    }}
    .file-icon {{
        width: 32px; height: 32px; flex-shrink: 0;
        background: var(--primary-soft);
        color: var(--primary);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 700;
    }}
    .file-body {{ flex: 1; min-width: 0; }}
    .file-name {{
        font-size: 12.5px; font-weight: 600; color: var(--text);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        direction: ltr; text-align: left;
    }}
    .file-meta {{
        font-size: 11px; color: var(--muted); margin-top: 2px;
        direction: ltr; text-align: left;
    }}

    /* Delete button (small, ghost) */
    div[data-testid="column"] .stButton > button[kind="secondary"],
    .row-delete .stButton > button {{
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
        border-radius: 10px !important;
        padding: 6px 8px !important;
        min-height: 38px !important;
        font-size: 13px !important;
    }}
    .row-delete .stButton > button:hover {{
        color: var(--danger) !important;
        border-color: var(--danger) !important;
        background: rgba(239,68,68,0.06) !important;
    }}

    /* Streamlit button — default */
    .stButton > button {{
        background: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 8px 14px !important;
        transition: all .18s ease !important;
        box-shadow: none !important;
    }}
    .stButton > button:hover {{
        background: var(--card-hover) !important;
        border-color: var(--border-strong) !important;
        transform: translateY(-1px);
    }}

    /* Primary theme toggle */
    .theme-toggle .stButton > button {{
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        width: 100%;
        display: flex; align-items: center; justify-content: center;
        gap: 8px;
        font-size: 13px !important;
    }}
    .theme-toggle .stButton > button:hover {{
        border-color: var(--primary) !important;
        color: var(--primary) !important;
    }}

    /* File uploader — modern dropzone */
    [data-testid="stFileUploader"] section {{
        background: var(--card) !important;
        border: 1.5px dashed var(--border-strong) !important;
        border-radius: 14px !important;
        padding: 18px !important;
        transition: all .2s ease !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--primary) !important;
        background: var(--primary-soft) !important;
    }}
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {{
        color: var(--muted) !important;
        font-size: 12px !important;
    }}
    [data-testid="stFileUploader"] button {{
        background: var(--primary) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }}
    [data-testid="stFileUploader"] button:hover {{
        background: var(--primary-hover) !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: var(--muted) !important;
    }}

    /* Status pill */
    .status-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 12px; border-radius: 10px;
        background: var(--card); border: 1px solid var(--border);
        margin-bottom: 6px;
        font-size: 12px;
    }}
    .status-key {{ color: var(--muted); font-weight: 500; }}
    .status-val {{ color: var(--text); font-weight: 600; direction: ltr; }}
    .dot {{
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        margin-inline-end: 6px; vertical-align: middle;
    }}
    .dot-on {{ background: var(--success); box-shadow: 0 0 0 3px rgba(34,197,94,0.15); }}
    .dot-off {{ background: var(--danger); box-shadow: 0 0 0 3px rgba(239,68,68,0.15); }}

    /* Empty state — sidebar */
    .empty-hint {{
        border: 1px dashed var(--border-strong);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        color: var(--muted);
        font-size: 12.5px;
        background: transparent;
    }}

    /* ============ منطقة المحادثة ============ */
    /* Compact top row */
    .top-row {{
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 18px;
    }}
    .top-title {{
        font-size: 15px; font-weight: 600; color: var(--text);
        display: flex; align-items: center; gap: 10px;
    }}
    .top-badge {{
        font-size: 11px; font-weight: 500;
        padding: 4px 10px; border-radius: 999px;
        background: var(--primary-soft); color: var(--primary);
        direction: ltr; letter-spacing: 0.02em;
    }}

    /* Empty chat state */
    .empty-chat {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center;
        padding: 80px 20px 40px 20px;
        color: var(--muted);
    }}
    .empty-chat h3 {{
        color: var(--text); font-size: 22px; font-weight: 600;
        margin: 18px 0 6px 0; letter-spacing: -0.01em;
    }}
    .empty-chat p {{
        font-size: 14px; color: var(--muted); margin: 0;
        max-width: 420px;
    }}

    /* Chat bubbles */
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        border: none !important;
        padding: 6px 0 !important;
        margin-bottom: 8px !important;
        animation: fadeInUp .28s ease;
    }}
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{
        background: var(--assistant-bubble);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 18px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    /* User bubble: right-aligned, gradient */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        flex-direction: row-reverse;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
        background: var(--user-bubble) !important;
        border: none !important;
        color: var(--user-bubble-text) !important;
        max-width: 78%;
        margin-left: auto;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {{
        color: var(--user-bubble-text) !important;
    }}
    [data-testid="stChatMessage"] p {{
        font-size: 14.5px !important;
        line-height: 1.75 !important;
        margin: 0 !important;
    }}
    [data-testid="stChatMessageAvatar"] {{
        border-radius: 10px !important;
    }}

    /* Source chips */
    .sources {{
        margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px;
    }}
    .chip {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        border: 1px solid var(--primary-ring);
        font-size: 11.5px; font-weight: 500;
        direction: ltr;
        transition: all .15s ease;
    }}
    .chip:hover {{ transform: translateY(-1px); }}

    /* Chat input — floating */
    [data-testid="stChatInput"] {{
        background: var(--card) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 24px -12px rgba(0,0,0,0.35) !important;
        transition: all .2s ease !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px var(--primary-ring) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: var(--text) !important;
        font-size: 14.5px !important;
        font-family: 'Inter','Cairo',sans-serif !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--dim) !important;
    }}
    [data-testid="stChatInput"] button {{
        background: var(--primary) !important;
        color: #fff !important;
        border-radius: 10px !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        background: var(--primary-hover) !important;
    }}

    /* Divider */
    hr, [data-testid="stSidebar"] hr {{
        border-color: var(--border) !important;
        margin: 14px 0 !important;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: var(--card) !important;
        font-size: 13px !important;
    }}

    /* Spinner */
    .stSpinner > div {{ border-top-color: var(--primary) !important; }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    /* RTL handling for Arabic-only text nodes */
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] > div {{
        unicode-bidi: plaintext;
    }}

    /* Responsive */
    @media (max-width: 768px) {{
        .block-container {{ padding: 1rem 0.75rem 6rem !important; }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
            max-width: 90%;
        }}
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# منطق إدارة الملفات والفهرسة — لم يتغيّر
# ============================================================
def rebuild_index():
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
    dest_path = DATA_DIR / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.file_registry[uploaded_file.name] = str(dest_path)


def remove_file(filename: str) -> None:
    filepath = st.session_state.file_registry.pop(filename, None)
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            logger.error(f"فشل حذف الملف {filename} من القرص: {e}")
    rebuild_index()


# ============================================================
# الشريط الجانبي
# ============================================================
with st.sidebar:
    # Brand
    st.markdown(
        f"""
<div class="brand">
    {NEXUS_LOGO_SVG}
    <div class="brand-text">
        <span class="brand-name">Nexus RAG</span>
        <span class="brand-tag">Document Intelligence</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Theme toggle
    st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
    is_dark = st.session_state.theme == "dark"
    toggle_label = "☀️  Light mode" if is_dark else "🌙  Dark mode"
    if st.button(toggle_label, use_container_width=True, key="theme_toggle"):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="side-label">Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload documents",
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
            with st.spinner("Indexing documents..."):
                rebuild_index()
            st.rerun()

    # Documents
    st.markdown('<div class="side-label">Documents</div>', unsafe_allow_html=True)

    if not st.session_state.file_registry:
        st.markdown(
            '<div class="empty-hint">No documents yet.<br>Upload a PDF or TXT to begin.</div>',
            unsafe_allow_html=True,
        )
    else:
        for filename in list(st.session_state.file_registry.keys()):
            n_chunks = sum(
                1
                for c in st.session_state.all_chunks
                if c.metadata.get("source") == filename
            )
            ext = Path(filename).suffix.replace(".", "").upper() or "DOC"
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(
                    f"""
<div class="file-card" title="{filename}">
    <div class="file-icon">{ext[:3]}</div>
    <div class="file-body">
        <div class="file-name">{filename}</div>
        <div class="file-meta">{n_chunks} chunks</div>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown('<div class="row-delete">', unsafe_allow_html=True)
                if st.button("✕", key=f"remove_{filename}", help=f"Remove {filename}"):
                    with st.spinner("Updating..."):
                        remove_file(filename)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # Status
    st.markdown('<div class="side-label">Status</div>', unsafe_allow_html=True)
    engine_on = bool(GROQ_API_KEY)
    dot_class = "dot-on" if engine_on else "dot-off"
    engine_txt = "Connected" if engine_on else "Offline"

    st.markdown(
        f"""
<div class="status-row"><span class="status-key">Engine</span>
  <span class="status-val"><span class="dot {dot_class}"></span>{engine_txt}</span></div>
<div class="status-row"><span class="status-key">Files</span>
  <span class="status-val">{len(st.session_state.file_registry)}</span></div>
<div class="status-row"><span class="status-key">Chunks</span>
  <span class="status-val">{len(st.session_state.all_chunks)}</span></div>
""",
        unsafe_allow_html=True,
    )

    # Settings — clear chat
    if st.session_state.chat_history:
        st.markdown('<div class="side-label">Settings</div>', unsafe_allow_html=True)
        if st.button("🧹  Clear conversation", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# الواجهة الرئيسية
# ============================================================
st.markdown(
    f"""
<div class="top-row">
    <div class="top-title">{NEXUS_LOGO_SVG}<span>Nexus RAG</span></div>
    <div class="top-badge">Hybrid · Cited</div>
</div>
""",
    unsafe_allow_html=True,
)

# Empty state
if not st.session_state.chat_history:
    st.markdown(
        f"""
<div class="empty-chat">
    {get_nexus_logo(56)}
    <h3>Ask your documents anything</h3>
    <p>Upload PDFs or text files, then ask questions in Arabic or English. Answers are grounded in your sources.</p>
</div>
""",
        unsafe_allow_html=True,
    )

# History
for msg in st.session_state.chat_history:
    with st.chat_message(
        msg["role"], avatar="🧠" if msg["role"] == "assistant" else "🧑"
    ):
        st.markdown(f'<div dir="auto">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            chips = "".join(
                [f'<span class="chip">📄 {s}</span>' for s in msg["sources"]]
            )
            st.markdown(f'<div class="sources">{chips}</div>', unsafe_allow_html=True)

# Input
query = st.chat_input("Ask a question about your documents…")

if query:
    if not st.session_state.file_registry:
        st.warning("⚠️ Please upload at least one document before asking questions.")
    elif not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY is not set. Cannot generate an answer.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f'<div dir="auto">{query}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching and reasoning..."):
                result = generate_rag_response(query, st.session_state.all_chunks)
                answer = result["answer"]
                sources = result.get("source_documents", [])

                source_labels = []
                for doc in sources:
                    src = doc.metadata.get("source", "document")
                    page = doc.metadata.get("page")
                    label = f"{src}" + (f" · p.{page}" if page else "")
                    if label not in source_labels:
                        source_labels.append(label)

            st.markdown(f'<div dir="auto">{answer}</div>', unsafe_allow_html=True)

            if source_labels:
                chips = "".join(
                    [f'<span class="chip">📄 {s}</span>' for s in source_labels]
                )
                st.markdown(
                    f'<div class="sources">{chips}</div>', unsafe_allow_html=True
                )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": source_labels}
        )
