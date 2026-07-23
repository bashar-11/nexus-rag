"""
Nexus RAG — Intelligent Document Assistant
واجهة مستخدم بهوية بصرية موحّدة (Nexus RAG) مع دعم الوضع الداكن/الفاتح.

ملاحظات التصميم:
- هوية بصرية: اسم المنتج "Nexus RAG" + لوجو SVG (عقدة معرفة مركزية متصلة
  بعقد فرعية) يتكيّف لونه مع الوضع الحالي (داكن/فاتح).
- نظام ألوان مزدوج مبني على متغيرات CSS واحدة (:root)، فتبديل الوضع
  لا يحتاج تكرار الأنماط — فقط تبديل قيم المتغيرات.
- الوضع الداكن: Indigo عميق + Amber دافئ (الهوية الأساسية).
- الوضع الفاتح: خلفية رمادية-باردة نظيفة + نفس عائلة الـ Amber لكن بتشبّع
  أعلى قليلاً لضمان التباين على الأبيض (تفادي تركيبة "كريمي + طيني" الشائعة).
- كل المسارات والاستدعاءات (imports, functions) كما هي بدون أي تغيير
  حتى يعمل الملف كبديل مباشر (drop-in replacement) للملف القديم.
"""

import streamlit as st
import textwrap as _tw
_orig_md = st.markdown
def _md(body="", unsafe_allow_html=False, **kw):
    if isinstance(body, str) and unsafe_allow_html:
        body = _tw.dedent(body)
    return _orig_md(body, unsafe_allow_html=unsafe_allow_html, **kw)
st.markdown = _md
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
    page_title="Nexus RAG — مساعد الوثائق الذكي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# الحالة (Session State) — بما فيها المظهر (Theme)
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
    st.session_state.theme = "dark"  # الافتراضي

# ============================================================
# اللوجو (SVG مضمّن) — عقدة مركزية متصلة بعقد معرفة
# دالة عشان لون خلفية الدائرة يتكيّف مع الوضع الحالي
# ============================================================
def nexus_logo(size: int = 42, circle_bg: str = "#1A1E29") -> str:
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="ng" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#F5B84A"/>
          <stop offset="100%" stop-color="#E8873D"/>
        </linearGradient>
      </defs>
      <circle cx="32" cy="32" r="30" fill="{circle_bg}" stroke="url(#ng)" stroke-width="1.5"/>
      <line x1="32" y1="32" x2="14" y2="14" stroke="#F5B84A" stroke-width="1.2" opacity="0.6"/>
      <line x1="32" y1="32" x2="50" y2="14" stroke="#F5B84A" stroke-width="1.2" opacity="0.6"/>
      <line x1="32" y1="32" x2="12" y2="40" stroke="#F5B84A" stroke-width="1.2" opacity="0.6"/>
      <line x1="32" y1="32" x2="52" y2="42" stroke="#F5B84A" stroke-width="1.2" opacity="0.6"/>
      <line x1="32" y1="32" x2="30" y2="56" stroke="#F5B84A" stroke-width="1.2" opacity="0.6"/>
      <circle cx="14" cy="14" r="3" fill="#F5B84A"/>
      <circle cx="50" cy="14" r="3" fill="#F5B84A"/>
      <circle cx="12" cy="40" r="3" fill="#F5B84A"/>
      <circle cx="52" cy="42" r="3" fill="#F5B84A"/>
      <circle cx="30" cy="56" r="3" fill="#F5B84A"/>
      <circle cx="32" cy="32" r="7" fill="url(#ng)"/>
      <circle cx="32" cy="32" r="7" fill="none" stroke="{circle_bg}" stroke-width="1.5"/>
    </svg>
    """

# ============================================================
# رموز الألوان لكل وضع — نفس هوية Nexus (Indigo/Amber) بنسختين
# ============================================================
THEMES = {
    "dark": {
        "bg": "#0F111A", "bg-2": "#12141C",
        "surface": "#171B26", "surface-alt": "#1E2333", "surface-hover": "#252B3D",
        "border": "#2A3145", "border-strong": "#3A4260",
        "text": "#EDEEF3", "text-muted": "#8B92A8", "text-dim": "#5C6480",
        "accent": "#F5B84A", "accent-2": "#E8873D",
        "accent-soft": "rgba(245, 184, 74, 0.12)", "accent-glow": "rgba(245, 184, 74, 0.28)",
        "user-bubble": "#22304F", "assistant-bubble": "#1A1F2E",
        "success": "#4CD964", "danger": "#E5626A",
        "overlay-soft": "rgba(255,255,255,0.06)", "overlay-faint": "rgba(255,255,255,0.015)",
        "logo-bg": "#1A1E29",
        "glow-1": "rgba(245,184,74,0.08)", "glow-2": "rgba(80,120,220,0.06)",
        "shadow": "0 10px 28px rgba(0,0,0,0.35)",
        "scheme": "dark",
    },
    "light": {
        "bg": "#F3F5F9", "bg-2": "#FFFFFF",
        "surface": "#FFFFFF", "surface-alt": "#F6F8FC", "surface-hover": "#ECEFF6",
        "border": "#E1E5EF", "border-strong": "#C9CFDF",
        "text": "#1C2030", "text-muted": "#5C6178", "text-dim": "#8A8FA3",
        "accent": "#D97B2E", "accent-2": "#B85F1E",
        "accent-soft": "rgba(217, 123, 46, 0.10)", "accent-glow": "rgba(217, 123, 46, 0.22)",
        "user-bubble": "#E8EDFB", "assistant-bubble": "#F6F8FC",
        "success": "#1FA854", "danger": "#D8434C",
        "overlay-soft": "rgba(20,25,40,0.045)", "overlay-faint": "rgba(20,25,40,0.02)",
        "logo-bg": "#FFFFFF",
        "glow-1": "rgba(217,123,46,0.06)", "glow-2": "rgba(80,120,220,0.05)",
        "shadow": "0 10px 28px rgba(20,25,40,0.08)",
        "scheme": "light",
    },
}

_theme = THEMES[st.session_state.theme]
_root_vars = "\n".join([f"            --{k}: {v};" for k, v in _theme.items() if k != "scheme"])

# ============================================================
# CSS: نظام تصميم Nexus RAG (متغيرات ديناميكية حسب الوضع)
# ============================================================
st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
{_root_vars}
            --radius: 10px;
            color-scheme: {_theme["scheme"]};
        }}

        html, body {{ direction: rtl; }}

        .stApp {{
            background:
                radial-gradient(circle at 12% -5%, var(--glow-1), transparent 45%),
                radial-gradient(circle at 88% 100%, var(--glow-2), transparent 40%),
                var(--bg);
            color: var(--text);
            font-family: 'Cairo', 'Inter', sans-serif;
            transition: background 0.25s ease, color 0.25s ease;
        }}

        #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.2rem; padding-bottom: 6rem; max-width: 1100px; }}

        /* ===== الشريط الجانبي ===== */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--surface) 0%, var(--bg-2) 100%);
            border-left: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

        /* ===== مبدّل الوضع (Theme Switch) ===== */
        div[role="radiogroup"] {{
            display: flex;
            gap: 4px;
            background: var(--surface-alt);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 3px;
            margin-bottom: 16px;
        }}
        div[role="radiogroup"] label {{
            flex: 1;
            justify-content: center;
            border-radius: 999px;
            padding: 5px 6px;
            margin: 0 !important;
            font-family: 'Cairo', sans-serif;
            font-size: 12.5px;
            transition: all 0.15s ease;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            background: var(--accent-soft);
            box-shadow: inset 0 0 0 1px var(--accent-glow);
        }}
        div[role="radiogroup"] label p {{ color: var(--text) !important; }}

        /* ===== ترويسة العلامة التجارية ===== */
        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 2px 4px 14px 4px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 18px;
        }}
        .brand-text {{ display: flex; flex-direction: column; line-height: 1.15; }}
        .brand-name {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 20px;
            letter-spacing: -0.01em;
            color: var(--text);
            direction: ltr;
        }}
        .brand-name span {{ color: var(--accent); }}
        .brand-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10.5px;
            color: var(--text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 2px;
            direction: ltr;
        }}

        /* ===== ترويسة الصفحة الرئيسية ===== */
        .app-hero {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 4px 0 18px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 22px;
        }}
        .app-hero-title {{
            font-family: 'Cairo', sans-serif;
            font-weight: 700;
            font-size: 26px;
            color: var(--text);
            margin: 0;
        }}
        .app-hero-sub {{
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 6px;
            max-width: 640px;
        }}
        .pill {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid var(--accent-glow);
            padding: 4px 10px;
            border-radius: 999px;
            letter-spacing: 0.05em;
            direction: ltr;
            display: inline-block;
        }}

        /* ===== تسميات الأقسام في الشريط الجانبي ===== */
        .sidebar-label {{
            font-family: 'Cairo', sans-serif;
            font-size: 13px;
            font-weight: 700;
            color: var(--text-muted);
            margin: 16px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .sidebar-label::before {{
            content: "";
            width: 3px; height: 14px;
            background: var(--accent);
            border-radius: 2px;
        }}

        /* ===== بطاقة الملف ===== */
        .file-card {{
            background: var(--surface-alt);
            border: 1px solid var(--border);
            border-right: 3px solid var(--accent);
            border-radius: var(--radius);
            padding: 10px 12px;
            margin-bottom: 6px;
            transition: all 0.18s ease;
        }}
        .file-card:hover {{
            background: var(--surface-hover);
            border-color: var(--border-strong);
        }}
        .file-card .fname {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text);
            direction: ltr;
            text-align: left;
            unicode-bidi: plaintext;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            display: block;
        }}
        .file-card .fmeta {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
            font-family: 'Cairo', sans-serif;
        }}

        /* ===== رفع الملفات ===== */
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--surface-alt);
            border: 1.5px dashed var(--border-strong);
            border-radius: var(--radius);
            transition: all 0.18s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: var(--accent);
            background: var(--accent-soft);
        }}

        /* ===== أزرار ===== */
        .stButton>button {{
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--surface-alt);
            color: var(--text);
            font-family: 'Cairo', sans-serif;
            font-weight: 500;
            padding: 6px 12px;
            transition: all 0.15s ease;
        }}
        .stButton>button:hover {{
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
        }}

        /* ===== فقاعات المحادثة ===== */
        div[data-testid="stChatMessage"] {{
            background: var(--assistant-bubble);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 10px;
            box-shadow: var(--shadow);
        }}
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            background: var(--user-bubble);
            border-color: var(--accent-glow);
        }}

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] h1,
        div[data-testid="stChatMessage"] h2,
        div[data-testid="stChatMessage"] h3,
        div[data-testid="stChatMessage"] h4 {{
            unicode-bidi: plaintext;
            color: var(--text);
        }}
        div[data-testid="stChatMessage"] h1,
        div[data-testid="stChatMessage"] h2,
        div[data-testid="stChatMessage"] h3 {{
            color: var(--accent);
            font-family: 'Inter', 'Cairo', sans-serif;
        }}
        div[data-testid="stChatMessage"] code {{
            background: var(--overlay-soft);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}

        /* ===== رقاقات المصادر ===== */
        .sources-wrap {{
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px dashed var(--border);
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .source-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid var(--accent-glow);
            border-radius: 999px;
            padding: 4px 10px;
            direction: ltr;
            unicode-bidi: plaintext;
        }}
        .source-chip::before {{
            content: "◆";
            font-size: 9px;
            opacity: 0.7;
        }}

        /* ===== الحالة الفارغة ===== */
        .empty-state {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            padding: 22px 12px;
            border: 1.5px dashed var(--border);
            border-radius: var(--radius);
            background: var(--overlay-faint);
            line-height: 1.8;
        }}
        .empty-state .icon {{
            font-size: 22px;
            display: block;
            margin-bottom: 6px;
            opacity: 0.6;
        }}

        /* الشاشة الترحيبية */
        .welcome {{
            background: linear-gradient(135deg, var(--surface) 0%, var(--surface-alt) 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 34px 28px;
            text-align: center;
            margin: 20px 0;
            box-shadow: var(--shadow);
        }}
        .welcome h3 {{
            font-family: 'Cairo', sans-serif;
            font-weight: 700;
            font-size: 20px;
            color: var(--text);
            margin: 12px 0 6px 0;
        }}
        .welcome p {{
            color: var(--text-muted);
            font-size: 14px;
            max-width: 520px;
            margin: 0 auto;
            line-height: 1.8;
        }}
        .features {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 24px;
        }}
        .feature {{
            background: var(--surface-alt);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 14px 12px;
            text-align: center;
        }}
        .feature .fi {{ font-size: 20px; margin-bottom: 6px; }}
        .feature .ft {{
            font-family: 'Cairo', sans-serif;
            font-weight: 600;
            font-size: 13px;
            color: var(--text);
        }}
        .feature .fd {{
            font-size: 11.5px;
            color: var(--text-muted);
            margin-top: 4px;
            line-height: 1.6;
        }}
        @media (max-width: 640px) {{
            .features {{ grid-template-columns: 1fr; }}
        }}

        /* شريط الحالة السفلي */
        .status-bar {{
            font-size: 12px;
            color: var(--text-muted);
            line-height: 2;
            font-family: 'Cairo', sans-serif;
        }}
        .status-dot {{
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-left: 6px;
            box-shadow: 0 0 8px currentColor;
        }}

        /* إدخال المحادثة */
        [data-testid="stChatInput"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

NEXUS_LOGO_SVG = nexus_logo(circle_bg=_theme["logo-bg"])


# ============================================================
# منطق إدارة الملفات والفهرسة (بدون تغيير في التوقيعات)
# ============================================================
def rebuild_index():
    """إعادة بناء الفهرس (Vector Store) بالكامل من كل الملفات الحالية."""
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
# الشريط الجانبي
# ============================================================
with st.sidebar:
    # ===== مبدّل الوضع الداكن/الفاتح =====
    theme_label = st.radio(
        "المظهر",
        options=["🌙  داكن", "☀️  فاتح"],
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_switch",
    )
    chosen_theme = "dark" if "🌙" in theme_label else "light"
    if chosen_theme != st.session_state.theme:
        st.session_state.theme = chosen_theme
        st.rerun()

    # الهوية البصرية
    st.markdown(
        f"""
        <div class="brand">
            {NEXUS_LOGO_SVG}
            <div class="brand-text">
                <div class="brand-name">Nexus<span>RAG</span></div>
                <div class="brand-tag">Intelligent Docs · v1.0</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            """
            <div class="empty-state">
                <span class="icon">📥</span>
                لا توجد ملفات بعد.<br>
                ارفع PDF أو TXT للبدء.
            </div>
            """,
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
                    f"""<div class="file-card" title="{filename}">
                            <span class="fname">📄 {filename}</span>
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
    dot_color = "var(--success)" if GROQ_API_KEY else "var(--danger)"
    status_label = "متصل" if GROQ_API_KEY else "غير مضبوط"
    st.markdown(
        f"""
        <div class="status-bar">
            <span class="status-dot" style="background:{dot_color}; color:{dot_color};"></span>
            حالة النموذج (GROQ): <b style="color:var(--text);">{status_label}</b><br>
            الملفات: <b style="color:var(--text);">{len(st.session_state.file_registry)}</b><br>
            المقاطع المفهرسة: <b style="color:var(--text);">{total_chunks}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history and st.button("🧹 مسح المحادثة", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================
# الواجهة الرئيسية
# ============================================================
st.markdown(
    f"""
    <div class="app-hero">
        <div>
            <h1 class="app-hero-title">مساعد الوثائق الذكي</h1>
            <div class="app-hero-sub">
                اسأل بلغتك الطبيعية — إجابات دقيقة من مستنداتك، موثّقة بالمصادر.
            </div>
        </div>
        <span class="pill">RAG · Hybrid Retrieval</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# شاشة ترحيبية عند عدم وجود محادثة
if not st.session_state.chat_history:
    st.markdown(
        f"""
        <div class="welcome">
            {NEXUS_LOGO_SVG}
            <h3>مرحبًا بك في Nexus RAG</h3>
            <p>ارفع مستنداتك من الشريط الجانبي، ثم اطرح أسئلتك — سنستخرج الإجابة من محتواك مع الاستشهاد بالمصادر.</p>
            <div class="features">
                <div class="feature">
                    <div class="fi">🎯</div>
                    <div class="ft">إجابات دقيقة</div>
                    <div class="fd">مبنية حصريًا على مستنداتك</div>
                </div>
                <div class="feature">
                    <div class="fi">🔗</div>
                    <div class="ft">استشهاد بالمصادر</div>
                    <div class="fd">كل إجابة موثّقة برقم الصفحة</div>
                </div>
                <div class="feature">
                    <div class="fi">⚡</div>
                    <div class="ft">استرجاع هجين</div>
                    <div class="fd">بحث دلالي + كلمات مفتاحية</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# عرض سجل المحادثة
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "🧑"):
        st.markdown(f'<div dir="auto">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{s}</span>' for s in msg["sources"]
            )
            st.markdown(f'<div class="sources-wrap">{chips}</div>', unsafe_allow_html=True)

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
            st.markdown(f'<div dir="auto">{query}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("جاري البحث والتحليل..."):
                result = generate_rag_response(query, st.session_state.all_chunks)
                answer = result["answer"]
                sources = result.get("source_documents", [])

                source_labels = []
                for doc in sources:
                    src = doc.metadata.get("source", "مستند")
                    page = doc.metadata.get("page")
                    source_labels.append(f"{src}" + (f" · p{page}" if page else ""))
                source_labels = list(dict.fromkeys(source_labels))

            st.markdown(f'<div dir="auto">{answer}</div>', unsafe_allow_html=True)
            if source_labels:
                chips = "".join(f'<span class="source-chip">{s}</span>' for s in source_labels)
                st.markdown(f'<div class="sources-wrap">{chips}</div>', unsafe_allow_html=True)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": source_labels}
        )
