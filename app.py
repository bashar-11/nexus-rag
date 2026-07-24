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
    page_title="Nexus RAG | مساعد الوثائق الذكي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# الحالة (Session State)
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
# رموز الألوان والتصميم (Theme Dictionary)
# ============================================================
THEMES = {
    "dark": {
        "bg": "#090B10", "bg-2": "#0F1219",
        "surface": "#141824", "surface-alt": "#1B202F", "surface-hover": "#22293C",
        "border": "#283044", "border-strong": "#39435C",
        "text": "#F3F4F6", "text-muted": "#9CA3AF", "text-dim": "#6B7280",
        "accent": "#F59E0B", "accent-2": "#D97706",
        "accent-soft": "rgba(245, 158, 11, 0.15)", "accent-glow": "rgba(245, 158, 11, 0.3)",
        "user-bubble": "#1E2536", "assistant-bubble": "#111520",
        "success": "#10B981", "danger": "#EF4444",
        "logo-bg": "#141824",
        "scheme": "dark",
    },
    "light": {
        "bg": "#F9FAFB", "bg-2": "#F3F4F6",
        "surface": "#FFFFFF", "surface-alt": "#F3F4F6", "surface-hover": "#E5E7EB",
        "border": "#E5E7EB", "border-strong": "#D1D5DB",
        "text": "#111827", "text-muted": "#4B5563", "text-dim": "#9CA3AF",
        "accent": "#D97706", "accent-2": "#B45309",
        "accent-soft": "rgba(217, 119, 6, 0.12)", "accent-glow": "rgba(217, 119, 6, 0.25)",
        "user-bubble": "#F3F4F6", "assistant-bubble": "#FFFFFF",
        "success": "#059669", "danger": "#DC2626",
        "logo-bg": "#FFFFFF",
        "scheme": "light",
    },
}

_theme = THEMES[st.session_state.theme]
_root_vars = "\n".join([f"            --{k}: {v};" for k, v in _theme.items() if k != "scheme"])

# ============================================================
# اللوجو
# ============================================================
def get_nexus_logo(size: int = 45, circle_bg: str = "#1A1E29") -> str:
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
    <linearGradient id="ng" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#F5B84A"/>
        <stop offset="100%" stop-color="#E8873D"/>
    </linearGradient>
    </defs>
    <circle cx="32" cy="32" r="30" fill="{circle_bg}" stroke="url(#ng)" stroke-width="1.5"/>
    <line x1="32" y1="32" x2="14" y2="14" stroke="#F5B84A" stroke-width="1.2" opacity="0.7"/>
    <line x1="32" y1="32" x2="50" y2="14" stroke="#F5B84A" stroke-width="1.2" opacity="0.7"/>
    <line x1="32" y1="32" x2="12" y2="40" stroke="#F5B84A" stroke-width="1.2" opacity="0.7"/>
    <line x1="32" y1="32" x2="52" y2="42" stroke="#F5B84A" stroke-width="1.2" opacity="0.7"/>
    <line x1="32" y1="32" x2="30" y2="56" stroke="#F5B84A" stroke-width="1.2" opacity="0.7"/>
    <circle cx="14" cy="14" r="3.5" fill="#F5B84A"/>
    <circle cx="50" cy="14" r="3.5" fill="#F5B84A"/>
    <circle cx="12" cy="40" r="3.5" fill="#F5B84A"/>
    <circle cx="52" cy="42" r="3.5" fill="#F5B84A"/>
    <circle cx="30" cy="56" r="3.5" fill="#F5B84A"/>
    <circle cx="32" cy="32" r="8" fill="url(#ng)"/>
    <circle cx="32" cy="32" r="8" fill="none" stroke="{circle_bg}" stroke-width="2"/>
</svg>
"""

NEXUS_LOGO_SVG = get_nexus_logo(circle_bg=_theme["logo-bg"])

# ============================================================
# حقن CSS المخصص
# تم وضع نصوص الـ HTML بدون مسافات بادئة لتجنب تحولها إلى أكواد في Markdown
# ============================================================
CUSTOM_CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Inter:wght@500;700&display=swap" rel="stylesheet">
<style>
    :root {{
{_root_vars}
        color-scheme: {_theme["scheme"]};
    }}
    
    /* التهيئة الأساسية ودعم اللغة العربية */
    html, body, [class*="css"], [class*="st-"] {{
        font-family: 'Cairo', 'Inter', sans-serif !important;
        direction: rtl;
        text-align: right;
    }}
    
    .stApp {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }}
    
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    
    /* الشريط الجانبي */
    section[data-testid="stSidebar"] {{
        background-color: var(--surface) !important;
        border-left: 1px solid var(--border) !important;
    }}
    
    /* ترويسة العلامة التجارية */
    .brand-container {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 20px;
        direction: ltr;
        text-align: left;
    }}
    .brand-title {{
        font-family: 'Inter', sans-serif !important;
        font-size: 24px;
        font-weight: 800;
        color: var(--text);
        margin: 0;
        line-height: 1.1;
    }}
    .brand-title span {{ color: var(--accent); }}
    .brand-subtitle {{
        font-size: 11px;
        color: var(--text-muted);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 4px;
    }}

    /* الترويسة الرئيسية */
    .hero-section {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 30px;
        margin-top: -20px;
    }}
    .hero-title {{
        font-size: 28px;
        font-weight: 800;
        color: var(--text);
        margin: 0 0 8px 0;
    }}
    .hero-desc {{
        font-size: 15px;
        color: var(--text-muted);
        margin: 0;
    }}
    .hero-badge {{
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--accent-glow);
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 700;
        direction: ltr;
        font-family: 'Inter', sans-serif !important;
    }}

    /* الشاشة الترحيبية */
    .welcome-box {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.1);
    }}
    .welcome-title {{
        font-size: 24px;
        font-weight: 700;
        color: var(--text);
        margin: 20px 0 10px 0;
    }}
    .welcome-text {{
        color: var(--text-muted);
        font-size: 15px;
        margin-bottom: 30px;
    }}
    .features-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
    }}
    .feature-card {{
        background: var(--surface-alt);
        border: 1px solid var(--border);
        padding: 20px;
        border-radius: 12px;
    }}
    .feature-icon {{ font-size: 28px; margin-bottom: 12px; }}
    .feature-title {{ font-size: 16px; font-weight: 700; color: var(--text); margin-bottom: 6px; }}
    .feature-desc {{ font-size: 13px; color: var(--text-muted); line-height: 1.6; }}

    /* بطاقات الملفات */
    .file-card {{
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-right: 4px solid var(--accent);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }}
    .file-name {{
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        direction: ltr;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .file-meta {{
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 6px;
    }}

    /* فقاعات المحادثة */
    [data-testid="stChatMessage"] {{
        background-color: var(--assistant-bubble) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background-color: var(--user-bubble) !important;
        border-color: var(--accent-glow) !important;
    }}
    [data-testid="stChatMessage"] p {{
        font-size: 15px !important;
        line-height: 1.8 !important;
        color: var(--text) !important;
    }}

    /* مصادر الإجابة */
    .sources-container {{
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px dashed var(--border);
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .source-tag {{
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--accent-glow);
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        direction: ltr;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}

    /* أزرار Streamlit */
    .stButton > button {{
        background-color: var(--surface-alt) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }}
    
    /* زر الثيم تحديداً */
    .theme-btn-container .stButton > button {{
        background-color: var(--accent-soft) !important;
        color: var(--accent) !important;
        border: 1px solid var(--accent-glow) !important;
        padding: 10px !important;
    }}
    .theme-btn-container .stButton > button:hover {{
        background-color: var(--accent) !important;
        color: #fff !important;
    }}

    /* مربع إدخال النص */
    [data-testid="stChatInput"] {{
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: var(--accent) !important;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# منطق إدارة الملفات والفهرسة (بدون أي تعديل)
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
# الشريط الجانبي (Sidebar)
# ============================================================
with st.sidebar:
    # 1. العلامة التجارية
    BRAND_HTML = f"""
<div class="brand-container">
    {NEXUS_LOGO_SVG}
    <div>
        <h1 class="brand-title">Nexus<span>RAG</span></h1>
        <div class="brand-subtitle">Knowledge Hub</div>
    </div>
</div>
"""
    st.markdown(BRAND_HTML, unsafe_allow_html=True)

    # 2. زر تبديل الوضع الأنيق
    st.markdown('<div class="theme-btn-container">', unsafe_allow_html=True)
    theme_text = "☀️ تفعيل الوضع الفاتح" if st.session_state.theme == "dark" else "🌙 تفعيل الوضع الداكن"
    if st.button(theme_text, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

    # 3. مكتبة المستندات
    st.markdown("### 📚 إدارة المستندات")
    
    uploaded_files = st.file_uploader(
        "إضافة ملف (PDF أو TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
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

    st.markdown("---")
    st.markdown("### 🗂️ الملفات الحالية")

    if not st.session_state.file_registry:
        st.info("📥 لم يتم رفع أي ملفات. يرجى رفع ملفات PDF أو TXT للبدء.")
    else:
        for filename in list(st.session_state.file_registry.keys()):
            n_chunks = sum(1 for c in st.session_state.all_chunks if c.metadata.get("source") == filename)
            
            # عرض كل ملف مع زر الحذف بجانبه
            col1, col2 = st.columns([5, 1])
            with col1:
                FILE_CARD_HTML = f"""
<div class="file-card" title="{filename}">
    <div class="file-name">📄 {filename}</div>
    <div class="file-meta">مقاطع مفهرسة: {n_chunks}</div>
</div>
"""
                st.markdown(FILE_CARD_HTML, unsafe_allow_html=True)
            with col2:
                if st.button("🗑", key=f"remove_{filename}", help=f"إزالة {filename}"):
                    with st.spinner("جاري التحديث..."):
                        remove_file(filename)
                    st.rerun()

    st.divider()
    
    # معلومات الحالة
    total_chunks = len(st.session_state.all_chunks)
    model_status = "متصل ✅" if GROQ_API_KEY else "غير متصل ❌"
    
    st.markdown(f"""
    <div style="font-size: 13px; color: var(--text-muted); line-height: 2;">
        <b>حالة النظام:</b><br>
        المحرك (Groq): {model_status}<br>
        إجمالي الملفات: {len(st.session_state.file_registry)}<br>
        إجمالي المقاطع: {total_chunks}
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    if st.session_state.chat_history:
        if st.button("🧹 مسح المحادثة الحالية", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# الواجهة الرئيسية (Main UI)
# ============================================================
HERO_HTML = """
<div class="hero-section">
    <div>
        <h1 class="hero-title">مساعد الوثائق الذكي</h1>
        <p class="hero-desc">اسأل بلغتك الطبيعية — إجابات دقيقة من مستنداتك، موثّقة بالمصادر.</p>
    </div>
    <div class="hero-badge">RAG · Hybrid Engine</div>
</div>
"""
st.markdown(HERO_HTML, unsafe_allow_html=True)

# 1. الشاشة الترحيبية (إذا لم يكن هناك محادثة)
if not st.session_state.chat_history:
    WELCOME_HTML = f"""
<div class="welcome-box">
    {NEXUS_LOGO_SVG}
    <h2 class="welcome-title">مرحباً بك في Nexus RAG</h2>
    <p class="welcome-text">النظام جاهز. قم برفع مستنداتك من الشريط الجانبي ثم اطرح أسئلتك ليبدأ الاستخراج الذكي.</p>
    
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">دقة متناهية</div>
            <div class="feature-desc">إجابات مبنية حصرياً على محتوى المستندات المرفوعة لتجنب التأليف.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔗</div>
            <div class="feature-title">توثيق المصادر</div>
            <div class="feature-desc">كل إجابة مزودة بمرجع للملف ورقم الصفحة التي تم استخراجها منها.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">محرك بحث هجين</div>
            <div class="feature-desc">يجمع بين قوة البحث الدلالي والبحث النصي لنتائج فائقة السرعة.</div>
        </div>
    </div>
</div>
"""
    st.markdown(WELCOME_HTML, unsafe_allow_html=True)

# 2. عرض سجل المحادثات
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])
        
        # عرض المصادر إن وجدت
        if msg.get("sources"):
            sources_html = '<div class="sources-container">'
            for source in msg["sources"]:
                sources_html += f'<div class="source-tag">📄 {source}</div>'
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

# 3. إدخال المستخدم
query = st.chat_input("اكتب سؤالك هنا...")

if query:
    if not st.session_state.file_registry:
        st.warning("⚠️ يرجى رفع ملف واحد على الأقل قبل طرح الأسئلة.")
    elif not GROQ_API_KEY:
        st.error("⚠️ مفتاح GROQ_API_KEY غير مضبوط. لا يمكن توليد الإجابة.")
    else:
        # إضافة سؤال المستخدم
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)

        # توليد الإجابة
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("جاري البحث والتحليل..."):
                result = generate_rag_response(query, st.session_state.all_chunks)
                answer = result["answer"]
                sources = result.get("source_documents", [])

                # تنسيق المصادر
                source_labels = []
                for doc in sources:
                    src = doc.metadata.get("source", "مستند")
                    page = doc.metadata.get("page")
                    label = f"{src}" + (f" · صفحة {page}" if page else "")
                    if label not in source_labels:
                        source_labels.append(label)

            st.markdown(answer)
            
            # رسم رقاقات المصادر
            if source_labels:
                sources_html = '<div class="sources-container">'
                for source in source_labels:
                    sources_html += f'<div class="source-tag">📄 {source}</div>'
                sources_html += '</div>'
                st.markdown(sources_html, unsafe_allow_html=True)

        # الحفظ في سجل المحادثة
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer, 
            "sources": source_labels
        })