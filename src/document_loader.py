import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, logger


def _clean_text(text: str) -> str:
    """
    تنظيف موحّد للنص: إزالة المسافات/الأسطر الزائدة التي تُضعف
    جودة الـ embeddings وتُدخل "ضوضاء" في الـ chunks.
    """
    if not text:
        return ""
    # توحيد الأسطر الجديدة المتكررة (3 أسطر فارغة أو أكثر -> سطرين فقط)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # إزالة المسافات المتكررة داخل السطر الواحد
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_document(file_path: str) -> List[Document]:
    """
    تحميل وتنظيف نصوص ملفات PDF و TXT.
    كل صفحة/ملف يحصل على metadata نظيفة (اسم الملف فقط، بدون المسار الكامل)
    عشان الاقتباسات اللي بترجع للمستخدم تبقى واضحة ومفيدة.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف غير موجود في المسار: {file_path}")

    file_name = os.path.basename(file_path)
    docs: List[Document] = []

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        for i, doc in enumerate(raw_docs):
            cleaned_text = _clean_text(doc.page_content)
            # استبعاد الصفحات شبه الفارغة أو التالفة (مثل صفحات الغلاف الفارغة)
            if len(cleaned_text) > 10:
                doc.page_content = cleaned_text
                doc.metadata["source"] = file_name
                doc.metadata["page"] = doc.metadata.get("page", i) + 1
                docs.append(doc)

        if not docs:
            logger.warning(f"لم يتم استخراج أي محتوى صالح من الملف: {file_name}")

    elif file_path.endswith(".txt"):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()
        except UnicodeDecodeError:
            # fallback لملفات مش UTF-8 نظيفة بالكامل
            logger.warning(f"فشل قراءة {file_name} كـ UTF-8، تتم إعادة المحاولة مع تجاهل الأخطاء")
            loader = TextLoader(file_path, encoding="utf-8", errors="ignore")
            raw_docs = loader.load()

        for doc in raw_docs:
            cleaned_text = _clean_text(doc.page_content)
            if len(cleaned_text) > 10:
                doc.page_content = cleaned_text
                doc.metadata["source"] = file_name
                docs.append(doc)
    else:
        raise ValueError("نوع الملف غير مدعوم، يرجى رفع ملف PDF أو TXT فقط.")

    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    تجزئة المستندات إلى Chunks متناسبة، مع إضافة رقم تسلسلي لكل chunk
    داخل مصدره (chunk_index) عشان يسهل تتبعه في الاقتباسات ولتشخيص
    مشاكل الاسترجاع لاحقًا.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "۔ ", ". ", "؟ ", "? ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    # ترقيم الـ chunks لكل مصدر على حدة (مفيد جدًا للتشخيص ولمنطق "overview query")
    counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk_index"] = counters[source]

    logger.info(f"تم إنشاء {len(chunks)} chunk من {len(documents)} مستند/صفحة")
    return chunks
