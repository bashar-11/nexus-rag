from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    تحميل نموذج الـ Embeddings المجاني والمحلي.

    ملاحظة مهمة للجودة: normalize_embeddings=True ضروري هنا —
    بدونه تصبح مقارنة التشابه (cosine/L2) غير متسقة بين المتجهات
    المختلفة الأطوال، مما يُفسد ترتيب نتائج الاسترجاع.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
    )
    return embeddings
