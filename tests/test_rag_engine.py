"""
Tests for the generation layer: query classification, context formatting,
and the guard clauses that run before any network call.

`is_overview_query` and `_format_context` are pure functions, and the guards
return early, so the whole module can be tested without a Groq API key.
"""

from langchain_core.documents import Document

from src.rag_engine import _format_context, generate_rag_response, is_overview_query


class TestIsOverviewQuery:
    def test_detects_english_summary_requests(self):
        for query in [
            "summarize this document",
            "Give me an overview",
            "what is this about",
            "main topic?",
        ]:
            assert is_overview_query(query), query

    def test_detects_arabic_summary_requests(self):
        # Regression guard: the Arabic keywords are the reason the app is
        # bilingual. If they are lost or mangled, Arabic users silently get
        # fact-retrieval behaviour instead of a summary.
        for query in [
            "لخص لي المستند",
            "ملخص",
            "نظرة عامة",
            "ما هو هذا",
            "اشرح لي",
        ]:
            assert is_overview_query(query), query

    def test_ignores_specific_factual_questions(self):
        for query in [
            "What was the revenue in Q3?",
            "who signed the contract",
            "كم عدد النتائج",
            "ما هي النتائج في التقرير؟",
        ]:
            assert not is_overview_query(query), query

    def test_does_not_match_a_keyword_inside_a_longer_arabic_word(self):
        # "نتائج" (results) must not match inside "النتائج" (the results).
        # Without a boundary guard the bare stem matches and a factual question
        # gets treated as a summary request.
        assert not is_overview_query("كم عدد النتائج")
        assert not is_overview_query("النتائج النهائية")

    def test_arabic_summary_verbs_still_match(self):
        # The boundary guard must not be so strict that it breaks real hits.
        for query in ["لخص لي", "ملخص المستند", "أهم النقاط"]:
            assert is_overview_query(query), query


class TestFormatContext:
    def test_labels_pages_when_available(self):
        docs = [
            Document(page_content="body", metadata={"source": "a.pdf", "page": 4}),
            Document(page_content="body", metadata={"source": "a.pdf", "page": 5}),
        ]
        out = _format_context(docs)
        assert "a.pdf" in out
        assert "page 4" in out
        assert "page 5" in out

    def test_falls_back_when_source_missing(self):
        out = _format_context([Document(page_content="body", metadata={})])
        assert "unknown source" in out

    def test_separates_passages_with_a_delimiter(self):
        docs = [
            Document(page_content="first passage", metadata={"source": "a.pdf"}),
            Document(page_content="second passage", metadata={"source": "b.pdf"}),
        ]
        assert "---" in _format_context(docs)


class TestGenerateRagResponseGuards:
    def test_returns_a_friendly_message_when_no_documents(self):
        result = generate_rag_response("anything", [])
        assert result["source_documents"] == []
        assert "upload" in result["answer"].lower()
        # The old build leaked mojibake into this string, which reached the UI.
        assert "\ufffd" not in result["answer"]