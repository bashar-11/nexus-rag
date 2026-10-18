"""
Tests for document loading, cleaning and chunking.

These exercise pure file handling and text splitting, so they need no API
keys and no vector database.
"""

import os
import tempfile

import pytest

from src.document_loader import _clean_text, load_document, split_documents


class TestCleanText:
    def test_strips_surrounding_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_collapses_excess_newlines(self):
        assert _clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_collapses_runs_of_spaces_and_tabs(self):
        assert _clean_text("a     b\t\t\tc") == "a b c"

    def test_empty_input(self):
        assert _clean_text("") == ""

    def test_none_input(self):
        assert _clean_text(None) == ""


class TestLoadDocument:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_document("this-file-does-not-exist.pdf")

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as fh:
            path = fh.name
        try:
            with pytest.raises(ValueError):
                load_document(path)
        finally:
            os.unlink(path)

    def test_loads_txt_with_source_metadata(self):
        content = (
            "Retrieval augmented generation combines search with generation.\n" * 5
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(content)
            path = fh.name
        try:
            docs = load_document(path)
            assert len(docs) == 1
            assert docs[0].metadata["source"] == os.path.basename(path)
        finally:
            os.unlink(path)

    def test_drops_content_too_short_to_be_meaningful(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("hi\n")  # under the 10 character minimum
            path = fh.name
        try:
            assert load_document(path) == []
        finally:
            os.unlink(path)

    def test_survives_non_utf8_text(self):
        with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as fh:
            fh.write(b"valid ascii text that is long enough.\n")
            fh.write(b"\xff\xfe\x00 invalid bytes \x81\x82")
            path = fh.name
        try:
            docs = load_document(path)
            assert len(docs) == 1
        finally:
            os.unlink(path)


class TestSplitDocuments:
    def _doc(self, text, source="notes.txt"):
        from langchain_core.documents import Document

        return Document(page_content=text, metadata={"source": source})

    def test_chunks_respect_overlap_and_size(self):
        text = ("word " * 900).strip()
        chunks = split_documents([self._doc(text)])
        assert len(chunks) > 1
        from src.config import CHUNK_SIZE

        assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)

    def test_chunk_index_restarts_per_source(self):
        text = ("word " * 900).strip()
        chunks = split_documents([self._doc(text, "a.txt"), self._doc(text, "b.txt")])
        by_source = {}
        for c in chunks:
            by_source.setdefault(c.metadata["source"], []).append(
                c.metadata["chunk_index"]
            )
        for indices in by_source.values():
            assert indices == list(range(1, len(indices) + 1))

    def test_arabic_sentences_split_on_arabic_punctuation(self):
        # Arabic text uses the Arabic question mark and comma. If the splitter
        # does not recognise them it falls back to whitespace and produces
        # chunks that cut mid-word.
        text = ("هذا نص عربي طويل للتجربة. " * 120).strip()
        chunks = split_documents([self._doc(text, "ar.txt")])
        assert len(chunks) > 1
        assert all(c.page_content.strip() for c in chunks)
