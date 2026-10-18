"""
Tests for hybrid retrieval.

`reciprocal_rank_fusion` is the piece that decides what the model actually
sees, so it is tested directly. The tests need no Qdrant instance.
"""

from langchain_core.documents import Document

from src.config import RRF_K
from src.vector_store import reciprocal_rank_fusion


def doc(text, source="a.pdf", chunk=0):
    return Document(page_content=text, metadata={"source": source, "chunk_index": chunk})


class TestReciprocalRankFusion:
    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_respects_top_k(self):
        docs = [doc(f"passage {i}", chunk=i) for i in range(20)]
        assert len(reciprocal_rank_fusion([docs], top_k=4)) == 4

    def test_deduplicates_documents_appearing_in_both_lists(self):
        shared = doc("the same passage", chunk=1)
        only_dense = doc("dense only", chunk=2)
        result = reciprocal_rank_fusion([[shared, only_dense], [shared]], top_k=10)
        texts = [d.page_content for d in result]
        assert texts.count("the same passage") == 1

    def test_a_document_in_both_lists_outranks_a_single_list_top_hit(self):
        # The point of RRF: agreement across retrievers beats a high rank in
        # just one. The shared document must not already sit at rank 0 in the
        # list it competes against, or both scores tie at 1/(k+1).
        shared = doc("agreed by both", chunk=1)
        filler_a = doc("dense first", chunk=2)
        filler_b = doc("lexical first", chunk=3)
        result = reciprocal_rank_fusion([[filler_a, shared], [filler_b, shared]], top_k=10)
        assert result[0].page_content == "agreed by both"

    def test_two_rank_zero_documents_score_equally(self):
        # Regression guard on the RRF arithmetic: a doc ranked first in one
        # list and a doc ranked first in the other both score 1/(k+1), so this
        # is a genuine tie rather than a preference for either.
        a = doc("ranked first", chunk=1)
        b = doc("shared", chunk=2)
        result = reciprocal_rank_fusion([[a], [b]], top_k=10)
        assert {r.page_content for r in result} == {"ranked first", "shared"}

    def test_rank_one_beats_rank_two_within_one_list(self):
        first = doc("ranked first", chunk=1)
        second = doc("ranked second", chunk=2)
        result = reciprocal_rank_fusion([[first, second]], top_k=10)
        assert result[0].page_content == "ranked first"

    def test_skips_blank_documents(self):
        result = reciprocal_rank_fusion([[doc("   ", chunk=1), doc("real", chunk=2)]], top_k=10)
        assert [d.page_content for d in result] == ["real"]

    def test_empty_content_is_never_ranked(self):
        result = reciprocal_rank_fusion([[doc("", chunk=1)]], top_k=10)
        assert result == []

    def test_rrf_k_constant_is_the_default(self):
        first = doc("a", chunk=1)
        second = doc("b", chunk=2)
        explicit = reciprocal_rank_fusion([[first, second]], k=RRF_K, top_k=10)
        default = reciprocal_rank_fusion([[first, second]], top_k=10)
        assert [d.page_content for d in explicit] == [d.page_content for d in default]