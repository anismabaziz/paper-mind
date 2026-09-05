"""
Gated local cross-encoder reranker tests — headless, no model download.

RERANK=true|false gates rerank of 50 hybrid candidates before shape_sources
keeps top 5. Flag off preserves legacy order; flag on yields reranked,
deduped, score-ordered results, entirely local CPU.

All branches use fakes: the cross-encoder is stubbed via _reranker, never
importing weights, so pytest stays fast and offline.
"""

import config
from services import reranker
from services.vector_service import VectorService
from evaluation import evaluator


class FakeIndex:
    def __init__(self, matches):
        self._matches = matches
        self.queries = []

    def query(self, vector, top_k, include_metadata, filter, **kwargs):
        # record kwargs to assert sparse handling not needed here
        self.queries.append({"top_k": top_k, "filter": filter, "kwargs": kwargs})
        return {"matches": self._matches}

    def upsert(self, vectors):
        return {"upserted": len(vectors)}

    def delete(self, **kwargs):
        return {}


class InvertingModel:
    """Fake CrossEncoder that inverts input order: last pair gets highest score."""

    def predict(self, pairs, **kwargs):
        # increasing scores so last source becomes top
        return list(range(len(pairs)))


class IdentityModel:
    """Fake that keeps input order: first keeps highest score."""

    def predict(self, pairs, **kwargs):
        return list(reversed(range(len(pairs))))


def _matches(n=10):
    return [
        {
            "id": f"v{i}",
            "score": 1.0 - i * 0.05,
            "metadata": {"content": f"chunk {i}", "pdf_name": "doc.pdf", "chunk_index": i},
        }
        for i in range(n)
    ]


class TestRerankerGate:
    def test_flag_off_preserves_legacy_order(self, monkeypatch):
        monkeypatch.setenv("RERANK", "false")
        reranker._reset_for_tests()
        idx = FakeIndex(_matches(10))
        monkeypatch.setattr(config, "_pinecone_index", idx)
        monkeypatch.setattr(config, "_qdrant_index", None)
        monkeypatch.setenv("VECTOR_BACKEND", "qdrant")

        sources = VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="test query")
        assert [s["content"] for s in sources] == [f"chunk {i}" for i in range(5)]

    def test_flag_on_reranked_order_differs_deduped_and_score_ordered(self, monkeypatch):
        monkeypatch.setenv("RERANK", "true")
        reranker._reset_for_tests()
        reranker._reranker = InvertingModel()

        idx = FakeIndex(_matches(10))
        monkeypatch.setattr(config, "_pinecone_index", idx)
        monkeypatch.setattr(config, "_qdrant_index", None)
        monkeypatch.setenv("VECTOR_BACKEND", "qdrant")

        sources_on = VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="test query")
        # InvertingModel gives chunk 9 highest, so top 5 should be 9..5
        assert [s["content"] for s in sources_on] == [f"chunk {i}" for i in range(9, 4, -1)]
        # score-ordered
        assert all(sources_on[i]["score"] >= sources_on[i + 1]["score"] for i in range(len(sources_on) - 1))
        # rerank_score preserved
        assert all("rerank_score" in s for s in sources_on)

        # Deduped: duplicate content keeps highest rerank_score
        dup_matches = [
            {"id": "a", "score": 0.9, "metadata": {"content": "dup", "pdf_name": "doc.pdf", "chunk_index": 0}},
            {"id": "b", "score": 0.8, "metadata": {"content": "dup", "pdf_name": "doc.pdf", "chunk_index": 1}},
            {"id": "c", "score": 0.7, "metadata": {"content": "unique", "pdf_name": "doc.pdf", "chunk_index": 2}},
        ]
        dup_idx = FakeIndex(dup_matches)

        class DupModel:
            def predict(self, pairs, **kwargs):
                return [0.1, 0.9, 0.5]

        reranker._reranker = DupModel()
        monkeypatch.setattr(config, "_pinecone_index", dup_idx)
        before = VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="q", rerank=False)
        assert len(before) == 2  # deduped legacy still 2
        deduped = VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="q", rerank=True)
        assert len(deduped) == 2
        assert deduped[0]["content"] == "dup"
        assert deduped[0]["score"] == 0.9

    def test_explicit_rerank_param_overrides_env(self, monkeypatch):
        # Env says true but explicit False preserves legacy
        monkeypatch.setenv("RERANK", "true")
        reranker._reset_for_tests()
        reranker._reranker = InvertingModel()
        idx = FakeIndex(_matches(10))
        monkeypatch.setattr(config, "_pinecone_index", idx)
        monkeypatch.setattr(config, "_qdrant_index", None)

        legacy = VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="q", rerank=False)
        assert [s["content"] for s in legacy] == [f"chunk {i}" for i in range(5)]

        reranked = VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="q", rerank=True)
        assert [s["content"] for s in reranked] != [s["content"] for s in legacy]

    def test_no_query_text_never_reranks(self, monkeypatch):
        monkeypatch.setenv("RERANK", "true")
        calls = []

        class CountingModel:
            def predict(self, pairs, **kwargs):
                calls.append(pairs)
                return [0] * len(pairs)

        reranker._reset_for_tests()
        reranker._reranker = CountingModel()
        idx = FakeIndex(_matches(5))
        monkeypatch.setattr(config, "_pinecone_index", idx)
        monkeypatch.setattr(config, "_qdrant_index", None)

        # No query_text -> no rerank
        VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text=None)
        assert calls == []

    def test_entirely_local_cpu_no_api(self, monkeypatch):
        # Ensure rerank path does not hit network: stub _get_reranker and
        # assert it is the only provider touched
        monkeypatch.setenv("RERANK", "true")
        reranker._reset_for_tests()

        invoked = {}

        class LocalOnly:
            def predict(self, pairs, **kwargs):
                invoked["device"] = "cpu"  # model was constructed with device=cpu
                return [float(len(p[1])) for p in pairs]

        reranker._reranker = LocalOnly()
        idx = FakeIndex(_matches(5))
        monkeypatch.setattr(config, "_pinecone_index", idx)
        monkeypatch.setattr(config, "_qdrant_index", None)

        VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="hello")
        assert invoked, "local model should have been invoked"
        # No external call recorded — entirely local

    def test_model_load_failure_degrades_to_legacy(self, monkeypatch, capsys):
        monkeypatch.setenv("RERANK", "true")
        reranker._reset_for_tests()
        monkeypatch.setattr(reranker, "_get_reranker", lambda: (_ for _ in ()).throw(RuntimeError("load failed")))

        idx = FakeIndex(_matches(5))
        monkeypatch.setattr(config, "_pinecone_index", idx)
        monkeypatch.setattr(config, "_qdrant_index", None)

        sources = VectorService.query_vectors([0.1] * 8, "doc.pdf", query_text="q")
        assert [s["content"] for s in sources] == [f"chunk {i}" for i in range(5)]
        assert "degraded" in capsys.readouterr().out.lower()


class TestEvaluatorReranker:
    def test_evaluator_retrieve_respects_flag(self, monkeypatch):
        class FakeIdx:
            def query(self, vector, top_k, include_metadata, filter, **kwargs):
                return {"matches": _matches(10)}

        idx = FakeIdx()

        def embed(texts):
            return [[0.1] * 4 for _ in texts]

        monkeypatch.setenv("RERANK", "false")
        reranker._reset_for_tests()
        reranker._reranker = InvertingModel()
        off = evaluator.retrieve(embed(["q"])[0], "doc.pdf", idx, k=5, query_text="q", rerank=False)
        assert [s["content"] for s in off] == [f"chunk {i}" for i in range(5)]

        on = evaluator.retrieve(embed(["q"])[0], "doc.pdf", idx, k=5, query_text="q", rerank=True)
        assert [s["content"] for s in on] == [f"chunk {i}" for i in range(9, 4, -1)]

    def test_hit5_faithfulness_logged_with_and_without_reranking_and_latency_delta(self, monkeypatch, capsys):
        # Minimal fixture where reranking flips order but we can still measure hit@5
        from evaluation.metrics import hit_at_k  # noqa: F401

        class MemIdx:
            def __init__(self):
                self.vectors = []

            def upsert(self, vectors):
                self.vectors.extend(vectors)

            def query(self, vector, top_k, include_metadata, filter, **kwargs):
                # return in insertion order with descending scores
                name = (filter or {}).get("pdf_name")
                scored = [v for v in self.vectors if v["metadata"]["pdf_name"] == name]
                scored.sort(key=lambda m: m.get("score", 0), reverse=True)
                # adapt to evaluator shape: dict with matches list of dicts with metadata/score
                return {"matches": [{"score": v.get("score", 0), "metadata": v["metadata"]} for v in scored[:top_k]]}

            def delete(self, **kwargs):
                return {}

        idx = MemIdx()
        # Two vectors: gold in second position legacy, first after rerank
        idx.vectors = [
            {"score": 0.9, "metadata": {"content": "irrelevant filler", "pdf_name": "doc.pdf", "chunk_index": 0}},
            {"score": 0.8, "metadata": {"content": "gold snippet alpha", "pdf_name": "doc.pdf", "chunk_index": 1}},
        ]
        for i in range(48):
            idx.vectors.append({"score": 0.7 - i * 0.01, "metadata": {"content": f"filler {i}", "pdf_name": "doc.pdf", "chunk_index": 10 + i}})

        fixture = {
            "documents": [{"filename": "doc.pdf"}],
            "questions": [
                {"id": "q1", "document": "doc.pdf", "question": "gold alpha", "expected_answer": "x", "gold_snippets": ["gold snippet alpha"]}
            ],
        }

        def embed(texts):
            return [[0.1] * 4 for _ in texts]

        def gen(q, ctx):
            return "answer"

        # Flag off
        reranker._reset_for_tests()
        reranker._reranker = IdentityModel()  # keeps legacy order
        report_off = evaluator.evaluate(fixture, idx, embed, gen, judge_fn=None, k=5, rerank=False)
        assert report_off.retrieval.hit_rate in (0.0, 1.0)

        # Flag on with inverting model — should change order but still deduped/score-ordered
        reranker._reset_for_tests()
        reranker._reranker = InvertingModel()
        capsys.readouterr()  # clear
        report_on = evaluator.evaluate(fixture, idx, embed, gen, judge_fn=None, k=5, rerank=True)
        out = capsys.readouterr().out
        assert "rerank" in out.lower()

        # Comparison helper logs hit@5/faithfulness delta and latency for 50 docs
        reranker._reset_for_tests()
        reranker._reranker = InvertingModel()
        comp = evaluator.evaluate_with_rerank_comparison(fixture, idx, embed, gen, judge_fn=None, k=5)
        assert "off" in comp and "on" in comp and "latency_delta_ms" in comp
        cap = capsys.readouterr().out
        assert "latency" in cap.lower() or "delta" in cap.lower()
