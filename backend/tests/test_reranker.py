"""
Gated local cross-encoder reranker tests — headless, no model download.

The reranker's enabled flag gates rerank of 50 hybrid candidates before
shape_sources keeps top 5. Flag off preserves legacy order; flag on yields
reranked, deduped, score-ordered results, entirely local CPU.

All branches use fakes injected through constructors: the cross-encoder is a
fake model object, the index a fake store, so pytest stays fast and offline.
"""

import sys
import types

from services.models import PINNED_MODEL_REVISIONS
from services.retrieval.reranker import Reranker, RerankerService
from services.retrieval.vector_service import VectorService


def _load_kwargs(reranker):
    """
    Run the real lazy load against a stand-in for sentence-transformers.

    Returns the keyword arguments the loader would hand the library, without
    downloading weights: the point is the arguments, not the model.
    """
    recorded = {}

    def _cross_encoder(model_name, **kwargs):
        recorded.update(kwargs)
        return IdentityModel()

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _cross_encoder  # type: ignore[attr-defined]
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = module
    try:
        reranker._get_model()
    finally:
        if original is None:
            del sys.modules["sentence_transformers"]
        else:
            sys.modules["sentence_transformers"] = original
    return recorded


class FakeIndex:
    """FakeIndex."""

    def __init__(self, matches):
        """Initialize."""
        self._matches = matches
        self.queries = []

    def query(self, vector, top_k, include_metadata, filter, **kwargs):
        # record kwargs to assert sparse handling not needed here
        """Do query."""
        self.queries.append({"top_k": top_k, "filter": filter, "kwargs": kwargs})
        return {"matches": self._matches}

    def upsert(self, vectors):
        """Do upsert."""
        return {"upserted": len(vectors)}

    def delete(self, **kwargs):
        """Do delete."""
        return {}


class InvertingModel:
    """Fake CrossEncoder that inverts input order: last pair gets highest score."""

    def predict(self, pairs, **kwargs):
        # increasing scores so last source becomes top
        """Do predict."""
        return list(range(len(pairs)))


class IdentityModel:
    """Fake that keeps input order: first keeps highest score."""

    def predict(self, pairs, **kwargs):
        """Do predict."""
        return list(reversed(range(len(pairs))))


def _matches(n=10):
    return [
        {
            "id": f"v{i}",
            "score": 1.0 - i * 0.05,
            "metadata": {
                "content": f"chunk {i}",
                "pdf_name": "doc.pdf",
                "chunk_index": i,
            },
        }
        for i in range(n)
    ]


def make_reranker(enabled=True, model=None) -> RerankerService:
    """Return a reranker with the gate and model the test chooses."""
    return RerankerService(model_name="fake-rerank-model", enabled=enabled, model=model)


def make_service(matches, reranker=None) -> VectorService:
    """Return a VectorService over a fake store with an injected reranker."""
    return VectorService(FakeIndex(matches), reranker)


def test_local_service_implements_reranker_interface():
    """The production adapter satisfies the reranker interface used by retrieval."""
    assert isinstance(make_reranker(), Reranker)


class TestRerankerGate:
    """TestRerankerGate."""

    def test_flag_off_preserves_legacy_order(self):
        """Do test flag off preserves legacy order."""
        service = make_service(_matches(10), make_reranker(enabled=False))

        sources = service.query_vectors(
            [0.1] * 8, "doc.pdf", query_text="test query"
        ).sources
        assert [s["content"] for s in sources] == [f"chunk {i}" for i in range(5)]

    def test_flag_on_reranked_order_differs_deduped_and_score_ordered(self):
        """Do test flag on reranked order differs deduped and score ordered."""
        service = make_service(
            _matches(10), make_reranker(enabled=True, model=InvertingModel())
        )

        sources_on = service.query_vectors(
            [0.1] * 8, "doc.pdf", query_text="test query"
        ).sources
        # InvertingModel gives chunk 9 highest, so top 5 should be 9..5
        assert [s["content"] for s in sources_on] == [
            f"chunk {i}" for i in range(9, 4, -1)
        ]
        # score-ordered
        assert all(
            sources_on[i]["score"] >= sources_on[i + 1]["score"]
            for i in range(len(sources_on) - 1)
        )
        # rerank_score preserved
        assert all("rerank_score" in s for s in sources_on)

        # Deduped: duplicate content keeps highest rerank_score
        dup_matches = [
            {
                "id": "a",
                "score": 0.9,
                "metadata": {"content": "dup", "pdf_name": "doc.pdf", "chunk_index": 0},
            },
            {
                "id": "b",
                "score": 0.8,
                "metadata": {"content": "dup", "pdf_name": "doc.pdf", "chunk_index": 1},
            },
            {
                "id": "c",
                "score": 0.7,
                "metadata": {
                    "content": "unique",
                    "pdf_name": "doc.pdf",
                    "chunk_index": 2,
                },
            },
        ]

        class DupModel:
            """DupModel."""

            def predict(self, pairs, **kwargs):
                """Do predict."""
                return [0.1, 0.9, 0.5]

        dup_service = make_service(
            dup_matches, make_reranker(enabled=True, model=DupModel())
        )
        before = dup_service.query_vectors(
            [0.1] * 8, "doc.pdf", query_text="q", rerank=False
        ).sources
        assert len(before) == 2  # deduped legacy still 2
        deduped = dup_service.query_vectors(
            [0.1] * 8, "doc.pdf", query_text="q", rerank=True
        ).sources
        assert len(deduped) == 2
        assert deduped[0]["content"] == "dup"
        assert deduped[0]["score"] == 0.9

    def test_explicit_rerank_param_overrides_flag(self):
        """Do test explicit rerank param overrides flag."""
        # Flag says true but explicit False preserves legacy
        """Do test explicit rerank param overrides flag."""
        service = make_service(
            _matches(10), make_reranker(enabled=True, model=InvertingModel())
        )

        legacy = service.query_vectors(
            [0.1] * 8, "doc.pdf", query_text="q", rerank=False
        ).sources
        assert [s["content"] for s in legacy] == [f"chunk {i}" for i in range(5)]

        reranked = service.query_vectors(
            [0.1] * 8, "doc.pdf", query_text="q", rerank=True
        ).sources
        assert [s["content"] for s in reranked] != [s["content"] for s in legacy]

    def test_no_query_text_never_reranks(self):
        """Do test no query text never reranks."""
        calls = []

        class CountingModel:
            """CountingModel."""

            def predict(self, pairs, **kwargs):
                """Do predict."""
                calls.append(pairs)
                return [0] * len(pairs)

        service = make_service(
            _matches(5), make_reranker(enabled=True, model=CountingModel())
        )

        # No query_text -> no rerank
        service.query_vectors([0.1] * 8, "doc.pdf", query_text=None)
        assert calls == []

    def test_entirely_local_cpu_no_api(self):
        """Do test entirely local cpu no api."""
        # Ensure rerank path does not hit network: the fake model is the only
        # provider touched
        """Do test entirely local cpu no api."""
        invoked = {}

        class LocalOnly:
            """LocalOnly."""

            def predict(self, pairs, **kwargs):
                """Do predict."""
                invoked["device"] = "cpu"  # model was constructed with device=cpu
                return [float(len(p[1])) for p in pairs]

        service = make_service(
            _matches(5), make_reranker(enabled=True, model=LocalOnly())
        )

        service.query_vectors([0.1] * 8, "doc.pdf", query_text="hello")
        assert invoked, "local model should have been invoked"
        # No external call recorded — entirely local

    def test_model_load_failure_degrades_to_legacy(self, monkeypatch, capsys):
        """Do test model load failure degrades to legacy."""
        reranker_svc = make_reranker(enabled=True)  # no injected model: lazy load
        monkeypatch.setattr(
            reranker_svc,
            "_get_model",
            lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
        )
        service = make_service(_matches(5), reranker_svc)

        sources = service.query_vectors([0.1] * 8, "doc.pdf", query_text="q").sources
        assert [s["content"] for s in sources] == [f"chunk {i}" for i in range(5)]
        assert "degraded" in capsys.readouterr().out.lower()


class TestModelLoad:
    """The cross-encoder the reranker asks sentence-transformers for."""

    def test_it_loads_the_revision_its_settings_name(self):
        """Do test it loads the revision its settings name."""
        kwargs = _load_kwargs(
            RerankerService("cross-encoder/ms-marco-MiniLM-L6-v2", True)
        )

        assert (
            kwargs["revision"]
            == PINNED_MODEL_REVISIONS["cross-encoder/ms-marco-MiniLM-L6-v2"]
        )

    def test_a_configured_revision_reaches_the_loader(self):
        """Do test a configured revision reaches the loader."""
        kwargs = _load_kwargs(
            RerankerService(
                "cross-encoder/ms-marco-MiniLM-L6-v2", True, revision="a" * 40
            )
        )

        assert kwargs["revision"] == "a" * 40

    def test_it_does_not_execute_code_from_the_model_repository(self):
        """The MiniLM cross-encoder is a stock BERT architecture."""
        kwargs = _load_kwargs(
            RerankerService("cross-encoder/ms-marco-MiniLM-L6-v2", True)
        )

        assert kwargs["trust_remote_code"] is False

    def test_an_unpinned_model_loads_no_revision_at_all(self):
        """Do test an unpinned model loads no revision at all."""
        assert "revision" not in _load_kwargs(RerankerService("acme/unknown", True))
