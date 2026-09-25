"""
End-to-end evaluation of the RAG pipeline against the ground-truth fixture.

Every external capability is injected:

- ``embed_fn``: texts -> list of embedding vectors
- ``index``: Qdrant-compatible store with ``upsert``, ``query``, ``delete``
- ``generate_fn``: (query, context) -> answer text
- ``judge_fn``: judge prompt -> verdict reply (see evaluation.judge)

Tests wire deterministic fakes into all four; the CLI wires the real
providers, and only behind ``--live``. ``uv run pytest`` stays headless
(no Qdrant/LLM; heavy models mocked or skipped).

Gates per phase (recorded on ``sample_docs`` via this module):
``hit@5``/``recall@5`` + per-question breakdown and ingest ``sec/PDF``
(see :func:`index_document_timed`; ``POST /process-file`` also logs
parse/embed/upsert wall time). Free local path uses Qdrant on
``http://localhost:6333`` with no API keys (retrieval-only ``--no-judge``).
"""

import json
from dataclasses import asdict, dataclass, field
from typing import cast
from pathlib import Path

from services.chat_context import render_prior_turns
from settings import get_settings

from evaluation import judge as judge_module
from evaluation.metrics import RetrievalReport, hit_at_k, recall_at_k, summarize
from services.parsing.document_parser import DocumentIngestor
from services.retrieval.base import (
    RetrievalMethod,
    RetrievalResult,
    VectorStoreConfigurationError,
)
from services.retrieval.hybrid import DEFAULT_FETCH_K, build_sparse_vector
from services.retrieval.query_expansion import expand_query
from services.retrieval.reranker import RerankerService
from services.retrieval.vector_service import (
    build_vectors_from_chunks,
    matches_to_sources,
    shape_sources,
)


def _is_rerank_enabled() -> bool:
    from settings import get_settings

    return get_settings().rerank.enabled


def _ingestor() -> DocumentIngestor:
    """Build the parse-and-chunk pipeline from the installed Settings."""
    from settings import get_settings

    s = get_settings()
    return DocumentIngestor(
        s.parsing.use_docling,
        s.chunking.chunk_size_tokens,
        s.chunking.chunk_overlap_tokens,
    )


def _reranker_from_settings() -> RerankerService:
    """Build the gated reranker from the installed Settings."""
    from settings import get_settings

    s = get_settings()
    return RerankerService(s.rerank.rerank_model, enabled=s.rerank.enabled)


BACKEND_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = Path(__file__).parent / "fixture.json"
SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

FETCH_K = DEFAULT_FETCH_K
DEFAULT_K = 5


@dataclass
class EvaluationReport:
    """EvaluationReport."""

    retrieval: RetrievalReport
    faithfulness: dict
    per_question: list = field(default_factory=list)

    def as_dict(self):
        """Do as dict."""
        return {
            "retrieval": asdict(self.retrieval),
            "faithfulness": self.faithfulness,
            "per_question": self.per_question,
        }


def load_fixture(path=FIXTURE_PATH) -> dict:
    """Do load fixture."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_document(filename: str, docs_dir=SAMPLE_DOCS_DIR) -> bytes:
    """Do read document."""
    return (Path(docs_dir) / filename).read_bytes()


def index_document(
    filename: str, index, embed_fn, docs_dir=SAMPLE_DOCS_DIR, pdf_name=None
):
    """
    Parse, chunk, embed, and upsert one sample document.

        ``pdf_name`` names the stored vectors (defaults to ``filename``), so a
        live run can namespace them under an eval- prefix without changing
        which file is read. Uses the same parser and vector shape as the
        /process-file route, so the evaluator exercises the real ingestion
        path. Hashed term-frequency vectors are stored alongside dense vectors
        for hybrid retrieval.
        Returns chunk count; timing is logged via :func:`index_document_timed`
        for gate reports (sec/PDF).
    """
    raw = read_document(filename, docs_dir)
    chunks = _ingestor().get_chunk_objects(filename, raw)
    embeddings = embed_fn([chunk.text for chunk in chunks])
    vectors = build_vectors_from_chunks(embeddings, chunks, pdf_name or filename)
    index.upsert(vectors)
    return len(chunks)


def index_document_timed(
    filename: str, index, embed_fn, docs_dir=SAMPLE_DOCS_DIR, pdf_name=None
):
    """
    Like :func:`index_document` but returns ``(chunks, elapsed_seconds)``.

    for gate reports (ingest sec/PDF). Phase timing mirrors
    ``POST /process-file`` parse/embed/upsert logging.
    """
    import time as _time

    t0 = _time.time()
    chunks = index_document(
        filename, index, embed_fn, docs_dir=docs_dir, pdf_name=pdf_name
    )
    elapsed = _time.time() - t0
    return chunks, elapsed


def remove_document(filename: str, index):
    """Do remove document."""
    index.delete(filter={"pdf_name": filename})


def retrieve(
    query_embedding,
    filename,
    index,
    k=DEFAULT_K,
    prefix="",
    query_text=None,
    rerank=None,
    reranker=None,
):
    """Fetch and shape candidates without changing the selected method."""
    sparse = None
    method: RetrievalMethod = "dense"
    if query_text is not None:
        sparse = build_sparse_vector(query_text)
        if sparse["indices"]:
            method = "hybrid"
        else:
            sparse = None
    if method == "hybrid" and sparse is None:
        raise VectorStoreConfigurationError(
            "Hybrid retrieval requires a non-empty sparse query vector"
        )
    query_args = {
        "vector": query_embedding,
        "top_k": FETCH_K,
        "include_metadata": True,
        "filter": {"pdf_name": f"{prefix}{filename}"},
        "method": method,
    }
    if sparse is not None:
        query_args["sparse_vector"] = sparse
    results = index.query(**query_args)
    matches = (
        results.get("matches", [])
        if isinstance(results, dict)
        else getattr(results, "matches", [])
    )
    actual_method: RetrievalMethod = method
    if isinstance(results, dict) and "method" in results:
        actual_method = cast(RetrievalMethod, results["method"])
    if actual_method not in {"dense", "sparse", "hybrid"}:
        raise VectorStoreConfigurationError(
            f"Vector store reported an invalid retrieval method: {actual_method}"
        )
    sources = matches_to_sources(matches, filename)
    rerank_service = reranker if reranker is not None else _reranker_from_settings()
    sources = rerank_service.maybe_rerank(query_text, sources, enabled=rerank)
    shaped = shape_sources(sources)[:k]
    return RetrievalResult(
        sources=shaped,
        method=actual_method,
        outcome="success" if shaped else "empty",
    )


def evaluate(
    fixture: dict,
    index,
    embed_fn,
    generate_fn,
    judge_fn=None,
    k: int = DEFAULT_K,
    docs_dir=SAMPLE_DOCS_DIR,
    prefix="",
    rerank=None,
    reranker=None,
) -> EvaluationReport:
    """
    Run every fixture question through retrieval and generation.

        ``judge_fn`` may be None to skip faithfulness scoring (retrieval-only
        runs and tests that focus on the metrics).
        ``rerank`` overrides the reranker's enabled flag per-run (None = flag).
        ``reranker`` injects a pre-built reranker; without one, a
        settings-built service is used for the whole run.
    """
    import time

    rerank_service = reranker if reranker is not None else _reranker_from_settings()
    question_results = []
    per_question = []
    faithfulness_scores = []
    rerank_latencies: list[float] = []

    for item in fixture["questions"]:
        filename = item["document"]
        # A follow-up case carries the earlier user questions it refers back
        # to, so evaluation expands the query exactly as the application does.
        prior_turns = [
            {"question": question, "answer": None}
            for question in item.get("follow_up", [])
        ]
        expansion = expand_query(
            item["question"],
            item.get("follow_up", []),
            max_chars=get_settings().query_context.max_expansion_chars,
        )
        query_embedding = embed_fn([expansion.expanded_query])[0]
        t0 = time.time()
        retrieval_result = retrieve(
            query_embedding,
            filename,
            index,
            k=k,
            prefix=prefix,
            query_text=expansion.expanded_query,
            rerank=rerank,
            reranker=rerank_service,
        )
        sources = retrieval_result.sources
        # Record latency delta proxy: rerank timing is printed inside reranker,
        # but we also capture per-query retrieval time for the report if rerank on
        if rerank is True or (rerank is None and _is_rerank_enabled()):
            rerank_latencies.append(time.time() - t0)
        retrieved_texts = [s["content"] for s in sources]

        result = {
            "id": item["id"],
            "hit_at_k": hit_at_k(retrieved_texts, item["gold_snippets"], k),
            "recall_at_k": recall_at_k(retrieved_texts, item["gold_snippets"], k),
        }

        detail = {
            "id": item["id"],
            **result,
            "retrieved_chunks": len(retrieved_texts),
            "retrieval_method": retrieval_result.method,
            **expansion.to_dict(),
        }

        if judge_fn is not None:
            context = "\n\n".join(retrieved_texts)
            # A follow-up is answered with the same transcript the application
            # would send, so the reported faithfulness describes the real shape
            # of the request rather than a contextless question.
            answer = generate_fn(
                item["question"], context, render_prior_turns(prior_turns)
            )
            verdict, score = judge_module.judge_faithfulness(
                item["question"], answer, context, judge_fn
            )
            faithfulness_scores.append(score)
            detail.update({"verdict": verdict, "faithfulness": score, "answer": answer})

        question_results.append(result)
        per_question.append(detail)

    report = EvaluationReport(
        retrieval=summarize(question_results, k),
        faithfulness={
            "mean": (
                sum(faithfulness_scores) / len(faithfulness_scores)
                if faithfulness_scores
                else None
            ),
            "judged": len(faithfulness_scores),
            "faithful": sum(1 for s in faithfulness_scores if s == 1.0),
        },
        per_question=per_question,
    )
    # Log latency delta for 50 docs when reranking was active (cheap observability)
    if rerank_latencies:
        avg_ms = sum(rerank_latencies) / len(rerank_latencies) * 1000
        print(
            f"Evaluator rerank: avg retrieval {avg_ms:.1f}ms/query over {len(rerank_latencies)} queries "
            f"(rerank={'on' if (rerank is True or (rerank is None and _is_rerank_enabled())) else 'off'}, FETCH_K={FETCH_K})"
        )
    return report


def evaluate_with_rerank_comparison(
    fixture: dict,
    index,
    embed_fn,
    generate_fn,
    judge_fn=None,
    k: int = DEFAULT_K,
    docs_dir=SAMPLE_DOCS_DIR,
    prefix="",
    reranker=None,
) -> dict:
    """
    Run the fixture twice — without and with reranking — and log hit@k /.

    faithfulness deltas plus latency for 50 candidates. Returns a dict with
    both reports for the caller to inspect. Used by the live CLI and docs
    to demonstrate the gated reranker gate.
    """
    import time

    t0 = time.time()
    report_off = evaluate(
        fixture,
        index,
        embed_fn,
        generate_fn,
        judge_fn,
        k=k,
        docs_dir=docs_dir,
        prefix=prefix,
        rerank=False,
        reranker=reranker,
    )
    off_ms = (time.time() - t0) / max(len(fixture.get("questions", [])), 1) * 1000

    t1 = time.time()
    report_on = evaluate(
        fixture,
        index,
        embed_fn,
        generate_fn,
        judge_fn,
        k=k,
        docs_dir=docs_dir,
        prefix=prefix,
        rerank=True,
        reranker=reranker,
    )
    on_ms = (time.time() - t1) / max(len(fixture.get("questions", [])), 1) * 1000

    delta_ms = on_ms - off_ms
    print(
        f"Rerank comparison: hit@{k} {report_off.retrieval.hit_rate:.2f} -> {report_on.retrieval.hit_rate:.2f} "
        f"(delta {report_on.retrieval.hit_rate - report_off.retrieval.hit_rate:+.2f}), "
        f"faithfulness {report_off.faithfulness.get('mean')} -> {report_on.faithfulness.get('mean')}, "
        f"latency {off_ms:.1f}ms -> {on_ms:.1f}ms (delta {delta_ms:+.1f}ms for 50 candidates, reranked to 5)"
    )
    return {
        "off": report_off.as_dict(),
        "on": report_on.as_dict(),
        "latency_delta_ms": delta_ms,
    }
