"""
    End-to-end evaluation of the RAG pipeline against the ground-truth fixture.

    Every external capability is injected:

    - ``embed_fn``: texts -> list of embedding vectors
    - ``index``: Pinecone-compatible store with ``upsert``, ``query``, ``delete``
    - ``generate_fn``: (query, context) -> answer text
    - ``judge_fn``: judge prompt -> verdict reply (see evaluation.judge)

    Tests wire deterministic fakes into all four; the CLI wires the real
    providers, and only behind ``--live``.
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evaluation import judge as judge_module
from evaluation.metrics import RetrievalReport, hit_at_k, recall_at_k, summarize
from services.document_parser import DocumentParser
from services.vector_service import matches_to_sources, shape_sources


def _is_rerank_enabled() -> bool:
    try:
        import config as _cfg

        return _cfg.is_rerank_enabled()
    except Exception:
        import os

        return os.getenv("RERANK", "false").lower() in ("1", "true", "yes")

BACKEND_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = Path(__file__).parent / "fixture.json"
SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

# Retrieval is fetched generously and scored at k; the index top_k also
# goes through source shaping, which needs headroom to dedupe.
FETCH_K = 10
DEFAULT_K = 5


@dataclass
class EvaluationReport:
    retrieval: RetrievalReport
    faithfulness: dict
    per_question: list = field(default_factory=list)

    def as_dict(self):
        return {
            "retrieval": asdict(self.retrieval),
            "faithfulness": self.faithfulness,
            "per_question": self.per_question,
        }


def load_fixture(path=FIXTURE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_document(filename: str, docs_dir=SAMPLE_DOCS_DIR) -> bytes:
    return (Path(docs_dir) / filename).read_bytes()


def index_document(
    filename: str, index, embed_fn, docs_dir=SAMPLE_DOCS_DIR, pdf_name=None
):
    """
        Parse, chunk, embed, and upsert one sample document.

            ``pdf_name`` names the stored vectors (defaults to ``filename``), so a
            live run can namespace them under an eval- prefix without changing
            which file is read. Uses the same parser seam and vector shape as the
            /process-file route, so the evaluator exercises the real ingestion
            path. Sparse BM25 vectors are stored alongside dense for hybrid retrieval.
    """
    raw = read_document(filename, docs_dir)
    chunks, page_numbers = DocumentParser.get_chunks(filename, raw)
    embeddings = embed_fn(chunks)
    # Build BM25 sparse vectors (hash-based TF) for hybrid indexing
    try:
        from services.hybrid import build_sparse_vectors
    except Exception:
        build_sparse_vectors = None  # type: ignore
    if build_sparse_vectors is not None:
        sparse_vectors = build_sparse_vectors(chunks)
    else:
        sparse_vectors = [{"indices": [], "values": []} for _ in chunks]
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": embeddings[i],
            "sparse_values": sparse_vectors[i],
            "sparse_vector": sparse_vectors[i],
            "metadata": {
                "content": chunks[i],
                "pdf_name": pdf_name or filename,
                "chunk_index": i,
                "page_no": page_numbers[i],
                "content_hash": hashlib.sha256(chunks[i].encode("utf-8")).hexdigest(),
            },
        }
        for i in range(len(chunks))
    ]
    index.upsert(vectors)
    return len(chunks)


def remove_document(filename: str, index):
    index.delete(filter={"pdf_name": filename})


def retrieve(query_embedding, filename, index, k=DEFAULT_K, prefix="", query_text=None, rerank=None):
    """
        Fetch candidates and shape them exactly like the /response route.

        When ``query_text`` is provided a hybrid dense+BM25 sparse query is
        issued (single Qdrant hybrid via RRF), mirroring ``VectorService``.
        When ``RERANK=true`` (or ``rerank=True``) the FETCH_K candidates are
        reranked with a local cross-encoder before shaping to ``k`` (top 5).
        Entirely local CPU, no API.
    """
    sparse = None
    if query_text is not None:
        try:
            from services.hybrid import build_sparse_vector

            sparse = build_sparse_vector(query_text)
            if not sparse["indices"]:
                sparse = None
        except Exception:
            sparse = None

    if sparse is not None:
        try:
            results = index.query(
                vector=query_embedding,
                top_k=FETCH_K,
                include_metadata=True,
                filter={"pdf_name": f"{prefix}{filename}"},
                sparse_vector=sparse,
                sparse_values=sparse,
            )
        except TypeError:
            results = index.query(
                vector=query_embedding,
                top_k=FETCH_K,
                include_metadata=True,
                filter={"pdf_name": f"{prefix}{filename}"},
            )
    else:
        results = index.query(
            vector=query_embedding,
            top_k=FETCH_K,
            include_metadata=True,
            filter={"pdf_name": f"{prefix}{filename}"},
        )
    matches = (
        results.get("matches", [])
        if isinstance(results, dict)
        else getattr(results, "matches", [])
    )
    sources = matches_to_sources(matches, filename)

    # Gated local reranker mirroring VectorService — single shared gate
    from services.reranker import maybe_rerank

    sources = maybe_rerank(query_text, sources, enabled=rerank)

    return shape_sources(sources)[:k]


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
) -> EvaluationReport:
    """
        Run every fixture question through retrieval and generation.

            ``judge_fn`` may be None to skip faithfulness scoring (retrieval-only
            runs and tests that focus on the metrics).
            ``rerank`` overrides the ``RERANK`` env flag per-run (None = env).
    """
    import time

    question_results = []
    per_question = []
    faithfulness_scores = []
    rerank_latencies: list[float] = []

    for item in fixture["questions"]:
        filename = item["document"]
        query_embedding = embed_fn([item["question"]])[0]
        t0 = time.time()
        sources = retrieve(query_embedding, filename, index, k=k, prefix=prefix, query_text=item["question"], rerank=rerank)
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

        detail = {"id": item["id"], **result, "retrieved_chunks": len(retrieved_texts)}

        if judge_fn is not None:
            context = "\n\n".join(retrieved_texts)
            answer = generate_fn(item["question"], context)
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
            "faithful": sum(
                1 for s in faithfulness_scores if s == 1.0
            ),
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
) -> dict:
    """
    Run the fixture twice — without and with reranking — and log hit@k /
    faithfulness deltas plus latency for 50 candidates. Returns a dict with
    both reports for the caller to inspect. Used by the live CLI and docs
    to demonstrate the gated reranker gate.
    """
    import time

    t0 = time.time()
    report_off = evaluate(fixture, index, embed_fn, generate_fn, judge_fn, k=k, docs_dir=docs_dir, prefix=prefix, rerank=False)
    off_ms = (time.time() - t0) / max(len(fixture.get("questions", [])), 1) * 1000

    t1 = time.time()
    report_on = evaluate(fixture, index, embed_fn, generate_fn, judge_fn, k=k, docs_dir=docs_dir, prefix=prefix, rerank=True)
    on_ms = (time.time() - t1) / max(len(fixture.get("questions", [])), 1) * 1000

    delta_ms = on_ms - off_ms
    print(
        f"Rerank comparison: hit@{k} {report_off.retrieval.hit_rate:.2f} -> {report_on.retrieval.hit_rate:.2f} "
        f"(delta {report_on.retrieval.hit_rate - report_off.retrieval.hit_rate:+.2f}), "
        f"faithfulness {report_off.faithfulness.get('mean')} -> {report_on.faithfulness.get('mean')}, "
        f"latency {off_ms:.1f}ms -> {on_ms:.1f}ms (delta {delta_ms:+.1f}ms for 50 candidates, reranked to 5)"
    )
    return {"off": report_off.as_dict(), "on": report_on.as_dict(), "latency_delta_ms": delta_ms}
