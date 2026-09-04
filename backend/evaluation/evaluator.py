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
            path.
    """
    raw = read_document(filename, docs_dir)
    chunks, page_numbers = DocumentParser.get_chunks(filename, raw)
    embeddings = embed_fn(chunks)
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": embeddings[i],
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


def retrieve(query_embedding, filename, index, k=DEFAULT_K, prefix=""):
    """
        Fetch candidates and shape them exactly like the /response route.
    """
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
    return shape_sources(matches_to_sources(matches, filename))[:k]


def evaluate(
    fixture: dict,
    index,
    embed_fn,
    generate_fn,
    judge_fn=None,
    k: int = DEFAULT_K,
    docs_dir=SAMPLE_DOCS_DIR,
    prefix="",
) -> EvaluationReport:
    """
        Run every fixture question through retrieval and generation.

            ``judge_fn`` may be None to skip faithfulness scoring (retrieval-only
            runs and tests that focus on the metrics).
    """
    question_results = []
    per_question = []
    faithfulness_scores = []

    for item in fixture["questions"]:
        filename = item["document"]
        query_embedding = embed_fn([item["question"]])[0]
        sources = retrieve(query_embedding, filename, index, k=k, prefix=prefix)
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
    return report
