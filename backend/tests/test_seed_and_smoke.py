"""Seed and smoke helpers stay keyless and deterministic."""

import importlib.util
import pathlib

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str):
    """Load a script module by filename without importing app code."""
    path = BACKEND_DIR / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_docs_are_licensed_and_bounded():
    """The seed set is small and every file declares its license."""
    seed = _load("seed_sample_docs.py")
    docs = seed.choose_seed_docs()
    assert 1 <= len(docs) <= 2, "seed stays bounded at one or two documents"
    for doc in docs:
        assert doc.exists(), f"seed document missing: {doc}"
        assert doc.suffix == ".pdf"
    licenses = seed.SEED_LICENSES
    for doc in docs:
        assert doc.name in licenses, f"{doc.name} must declare its license"
        assert licenses[doc.name] in ("CC0-1.0", "CC-BY-4.0")


def test_seed_skips_documents_already_present():
    """Seeding matches on original filename so reruns skip work."""
    seed = _load("seed_sample_docs.py")
    files = [
        {"original_filename": "papermind-rag-primer.pdf"},
        {"original_filename": "other.pdf"},
    ]
    assert seed.find_existing(files, "papermind-rag-primer.pdf") is not None
    assert seed.find_existing(files, "papermind-team-notes.pdf") is None


def test_seed_proof_query_needs_no_model_key():
    """The proof question abstains before generation, so no key is needed."""
    seed = _load("seed_sample_docs.py")
    query = seed.build_abstention_query()
    assert isinstance(query, str) and len(query.strip()) > 0
    # A nonsense query abstains before generation, so no provider key is needed.
    assert "xqzzy" in query or "nonexistent" in query


def test_smoke_accepts_cited_answer_or_deterministic_abstention():
    """Smoke passes on citations or abstention, never on an error."""
    smoke = _load("smoke_setup.py")
    done = {
        "done": True,
        "sources": [{"content": "A RAG pipeline has five stages", "page": 1}],
    }
    abstained = {"abstained": True, "reason": "no_evidence"}
    provider_error = {"error": "provider down"}
    assert smoke.classify_proof(done) == "cited-answer"
    assert smoke.classify_proof(abstained) == "abstention"
    assert smoke.classify_proof(provider_error) == "failure"


def test_smoke_terminal_parsing_reads_sse():
    """The smoke check reads the stream's terminal event."""
    smoke = _load("smoke_setup.py")
    sse = (
        'event: start\ndata: {"turn_id": "1"}\n\n'
        'event: abstained\ndata: {"abstained": true, "reason": "no_evidence"}\n\n'
    )
    terminal = smoke.parse_sse_terminal(sse)
    assert terminal[0] == "abstained"
    assert terminal[1]["reason"] == "no_evidence"


def test_smoke_recognizes_the_keyless_refusal():
    """A workspace with no saved key fails closed with a known category."""
    smoke = _load("smoke_setup.py")
    assert (
        smoke.is_keyless_refusal(
            400, '{"error": "x", "category": "no_provider_configured"}'
        )
        is True
    )
    assert smoke.is_keyless_refusal(400, '{"error": "x"}') is False
    assert smoke.is_keyless_refusal(500, "") is False
