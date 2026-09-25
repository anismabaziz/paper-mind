"""
Tests for the evaluation package.

Everything runs offline against deterministic fakes: a hash-based
embedder, an in-memory cosine index, a canned generator, and a canned
judge. The only real component exercised is the document parser, which is
also real in production ingestion.

Metric values are asserted on a synthetic fixture whose vectors make
retrieval fully controllable; the real fixture run is a plumbing check,
because a lexical fake embedder says nothing about real embedding quality.
"""

import hashlib
import json
import math
import re

import pytest

from evaluation import cli, evaluator, judge
from evaluation.build_primer_pdf import SOURCE as PRIMER_SOURCE
from evaluation.metrics import hit_at_k, recall_at_k, summarize
from services.parsing.document_parser import resolve_parser
from services.retrieval.base import VectorStoreConfigurationError
from services.retrieval.hybrid import RRF_K, build_sparse_vector

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "where",
    "which",
    "with",
}
HASH_DIMS = 512


def _tokens(text):
    return [
        token
        for token in re.findall(r"[a-z0-9%°]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def hash_embed(texts):
    """
    Deterministic TF embeddings over content words (stable hashing).

        Good enough to exercise retrieval plumbing; deliberately not a
        semantic model, so tests never assert retrieval perfection from it.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * HASH_DIMS
        for token in _tokens(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % HASH_DIMS] += 1.0
        vectors.append([1 + math.log(v) if v else 0.0 for v in vec])
    return vectors


class InMemoryIndex:
    """Qdrant-shaped store: upsert, filtered cosine query, delete."""

    def __init__(self):
        """Initialize."""
        self.vectors = []

    def upsert(self, vectors):
        """Do upsert."""
        self.vectors.extend(vectors)

    def query(
        self,
        vector,
        top_k,
        include_metadata=False,
        filter=None,
        method="dense",
        sparse_vector=None,
        **kwargs,
    ):
        """Run dense, sparse, or hybrid retrieval without hidden fallback."""
        if method not in {"dense", "sparse", "hybrid"}:
            raise VectorStoreConfigurationError(
                f"Unsupported retrieval method: {method}"
            )
        name = (filter or {}).get("pdf_name")
        candidates = [
            candidate
            for candidate in self.vectors
            if candidate["metadata"]["pdf_name"] == name
        ]
        dense = [
            {
                "id": candidate.get("id", candidate["metadata"]["content"]),
                "score": score,
                "metadata": dict(candidate["metadata"]),
            }
            for candidate in candidates
            if (score := self._cosine(vector, candidate["values"])) > 0
        ]
        dense.sort(key=lambda match: match["score"], reverse=True)
        if method == "dense":
            matches = dense
        else:
            if sparse_vector is None:
                raise VectorStoreConfigurationError("Sparse query vector is required")
            sparse = [
                {
                    "id": vector.get("id", vector["metadata"]["content"]),
                    "score": score,
                    "metadata": dict(vector["metadata"]),
                }
                for vector in candidates
                if (
                    score := self._sparse_dot(
                        sparse_vector,
                        vector.get("sparse_vector")
                        or build_sparse_vector(vector["metadata"]["content"]),
                    )
                )
                > 0
            ]
            sparse.sort(key=lambda match: match["score"], reverse=True)
            matches = sparse if method == "sparse" else self._fuse(dense, sparse)
        return {
            "matches": matches[:top_k],
            "method": method,
            "outcome": "success" if matches else "empty",
        }

    def delete(self, filter=None):
        """Do delete."""
        name = (filter or {}).get("pdf_name")
        self.vectors = [v for v in self.vectors if v["metadata"]["pdf_name"] != name]

    @staticmethod
    def _sparse_dot(query_sparse, document_sparse):
        query_values = dict(zip(query_sparse["indices"], query_sparse["values"]))
        document_values = dict(
            zip(document_sparse["indices"], document_sparse["values"])
        )
        return sum(
            value * document_values.get(index, 0.0)
            for index, value in query_values.items()
        )

    @staticmethod
    def _fuse(dense, sparse):
        fused = {}
        for matches in (dense, sparse):
            for rank, match in enumerate(matches, start=1):
                entry = fused.setdefault(match["id"], {**match, "score": 0.0})
                entry["score"] += 1.0 / (RRF_K + rank)
        return sorted(fused.values(), key=lambda match: match["score"], reverse=True)

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0


def canned_generate(query, context, prior_turns=""):
    """Do canned generate."""
    return "An answer fully supported by the provided context."


def faithful_judge(prompt):
    """Do faithful judge."""
    return "faithful"


@pytest.fixture
def fixture():
    """Do fixture."""
    return evaluator.load_fixture()


@pytest.fixture
def indexed_index(fixture):
    """Do indexed index."""
    index = InMemoryIndex()
    for doc in fixture["documents"]:
        evaluator.index_document(doc["filename"], index, hash_embed)
    return index


def one_hot_embed(texts, token_map, dims=None):
    """
    Embedding that maps marker tokens to fixed unit vectors.

        ``token_map`` maps a token to an axis, so tests point queries at
        exactly the chunks they should retrieve.
    """
    dims = dims or (max(token_map.values()) + 1)
    vectors = []
    for text in texts:
        vec = [0.0] * dims
        for token in _tokens(text):
            if token in token_map:
                vec[token_map[token]] = 1.0
        vectors.append(vec)
    return vectors


class TestMetrics:
    """TestMetrics."""

    def test_hit_at_k_true_within_top_k(self):
        """Do test hit at k true within top k."""
        retrieved = ["irrelevant", "the count was 41 jumps in total"]
        assert hit_at_k(retrieved, ["41 jumps"], 2)

    def test_hit_at_k_false_beyond_top_k(self):
        """Do test hit at k false beyond top k."""
        retrieved = ["irrelevant", "gold lives here: 41 jumps"]
        assert not hit_at_k(retrieved, ["41 jumps"], 1)

    def test_hit_ignores_case_and_whitespace(self):
        """Do test hit ignores case and whitespace."""
        assert hit_at_k(
            ["COUNTED   39 of the 41\nprogram jumps"], ["39 of the 41 program jumps"], 1
        )

    def test_recall_fraction_over_all_gold_snippets(self):
        """Do test recall fraction over all gold snippets."""
        retrieved = ["first gold: alpha", "nothing here"]
        assert recall_at_k(retrieved, ["alpha", "beta"], 2) == 0.5

    def test_recall_empty_snippets_is_zero(self):
        """Do test recall empty snippets is zero."""
        assert recall_at_k(["anything"], [], 1) == 0.0

    def test_summarize_averages_question_results(self):
        """Do test summarize averages question results."""
        results = [
            {"hit_at_k": True, "recall_at_k": 1.0},
            {"hit_at_k": False, "recall_at_k": 0.0},
        ]
        report = summarize(results, k=3)
        assert report.questions == 2
        assert report.hit_rate == 0.5
        assert report.recall == 0.5

    def test_summarize_empty_fixture(self):
        """Do test summarize empty fixture."""
        report = summarize([], k=3)
        assert report.hit_rate == 0.0 and report.recall == 0.0


class TestJudgeParsing:
    """TestJudgeParsing."""

    def test_parse_clean_verdict(self):
        """Do test parse clean verdict."""
        assert judge.parse_verdict("faithful") == ("faithful", 1.0)
        assert judge.parse_verdict("partial") == ("partial", 0.5)
        assert judge.parse_verdict("unfaithful") == ("unfaithful", 0.0)

    def test_parse_verdict_inside_a_sentence(self):
        """Do test parse verdict inside a sentence."""
        assert judge.parse_verdict('The verdict is: "Unfaithful"') == (
            "unfaithful",
            0.0,
        )

    def test_unparseable_reply_scores_zero(self):
        """Do test unparseable reply scores zero."""
        assert judge.parse_verdict("I think it is fine") == ("unparseable", 0.0)

    def test_judge_faithfulness_formats_the_prompt(self):
        """Do test judge faithfulness formats the prompt."""
        captured = {}

        def spy(prompt):
            """Do spy."""
            captured["prompt"] = prompt
            return "partial"

        verdict, score = judge.judge_faithfulness("Q", "A", "CTX", spy)
        assert (verdict, score) == ("partial", 0.5)
        assert "Q" in captured["prompt"]
        assert "CTX" in captured["prompt"]
        assert "A" in captured["prompt"]


class TestFixtureIntegrity:
    """TestFixtureIntegrity."""

    def test_fixture_references_committed_documents(self, fixture):
        """Do test fixture references committed documents."""
        for doc in fixture["documents"]:
            assert (evaluator.SAMPLE_DOCS_DIR / doc["filename"]).is_file()

    def test_every_question_has_gold_snippets_in_its_document(self, fixture):
        """
        The fixture stays honest: gold snippets must survive the real.

                parser, meaning retrieval could actually find them.
        """
        for item in fixture["questions"]:
            text = resolve_parser(item["document"]).extract_text(
                evaluator.read_document(item["document"])
            )
            normalized = " ".join(text.lower().split())
            assert item["gold_snippets"], item["id"]
            for snippet in item["gold_snippets"]:
                assert " ".join(snippet.lower().split()) in normalized, (
                    f"{item['id']}: gold snippet not in document: {snippet!r}"
                )

    def test_in_repo_primer_source_is_cc0_and_matches_its_pdf(self):
        """Do test in repo primer source is cc0 and matches its pdf."""
        text = PRIMER_SOURCE.read_text(encoding="utf-8")
        assert "CC0" in text


def test_retrieve_does_not_fall_back_when_store_rejects_sparse():
    """A misconfigured store fails instead of silently running dense retrieval."""

    class UnsupportedStore:
        def __init__(self):
            self.calls = 0

        def query(self, *args, **kwargs):
            self.calls += 1
            raise TypeError("sparse retrieval is required")

    store = UnsupportedStore()
    with pytest.raises(TypeError, match="sparse retrieval is required"):
        evaluator.retrieve([0.1], "doc.pdf", store, query_text="keyword")

    assert store.calls == 1


class TestEvaluator:
    """TestEvaluator."""

    @staticmethod
    def synthetic_setup():
        """
        Fixture + index where retrieval is pinned by marker tokens.

                Each question has a marker token ("both"/"one"/"none") that only
                its own seed chunks contain, so what the top-k returns is fully
                controlled by the test.
        """
        fixture = {
            "documents": [{"filename": "synthetic.pdf"}],
            "questions": [
                {
                    "id": "both-gold",
                    "document": "synthetic.pdf",
                    "question": "both",
                    "expected_answer": "expected",
                    "gold_snippets": ["gold alpha", "gold beta"],
                },
                {
                    "id": "one-gold",
                    "document": "synthetic.pdf",
                    "question": "one",
                    "expected_answer": "expected",
                    "gold_snippets": ["gold alpha"],
                },
                {
                    "id": "no-gold",
                    "document": "synthetic.pdf",
                    "question": "none",
                    "expected_answer": "expected",
                    "gold_snippets": ["gold alpha"],
                },
            ],
        }
        token_map = {
            "gold": 0,
            "alpha": 1,
            "beta": 2,
            "marker": 3,
            "both": 4,
            "one": 5,
            "none": 6,
        }
        seeds = {
            "both-gold": [
                "both gold alpha chunk",
                "both gold beta chunk",
            ],
            "one-gold": ["one gold alpha chunk"],
            # Deliberately lacks its gold snippet: this question must miss.
            "no-gold": ["none marker chunk with nothing useful"],
        }
        # Noise shares the "marker" axis with every query, so it always
        # out-scores irrelevant seeds (which score 0) and fills the
        # remaining top-k slots without ever containing gold text.

        index = InMemoryIndex()
        for question in fixture["questions"]:
            for i, content in enumerate(seeds[question["id"]]):
                index.vectors.append(
                    {
                        "id": f"{question['id']}-{i}",
                        "values": one_hot_embed([content], token_map)[0],
                        "metadata": {
                            "content": content,
                            "pdf_name": "synthetic.pdf",
                            "chunk_index": i,
                        },
                    }
                )
        for i in range(8):
            # Unique filler per chunk so source shaping cannot dedupe them.
            content = f"marker plain filler words about nothing in particular part {i}"
            index.vectors.append(
                {
                    "id": f"noise-{i}",
                    "values": one_hot_embed([content], token_map)[0],
                    "metadata": {
                        "content": content,
                        "pdf_name": "synthetic.pdf",
                        "chunk_index": 10 + i,
                    },
                }
            )
        return fixture, index, (lambda texts: one_hot_embed(texts, token_map))

    def test_metrics_on_controllable_retrieval(self):
        """Do test metrics on controllable retrieval."""
        fixture, index, embed = self.synthetic_setup()
        report = evaluator.evaluate(
            fixture, index, embed, canned_generate, judge_fn=None
        )
        assert report.retrieval.questions == 3
        per_id = {q["id"]: q for q in report.per_question}
        assert per_id["both-gold"]["retrieval_method"] == "hybrid"
        assert per_id["both-gold"]["hit_at_k"] is True
        assert per_id["both-gold"]["recall_at_k"] == 1.0
        assert per_id["one-gold"]["hit_at_k"] is True
        assert per_id["one-gold"]["recall_at_k"] == 1.0
        assert per_id["no-gold"]["hit_at_k"] is False
        assert per_id["no-gold"]["recall_at_k"] == 0.0
        assert report.retrieval.hit_rate == pytest.approx(2 / 3)
        assert report.retrieval.recall == pytest.approx(2 / 3)

    def test_a_follow_up_case_expands_its_query_like_the_application(self):
        """Both query forms reach the report, and retrieval uses the expansion."""
        fixture = {
            "documents": [{"filename": "synthetic.pdf"}],
            "questions": [
                {
                    "id": "follow-up",
                    "document": "synthetic.pdf",
                    "question": "What about the second one?",
                    "follow_up": ["What are the alpha and beta methods?"],
                    "expected_answer": "expected",
                    "gold_snippets": ["gold alpha"],
                },
            ],
        }
        token_map = {"gold": 0, "alpha": 1, "beta": 2}
        index = InMemoryIndex()
        index.vectors.append(
            {
                "id": "1",
                "values": one_hot_embed(["gold alpha chunk"], token_map)[0],
                "metadata": {
                    "content": "gold alpha chunk",
                    "pdf_name": "synthetic.pdf",
                    "chunk_index": 0,
                },
            }
        )
        embedded = []

        def embed(texts):
            embedded.extend(texts)
            return one_hot_embed(texts, token_map)

        report = evaluator.evaluate(
            fixture, index, embed, canned_generate, judge_fn=None
        )

        detail = report.per_question[0]
        assert detail["original_query"] == "What about the second one?"
        assert detail["query_expansion"] == "deterministic"
        assert detail["expanded_query"].endswith("What about the second one?")
        assert embedded == [detail["expanded_query"]]

    def test_partial_recall_when_gold_exceeds_top_k(self):
        """
        One gold snippet beyond the top-k chunk lowers recall but the.

                question can still count as a hit.
        """
        fixture, index, embed = self.synthetic_setup()
        report = evaluator.evaluate(
            fixture, index, embed, canned_generate, judge_fn=None, k=1
        )
        per_id = {q["id"]: q for q in report.per_question}
        assert per_id["both-gold"]["hit_at_k"] is True
        assert per_id["both-gold"]["recall_at_k"] == pytest.approx(0.5)

    def test_faithfulness_is_scored_when_a_judge_is_wired(self, fixture, indexed_index):
        """Do test faithfulness is scored when a judge is wired."""
        report = evaluator.evaluate(
            fixture,
            indexed_index,
            hash_embed,
            canned_generate,
            judge_fn=faithful_judge,
        )
        assert report.faithfulness["judged"] == len(fixture["questions"])
        assert report.faithfulness["mean"] == 1.0
        assert report.per_question[0]["verdict"] == "faithful"

    def test_real_fixture_plumbing_end_to_end(self, fixture, indexed_index):
        """
        Smoke check on the committed fixture: every question flows.

                through parsing, indexing, retrieval, and reporting.
        """
        report = evaluator.evaluate(
            fixture, indexed_index, hash_embed, canned_generate, judge_fn=None
        )
        assert report.retrieval.questions == len(fixture["questions"])
        assert len(report.per_question) == len(fixture["questions"])
        assert {q["id"] for q in report.per_question} == {
            q["id"] for q in fixture["questions"]
        }
        assert 0.0 <= report.retrieval.hit_rate <= 1.0
        assert 0.0 <= report.retrieval.recall <= 1.0

    def test_prefixed_live_style_run_matches_fixture_documents(self, fixture):
        """
        Regression: a live run indexes docs under an eval- prefix; the.

                retrieval filter must use the same prefixed name or every query
                silently matches nothing.
        """
        index = InMemoryIndex()
        for doc in fixture["documents"]:
            evaluator.index_document(
                doc["filename"],
                index,
                hash_embed,
                pdf_name=f"{cli.EVAL_PREFIX}{doc['filename']}",
            )
        report = evaluator.evaluate(
            fixture,
            index,
            hash_embed,
            canned_generate,
            judge_fn=None,
            prefix=cli.EVAL_PREFIX,
        )
        assert report.retrieval.hit_rate > 0.0

    def test_index_and_remove_round_trip(self, fixture):
        """Do test index and remove round trip."""
        index = InMemoryIndex()
        filename = fixture["documents"][0]["filename"]
        evaluator.index_document(filename, index, hash_embed)
        assert index.vectors
        evaluator.remove_document(filename, index)
        assert index.vectors == []


class TestLiveGate:
    """TestLiveGate."""

    def test_cli_refuses_without_live_flag(self, capsys):
        """Do test cli refuses without live flag."""
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert "live" in str(exc.value.code)

    def test_evaluation_judge_errors_redact_the_api_key(self, monkeypatch):
        """A provider exception cannot print the evaluation key."""
        import settings as settings_module
        from services.embeddings import local_embeddings
        from services.llm import google_provider

        secret = "sk-evaluation-secret"

        class FakeEmbedding:
            def __init__(self, model):
                self.model = model

            def embed_texts(self, texts):
                return [[0.0] for _ in texts]

        class FakeModels:
            def generate_content(self, **kwargs):
                raise RuntimeError(f"judge rejected {secret}")

        class FakeGoogleClient:
            models = FakeModels()

        monkeypatch.setattr(
            settings_module,
            "get_settings",
            lambda: type(
                "Settings",
                (),
                {"embedding": type("Embedding", (), {"embedding_model": "fake"})()},
            )(),
        )
        monkeypatch.setattr(local_embeddings, "LocalEmbeddingService", FakeEmbedding)
        monkeypatch.setattr(google_provider, "_client", lambda key: FakeGoogleClient())

        _, _, judge = cli.make_live_components("google", "gemini-2.5-flash", secret)
        with pytest.raises(RuntimeError) as exc:
            judge("judge this")

        assert secret not in str(exc.value)

    def test_cli_refuses_a_cross_provider_judge(self, monkeypatch):
        """Evaluation never sends a Groq key to the Google judge."""
        monkeypatch.setattr(
            cli,
            "load_fixture",
            lambda: pytest.fail("cross-provider evaluation reached the fixture"),
        )

        with pytest.raises(SystemExit) as exc:
            cli.run(
                live=True,
                judge=True,
                k=5,
                provider="groq",
                model="openai/gpt-oss-20b",
                api_key="groq-secret",
            )

        assert "Google" in str(exc.value.code)

    def test_cli_json_output_shape(self, fixture, indexed_index, monkeypatch, capsys):
        """Do test cli json output shape."""
        monkeypatch.setattr(
            cli,
            "run",
            lambda live, judge, k: {
                "retrieval": {"k": k, "questions": 1, "hit_rate": 1.0, "recall": 1.0},
                "faithfulness": {"mean": 1.0, "judged": 1, "faithful": 1},
                "per_question": [],
            },
        )
        cli.main(["--live", "--json", "--k", "3"])
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"retrieval", "faithfulness", "per_question"}
        assert payload["retrieval"]["k"] == 3

    def test_cli_defaults_to_an_active_catalog_model(self, monkeypatch, capsys):
        """The live evaluator uses a current model without a literal override."""
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {
                "retrieval": {"k": 3, "questions": 0, "hit_rate": 0.0, "recall": 0.0},
                "faithfulness": {"mean": 0.0, "judged": 0, "faithful": 0},
                "per_question": [],
            }

        monkeypatch.setattr(cli, "run", fake_run)
        cli.main(["--live", "--no-judge", "--json"])

        assert captured["provider"] == "google"
        assert captured["model"] == "gemini-2.5-flash"
        assert json.loads(capsys.readouterr().out)["retrieval"]["k"] == 3

    def test_cli_rejects_a_retired_model_before_running(self, monkeypatch):
        """A retired model cannot reach the live evaluation runner."""
        monkeypatch.setattr(
            cli,
            "run",
            lambda **kwargs: pytest.fail("retired model reached the evaluator"),
        )

        with pytest.raises(SystemExit) as exc:
            cli.main(["--live", "--no-judge", "--model", "gemini-2.0-flash"])

        assert exc.value.code == 2
