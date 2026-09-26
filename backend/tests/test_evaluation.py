"""
Running the labeled case set through the production answer path.

The evaluator's job is to describe the application, so these tests build the
application: real repositories over an in-memory database, the real parser and
ingestion worker, the real vector service over a store that keeps both vector
representations, and the real answer path. Only the model, the embedding
weights, and the vector database are substituted, each behind the same
interface the production one implements.

What is asserted is what a report says: the outcome of every case, the run
record behind it, and that the same case answered over HTTP and through the
evaluator is the same answer.
"""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from composition import Services
from evaluation import cli, evaluator, judge
from evaluation.harness import build_environment, remove_documents
from services.accounts.chat_settings_service import model_for
from services.llm.base import ChatBudget, LLMProvider
from services.retrieval.base import VectorStoreConfigurationError
from services.retrieval.vector_service import VectorService
from tests.evaluation_support import (
    HashingEmbeddingService,
    InMemoryStorage,
    InMemoryVectorStore,
    ScriptedChatFactory,
)
from tests.sse import parse_sse

MODEL = model_for("google", "gemini-2.5-flash")

ANSWER = "A RAG pipeline has five stages."
GOLD = "A RAG pipeline has five stages"
PRIMER = "papermind-rag-primer.pdf"


def in_memory_sessions():
    """Session factory over a fresh in-memory database."""
    from db import Base

    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def session_factory():
    """Session factory over one in-memory database."""
    return in_memory_sessions()


@pytest.fixture
def settings_obj():
    """Return the pinned settings the suite runs with."""
    import settings as settings_module

    return settings_module.get_settings()


@pytest.fixture
def store():
    """Return a vector store with the whole retrieval contract."""
    return InMemoryVectorStore()


@pytest.fixture
def embeddings():
    """Return deterministic embeddings."""
    return HashingEmbeddingService()


@pytest.fixture
def chat():
    """Return a provider factory whose next answer the test scripts."""
    factory = ScriptedChatFactory()
    factory.answer(ANSWER)
    return factory


@pytest.fixture
def storage(tmp_path, settings_obj):
    """Return local storage in a directory the test owns."""
    return InMemoryStorage(tmp_path / "evaluation-storage")


class RunBuilder:
    """
    Builds evaluation runs over the suite's collaborators.

    A test that needs a run to look different — a second database, a store that
    cannot serve hybrid, a clock it controls — says so here instead of repeating
    the same seven arguments.
    """

    def __init__(self, fixture, settings_obj, storage, embeddings, store, chat):
        """Bind the collaborators every run shares."""
        self._fixture = fixture
        self._settings = settings_obj
        self._storage = storage
        self._embeddings = embeddings
        self._store = store
        self._chat = chat

    def __call__(
        self,
        session_factory=None,
        store=None,
        documents_prefix="",
        clock=None,
    ):
        """Index the fixture documents into a working application."""
        return build_environment(
            self._fixture,
            settings=self._settings,
            session_factory=session_factory or in_memory_sessions(),
            storage=self._storage,
            embedding_service=self._embeddings,
            vector_service=VectorService(store or self._store),
            chat_provider_factory=self._chat,
            documents_prefix=documents_prefix,
            **({"clock": clock} if clock is not None else {}),
        )


@pytest.fixture
def fixture():
    """Return the committed labeled case set."""
    return evaluator.load_fixture()


@pytest.fixture
def build_run(fixture, settings_obj, storage, embeddings, store, chat):
    """Return the builder a test uses to start a run."""
    return RunBuilder(fixture, settings_obj, storage, embeddings, store, chat)


@pytest.fixture
def environment(build_run):
    """Return a working application with the fixture documents indexed in it."""
    environment = build_run()
    yield environment
    remove_documents(environment)


def one_case(fixture, document=PRIMER):
    """Return a single-case fixture so a test can script one outcome."""
    cases = [case for case in fixture["questions"] if case["document"] == document]
    return {"documents": fixture["documents"], "questions": cases[:1]}


def faithful_judge(prompt):
    """Grade every answer as faithful."""
    return "faithful"


class FailingProvider(LLMProvider):
    """A real provider whose one-shot call always fails."""

    name = "failing"

    def __init__(self, api_key: str):
        """Bind the key the failure will try to echo back."""
        super().__init__(api_key, "judge-model", budget=ChatBudget())

    def _build_client(self):
        """Return no client because this provider has no SDK."""
        return None

    def verify(self) -> None:
        """Verification is not exercised over the judge."""
        return None

    def _generate_response(self, query: str, context: str, history: str = "") -> str:
        """Fail the way a rejected key does, quoting the key back."""
        raise RuntimeError(f"judge rejected {self.api_key}")

    def _stream_response(self, query: str, context: str, history: str = ""):
        """Streaming is not exercised over the judge."""
        raise NotImplementedError


class TestMetrics:
    """TestMetrics."""

    def test_hit_at_k_true_within_top_k(self):
        """Do test hit at k true within top k."""
        from evaluation.metrics import hit_at_k

        assert hit_at_k(
            ["irrelevant", "the count was 41 jumps in total"], ["41 jumps"], 2
        )

    def test_recall_fraction_over_all_gold_snippets(self):
        """Do test recall fraction over all gold snippets."""
        from evaluation.metrics import recall_at_k

        assert (
            recall_at_k(["first gold: alpha", "nothing here"], ["alpha", "beta"], 2)
            == 0.5
        )

    def test_summarize_averages_question_results(self):
        """Do test summarize averages question results."""
        from evaluation.metrics import summarize

        report = summarize(
            [
                {"hit_at_k": True, "recall_at_k": 1.0},
                {"hit_at_k": False, "recall_at_k": 0.0},
            ],
            k=3,
        )
        assert report.questions == 2
        assert report.hit_rate == 0.5


class TestJudgeProtocol:
    """A verdict word, its score, and the prompt that asks for one."""

    def test_a_clean_verdict_is_read_as_its_score(self):
        """Do test a clean verdict is read as its score."""
        assert judge.parse_verdict("faithful") == ("faithful", 1.0)
        assert judge.parse_verdict("partial") == ("partial", 0.5)
        assert judge.parse_verdict("unfaithful") == ("unfaithful", 0.0)

    def test_a_verdict_inside_a_sentence_is_still_read(self):
        """Do test a verdict inside a sentence is still read."""
        assert judge.parse_verdict('The verdict is: "Unfaithful"') == (
            "unfaithful",
            0.0,
        )

    def test_a_reply_that_is_not_a_verdict_scores_zero(self):
        """Do test a reply that is not a verdict scores zero."""
        assert judge.parse_verdict("I think it is fine") == ("unparseable", 0.0)

    def test_the_prompt_carries_the_question_the_answer_and_the_context(self):
        """Do test the prompt carries the question the answer and the context."""
        captured = {}

        def spy(prompt):
            """Do spy."""
            captured["prompt"] = prompt
            return "partial"

        verdict, score = judge.judge_faithfulness("Q", "A", "CTX", spy)

        assert (verdict, score) == ("partial", 0.5)
        assert "Q" in captured["prompt"]
        assert "A" in captured["prompt"]
        assert "CTX" in captured["prompt"]


class TestFixtureIntegrity:
    """TestFixtureIntegrity."""

    def test_fixture_references_committed_documents(self, fixture):
        """Do test fixture references committed documents."""
        from evaluation.harness import SAMPLE_DOCS_DIR

        for doc in fixture["documents"]:
            assert (SAMPLE_DOCS_DIR / doc["filename"]).is_file()

    def test_every_question_has_gold_snippets_in_its_document(self, fixture):
        """Do test every question has gold snippets in its document."""
        from evaluation.harness import read_document
        from services.parsing.document_parser import resolve_parser

        for item in fixture["questions"]:
            text = resolve_parser(item["document"]).extract_text(
                read_document(item["document"])
            )
            normalized = " ".join(text.lower().split())
            assert item["gold_snippets"], item["id"]
            for snippet in item["gold_snippets"]:
                assert " ".join(snippet.lower().split()) in normalized, item["id"]


class TestProductionPath:
    """The run asks the application, and the report describes what it did."""

    def test_every_case_is_answered_through_the_answer_path(
        self, fixture, environment, chat
    ):
        """Each case ends as a stored answer with hybrid retrieval behind it."""
        report = evaluator.evaluate(fixture, environment, MODEL)

        assert len(report.cases) == len(fixture["questions"])
        assert {case.outcome for case in report.cases} == {evaluator.ANSWERED}
        assert {case.retrieval["method"] for case in report.cases} == {"hybrid"}
        assert all(case.generated for case in report.cases)

    def test_an_answer_is_recorded_on_the_documents_conversation(
        self, fixture, environment
    ):
        """The run's answers are on the Conversation, not only in the report."""
        report = evaluator.evaluate(fixture, environment, MODEL)
        conversation_id = environment.conversation_id(PRIMER)
        stored = environment.repositories.conversations.get_turns(conversation_id)

        primer = [case for case in report.cases if case.document == PRIMER]
        assert [turn["status"] for turn in stored] == ["answered"] * len(primer)
        assert stored[0]["answer"] == ANSWER

    def test_the_run_record_names_the_index_prompt_provider_model_and_settings(
        self, fixture, environment
    ):
        """A number is only comparable when the configuration behind it is named."""
        report = evaluator.evaluate(fixture, environment, MODEL)

        run = report.run
        assert run["provider"] == "google"
        assert run["model"] == "gemini-2.5-flash"
        assert run["prompt_version"] == "grounded-claims-v1"
        assert run["retrieval_methods"] == ["hybrid"]
        assert run["settings"]["embedding_model"]
        assert run["settings"]["context_token_budget"] > 0
        primer = run["documents"][PRIMER]
        assert primer["is_processed"] is True
        assert primer["index_generation"] == 1
        assert json.loads(primer["index_manifest"])["index_generation"] == 1
        assert primer["indexed_seconds"] >= 0

    def test_a_run_where_every_case_failed_still_names_its_configuration(
        self, fixture, environment, chat
    ):
        """The record describes what was measured, not what happened to succeed."""
        chat.fail_with(RuntimeError("provider down"), before_output=True)

        report = evaluator.evaluate(fixture, environment, MODEL)

        assert {case.outcome for case in report.cases} == {evaluator.PROVIDER_ERROR}
        assert report.run["prompt_version"] == "grounded-claims-v1"
        assert report.run["provider"] == "google"
        assert report.run["model"] == "gemini-2.5-flash"
        assert report.run["settings"]["context_token_budget"] > 0

    def test_each_case_records_the_index_and_retrieval_it_was_answered_from(
        self, fixture, environment
    ):
        """Provenance travels with the case, not only with the run."""
        report = evaluator.evaluate(fixture, environment, MODEL)

        provenance = report.cases[0].provenance
        assert provenance["index_generation"] == 1
        assert provenance["index_state"] == "ready"
        assert provenance["retrieval_method"] == "hybrid"
        assert provenance["retrieval_seconds"] >= 0
        assert provenance["prompt_version"] == "grounded-claims-v1"

    def test_retrieval_is_measured_against_what_the_model_was_shown(
        self, fixture, environment
    ):
        """Hit and recall describe the evidence the answer path supplied."""
        case = one_case(fixture)
        report = evaluator.evaluate(case, environment, MODEL)

        result = report.cases[0]
        assert result.retrieved_chunks == len(result.retrieved_texts)
        assert result.retrieved_chunks > 0
        assert result.hit_at_k is (GOLD in " ".join(result.retrieved_texts))
        assert report.retrieval.questions == 1

    def test_a_run_measures_retrieval_with_the_clock_it_was_given(
        self, fixture, build_run
    ):
        """A run's latency is a measurement, so the clock is an input."""
        ticks = iter(float(index) for index in range(10_000))
        environment = build_run(clock=lambda: next(ticks))
        try:
            report = evaluator.evaluate(one_case(fixture), environment, MODEL)

            assert report.cases[0].provenance["retrieval_seconds"] == 1.0
        finally:
            remove_documents(environment)

    def test_a_wired_judge_scores_only_the_answers_a_model_wrote(
        self, fixture, environment
    ):
        """Faithfulness is counted over generated answers, and names them."""
        report = evaluator.evaluate(
            fixture,
            environment,
            MODEL,
            judge_fn=faithful_judge,
        )

        assert report.faithfulness["judged"] == len(fixture["questions"])
        assert report.faithfulness["mean"] == 1.0
        assert all(case.verdict == "faithful" for case in report.cases)


class TestCaseOutcomes:
    """Every way a case can end is recorded as the outcome it was."""

    def test_a_provider_failure_is_its_own_outcome_and_is_never_graded(
        self, fixture, environment, chat
    ):
        """A provider that dies produces a provider error, not a bad answer."""
        case = one_case(fixture)
        chat.fail_with(RuntimeError("provider down"), before_output=True)

        report = evaluator.evaluate(
            case,
            environment,
            MODEL,
            judge_fn=faithful_judge,
        )

        result = report.cases[0]
        assert result.outcome == evaluator.PROVIDER_ERROR
        assert result.detail == "provider"
        assert result.generated is False
        assert result.verdict is None
        assert report.faithfulness["judged"] == 0

    def test_a_stalled_provider_is_recorded_as_a_timeout(
        self, fixture, environment, chat
    ):
        """Running out of time is not the same as the model being down."""
        from tests.evaluation_support import timeout_error

        case = one_case(fixture)
        chat.fail_with(timeout_error())

        report = evaluator.evaluate(case, environment, MODEL)

        assert report.cases[0].outcome == evaluator.PROVIDER_ERROR
        assert report.cases[0].detail == "timeout"

    def test_an_unrepairable_citation_is_recorded_as_a_citation_error(
        self, fixture, environment, chat
    ):
        """A claim citing a Passage that was never supplied fails the case."""
        case = one_case(fixture)
        chat.uncited(ANSWER)

        report = evaluator.evaluate(case, environment, MODEL)

        result = report.cases[0]
        assert result.outcome == evaluator.CITATION_ERROR
        assert result.generated is False
        assert result.answer is None

    def test_an_answer_that_cannot_be_saved_is_a_persistence_error(
        self, fixture, environment, chat, monkeypatch
    ):
        """A store failure is its own outcome, and the Turn is closed failed."""
        case = one_case(fixture)
        monkeypatch.setattr(
            environment.repositories.conversations,
            "complete_turn",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("postgres down")),
        )

        report = evaluator.evaluate(case, environment, MODEL)

        assert report.cases[0].outcome == evaluator.PERSISTENCE_ERROR
        assert report.cases[0].generated is False

    def test_a_question_the_evidence_does_not_reach_is_recorded_as_an_abstention(
        self, fixture, environment, store, chat
    ):
        """A Document with no matching Passages abstains without spending a call."""
        store.delete(filter={"pdf_name": environment.stored_name(PRIMER)})
        case = one_case(fixture)

        report = evaluator.evaluate(case, environment, MODEL)

        result = report.cases[0]
        assert result.outcome == evaluator.ABSTAINED
        assert result.detail == "no_evidence"
        assert result.generated is False
        assert chat.streamed == []

    def test_a_document_that_cannot_be_asked_is_refused_rather_than_missed(
        self, fixture, build_run, chat
    ):
        """A Document with no index is not a retrieval that found nothing."""
        environment = build_run()
        try:
            case = one_case(fixture)
            environment.repositories.files.set_processed(
                environment.stored_name(PRIMER), False
            )

            report = evaluator.evaluate(case, environment, MODEL)

            result = report.cases[0]
            assert result.outcome == evaluator.REFUSED
            assert result.detail == "index_pending"
            assert result.hit_at_k is None
            assert report.retrieval.questions == 0
            assert chat.streamed == []
        finally:
            remove_documents(environment)

    def test_fallback_context_is_never_graded_as_a_generated_answer(
        self, fixture, environment, chat
    ):
        """Retrieved text quoted back after a failure is not an answer."""
        case = one_case(fixture)
        # The provider answers, but its stored answer is the one-shot
        # fallback: document text wearing the shape of an answer. The run must
        # name that for what it is and keep it away from the grader.
        chat.answer(ANSWER)
        fallback = (
            "I couldn't use the language model right now, so here is relevant "
            f"context from your document:\n\n[1] {GOLD}"
        )
        original = environment.answer_service.stream

        def stream_with_fallback(resolved):
            for event in original(resolved):
                if event.name == "done":
                    yield type(event)("done", {**event.payload, "answer": fallback})
                else:
                    yield event

        environment.answer_service.stream = stream_with_fallback

        report = evaluator.evaluate(
            case,
            environment,
            MODEL,
            judge_fn=faithful_judge,
        )

        result = report.cases[0]
        assert result.outcome == evaluator.CONTEXT_FALLBACK
        assert result.generated is False
        assert result.verdict is None
        assert report.faithfulness["judged"] == 0

    def test_every_outcome_a_case_can_have_is_reported_even_at_zero(
        self, fixture, environment
    ):
        """A report names the outcomes that did not happen, not only the ones that did."""
        report = evaluator.evaluate(fixture, environment, MODEL)

        assert set(report.outcomes) == set(evaluator.OUTCOMES)
        assert report.outcomes[evaluator.ABSTAINED] == 0


class TestRetrievalContract:
    """A store that cannot serve the app's retrieval fails the run."""

    def test_a_store_that_refuses_sparse_fails_the_run(self, build_run):
        """A store that cannot serve sparse queries is a failed run."""
        with pytest.raises(VectorStoreConfigurationError) as failure:
            build_run(store=InMemoryVectorStore(supports_hybrid=False))

        assert "cannot serve sparse or hybrid" in str(failure.value)

    def test_a_store_that_answers_hybrid_with_dense_fails_the_run(self, build_run):
        """
        The silent case is the one that matters.

            A store that accepts a hybrid query and answers it densely would
            otherwise report a lower hit rate that no reader could explain, so
            the run stops and names the degradation.
        """

        class DegradingStore(InMemoryVectorStore):
            def query(
                self, vector, top_k, include_metadata=True, filter=None, **kwargs
            ):
                if kwargs.get("method") == "hybrid":
                    kwargs = {**kwargs, "method": "dense"}
                return super().query(vector, top_k, include_metadata, filter, **kwargs)

        with pytest.raises(VectorStoreConfigurationError) as failure:
            build_run(store=DegradingStore())

        assert "answered a hybrid question with dense results" in str(failure.value)

    def test_an_unreachable_store_is_reported_rather_than_answered(
        self, fixture, build_run
    ):
        """Retrieval reports the failure rather than quietly answering densely."""
        from services.retrieval.base import VectorStoreUnavailableError

        class UnreachableAfterIndexing(InMemoryVectorStore):
            unreachable = False

            def query(self, *args, **kwargs):
                if self.unreachable:
                    raise VectorStoreUnavailableError("store is down")
                return super().query(*args, **kwargs)

        store = UnreachableAfterIndexing()
        environment = build_run(store=store)
        try:
            store.unreachable = True
            report = evaluator.evaluate(one_case(fixture), environment, MODEL)

            assert report.cases[0].outcome == evaluator.REFUSED
            assert report.cases[0].detail == "vector_store_unavailable"
        finally:
            remove_documents(environment)


class TestMultiTurn:
    """A follow-up case reads the Turns the run actually committed."""

    def test_a_follow_up_reads_the_earlier_questions_from_the_conversation(
        self, fixture, environment, chat
    ):
        """The expansion is built from stored Turns, not from the fixture."""
        case = {
            "documents": fixture["documents"],
            "questions": [
                {
                    "id": "follow-up",
                    "document": PRIMER,
                    "question": "What about the second one?",
                    "follow_up": ["What are the stages of a RAG pipeline?"],
                    "gold_snippets": [GOLD],
                }
            ],
        }

        report = evaluator.evaluate(case, environment, MODEL)

        result = report.cases[0]
        assert result.retrieval["query_expansion"] == "deterministic"
        assert result.retrieval["expanded_query"].endswith("What about the second one?")
        assert result.outcome == evaluator.ANSWERED


class TestSameResultOverHttp:
    """The report describes what the HTTP application does."""

    def test_the_same_case_answers_the_same_way_over_http_and_in_the_evaluator(
        self, fixture, build_run, settings_obj, storage, embeddings, chat
    ):
        """One case, asked two ways against equal state, is one answer."""
        from app import create_app
        from services.accounts.secrets_service import encrypt_api_key

        case = one_case(fixture)
        question = case["questions"][0]["question"]
        # Two applications over separate databases and separate stores, so
        # neither ask can see the other's Turn or the other's vectors. The only
        # thing that differs is which door the same question came through.
        over_http = build_run(store=InMemoryVectorStore())
        through_evaluator = build_run(store=InMemoryVectorStore())
        try:
            application = create_app(
                settings_obj,
                services=Services(
                    settings=settings_obj,
                    repositories=over_http.repositories,
                    storage=storage,
                    parser=over_http.parser,
                    embedding_service=embeddings,
                    vector_service=over_http.vector_service,
                    chat_provider_factory=chat,
                    api_key_verifier=lambda credentials: (True, None),
                ),
            )
            over_http.repositories.app_settings.upsert_app_settings(
                "google", "gemini-2.5-flash", encrypt_api_key("sk-test-key")
            )

            chat.answer(ANSWER)
            response = application.test_client().post(
                "/response",
                json={"query": question, "filename": over_http.stored_name(PRIMER)},
            )
            done = [
                payload
                for name, payload in parse_sse(response.get_data(as_text=True))
                if name == "done"
            ][0]

            chat.answer(ANSWER)
            result = evaluator.evaluate(
                case,
                through_evaluator,
                MODEL,
            ).cases[0]

            assert result.answer == done["answer"] == ANSWER
            assert result.retrieval["method"] == done["retrieval"]["method"] == "hybrid"
            assert result.retrieved_texts == [
                source["content"] for source in done["sources"]
            ]
            # Both asks reached the model with the same question, the same
            # evidence, and no transcript, which is what makes one number
            # describe the other.
            assert (
                chat.streamed[0]
                == chat.streamed[1]
                == (
                    question,
                    result.context,
                    "",
                )
            )
        finally:
            remove_documents(over_http)
            remove_documents(through_evaluator)


class TestCliConfiguration:
    """Keys come from the environment, and the two roles are configured apart."""

    def test_cli_refuses_without_live_flag(self):
        """Do test cli refuses without live flag."""
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert "live" in str(exc.value.code)

    def test_a_key_is_only_ever_read_from_the_environment(self, monkeypatch):
        """A key on the command line is visible to every process on the machine."""
        monkeypatch.setattr(cli, "run", lambda **kwargs: _empty_report())

        with pytest.raises(SystemExit) as exc:
            cli.main(["--live", "--no-judge", "--api-key", "sk-secret"])

        assert exc.value.code == 2
        assert cli.secret_from_env(cli.GENERATOR_API_KEY_ENV) == ""

    def test_a_missing_generator_key_names_the_variable_to_set(self, monkeypatch):
        """Do test a missing generator key names the variable to set."""
        monkeypatch.delenv(cli.GENERATOR_API_KEY_ENV, raising=False)
        monkeypatch.setattr("settings.get_settings", lambda: _pinned())

        with pytest.raises(SystemExit) as exc:
            cli.run(live=True, judge=False, k=5)

        assert cli.GENERATOR_API_KEY_ENV in str(exc.value.code)

    def test_a_missing_judge_key_names_its_own_variable(self, monkeypatch):
        """Do test a missing judge key names its own variable."""
        monkeypatch.setenv(cli.GENERATOR_API_KEY_ENV, "generator-secret")
        monkeypatch.delenv(cli.JUDGE_API_KEY_ENV, raising=False)

        with pytest.raises(SystemExit) as exc:
            cli.run(
                live=True,
                judge=True,
                k=5,
                provider="google",
                model="gemini-2.5-flash",
                judge_provider="google",
                judge_model="gemini-2.5-flash",
            )

        assert cli.JUDGE_API_KEY_ENV in str(exc.value.code)

    def test_the_generator_and_the_judge_are_configured_independently(
        self, monkeypatch
    ):
        """One run may generate with one account and grade with another."""
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return _empty_report()

        monkeypatch.setattr(cli, "run", fake_run)
        cli.main(
            [
                "--live",
                "--no-judge",
                "--provider",
                "groq",
                "--model",
                "openai/gpt-oss-20b",
                "--judge-provider",
                "google",
                "--judge-model",
                "gemini-3.5-flash",
            ]
        )

        assert captured["provider"] == "groq"
        assert captured["model"] == "openai/gpt-oss-20b"
        assert captured["judge_provider"] == "google"
        assert captured["judge_model"] == "gemini-3.5-flash"

    def test_the_judge_reaches_the_provider_it_was_asked_for(self, monkeypatch):
        """A judge configured for one provider is never sent to another."""
        captured = {}

        def fake_build(credentials, use_cache=True):
            """Do fake build."""
            captured["credentials"] = credentials
            return FailingProvider("sk-judge-secret")

        monkeypatch.setattr(cli, "build_chat_provider", fake_build)
        cli.build_judge("groq", "openai/gpt-oss-120b", "groq-secret")

        credentials = captured["credentials"]
        assert credentials.provider == "groq"
        assert credentials.model == "openai/gpt-oss-120b"
        assert credentials.api_key == "groq-secret"

    def test_a_failed_judge_call_reports_a_category_and_no_key(self, monkeypatch):
        """A provider error can quote the key back, and never reaches a report."""
        secret = "sk-judge-secret"
        monkeypatch.setattr(
            cli,
            "build_chat_provider",
            lambda *a, **k: FailingProvider(secret),
        )
        judge_fn = cli.build_judge("google", "gemini-2.5-flash", secret)

        with pytest.raises(RuntimeError) as failure:
            judge_fn("grade this")

        assert secret not in str(failure.value)
        assert "the provider call failed" in str(failure.value)

    def test_the_cli_defaults_to_an_active_catalog_model(self, monkeypatch, capsys):
        """Do test the cli defaults to an active catalog model."""
        captured = {}

        def fake_run(**kwargs):
            """Do fake run."""
            captured.update(kwargs)
            return _empty_report()

        monkeypatch.setattr(cli, "run", fake_run)
        cli.main(["--live", "--no-judge", "--json"])

        assert captured["provider"] == "google"
        assert captured["model"] == "gemini-2.5-flash"
        assert json.loads(capsys.readouterr().out)["retrieval"]["k"] == 5

    def test_an_unsupported_generator_model_is_refused_before_running(
        self, monkeypatch
    ):
        """A retired model cannot reach the run."""
        monkeypatch.setattr(
            cli,
            "run",
            lambda **kwargs: pytest.fail("retired model reached the evaluator"),
        )

        with pytest.raises(SystemExit) as exc:
            cli.main(["--live", "--no-judge", "--model", "gemini-2.0-flash"])

        assert exc.value.code == 2

    def test_an_unsupported_judge_model_is_refused_before_running(self, monkeypatch):
        """A retired judge model cannot reach the run either."""
        monkeypatch.setattr(
            cli,
            "run",
            lambda **kwargs: pytest.fail("retired judge model reached the evaluator"),
        )

        with pytest.raises(SystemExit) as exc:
            cli.main(["--live", "--judge-model", "gemini-2.0-flash"])

        assert exc.value.code == 2

    def test_json_output_carries_the_run_record_and_every_case(
        self, monkeypatch, capsys
    ):
        """Do test json output carries the run record and every case."""
        monkeypatch.setattr(cli, "run", lambda **kwargs: _empty_report())
        cli.main(["--live", "--no-judge", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"run", "retrieval", "outcomes", "faithfulness", "cases"}
        assert payload["run"]["prompt_version"] == "grounded-claims-v1"
        assert payload["outcomes"]["answered"] == 0

    def test_the_reranker_gate_is_a_setting_not_a_branch(self, settings_obj):
        """An ablation changes configuration; the answer path is unchanged."""
        flipped = cli.with_rerank(settings_obj, True)
        assert flipped.rerank.enabled is True
        assert settings_obj.rerank.enabled is False
        assert flipped.query_context == settings_obj.query_context

    def test_comparing_reranking_reports_both_runs(self, monkeypatch, capsys):
        """An ablation is two runs of the same case set, reported side by side."""
        monkeypatch.setenv(cli.GENERATOR_API_KEY_ENV, "generator-secret")
        monkeypatch.delenv(cli.JUDGE_API_KEY_ENV, raising=False)
        monkeypatch.setattr("settings.get_settings", _pinned)
        seen = []

        def fake_once(fixture, app_settings, *args, **kwargs):
            seen.append(app_settings.rerank.enabled)
            return _empty_report()

        monkeypatch.setattr(cli, "_run_once", fake_once)
        report = cli.run(
            live=True,
            judge=False,
            k=5,
            provider="google",
            model="gemini-2.5-flash",
            compare_rerank=True,
        )

        assert seen == [False, True]
        assert set(report["rerank"]) == {"off", "on"}


def _pinned():
    from tests.conftest import TEST_SETTINGS

    return TEST_SETTINGS


def _empty_report() -> dict:
    return {
        "run": {
            "k": 5,
            "prompt_version": "grounded-claims-v1",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "settings": {},
            "documents": {},
            "retrieval_methods": [],
        },
        "retrieval": {"questions": 0, "k": 5, "hit_rate": 0.0, "recall": 0.0},
        "outcomes": {outcome: 0 for outcome in evaluator.OUTCOMES},
        "faithfulness": {"mean": None, "judged": 0, "faithful": 0},
        "cases": [],
    }
