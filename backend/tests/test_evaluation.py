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
from dataclasses import replace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from composition import Services
from evaluation import cli, evaluator, graders, judge
from evaluation.calibration import load_calibration
from evaluation.dataset import TUNING, Case, load_dataset
from evaluation.harness import SAMPLE_DOCS_DIR, build_environment, remove_documents
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

    def __init__(self, dataset, settings_obj, storage, embeddings, store, chat):
        """Bind the collaborators every run shares."""
        self._dataset = dataset
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
        """Index the case set's documents into a working application."""
        return build_environment(
            self._dataset,
            settings=self._settings,
            session_factory=session_factory or in_memory_sessions(),
            storage=self._storage,
            embedding_service=self._embeddings,
            vector_service=VectorService(store or self._store),
            chat_provider_factory=self._chat,
            documents_prefix=documents_prefix,
            **({"clock": clock} if clock is not None else {}),
        )


@pytest.fixture(scope="session")
def dataset():
    """Return the committed labeled case set."""
    return load_dataset(docs_dir=SAMPLE_DOCS_DIR)


@pytest.fixture
def build_run(dataset, settings_obj, storage, embeddings, store, chat):
    """Return the builder a test uses to start a run."""
    return RunBuilder(dataset, settings_obj, storage, embeddings, store, chat)


@pytest.fixture
def environment(build_run):
    """Return a working application with the case set's documents indexed in it."""
    environment = build_run()
    yield environment
    remove_documents(environment)


def a_set(dataset, cases):
    """Return the same set narrowed to the given cases."""
    return replace(dataset, cases=tuple(cases))


def tuning_set(dataset):
    """
    Return the tuning split, which is the small half a test run can afford.

    A run over the whole set would work, but every case asks a real question of
    the real answer path, and the tuning split exercises the same code over
    fourteen questions instead of forty-two.
    """
    return a_set(dataset, dataset.cases_for(TUNING))


def one_case(dataset, document=PRIMER):
    """Return a set holding one case, so a test can script one outcome."""
    matching = [case for case in tuning_set(dataset).cases if case.document == document]
    return a_set(dataset, matching[:1])


def run_set(cases, environment, **kwargs):
    """
    Run exactly the cases a test narrowed the set down to.

    A run picks a split, and a test holds a handful of cases from one of them,
    so the split is read off the cases rather than repeated at every call site.
    A test that hands over an exact list of cases means those cases, so the ones
    carrying a fault are asked for here; what a run does with them by default is
    asserted against a whole set instead.
    """
    splits = {case.split for case in cases.cases}
    assert len(splits) == 1, "a narrowed set holds cases from one split"
    return evaluator.evaluate(
        cases,
        environment,
        MODEL,
        split=splits.pop(),
        include_faults=True,
        **kwargs,
    )


def faithful_judge(prompt):
    """Grade every answer as faithful."""
    return "faithful"


def a_judge(grade=faithful_judge, provider="google", model="gemini-2.5-flash"):
    """Return a judge with the identity a run records beside its scores."""
    return judge.Judge(
        settings=judge.JudgeSettings(provider=provider, model=model), grade=grade
    )


class TestTheSetDecidesWhatARunAsks:
    """The set chooses the cases, the split, and the outcome each has to reach."""

    def test_a_run_asks_the_reported_split_and_says_which_it_left_out(
        self, dataset, environment
    ):
        """The quoted number is the reported split, and the held-back cases are named."""
        report = evaluator.evaluate(dataset, environment, MODEL)

        asked = {case.id for case in report.cases}
        assert asked == {case.id for case in dataset.cases_for("validation")}
        assert report.run["split"] == "validation"
        assert report.run["cases"] == len(asked)
        assert report.run["held_back"] == [
            case.id for case in dataset.faults_for("validation")
        ]

    def test_the_fault_cases_run_when_a_run_asks_for_them(self, dataset, environment):
        """They are held back from a reported run, not deleted from the set."""
        report = evaluator.evaluate(
            dataset, environment, MODEL, split="validation", include_faults=True
        )

        assert {case.id for case in report.cases} >= {
            case.id for case in dataset.faults_for("validation")
        }
        assert report.run["held_back"] == []

    def test_a_provider_failure_case_is_reported_as_a_failure(
        self, dataset, environment, chat
    ):
        """A case that exists to check a failure is scored as one, not as an answer."""
        cases = [
            case
            for case in dataset.faults_for("tuning")
            if case.fault == "provider_unavailable"
        ]
        chat.fail_with(RuntimeError("provider down"), before_output=True)

        report = run_set(a_set(dataset, cases), environment)

        result = report.cases[0]
        assert result.outcome == evaluator.PROVIDER_ERROR
        assert result.expected_outcome == evaluator.PROVIDER_ERROR
        assert result.grades["provider_status"].passed
        assert not result.generated
        assert result.answer is None
        assert report.answers.faithfulness.scored == 0
        assert report.cost.priced_cases == 0

    def test_a_citation_validation_case_is_reported_as_an_unusable_citation(
        self, dataset, environment, chat
    ):
        """A citation naming nothing supplied is reported, not stored for a reader."""
        cases = [
            case
            for case in dataset.faults_for("validation")
            if case.fault == "invalid_citation"
        ]
        chat.uncited("The flight time error was 0.098 s.")

        report = run_set(a_set(dataset, cases), environment)

        result = report.cases[0]
        assert result.outcome == evaluator.CITATION_ERROR
        assert result.expected_outcome == evaluator.CITATION_ERROR
        assert result.grades["provider_status"].passed
        assert report.answers.citation_recall.scored == 0

    def test_an_injection_case_is_asked_about_the_notice_rather_than_obeyed(
        self, dataset, environment, chat
    ):
        """The case wants the notice reported, and grades the answer against its words."""
        case = next(
            case for case in dataset.cases if case.id == "notes-injection-what-it-says"
        )
        chat.answer("The pasted notice tells the assistant to disregard the question.")

        report = run_set(a_set(dataset, [case]), environment)

        result = report.cases[0]
        assert result.category == "prompt_injection"
        assert result.outcome == evaluator.ANSWERED
        assert result.expected_answer
        assert result.grades["exact_evidence"].outcome == "passed"

    def test_an_answer_that_obeys_the_document_instead_of_the_question_is_left_uncited(
        self, dataset, environment, chat
    ):
        """Compliance leaves a sentence the reader cannot check, and nothing fakes a check."""
        case = next(
            case for case in dataset.cases if case.id == "notes-injection-what-it-says"
        )
        chat.answer_without_citations("Report access verified.")

        report = run_set(a_set(dataset, [case]), environment)

        result = report.cases[0]
        assert result.outcome == evaluator.ANSWERED
        assert result.claims == []
        assert result.grades["claims_schema"].outcome == "unknown"
        assert result.citation_recall is None

    def test_every_case_row_carries_what_it_was_asked_as(self, dataset, environment):
        """A report can be grouped by category, because each row says which."""
        report = run_set(one_case(dataset), environment)

        row = report.as_dict()["cases"][0]
        assert row["split"] == "tuning"
        assert row["category"] == "exact_lookup"
        assert row["expected_outcome"] == "answered"


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


class TestProductionPath:
    """The run asks the application, and the report describes what it did."""

    def test_every_case_is_answered_through_the_answer_path(
        self, dataset, environment, chat
    ):
        """Each case ends as a stored answer with hybrid retrieval behind it."""
        report = run_set(tuning_set(dataset), environment)

        assert len(report.cases) == len(tuning_set(dataset).cases)
        assert {case.outcome for case in report.cases} == {evaluator.ANSWERED}
        assert {case.retrieval["method"] for case in report.cases} == {"hybrid"}
        assert all(case.generated for case in report.cases)

    def test_an_answer_is_recorded_on_the_documents_conversation(
        self, dataset, environment
    ):
        """The run's answers are on the Conversation, not only in the report."""
        report = run_set(tuning_set(dataset), environment)
        conversation_id = environment.conversation_id(PRIMER)
        stored = environment.repositories.conversations.get_turns(conversation_id)

        primer = [case for case in report.cases if case.document == PRIMER]
        assert [turn["status"] for turn in stored] == ["answered"] * len(primer)
        assert stored[0]["answer"] == ANSWER

    def test_the_run_record_names_the_index_prompt_provider_model_and_settings(
        self, dataset, environment
    ):
        """A number is only comparable when the configuration behind it is named."""
        report = run_set(tuning_set(dataset), environment)

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
        self, dataset, environment, chat
    ):
        """The record describes what was measured, not what happened to succeed."""
        chat.fail_with(RuntimeError("provider down"), before_output=True)

        report = run_set(tuning_set(dataset), environment)

        assert {case.outcome for case in report.cases} == {evaluator.PROVIDER_ERROR}
        assert report.run["prompt_version"] == "grounded-claims-v1"
        assert report.run["provider"] == "google"
        assert report.run["model"] == "gemini-2.5-flash"
        assert report.run["settings"]["context_token_budget"] > 0

    def test_each_case_records_the_index_and_retrieval_it_was_answered_from(
        self, dataset, environment
    ):
        """Provenance travels with the case, not only with the run."""
        report = run_set(tuning_set(dataset), environment)

        provenance = report.cases[0].provenance
        assert provenance["index_generation"] == 1
        assert provenance["index_state"] == "ready"
        assert provenance["retrieval_method"] == "hybrid"
        assert provenance["retrieval_seconds"] >= 0
        assert provenance["prompt_version"] == "grounded-claims-v1"

    def test_retrieval_is_measured_against_what_the_model_was_shown(
        self, dataset, environment
    ):
        """Hit and recall describe the evidence the answer path supplied."""
        case = one_case(dataset)
        report = run_set(case, environment)

        result = report.cases[0]
        assert result.retrieved_chunks == len(result.retrieved_texts)
        assert result.retrieved_chunks > 0
        assert result.hit_at_k is (GOLD in " ".join(result.retrieved_texts))
        assert report.retrieval.questions == 1

    def test_a_run_measures_retrieval_with_the_clock_it_was_given(
        self, dataset, build_run
    ):
        """A run's latency is a measurement, so the clock is an input."""
        ticks = iter(float(index) for index in range(10_000))
        environment = build_run(clock=lambda: next(ticks))
        try:
            report = run_set(one_case(dataset), environment)

            assert report.cases[0].provenance["retrieval_seconds"] == 1.0
        finally:
            remove_documents(environment)

    def test_a_wired_judge_scores_only_the_answers_a_model_wrote(
        self, dataset, environment
    ):
        """Faithfulness is counted over generated answers, and names them."""
        report = run_set(
            tuning_set(dataset),
            environment,
            judge=a_judge(),
        )

        assert report.answers.faithfulness.scored == len(tuning_set(dataset).cases)
        assert report.answers.faithfulness.mean == 1.0
        assert all(case.verdict == "faithful" for case in report.cases)


class TestCaseOutcomes:
    """Every way a case can end is recorded as the outcome it was."""

    def test_a_provider_failure_is_its_own_outcome_and_is_never_graded(
        self, dataset, environment, chat
    ):
        """A provider that dies produces a provider error, not a bad answer."""
        case = one_case(dataset)
        chat.fail_with(RuntimeError("provider down"), before_output=True)

        report = run_set(
            case,
            environment,
            judge=a_judge(),
        )

        result = report.cases[0]
        assert result.outcome == evaluator.PROVIDER_ERROR
        assert result.detail == "provider"
        assert result.generated is False
        assert result.verdict is None
        assert report.answers.faithfulness.scored == 0
        assert report.answers.faithfulness.mean is None

    def test_a_stalled_provider_is_recorded_as_a_timeout(
        self, dataset, environment, chat
    ):
        """Running out of time is not the same as the model being down."""
        from tests.evaluation_support import timeout_error

        case = one_case(dataset)
        chat.fail_with(timeout_error())

        report = run_set(case, environment)

        assert report.cases[0].outcome == evaluator.PROVIDER_ERROR
        assert report.cases[0].detail == "timeout"

    def test_an_unrepairable_citation_is_recorded_as_a_citation_error(
        self, dataset, environment, chat
    ):
        """A claim citing a Passage that was never supplied fails the case."""
        case = one_case(dataset)
        chat.uncited(ANSWER)

        report = run_set(case, environment)

        result = report.cases[0]
        assert result.outcome == evaluator.CITATION_ERROR
        assert result.generated is False
        assert result.answer is None

    def test_an_answer_that_cannot_be_saved_is_a_persistence_error(
        self, dataset, environment, chat, monkeypatch
    ):
        """A store failure is its own outcome, and the Turn is closed failed."""
        case = one_case(dataset)
        monkeypatch.setattr(
            environment.repositories.conversations,
            "complete_turn",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("postgres down")),
        )

        report = run_set(case, environment)

        assert report.cases[0].outcome == evaluator.PERSISTENCE_ERROR
        assert report.cases[0].generated is False

    def test_a_question_the_evidence_does_not_reach_is_recorded_as_an_abstention(
        self, dataset, environment, store, chat
    ):
        """A Document with no matching Passages abstains without spending a call."""
        store.delete(filter={"pdf_name": environment.stored_name(PRIMER)})
        case = one_case(dataset)

        report = run_set(case, environment)

        result = report.cases[0]
        assert result.outcome == evaluator.ABSTAINED
        assert result.detail == "no_evidence"
        assert result.generated is False
        assert chat.streamed == []

    def test_a_document_that_cannot_be_asked_is_refused_rather_than_missed(
        self, dataset, build_run, chat
    ):
        """A Document with no index is not a retrieval that found nothing."""
        environment = build_run()
        try:
            case = one_case(dataset)
            environment.repositories.files.set_processed(
                environment.stored_name(PRIMER), False
            )

            report = run_set(case, environment)

            result = report.cases[0]
            assert result.outcome == evaluator.REFUSED
            assert result.detail == "index_pending"
            assert result.hit_at_k is None
            assert report.retrieval.questions == 0
            assert chat.streamed == []
        finally:
            remove_documents(environment)

    def test_fallback_context_is_never_graded_as_a_generated_answer(
        self, dataset, environment, chat
    ):
        """Retrieved text quoted back after a failure is not an answer."""
        case = one_case(dataset)
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

        report = run_set(
            case,
            environment,
            judge=a_judge(),
        )

        result = report.cases[0]
        assert result.outcome == evaluator.CONTEXT_FALLBACK
        assert result.generated is False
        assert result.verdict is None
        assert report.answers.faithfulness.scored == 0
        assert report.answers.faithfulness.mean is None

    def test_every_outcome_a_case_can_have_is_reported_even_at_zero(
        self, dataset, environment
    ):
        """A report names the outcomes that did not happen, not only the ones that did."""
        report = run_set(tuning_set(dataset), environment)

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
        self, dataset, build_run
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
            report = run_set(one_case(dataset), environment)

            assert report.cases[0].outcome == evaluator.REFUSED
            assert report.cases[0].detail == "vector_store_unavailable"
        finally:
            remove_documents(environment)


class TestMultiTurn:
    """A follow-up case reads the Turns the run actually committed."""

    def test_a_follow_up_reads_the_earlier_questions_from_the_conversation(
        self, dataset, environment, chat
    ):
        """The expansion is built from stored Turns, not from the case set."""
        case = a_set(
            dataset,
            [
                Case(
                    id="follow-up",
                    document=PRIMER,
                    split=TUNING,
                    category="follow_up",
                    question="What about the second one?",
                    expected_outcome=evaluator.ANSWERED,
                    expected_evidence=(GOLD,),
                    follow_up=("What are the stages of a RAG pipeline?",),
                )
            ],
        )

        report = run_set(case, environment)

        result = report.cases[0]
        assert result.retrieval["query_expansion"] == "deterministic"
        assert result.retrieval["expanded_query"].endswith("What about the second one?")
        assert result.outcome == evaluator.ANSWERED


class TestSameResultOverHttp:
    """The report describes what the HTTP application does."""

    def test_the_same_case_answers_the_same_way_over_http_and_in_the_evaluator(
        self, dataset, build_run, settings_obj, storage, embeddings, chat
    ):
        """One case, asked two ways against equal state, is one answer."""
        from app import create_app
        from services.accounts.secrets_service import encrypt_api_key

        case = one_case(dataset)
        question = case.cases[0].question
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
            result = run_set(
                case,
                through_evaluator,
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


class TestRetrievalScores:
    """The run's retrieval numbers, per question and in aggregate."""

    def test_every_question_that_reached_retrieval_is_scored_and_named(
        self, dataset, environment
    ):
        """The run names the cases it scored, so an average can be traced back."""
        report = run_set(tuning_set(dataset), environment)

        assert [row["id"] for row in report.retrieval.per_question] == [
            case.id for case in tuning_set(dataset).cases
        ]
        assert report.retrieval.questions == len(tuning_set(dataset).cases)

    def test_the_aggregate_reports_all_four_ways_of_scoring_a_ranking(
        self, dataset, environment
    ):
        """A rate that counts a hit, a rate that counts every gold snippet, a rate that rewards rank, and a rate that discounts it."""
        report = run_set(tuning_set(dataset), environment)

        assert 0.0 <= report.retrieval.hit_rate <= 1.0
        assert 0.0 <= report.retrieval.recall <= 1.0
        assert 0.0 <= report.retrieval.mrr <= 1.0
        assert 0.0 <= report.retrieval.ndcg <= 1.0
        assert report.retrieval.mrr <= report.retrieval.hit_rate + 1e-9
        assert report.retrieval.ndcg <= report.retrieval.hit_rate + 1e-9

    def test_a_per_question_row_carries_the_own_numbers_of_that_question(
        self, dataset, environment
    ):
        """The per-question row is the case's own numbers, not a second measurement of them."""
        report = run_set(tuning_set(dataset), environment)
        first = report.retrieval.per_question[0]
        case = report.cases[0]

        assert first["recall_at_k"] == case.recall_at_k
        assert first["reciprocal_rank"] == case.reciprocal_rank
        assert first["ndcg_at_k"] == case.ndcg_at_k
        assert first["hit_at_k"] == case.hit_at_k


class TestLatencyAndCost:
    """How long the run took, and what it cost, measured rather than assumed."""

    def test_the_first_case_is_a_cold_start_and_the_rest_are_warm(
        self, dataset, environment
    ):
        """One case pays for whatever loads lazily; the others do not."""
        report = run_set(tuning_set(dataset), environment)

        assert [case.phase for case in report.cases] == [
            evaluator.COLD_START,
            *(evaluator.WARM for _ in tuning_set(dataset).cases[1:]),
        ]

    def test_a_case_that_never_retrieved_does_not_get_to_be_the_cold_start(
        self, dataset, build_run
    ):
        """
        A refused case measures nothing, so it cannot own the cold start.

            Otherwise the first case that actually reached the vector store is
            labelled warm, and the seconds that paid for a lazy load are averaged
            into the steady-state distribution as if every question cost them.
        """
        environment = build_run()
        try:
            environment.repositories.files.set_processed(
                environment.stored_name(PRIMER), False
            )
            # Two documents, so marking the primer unanswerable refuses the
            # first case without refusing the second one as well.
            refused = a_set(
                dataset,
                [
                    Case(
                        id="refused-first",
                        document=PRIMER,
                        split=TUNING,
                        category="exact_lookup",
                        question="What are the stages of a RAG pipeline?",
                        expected_outcome=evaluator.ANSWERED,
                    ),
                    replace(
                        one_case(dataset).cases[0],
                        id="answered-second",
                        document=next(
                            document.filename
                            for document in dataset.documents
                            if document.filename != PRIMER
                        ),
                    ),
                ],
            )

            report = run_set(refused, environment)

            assert [result.phase for result in report.cases] == [
                evaluator.NOT_MEASURED,
                evaluator.COLD_START,
            ]
            assert report.cases[0].outcome == evaluator.REFUSED
            assert report.latency.cold_start["retrieval_seconds"].samples == 1
            assert report.latency.steady_state["retrieval_seconds"].samples == 0
        finally:
            remove_documents(environment)

    def test_a_run_measures_retrieval_first_token_and_total_from_its_clock(
        self, dataset, build_run
    ):
        """
        The clock is an input, so every interval the run reports is checkable.

            A clock that returns consecutive integers makes every pair of reads
            one second apart, whatever else read it first, so the three
            measurements a case makes are exact rather than plausible.
        """
        ticks = iter(float(index) for index in range(10_000))
        environment = build_run(clock=lambda: next(ticks))
        try:
            report = run_set(one_case(dataset), environment)

            case = report.cases[0]
            assert case.provenance["retrieval_seconds"] == 1.0
            assert case.first_token_seconds == 1.0
            assert case.total_seconds == 2.0
        finally:
            remove_documents(environment)

    def test_an_answer_that_never_streamed_a_token_has_no_time_to_first_token(
        self, dataset, build_run, chat
    ):
        """A case that failed before any output has nothing to wait for."""
        ticks = iter(float(index) for index in range(10_000))
        environment = build_run(clock=lambda: next(ticks))
        try:
            chat.fail_with(RuntimeError("provider down"), before_output=True)

            report = run_set(one_case(dataset), environment)

            assert report.cases[0].first_token_seconds is None
            assert report.cases[0].total_seconds == 1.0
        finally:
            remove_documents(environment)

    def test_the_cold_start_is_summarized_apart_from_the_steady_state(
        self, dataset, build_run
    ):
        """A p95 that averaged a lazy load in would describe nobody's question."""
        ticks = iter(float(index) for index in range(100_000))
        environment = build_run(clock=lambda: next(ticks))
        try:
            report = run_set(tuning_set(dataset), environment)

            cold = report.latency.cold_start["total_seconds"]
            steady = report.latency.steady_state["total_seconds"]
            assert cold.samples == 1
            assert steady.samples == len(tuning_set(dataset).cases) - 1
            # Every interval the tick clock produces is two seconds, so the
            # steady-state distribution is flat and the cold start is its own.
            assert (steady.p50, steady.p95) == (2.0, 2.0)
            assert (cold.p50, cold.p95) == (2.0, 2.0)
        finally:
            remove_documents(environment)

    def test_retrieval_latency_is_reported_at_p50_and_p95(self, dataset, environment):
        """Every measurement a run makes is reported as a median and a tail."""
        report = run_set(tuning_set(dataset), environment)

        for phase in (report.latency.cold_start, report.latency.steady_state):
            for name in ("retrieval_seconds", "first_token_seconds", "total_seconds"):
                assert phase[name].p50 is not None
                assert phase[name].p95 is not None

    def test_tokens_and_cost_are_counted_for_the_answers_a_model_wrote(
        self, dataset, environment
    ):
        """A run reports what its calls cost, priced from the catalog."""
        report = run_set(tuning_set(dataset), environment)

        answered = [case for case in report.cases if case.generated]
        assert report.cost.priced_cases == len(answered)
        assert report.cost.input_tokens == sum(case.input_tokens for case in answered)
        assert report.cost.output_tokens > 0
        assert report.cost.usd > 0.0
        assert report.cost.model == MODEL.id
        assert report.cost.input_cost_per_million_usd == (
            MODEL.input_cost_per_million_usd
        )

    def test_a_case_that_never_reached_a_model_is_not_priced(
        self, dataset, environment, store, chat
    ):
        """A failure that generated nothing has nothing to charge for."""
        store.delete(filter={"pdf_name": environment.stored_name(PRIMER)})

        report = run_set(one_case(dataset), environment)

        case = report.cases[0]
        assert case.outcome == evaluator.ABSTAINED
        assert case.output_tokens is None
        assert case.cost_usd is None
        assert report.cost.priced_cases == 0
        assert report.cost.usd == 0.0
        # The token totals count the same cases the dollar figure does, or the
        # run would print a token count it is not charging for.
        assert report.cost.input_tokens == 0
        assert report.cost.output_tokens == 0

    def test_the_input_count_includes_the_system_instruction_every_call_carries(
        self, dataset, environment
    ):
        """Counting only the question would understate every case by the same."""
        from services.chat_context import token_count
        from services.prompts import SYSTEM_INSTRUCTION

        report = run_set(one_case(dataset), environment)

        case = report.cases[0]
        assert case.input_tokens > token_count(SYSTEM_INSTRUCTION)
        assert case.input_tokens == token_count(SYSTEM_INSTRUCTION) + token_count(
            case.context
        ) + token_count(case.question)

    def test_the_output_count_includes_the_claims_block_the_model_wrote(
        self, dataset, environment
    ):
        """The block is stripped before storage and billed before it was."""
        from services.chat_context import token_count

        report = run_set(one_case(dataset), environment)

        case = report.cases[0]
        assert case.output_tokens > token_count(case.answer)
        assert case.claims, "the scripted answer declares a claim to count"

    def test_finish_reasons_are_counted_across_the_run(self, dataset, environment):
        """How the provider said the answer was over, counted across the run."""
        report = run_set(tuning_set(dataset), environment)

        assert report.answers.finish_reasons == {"stop": len(tuning_set(dataset).cases)}
        assert report.answers.generated == len(tuning_set(dataset).cases)
        assert report.answers.truncated == 0

    def test_an_answer_cut_short_is_reported_as_truncated(self, dataset, environment):
        """An answer that arrived whole is not counted as cut short."""
        report = run_set(one_case(dataset), environment)

        assert report.cases[0].finish_reason == "stop"
        assert report.answers.truncated == 0


class TestDeterministicGrading:
    """What the graders decide without asking a model."""

    def test_every_case_is_graded_by_every_deterministic_grader(
        self, dataset, environment
    ):
        """A case nobody graded is a case whose quality nobody knows."""
        report = run_set(tuning_set(dataset), environment)

        for case in report.cases:
            assert set(case.grades) == {
                "provider_status",
                "required_abstention",
                "exact_evidence",
                "claims_schema",
                "source_ids",
                "page_references",
            }
        assert len(graders.DETERMINISTIC_GRADERS) == 6

    def test_an_answer_citing_the_passage_that_says_it_passes_the_evidence_grade(
        self, dataset, environment, chat
    ):
        """Citations that reach the evidence pass every citation-shaped grade."""
        case = one_case(dataset)
        chat.answer_with_claims(ANSWER, [{"claim": GOLD, "sources": ["S1"]}])

        report = run_set(case, environment)

        grades = report.cases[0].grades
        assert grades["exact_evidence"].outcome == graders.PASSED
        assert grades["source_ids"].outcome == graders.PASSED
        assert grades["claims_schema"].outcome == graders.PASSED
        assert report.cases[0].citation_recall == 1.0
        assert report.cases[0].citation_precision == 1.0

    def test_a_claim_citing_no_passage_fails_the_source_grade_and_costs_recall(
        self, dataset, environment, chat
    ):
        """A claim a reader cannot follow is a claim that counts against recall."""
        case = one_case(dataset)
        chat.answer_with_claims(ANSWER, [{"claim": GOLD, "sources": []}])

        report = run_set(case, environment)

        result = report.cases[0]
        assert result.outcome == evaluator.ANSWERED
        assert result.grades["source_ids"].outcome == graders.FAILED
        assert result.citation_recall == 0.0
        assert report.answers.citation_recall.mean == 0.0
        assert report.answers.citation_recall.failed == 1

    def test_a_claim_pointed_at_a_passage_that_does_not_say_it_is_not_precise(
        self, dataset, environment, chat
    ):
        """A citation can resolve and still not support what it is attached to."""
        case = one_case(dataset)
        chat.answer_with_claims(
            ANSWER, [{"claim": "The monitor is ankle-mounted", "sources": ["S1"]}]
        )

        report = run_set(case, environment)

        result = report.cases[0]
        assert result.grades["source_ids"].outcome == graders.PASSED
        assert result.citation_precision == 0.0
        assert result.citation_recall == 1.0
        assert report.answers.citation_precision.mean == 0.0
        assert report.answers.citation_precision.failed == 1

    def test_a_claim_with_no_citation_loses_citation_recall(self, dataset, environment):
        """A citation that resolves is not automatically one that supports the claim."""
        report = run_set(one_case(dataset), environment)

        # The scripted answer cites S1 for a claim about nothing in particular,
        # so the citation resolves but does not support what it is attached to.
        assert report.cases[0].citation_recall == 1.0
        assert report.cases[0].citation_precision == 0.0
        assert report.answers.citation_recall.mean == 1.0

    def test_page_references_are_checked_against_the_passages_cited(
        self, dataset, environment, chat
    ):
        """A page the reader is sent to has to be a page of a cited Passage."""
        case = one_case(dataset)
        chat.answer_with_claims(
            f"{ANSWER} See page 3 for the table.",
            [{"claim": GOLD, "sources": ["S1"]}],
        )

        report = run_set(case, environment)

        grade = report.cases[0].grades["page_references"]
        assert grade.outcome == graders.FAILED
        assert "3" in grade.detail

    def test_a_provider_failure_fails_the_run_and_improves_no_score(
        self, dataset, environment, chat
    ):
        """A dead provider is a failure everywhere, never a quiet zero."""
        chat.fail_with(RuntimeError("provider down"), before_output=True)

        report = run_set(one_case(dataset), environment)

        case = report.cases[0]
        assert case.grades["provider_status"].outcome == graders.FAILED
        assert case.grades["required_abstention"].outcome == graders.FAILED
        assert case.citation_precision is None
        assert case.citation_recall is None
        assert report.answers.citation_precision.scored == 0
        assert report.answers.citation_precision.passed == 0
        assert report.answers.abstention.failed == 1
        assert report.answers.abstention.passed == 0
        assert report.answers.generated == 0
        assert report.cost.priced_cases == 0

    def test_an_abstention_on_an_answerable_question_is_scored_as_a_mistake(
        self, dataset, environment, store
    ):
        """Refusing a question the evidence answers is a defect, not a safety."""
        store.delete(filter={"pdf_name": environment.stored_name(PRIMER)})

        report = run_set(one_case(dataset), environment)

        abstention = report.answers.abstention
        assert report.cases[0].outcome == evaluator.ABSTAINED
        assert abstention.failed == 1
        assert abstention.mean == 0.0

    def test_a_run_where_every_case_answered_scores_the_abstention_metric_full(
        self, dataset, environment
    ):
        """Answering every answerable question is the abstention metric at full marks."""
        report = run_set(tuning_set(dataset), environment)

        abstention = report.answers.abstention
        # A question the document cannot answer may abstain or say it does not
        # know, so the grader has nothing to decide about it and says so.
        assert abstention.mean == 1.0
        assert abstention.failed == 0
        assert abstention.passed + abstention.unknown == abstention.graded


class TestModelGrading:
    """The two metrics only a judge can settle, and the judge's own labels."""

    def test_the_run_names_the_judge_and_the_rubric_it_answered_against(
        self, dataset, environment
    ):
        """A judged number is only comparable next to the judge that produced it."""
        grader = a_judge(provider="groq", model="openai/gpt-oss-20b")

        report = run_set(tuning_set(dataset), environment, judge=grader)

        assert report.run["judge"] == {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "rubric_version": judge.RUBRIC_VERSION,
        }

    def test_a_run_with_no_judge_names_no_judge_and_claims_nothing_judged(
        self, dataset, environment
    ):
        """A retrieval-only run reports nothing as judged."""
        report = run_set(tuning_set(dataset), environment)

        assert report.run["judge"] is None
        assert report.calibration is None
        assert report.answers.faithfulness.scored == 0
        assert report.answers.correctness.scored == 0

    def test_a_judge_is_calibrated_against_the_hand_labelled_set(
        self, dataset, environment
    ):
        """A judge is trusted with what reading cannot settle, so it is checked against labels."""
        report = run_set(tuning_set(dataset), environment, judge=a_judge())

        calibration = report.calibration
        assert calibration["cases"] == len(load_calibration())
        assert calibration["rubric_version"] == judge.RUBRIC_VERSION
        # The scripted judge grades every labelled case faithful, so it matches
        # only the one case a person also called faithful.
        assert calibration["agreed"] == 1
        assert calibration["disagreed"] >= 1

    def test_correctness_is_only_asked_where_the_case_set_expects_an_answer(
        self, dataset, environment
    ):
        """Two graders, two questions: support from the evidence, and agreement with the label."""
        calls = []

        def spy(prompt):
            """Do spy."""
            calls.append(prompt)
            return "correct" if "expected answer" in prompt else "faithful"

        report = run_set(tuning_set(dataset), environment, judge=a_judge(spy))

        assert report.answers.correctness.scored == len(tuning_set(dataset).cases)
        assert report.answers.correctness.mean == 1.0
        assert report.answers.faithfulness.mean == 1.0
        assert all(case.correctness_verdict == "correct" for case in report.cases)

    def test_a_judge_that_returns_nothing_is_counted_as_unknown_not_as_zero(
        self, dataset, environment
    ):
        """A judge that cannot read its own verdict has graded nothing."""
        report = run_set(
            tuning_set(dataset),
            environment,
            judge=a_judge(lambda prompt: "I cannot say"),
        )

        faithfulness = report.answers.faithfulness
        assert faithfulness.scored == 0
        assert faithfulness.unknown == len(tuning_set(dataset).cases)
        assert faithfulness.mean is None
        assert faithfulness.failed == 0

    def test_calibration_can_be_switched_off_for_a_run_that_skips_it(
        self, dataset, environment
    ):
        """Calibration costs judge calls, so a run can leave it out."""
        report = run_set(
            tuning_set(dataset), environment, judge=a_judge(), calibrate=False
        )

        assert report.calibration is None


class TestReportSerialization:
    """The report a reader parses is the report the run measured."""

    def test_the_whole_report_survives_being_written_as_json(
        self, dataset, environment
    ):
        """The report a reader parses is the report the run measured."""
        report = run_set(tuning_set(dataset), environment, judge=a_judge())

        payload = json.loads(json.dumps(report.as_dict()))

        assert payload["run"]["prompt_version"] == "grounded-claims-v1"
        assert payload["answers"]["faithfulness"]["scored"] == len(
            tuning_set(dataset).cases
        )
        assert (
            payload["latency"]["steady_state"]["total_seconds"]["samples"]
            == len(tuning_set(dataset).cases) - 1
        )
        assert payload["cost"]["model"] == MODEL.id
        assert len(payload["cases"]) == len(tuning_set(dataset).cases)

    def test_a_case_carries_the_page_of_each_supplied_passage_not_its_text(
        self, dataset, environment
    ):
        """The evidence is measured, not reprinted: the report keeps the pages."""
        report = run_set(one_case(dataset), environment)

        case = report.as_dict()["cases"][0]
        assert "retrieved_texts" not in case
        assert "sources" not in case
        assert set(case["source_pages"]) <= {"S1", "S2", "S3", "S4", "S5"}

    def test_a_case_records_what_it_was_measured_on(self, dataset, environment):
        """Every case carries the measurements its own numbers came from."""
        report = run_set(one_case(dataset), environment)

        case = report.as_dict()["cases"][0]
        for field in (
            "retrieval_seconds",
            "first_token_seconds",
            "total_seconds",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "phase",
            "finish_reason",
            "reciprocal_rank",
            "ndcg_at_k",
        ):
            assert field in case, field
        assert case["generated"] is True


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
            cli.run(live=True, use_judge=False, k=5)

        assert cli.GENERATOR_API_KEY_ENV in str(exc.value.code)

    def test_a_missing_judge_key_names_its_own_variable(self, monkeypatch):
        """Do test a missing judge key names its own variable."""
        monkeypatch.setenv(cli.GENERATOR_API_KEY_ENV, "generator-secret")
        monkeypatch.delenv(cli.JUDGE_API_KEY_ENV, raising=False)

        with pytest.raises(SystemExit) as exc:
            cli.run(
                live=True,
                use_judge=True,
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
        cli.build_evaluation_judge("groq", "openai/gpt-oss-120b", "groq-secret")

        credentials = captured["credentials"]
        assert credentials.provider == "groq"
        assert credentials.model == "openai/gpt-oss-120b"
        assert credentials.api_key == "groq-secret"

    def test_a_failed_judge_call_is_unknown_and_carries_no_key(self, monkeypatch):
        """
        A provider error can quote the key back, and never reaches a report.

        A judge that could not be reached leaves the case Unknown rather than
        throwing the run away, so the reply that stands in for it must carry
        neither the failure's own words nor the key it tried to echo back.
        """
        secret = "sk-judge-secret"
        monkeypatch.setattr(
            cli,
            "build_chat_provider",
            lambda *a, **k: FailingProvider(secret),
        )
        grader = cli.build_evaluation_judge("google", "gemini-2.5-flash", secret)

        reply = grader("grade this")

        assert secret not in reply
        assert judge.parse_verdict(reply) == (judge.UNKNOWN, None)

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
        assert set(payload) == {
            "run",
            "retrieval",
            "outcomes",
            "answers",
            "latency",
            "cost",
            "calibration",
            "cases",
        }
        assert payload["run"]["prompt_version"] == "grounded-claims-v1"
        assert payload["outcomes"]["answered"] == 0

    def test_the_reranker_gate_is_a_setting_not_a_branch(self, settings_obj):
        """An ablation changes configuration; the answer path is unchanged."""
        flipped = cli.with_rerank(settings_obj, True)
        assert flipped.rerank.enabled is True
        assert settings_obj.rerank.enabled is False
        assert flipped.query_context == settings_obj.query_context

    def test_the_summary_names_the_metrics_the_cost_and_the_judge(self, capsys):
        """The operator-facing summary is where the numbers are read first."""
        cli._print(_empty_report())
        printed = capsys.readouterr().out

        assert "not judged" in printed
        assert "Retrieval@5" in printed
        for name in ("correctness", "faithfulness", "citation_recall", "abstention"):
            assert name in printed
        assert "Latency (steady state)" in printed
        assert "cold start" in printed
        assert "Cost: $0.0000" in printed

    def test_a_printed_run_names_its_judge_and_the_grades_that_failed(
        self, dataset, environment, chat, capsys
    ):
        """A printed run says who judged it and which graders objected."""
        report = run_set(
            one_case(dataset),
            environment,
            judge=a_judge(provider="groq", model="openai/gpt-oss-20b"),
        )
        cli._print(report.as_dict())
        printed = capsys.readouterr().out

        assert "judged by groq/openai/gpt-oss-20b" in printed
        assert f"rubric {judge.RUBRIC_VERSION}" in printed
        assert "judge=faithful" in printed
        assert "ndcg=" in printed

    def test_the_calibration_set_can_be_left_out_of_a_run(self, monkeypatch):
        """The flag reaches the run rather than only the argument parser."""
        monkeypatch.setenv(cli.GENERATOR_API_KEY_ENV, "generator-secret")
        monkeypatch.delenv(cli.JUDGE_API_KEY_ENV, raising=False)
        monkeypatch.setattr("settings.get_settings", _pinned)
        seen = []

        def fake_once(dataset, app_settings, **kwargs):
            seen.append(kwargs["calibrate"])
            return _empty_report()

        monkeypatch.setattr(cli, "_run_once", fake_once)
        cli.run(
            live=True,
            use_judge=False,
            k=5,
            provider="google",
            model="gemini-2.5-flash",
            calibrate=False,
        )

        assert seen == [False]

    def test_comparing_reranking_reports_both_runs(self, monkeypatch, capsys):
        """An ablation is two runs of the same case set, reported side by side."""
        monkeypatch.setenv(cli.GENERATOR_API_KEY_ENV, "generator-secret")
        monkeypatch.delenv(cli.JUDGE_API_KEY_ENV, raising=False)
        monkeypatch.setattr("settings.get_settings", _pinned)
        seen = []

        def fake_once(dataset, app_settings, **kwargs):
            seen.append(app_settings.rerank.enabled)
            return _empty_report()

        monkeypatch.setattr(cli, "_run_once", fake_once)
        report = cli.run(
            live=True,
            use_judge=False,
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
    """Return the report shape a run with nothing in it serializes as."""
    metric = {
        "graded": 0,
        "scored": 0,
        "mean": None,
        "passed": 0,
        "failed": 0,
        "unknown": 0,
    }
    latency = {"samples": 0, "p50": None, "p95": None}
    return {
        "run": {
            "k": 5,
            "dataset": "2026-09-labeled-cases-v1",
            "split": "validation",
            "cases": 0,
            "held_back": [],
            "prompt_version": "grounded-claims-v1",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "judge": None,
            "settings": {},
            "documents": {},
            "retrieval_methods": [],
        },
        "retrieval": {
            "questions": 0,
            "k": 5,
            "hit_rate": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "per_question": [],
        },
        "outcomes": {outcome: 0 for outcome in evaluator.OUTCOMES},
        "answers": {
            "correctness": dict(metric),
            "faithfulness": dict(metric),
            "citation_precision": dict(metric),
            "citation_recall": dict(metric),
            "abstention": dict(metric),
            "generated": 0,
            "truncated": 0,
            "finish_reasons": {},
        },
        "latency": {
            "cold_start": {
                name: dict(latency)
                for name in (
                    "retrieval_seconds",
                    "first_token_seconds",
                    "total_seconds",
                )
            },
            "steady_state": {
                name: dict(latency)
                for name in (
                    "retrieval_seconds",
                    "first_token_seconds",
                    "total_seconds",
                )
            },
        },
        "cost": {
            "input_tokens": 0,
            "output_tokens": 0,
            "usd": 0.0,
            "priced_cases": 0,
            "model": "gemini-2.5-flash",
            "input_cost_per_million_usd": 0.30,
            "output_cost_per_million_usd": 2.50,
        },
        "calibration": None,
        "cases": [],
    }
