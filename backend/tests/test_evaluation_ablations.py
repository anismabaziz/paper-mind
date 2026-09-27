"""
Measuring one configuration against the same labeled questions.

The ablations here answer one question at a time — which retrieval method, how
deep a candidate pool, which chunk size — and every answer is measured on the
same cases through the same retrieval service, so the columns of a comparison
differ by the thing being compared and by nothing else. The tests assert that:
the same questions are asked, a difference is attributed to the questions that
caused it, and a change to the stored vectors is measured on an index built for
it rather than on the index that was already there.
"""

import pytest

from evaluation import ablations, experiments
from evaluation.dataset import TUNING, load_dataset
from evaluation.harness import SAMPLE_DOCS_DIR, remove_documents
from tests.evaluation_support import (
    InMemoryVectorStore,
    ScriptedChatFactory,
    build_environment_for,
)


class SparseReversesStore(InMemoryVectorStore):
    """A store that ranks its sparse matches worst-first, and dense ones as usual."""

    def query(self, vector, top_k, include_metadata=True, filter=None, **kwargs):
        """Answer the query, with the sparse ordering and its scores turned around."""
        result = super().query(
            vector, top_k, include_metadata=include_metadata, filter=filter, **kwargs
        )
        if kwargs.get("method") == "sparse":
            matches = []
            for rank, match in enumerate(reversed(result["matches"])):
                # The shaping step re-sorts by score, so the reversal has to
                # carry its scores with it.
                matches.append({**match, "score": 1.0 - rank / 100})
            result["matches"] = matches
        return result


@pytest.fixture
def settings_obj():
    """Return the pinned settings the suite runs with."""
    import settings as settings_module

    return settings_module.get_settings()


@pytest.fixture
def dataset():
    """Return the committed labeled case set."""
    return load_dataset(docs_dir=SAMPLE_DOCS_DIR)


@pytest.fixture
def cases(dataset):
    """Return the handful of tuning cases a run can afford to ask."""
    return dataset.cases_for(TUNING)[:2]


@pytest.fixture
def chat():
    """Return a provider factory; retrieval does not call it."""
    return ScriptedChatFactory()


def build(dataset, settings_obj, chat, tmp_path, store=None, prefix=""):
    """Index the set's documents into a working application."""
    return build_environment_for(
        dataset,
        settings_obj,
        tmp_path / f"storage-{prefix or 'default'}",
        store=store,
        prefix=prefix,
        chat_factory=chat,
    )


@pytest.fixture
def environment(dataset, settings_obj, chat, tmp_path):
    """Return a working application, removed after the test."""
    built = build(dataset, settings_obj, chat, tmp_path)
    yield built
    remove_documents(built)


@pytest.fixture
def reversed_environment(dataset, settings_obj, chat, tmp_path):
    """Return a working application whose sparse ranking is the wrong way round."""
    built = build(
        dataset, settings_obj, chat, tmp_path, SparseReversesStore(), prefix="reversed"
    )
    yield built
    remove_documents(built)


def measure(environment, dataset, cases, id):
    """Measure one declared experiment on the given cases."""
    return ablations.measure(
        environment, dataset, cases, experiments.experiment(id), k=5
    )


class TestWhatOneExperimentMeasures:
    """A result describes one configuration, measured on the cases it names."""

    def test_a_result_reports_retrieval_over_the_questions_it_asked(
        self, environment, dataset, cases
    ):
        """A result aggregates the four retrieval numbers over the cases it asked."""
        result = measure(environment, dataset, cases, "baseline")

        assert [row.id for row in result.cases] == [case.id for case in cases]
        assert result.retrieval.questions == len(cases)
        assert 0.0 <= result.retrieval.ndcg <= 1.0

    def test_a_result_records_the_method_the_service_reported(
        self, environment, dataset, cases
    ):
        """A dense variant must have asked dense retrieval, not fallen back to another method."""
        result = measure(environment, dataset, cases, "dense-only")

        assert {row.method for row in result.cases} == {"dense"}

    def test_a_result_reports_how_long_retrieval_took(
        self, environment, dataset, cases
    ):
        """The cost of a variant is in the seconds retrieval took, not in a guess."""
        result = measure(environment, dataset, cases, "baseline")

        assert result.latency["retrieval_seconds"].samples == len(cases)
        assert result.latency["retrieval_seconds"].p50 is not None

    def test_a_result_says_it_measured_no_answers(self, environment, dataset, cases):
        """Nothing asked a model, so the report must not leave answer fields looking empty."""
        result = measure(environment, dataset, cases, "baseline")

        assert result.as_dict()["evidence"]["answers"]["measured"] is False
        assert result.as_dict()["evidence"]["answers"]["reason"]


class TestDifferencesAreAttributed:
    """A difference between two columns is traced to the questions behind it."""

    def test_a_variant_is_measured_on_the_same_questions(
        self, environment, dataset, cases
    ):
        """A difference between two columns is only about configuration when the cases match."""
        baseline = measure(environment, dataset, cases, "baseline")
        dense = measure(environment, dataset, cases, "dense-only")

        assert [row.id for row in dense.cases] == [row.id for row in baseline.cases]

    def test_a_regression_names_the_question_behind_it(
        self, environment, reversed_environment, dataset, cases
    ):
        """A worse ranking is reported against the question that lost its place."""
        baseline = measure(environment, dataset, cases, "baseline")
        sparse = measure(reversed_environment, dataset, cases, "sparse-only")

        comparison = ablations.compare(baseline, sparse)

        assert comparison.regressions, "the reversed ranking should lose hits"
        assert all(
            row["id"] in {case.id for case in cases} for row in comparison.regressions
        )
        assert all(row["delta"]["ndcg_at_k"] < 0 for row in comparison.regressions)
        assert all(
            row["variant"]["hit_at_k"] <= row["baseline"]["hit_at_k"]
            for row in comparison.regressions
        )

    def test_an_improvement_is_named_too(
        self, environment, reversed_environment, dataset, cases
    ):
        """A better ranking is traced the same way a worse one is."""
        baseline = measure(reversed_environment, dataset, cases, "sparse-only")
        variant = measure(environment, dataset, cases, "dense-only")

        comparison = ablations.compare(baseline, variant)

        assert comparison.improvements
        assert all(row["delta"]["ndcg_at_k"] > 0 for row in comparison.improvements)
        assert all(
            row["variant"]["hit_at_k"] >= row["baseline"]["hit_at_k"]
            for row in comparison.improvements
        )

    def test_a_comparison_reports_the_change_in_the_aggregate(
        self, environment, dataset, cases
    ):
        """The summary of a comparison and the per-question rows must agree."""
        baseline = measure(environment, dataset, cases, "baseline")
        variant = measure(environment, dataset, cases, "dense-only")

        comparison = ablations.compare(baseline, variant)

        assert comparison.delta["ndcg"] == pytest.approx(
            variant.retrieval.ndcg - baseline.retrieval.ndcg
        )
        assert comparison.latency_delta["retrieval_seconds"]["p50"] == pytest.approx(
            variant.latency["retrieval_seconds"].p50
            - baseline.latency["retrieval_seconds"].p50
        )


class TestTheExperimentsAreRunInOrder:
    """A run reuses an index until something about the index changes."""

    def test_a_chunking_experiment_is_measured_on_an_index_built_for_it(
        self, environment, dataset, cases
    ):
        """Different chunks are different vectors, so the index is rebuilt for them."""
        asked_for = []

        def environment_for(candidate):
            asked_for.append(candidate.id)
            return environment

        ablations.run(
            [
                experiments.experiment(id)
                for id in ("baseline", "dense-only", "chunk-256-25", "hybrid-only")
            ],
            cases,
            environment_for,
            dataset=dataset,
            k=5,
        )

        assert asked_for == ["baseline", "chunk-256-25"]

    def test_every_environment_a_run_built_is_released(
        self, environment, dataset, cases
    ):
        """A run that indexed real documents has to take them with it when it ends."""
        released = []

        ablations.run(
            [experiments.experiment(id) for id in ("baseline", "chunk-256-25")],
            cases,
            lambda candidate: environment,
            dataset=dataset,
            k=5,
            close=released.append,
        )

        assert released == [environment, environment]

    def test_a_run_measures_every_experiment_it_was_given(
        self, environment, dataset, cases
    ):
        """A run returns one result per experiment, in the order asked."""
        results = ablations.run(
            [experiments.experiment(id) for id in ("baseline", "sparse-only")],
            cases,
            lambda candidate: environment,
            dataset=dataset,
            k=5,
        )

        assert [result.experiment for result in results] == ["baseline", "sparse-only"]
