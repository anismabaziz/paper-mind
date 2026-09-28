"""
A checked-in report a reviewer can read without running anything.

A number in isolation is a claim. What makes it checkable is everything around
it: the revision it was measured on, the version of the case set, the hashes of
the documents, the models and their revisions, the prompts, the settings, and
the environment. Those travel together in a manifest, and the manifest is
written where a reviewer can find it, next to the numbers it describes.

The other half of the contract is what happens when a report is produced a
second time. A second run has to say plainly whether it reproduced the first
one, and when it did not, which model revision moved — because a report whose
numbers changed for an unnamed reason is not a comparison.
"""

import json

import pytest

from evaluation import ablations, cli, experiments, evaluator, report
from evaluation.dataset import TUNING, load_dataset
from evaluation.harness import SAMPLE_DOCS_DIR, remove_documents
from services.accounts.chat_settings_service import model_for
from tests.evaluation_support import (
    ScriptedChatFactory,
    build_environment_for,
    timeout_error,
)

MODEL = model_for("google", "gemini-2.5-flash")

REVISION = {"revision": "0" * 40, "dirty": False, "subject": "feat: something"}


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
def chat():
    """Return a provider factory that answers the next question with a sentence."""
    factory = ScriptedChatFactory()
    factory.answer("A RAG pipeline has five stages.")
    return factory


@pytest.fixture
def environment(dataset, settings_obj, chat, tmp_path):
    """Return a working application, removed after the test."""
    built = build_environment_for(
        dataset, settings_obj, tmp_path / "storage", chat_factory=chat
    )
    yield built
    remove_documents(built)


@pytest.fixture
def cases(dataset):
    """Return the tuning cases a run over doubles can afford to ask."""
    return dataset.cases_for(TUNING)[:2]


def a_result(id="baseline", **changes):
    """Return a measured result shaped the way a report stores one."""
    return {
        "experiment": id,
        "family": experiments.experiment(id).family,
        "dataset": "2026-09-labeled-cases-v1",
        "split": TUNING,
        "retrieval": {
            "questions": 2,
            "k": 5,
            "hit_rate": 1.0,
            "recall": 0.75,
            "mrr": 0.8,
            "ndcg": 0.7,
            "per_question": [
                {"id": "primer-stages", "hit_at_k": True, "ndcg_at_k": 1.0},
                {"id": "primer-hit-rate", "hit_at_k": False, "ndcg_at_k": 0.0},
            ],
        },
        "latency": {
            "retrieval_seconds": {"samples": 2, "p50": 0.01, "p95": 0.02},
        },
        "evidence": report.evidence_for_retrieval(2),
        "cases": [
            {
                "id": "primer-stages",
                "category": "exact_lookup",
                "method": "hybrid",
                "hit_at_k": True,
                "ndcg_at_k": 1.0,
            },
            {
                "id": "primer-hit-rate",
                "category": "exact_lookup",
                "method": "hybrid",
                "hit_at_k": False,
                "ndcg_at_k": 0.0,
            },
        ],
        **changes,
    }


def a_manifest(dataset, settings_obj, results, **changes):
    """Return a manifest built from the same inputs a real run would use."""
    arguments = {
        "dataset": dataset,
        "settings": settings_obj,
        "results": results,
        "revision": REVISION,
    }
    return report.build_manifest(**{**arguments, **changes})


class TestWhatTheManifestRecords:
    """Everything a number cannot be checked without."""

    def test_the_manifest_names_the_revision_the_set_and_the_split(
        self, dataset, settings_obj
    ):
        """A number is comparable to another only if these match."""
        manifest = a_manifest(dataset, settings_obj, [a_result()])

        assert manifest["revision"] == REVISION
        assert manifest["dataset"]["version"] == dataset.version
        assert manifest["dataset"]["split"] == TUNING
        assert manifest["dataset"]["held_back"] == []
        assert manifest["dataset"]["cases"] == 2
        assert manifest["dataset"]["held_back"] == []

    def test_the_manifest_pins_every_document_by_hash(self, dataset, settings_obj):
        """A regenerated document under the same name is a different document."""
        manifest = a_manifest(dataset, settings_obj, [a_result()])

        pinned = {document["filename"]: document for document in manifest["documents"]}
        for document in dataset.documents:
            assert pinned[document.filename]["sha256"] == document.sha256
            assert pinned[document.filename]["license"] == document.license

    def test_the_manifest_names_the_models_and_their_revisions(
        self, dataset, settings_obj
    ):
        """A model revision is the one thing a checkout cannot pin."""
        settings_obj = settings_obj.model_copy(
            update={
                "embedding": settings_obj.embedding.model_copy(
                    update={"revision": "aaa111"}
                ),
                "rerank": settings_obj.rerank.model_copy(update={"revision": "bbb222"}),
            }
        )

        manifest = a_manifest(dataset, settings_obj, [a_result()])

        assert manifest["models"]["embedding"] == {
            "id": settings_obj.embedding.embedding_model,
            "revision": "aaa111",
        }
        assert manifest["models"]["reranker"]["revision"] == "bbb222"

    def test_the_manifest_records_the_prompt_and_the_settings_that_ran(
        self, dataset, settings_obj
    ):
        """The prompt and the settings decide the answer as much as the code does."""
        manifest = a_manifest(dataset, settings_obj, [a_result()])

        assert manifest["prompts"]["citation"]
        assert manifest["settings"]["chunk_size_tokens"] == (
            settings_obj.chunking.chunk_size_tokens
        )
        assert manifest["experiments"][0]["id"] == "baseline"

    def test_the_manifest_records_the_environment_it_ran_in(
        self, dataset, settings_obj
    ):
        """A retrieval implementation change moves the numbers without a code change."""
        manifest = a_manifest(dataset, settings_obj, [a_result()])

        assert manifest["environment"]["python"].startswith("3.")
        assert manifest["environment"]["retrieval"]["rrf_k"] > 0

    def test_the_environment_record_is_a_list_not_a_dump_of_the_machine(
        self, dataset, settings_obj
    ):
        """A committed report must never carry a key."""
        manifest = a_manifest(dataset, settings_obj, [a_result()])

        written = json.dumps(manifest)
        assert "PAPERMIND_EVAL_GENERATOR_API_KEY" not in written
        assert "api_key" not in written


def an_answer_run(dataset, environment, cases, judge=None):
    """Return the report shape a live run hands to the report writer."""
    from dataclasses import replace

    return evaluator.evaluate(
        replace(dataset, cases=tuple(cases)),
        environment,
        MODEL,
        api_key="not-a-real-key",
        judge=judge,
        split=TUNING,
    ).as_dict()


def a_judge(reply="faithful"):
    """Return a judge with the identity a run records beside its scores."""
    from evaluation.judge import Judge, JudgeSettings

    return Judge(
        settings=JudgeSettings(provider="google", model="gemini-2.5-flash"),
        grade=lambda prompt: reply,
    )


class TestAnAnswerPathReport:
    """A report of the answer path is read for what the answers said."""

    def test_a_judged_run_reports_its_judge_as_measured(
        self, dataset, environment, cases
    ):
        """A run that graded with a judge is not reported as unjudged."""
        result = report.evidence_from_run(
            an_answer_run(dataset, environment, cases, a_judge())
        )

        assert result["model_graded"]["measured"] is True
        assert result["model_graded"]["verdicts"] > 0

    def test_a_calibration_that_decided_nothing_is_not_calibration(
        self, dataset, environment, cases
    ):
        """Agreement over no decided cases is not a number."""
        result = report.evidence_from_run(
            an_answer_run(dataset, environment, cases, a_judge("I cannot say"))
        )

        assert result["human_calibration"]["measured"] is True
        assert result["human_calibration"]["agreement"] is None
        assert result["human_calibration"]["decided"] == 0

    def test_the_summary_of_an_answer_run_shows_the_answer_metrics(
        self, dataset, settings_obj, environment, cases
    ):
        """A report a reviewer opens for answer quality has to show them."""
        run_record = an_answer_run(dataset, environment, cases)
        results = [cli.result_of_run(run_record)]

        summary = report.render(
            report.build_manifest(
                dataset=dataset,
                settings=settings_obj,
                results=results,
                run_records=[run_record],
            ),
            results,
        )

        assert "## Answers" in summary
        assert "Correctness" in summary
        assert "Abstention" in summary


class TestAnAnswerPathRun:
    """A run that asked a model says which of its numbers a model decided."""

    def test_a_run_without_a_judge_reports_no_judged_metrics(
        self, dataset, environment, cases
    ):
        """Nothing is reported as judged that was not judged."""
        from dataclasses import replace

        run = evaluator.evaluate(
            replace(dataset, cases=tuple(cases)), environment, MODEL, split=TUNING
        )

        evidence = report.evidence_from_run(run.as_dict())

        assert evidence["model_graded"] == {
            "measured": False,
            "reason": "no judge was configured for this run",
        }
        assert evidence["human_calibration"]["measured"] is False
        assert evidence["deterministic"]["graded"] > 0
        assert evidence["tokens"]["measured"] is True

    def test_a_provider_failure_is_reported_as_a_failure_not_as_a_number(
        self, dataset, environment, cases
    ):
        """Fallback text cannot improve a quality score."""
        from dataclasses import replace

        environment_chat = environment
        environment_chat.chat_provider_factory.fail_with(timeout_error())
        run = evaluator.evaluate(
            replace(dataset, cases=tuple(cases)),
            environment_chat,
            MODEL,
            split=TUNING,
        )

        evidence = report.evidence_from_run(run.as_dict())

        assert evidence["provider_failures"] == {"measured": True, "failures": 2}
        assert evidence["answers"]["measured"] is False


class TestWritingAReport:
    """A report is a directory a reviewer opens."""

    def test_the_report_directory_holds_the_manifest_the_results_and_a_summary(
        self, dataset, settings_obj, tmp_path
    ):
        """A report is a directory a reviewer opens."""
        results = [a_result("baseline"), a_result("dense-only")]
        manifest = a_manifest(dataset, settings_obj, results)

        report.write_report(tmp_path / "reports" / "baseline-v1", manifest, results)

        written = tmp_path / "reports" / "baseline-v1"
        assert json.loads((written / "manifest.json").read_text()) == manifest
        assert sorted(path.name for path in (written / "results").iterdir()) == [
            "baseline.json",
            "dense-only.json",
        ]

    def test_the_summary_shows_each_experiment_and_the_questions_that_moved(
        self, dataset, settings_obj, tmp_path
    ):
        """A mean that moved is a question, not an answer."""
        results = [
            a_result(
                "dense-only",
                comparison={
                    "baseline": "baseline",
                    "variant": "dense-only",
                    "delta": {"ndcg": -0.2},
                    "latency_delta": {
                        "retrieval_seconds": {"p50": 0.001, "p95": 0.002}
                    },
                    "regressions": [
                        {
                            "id": "primer-hit-rate",
                            "baseline": {"ndcg_at_k": 1.0},
                            "variant": {"ndcg_at_k": 0.0},
                            "delta": {"ndcg_at_k": -1.0},
                        }
                    ],
                    "improvements": [],
                },
            )
        ]
        manifest = a_manifest(dataset, settings_obj, results)

        summary = report.render(manifest, results)

        assert "dense-only" in summary
        assert "primer-hit-rate" in summary
        assert manifest["dataset"]["version"] in summary

    def test_the_summary_names_the_cases_a_reported_run_holds_back(
        self, dataset, settings_obj
    ):
        """A report that leaves cases out says which, rather than how many."""
        results = [
            a_result(held_back=["eval-failure-citation", "notes-failure-provider"])
        ]

        summary = report.render(a_manifest(dataset, settings_obj, results), results)

        assert "held back" in summary
        assert "eval-failure-citation" in summary
        assert "notes-failure-provider" in summary

    def test_the_summary_of_the_tuning_run_says_it_is_where_values_were_chosen(
        self, dataset, settings_obj
    ):
        """A number from the tuning half is a decision, and says so."""
        results = [a_result()]

        summary = report.render(a_manifest(dataset, settings_obj, results), results)

        assert "chosen on" in summary
        assert "tuning half" in summary

    def test_the_summary_says_what_the_run_did_not_measure(self, dataset, settings_obj):
        """An empty cell would read as a measurement that found nothing."""
        summary = report.render(
            a_manifest(dataset, settings_obj, [a_result()]), [a_result()]
        )

        assert "not measured" in summary
        assert "cost" in summary


class TestRunningItAgain:
    """A second run either reproduces the first one or says what moved."""

    def test_the_same_measurements_reproduce_the_manifest(
        self, dataset, settings_obj, tmp_path
    ):
        """Timings move every run, so only the measurements are digested."""
        first = a_manifest(dataset, settings_obj, [a_result()])
        # Timings are a property of the machine and the hour, so they move
        # without the measurement having moved.
        again = a_manifest(
            dataset,
            settings_obj,
            [a_result(latency={"retrieval_seconds": {"samples": 2, "p50": 0.02}})],
        )

        report.write_report(tmp_path / "first", first, [a_result()])
        report.write_report(tmp_path / "again", again, [a_result()])

        outcome = report.reproduction(tmp_path / "again", tmp_path / "first")
        assert outcome["reproduced"] is True
        assert outcome["changes"] == []

    def test_a_reindex_of_the_same_documents_still_reproduces(
        self, dataset, settings_obj, tmp_path
    ):
        """Indexing seconds are a property of the machine, not of the measurement."""
        first = a_manifest(
            dataset,
            settings_obj,
            [
                a_result(
                    index={
                        "primer.pdf": {
                            "index_manifest": "m1",
                            "index_generation": 1,
                            "is_processed": True,
                            "indexed_seconds": 12.5,
                        }
                    }
                )
            ],
        )
        again = a_manifest(
            dataset,
            settings_obj,
            [
                a_result(
                    index={
                        "primer.pdf": {
                            "index_manifest": "m1",
                            "index_generation": 1,
                            "is_processed": True,
                            "indexed_seconds": 31.0,
                        }
                    }
                )
            ],
        )
        report.write_report(tmp_path / "first", first, [])
        report.write_report(tmp_path / "again", again, [])

        outcome = report.reproduction(tmp_path / "again", tmp_path / "first")

        assert outcome["reproduced"] is True
        assert outcome["measurements_changed"] is False

    def test_moved_numbers_are_reported_without_failing_the_setup(
        self, dataset, settings_obj, tmp_path
    ):
        """
        The same setup can measure different numbers, and that is a finding.

        Retrieval runs against an approximate index, so a re-run that matched
        the revision, the set, the models, and the settings is still a
        reproduction; how far the numbers moved is reported beside it.
        """
        first = a_manifest(dataset, settings_obj, [a_result()])
        moved = a_result(
            retrieval={
                "questions": 2,
                "k": 5,
                "hit_rate": 0.5,
                "recall": 0.5,
                "mrr": 0.5,
                "ndcg": 0.4,
                "per_question": [],
            }
        )
        again = a_manifest(dataset, settings_obj, [moved])
        report.write_report(tmp_path / "first", first, [a_result()])
        report.write_report(tmp_path / "again", again, [moved])

        outcome = report.reproduction(tmp_path / "again", tmp_path / "first")

        assert outcome["reproduced"] is True
        assert outcome["measurements_changed"] is True
        assert outcome["measurement_deltas"] == [
            {"experiment": "baseline", "largest_move": pytest.approx(0.5)}
        ]

    def test_a_changed_model_revision_is_named(self, dataset, settings_obj, tmp_path):
        """A model that moved behind the code is the difference a checkout cannot explain."""
        first = a_manifest(dataset, settings_obj, [a_result()])
        moved = settings_obj.model_copy(
            update={
                "rerank": settings_obj.rerank.model_copy(update={"revision": "ccc333"})
            }
        )
        again = a_manifest(dataset, moved, [a_result()])
        report.write_report(tmp_path / "first", first, [a_result()])
        report.write_report(tmp_path / "again", again, [a_result()])

        outcome = report.reproduction(tmp_path / "again", tmp_path / "first")

        assert outcome["reproduced"] is False
        assert outcome["model_revisions_changed"] == [
            {"field": "models.reranker.revision", "before": "", "after": "ccc333"}
        ]

    def test_a_changed_revision_is_named_separately_from_a_changed_model(
        self, dataset, settings_obj, tmp_path
    ):
        """New code and a new model are two different findings."""
        first = a_manifest(dataset, settings_obj, [a_result()])
        again = a_manifest(
            dataset,
            settings_obj,
            [a_result()],
            revision={"revision": "f" * 40, "dirty": True, "subject": "wip"},
        )
        report.write_report(tmp_path / "first", first, [a_result()])
        report.write_report(tmp_path / "again", again, [a_result()])

        outcome = report.reproduction(tmp_path / "again", tmp_path / "first")

        assert [change["field"] for change in outcome["changes"]] == [
            "revision.dirty",
            "revision.revision",
            "revision.subject",
        ]
        assert outcome["model_revisions_changed"] == []


class TestAMeasurementFeedsTheReport:
    """A result measured through the real retrieval service is reportable."""

    def test_a_measured_experiment_becomes_a_report_result(
        self, dataset, settings_obj, environment, cases, tmp_path
    ):
        """A result measured through the real service is publishable as it stands."""
        measured = ablations.measure(
            environment, dataset, cases, experiments.experiment("baseline"), k=5
        )
        results = [measured.as_dict()]
        results[0]["comparison"] = ablations.compare(measured, measured).as_dict()

        written = tmp_path / "reports" / "measured"
        report.write_report(
            written, a_manifest(dataset, settings_obj, results), results
        )

        stored = json.loads((written / "results" / "baseline.json").read_text())
        assert stored["retrieval"]["questions"] == len(cases)
        assert stored["dataset"] == dataset.version
        assert report.reproduction(written, written)["reproduced"] is True
