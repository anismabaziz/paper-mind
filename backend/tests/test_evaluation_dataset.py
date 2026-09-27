"""
The labeled case set, and what it refuses to load.

A case set is only worth a run if somebody read it, so most of these tests are
about refusal. Each one hands the loader a set with a single thing wrong with it
and asserts that the thing is named, because a validator that quietly accepts a
set is a validator nobody can rely on to have run.

The last class checks the labels against the documents themselves, which costs
a parse per document and is asked once rather than on every load.
"""

import json
from typing import Any

import pytest

from evaluation import dataset as dataset_module
from evaluation.dataset import (
    ANSWERED,
    CATEGORIES,
    CITATION_VALIDATION,
    CURRENT_DATASET,
    DATASETS_DIR,
    EXACT_LOOKUP,
    FOLLOW_UP,
    MAX_CASES,
    MIN_CASES,
    MULTI_SECTION,
    NUMERIC,
    PARAPHRASE,
    TABLE,
    PROVIDER_FAILURE,
    REPORTED,
    REQUIRED_OUTCOMES,
    TUNING,
    UNANSWERABLE,
    Case,
    DatasetInvalid,
    content_hash,
    evidence_problems,
    load_dataset,
    validate_payload,
)
from evaluation.dataset import abstention_required
from evaluation.harness import SAMPLE_DOCS_DIR
from evaluation.metrics import contains_snippet
from services.parsing.pdf_service import PDFParser

PRIMER = "papermind-rag-primer.pdf"
PRIMER_TEXT = "A RAG pipeline has five stages"


@pytest.fixture(scope="module")
def dataset():
    """Return the committed case set."""
    return load_dataset(docs_dir=SAMPLE_DOCS_DIR)


def case_payload(**overrides: Any) -> dict[str, Any]:
    """Return one valid case entry, with whatever the test changes about it."""
    entry: dict[str, Any] = {
        "id": "primer-stages",
        "document": PRIMER,
        "split": TUNING,
        "category": "exact_lookup",
        "question": "How many stages does the pipeline have?",
        "expected_outcome": ANSWERED,
        "expected_answer": "Five stages.",
        "expected_evidence": [PRIMER_TEXT],
        "rubric": "A correct answer names five stages.",
    }
    entry.update(overrides)
    return entry


def filled_cases(count: int = MIN_CASES) -> list[dict[str, Any]]:
    """Return that many valid cases, numbered apart from each other."""
    return [case_payload(id=f"primer-case-{number}") for number in range(1, count + 1)]


def dataset_payload(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return a valid payload whose cases are numbered apart from each other."""
    if cases is None:
        cases = filled_cases()
    return {
        "version": "test-set-v1",
        "reviewed_on": "2026-09-27",
        "documents": [
            {
                "filename": PRIMER,
                "version": "1",
                "sha256": content_hash(SAMPLE_DOCS_DIR / PRIMER),
                "origin": "authored in-repo for this project",
                "license": "CC0 1.0 (public domain)",
            }
        ],
        "cases": cases,
    }


class TestCommittedSet:
    """The set that ships is a reviewed set, and says so."""

    def test_it_loads(self, dataset):
        """Every reference, label, split, and content hash holds."""
        assert dataset.version == CURRENT_DATASET
        assert MIN_CASES <= len(dataset.cases) <= MAX_CASES

    def test_every_case_carries_a_split_a_category_and_an_outcome(self, dataset):
        """Each case names the split it belongs to, how it is asked, and its outcome."""
        for case in dataset.cases:
            assert case.split in (TUNING, REPORTED), case.id
            assert case.category in CATEGORIES, case.id
            assert case.expected_outcome in REQUIRED_OUTCOMES, case.id

    def test_ids_are_unique_and_stable_slugs(self, dataset):
        """Ids survive a re-read and read the same in every report."""
        ids = [case.id for case in dataset.cases]
        assert len(set(ids)) == len(ids)
        assert all(case.id.islower() for case in dataset.cases)

    def test_it_ships_the_documents_its_cases_are_labelled_against(self, dataset):
        """Each pinned document is on disk, at the bytes it was reviewed at."""
        for document in dataset.documents:
            path = SAMPLE_DOCS_DIR / document.filename
            assert path.is_file(), document.filename
            assert content_hash(path) == document.sha256, document.filename

    def test_every_document_declares_its_version_origin_and_license(self, dataset):
        """A third-party document states where it came from and under what terms."""
        for document in dataset.documents:
            assert document.version, document.filename
            assert document.origin, document.filename
            assert document.license, document.filename

    def test_every_answerable_case_has_evidence_and_a_rubric(self, dataset):
        """A reference reviewer can reproduce the expected answer from the case."""
        for case in dataset.cases:
            if case.expected_outcome != ANSWERED or case.category == UNANSWERABLE:
                continue
            assert case.expected_answer, case.id
            assert case.expected_evidence, case.id
            assert case.rubric, case.id

    def test_every_unanswerable_case_names_the_gap_it_leaves(self, dataset):
        """The unsupported questions are labelled, not merely absent."""
        for case in dataset.cases:
            if case.category != UNANSWERABLE:
                continue
            assert case.absent_terms, case.id
            assert case.rubric, case.id
            assert not case.expected_evidence, case.id

    def test_the_fault_cases_declare_the_failure_they_are_for(self, dataset):
        """A case that only runs with a fault injected asks for that outcome."""
        faults = [case for case in dataset.cases if case.faulty]
        assert faults, "the set holds no case that checks a failure is reported"
        for case in faults:
            assert case.fault in dataset_module.FAULTS, case.id
            assert case.expected_outcome == dataset_module.FAULTS[case.fault], case.id

    def test_it_covers_every_way_of_failing(self, dataset):
        """Each category the set is balanced over has cases in it."""
        counts = dataset.counts_by_category()
        for category in CATEGORIES:
            assert counts.get(category), category

    def test_the_reading_cases_do_not_restate_their_own_evidence(self, dataset):
        """The cases meant to be answered by reading cannot be matched by words."""
        for case in dataset.cases:
            if case.category not in dataset_module.NEEDS_READING:
                continue
            for snippet in case.expected_evidence:
                restated = dataset_module.restated_evidence(case.question, snippet)
                assert restated <= dataset_module.MAX_QUOTED_EVIDENCE, (
                    case.id,
                    snippet,
                )

    def test_the_tuning_split_covers_what_a_configuration_is_chosen_on(self, dataset):
        """The tuning half has to carry the failures a configuration can cause."""
        tuning = dataset.cases_for(TUNING)
        covered = {case.category for case in tuning}
        assert {
            EXACT_LOOKUP,
            PARAPHRASE,
            NUMERIC,
            TABLE,
            MULTI_SECTION,
            UNANSWERABLE,
        } <= covered

    def test_a_reported_run_would_hold_back_the_cases_that_need_a_fault(self, dataset):
        """The reported split is bigger than the tuning split, faults aside."""
        reported = dataset.cases_for(REPORTED)
        assert len(reported) > len(dataset.cases_for(TUNING))
        assert all(not case.faulty for case in reported)
        assert dataset.faults_for(REPORTED) or dataset.faults_for(TUNING)


class TestSelectingCases:
    """A run asks for a split and gets that split."""

    def test_it_returns_only_that_split(self, dataset):
        """A reported run cannot be handed a tuning case by accident."""
        assert {case.split for case in dataset.cases_for(REPORTED)} == {REPORTED}
        assert {case.split for case in dataset.cases_for(TUNING)} == {TUNING}

    def test_it_holds_the_fault_cases_back_until_they_are_asked_for(self, dataset):
        """An injected failure is not counted as a quality result."""
        faults = dataset.faults_for(REPORTED)
        assert faults
        assert all(not case.faulty for case in dataset.cases_for(REPORTED))
        assert len(dataset.cases_for(REPORTED, include_faults=True)) == (
            len(dataset.cases_for(REPORTED)) + len(faults)
        )

    def test_an_unknown_split_is_refused(self, dataset):
        """A typo in a split name does not quietly run the whole set."""
        with pytest.raises(ValueError, match="not a split"):
            dataset.cases_for("holdout")

    def test_the_whole_set_round_trips_through_its_own_serialisation(self, dataset):
        """What the loader reads is what the loader would write back."""
        again = validate_payload(dataset.to_dict(), docs_dir=SAMPLE_DOCS_DIR)
        assert again.to_dict() == dataset.to_dict()


class TestRefusingSets:
    """Each rule is a test, so a rule that stops firing is a failing test."""

    def test_a_case_against_a_document_nobody_shipped(self):
        """A reference has to resolve."""
        with pytest.raises(DatasetInvalid, match="which the set does not ship"):
            validate_payload(
                dataset_payload([case_payload(document="absent.pdf")]),
                docs_dir=SAMPLE_DOCS_DIR,
            )

    def test_a_document_with_no_license(self):
        """Provenance is not optional, least of all for a third-party file."""
        payload = dataset_payload()
        payload["documents"][0]["license"] = ""
        with pytest.raises(DatasetInvalid, match="declares no license"):
            validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)

    def test_a_document_whose_bytes_moved(self):
        """A regenerated file cannot keep answering to the old expectations."""
        payload = dataset_payload()
        payload["documents"][0]["sha256"] = "0" * 64
        with pytest.raises(DatasetInvalid, match="no longer matches the content hash"):
            validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)

    def test_a_document_with_no_version(self):
        """A pin without a version cannot be told apart from its replacement."""
        payload = dataset_payload()
        payload["documents"][0]["version"] = ""
        with pytest.raises(DatasetInvalid, match="declares no version"):
            validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)

    def test_a_set_with_no_version(self):
        """A run has to record which set produced its numbers."""
        payload = dataset_payload()
        payload["version"] = ""
        with pytest.raises(DatasetInvalid, match="declares no version"):
            validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)

    def test_an_answerable_case_with_no_evidence(self):
        """An expected answer nothing can be checked against is a guess."""
        _refuse({"expected_evidence": []}, "names no evidence")

    def test_an_answerable_case_with_no_rubric(self):
        """Nothing says what a reviewer should look for."""
        _refuse({"rubric": ""}, "declares no rubric")

    def test_an_answerable_case_with_no_expected_answer(self):
        """Nothing says what the answer was worth."""
        _refuse({"expected_answer": None}, "declares no expected answer")

    def test_an_unanswerable_case_that_names_evidence(self):
        """Nothing in the document answers it, so there is no passage to point at."""
        _refuse(
            {
                "category": UNANSWERABLE,
                "expected_evidence": [PRIMER_TEXT],
                "absent_terms": ["spinster"],
            },
            "names no evidence",
        )

    def test_an_unanswerable_case_that_names_no_gap(self):
        """Nothing then checks that the document cannot answer it."""
        _refuse(
            {"category": UNANSWERABLE, "absent_terms": []},
            "names no absent term",
        )

    def test_an_unanswerable_case_that_expects_a_failure(self):
        """Neither abstaining nor declining is a provider failure."""
        _refuse(
            {
                "category": UNANSWERABLE,
                "expected_outcome": "provider_error",
                "expected_answer": None,
                "rubric": "",
                "absent_terms": ["spinster"],
            },
            "the run may abstain or answer that it does not know",
        )

    def test_a_follow_up_with_no_earlier_question(self):
        """A follow-up is only a follow-up against something that came before."""
        _refuse({"category": FOLLOW_UP, "follow_up": []}, "records no earlier question")

    def test_a_fault_case_that_expects_the_wrong_outcome(self):
        """The case has to say what the injected failure must come out as."""
        _refuse(
            {
                "category": PROVIDER_FAILURE,
                "fault": "provider_unavailable",
                "expected_outcome": ANSWERED,
                "expected_answer": None,
                "expected_evidence": [],
                "rubric": "",
            },
            "has to come out as 'provider_error'",
        )

    def test_a_fault_case_that_still_carries_an_answer(self):
        """There is no answer to grade when the run is made to fail."""
        _refuse(
            {
                "category": CITATION_VALIDATION,
                "fault": "invalid_citation",
                "expected_outcome": "citation_error",
                "expected_answer": "An answer nobody will read.",
            },
            "no answer to grade",
        )

    def test_a_case_naming_a_fault_nobody_injects(self):
        """A fault the run cannot produce is a case that never runs."""
        _refuse({"fault": "gremlins"}, "names fault 'gremlins'")

    def test_a_case_whose_question_repeats_its_evidence(self):
        """A paraphrase case that quotes its own answer measures the matcher."""
        _refuse(
            {
                "category": "paraphrase",
                "question": "What does A RAG pipeline has five stages mean here?",
                "expected_evidence": [PRIMER_TEXT],
            },
            "lexical match alone could pass it",
        )

    def test_two_cases_with_one_id(self):
        """A report keyed on the id cannot hold both of them."""
        payload = dataset_payload()
        payload["cases"][1] = dict(payload["cases"][0])
        with pytest.raises(DatasetInvalid, match="appears twice"):
            validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)

    def test_a_case_id_that_is_not_a_stable_slug(self):
        """Ids travel between reports, so they cannot be free text."""
        _refuse({"id": "Case 16: the good one"}, "is not a slug")

    def test_a_set_with_no_unanswerable_case(self):
        """A set with nothing unsupported in it flatters every abstention score."""
        payload = dataset_payload(filled_cases())
        with pytest.raises(DatasetInvalid, match="holds no unanswerable case"):
            validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)

    def test_a_set_that_is_smaller_than_it_claims_to_be_reviewed(self):
        """A trimmed set needs a version of its own, not a shorter range."""
        payload = dataset_payload(
            [case_payload(id=f"primer-case-{number}") for number in range(1, 12)]
        )
        with pytest.raises(DatasetInvalid, match="a reviewed set is between"):
            validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)

    def test_a_reported_split_smaller_than_the_tuning_split(self):
        """The quoted number has to come from the bigger set."""
        cases = [
            case_payload(id=f"primer-case-{number}", split=TUNING)
            for number in range(1, MIN_CASES - 5)
        ]
        cases += [
            case_payload(id=f"primer-reported-{number}", split=REPORTED)
            for number in range(1, 4)
        ]
        with pytest.raises(DatasetInvalid, match="decided by the smaller set"):
            validate_payload(dataset_payload(cases), docs_dir=SAMPLE_DOCS_DIR)

    def test_every_problem_is_reported_at_once(self):
        """A reviewer gets the whole list rather than the first thing."""
        with pytest.raises(DatasetInvalid) as refused:
            validate_payload(
                dataset_payload([case_payload(rubric="", expected_evidence=[])]),
                docs_dir=SAMPLE_DOCS_DIR,
            )
        assert len(refused.value.problems) >= 2


def _refuse(overrides: dict[str, Any], expected: str) -> None:
    """Assert a set whose first case carries the given overrides is refused."""
    payload = dataset_payload()
    payload["cases"][0].update(overrides)
    with pytest.raises(DatasetInvalid, match=expected):
        validate_payload(payload, docs_dir=SAMPLE_DOCS_DIR)


class TestLabelsAgainstTheDocuments:
    """The last checks need the documents, so they cost a parse each."""

    def test_every_label_holds_against_the_documents(self, dataset):
        """No passage is missing, and no unanswerable question can be answered."""
        assert evidence_problems(dataset, SAMPLE_DOCS_DIR) == []

    def test_the_multi_section_cases_really_span_pages(self, dataset):
        """A case labelled as needing several sections draws on more than one page."""
        cases = [case for case in dataset.cases if case.category == MULTI_SECTION]
        assert len(cases) >= 3
        for case in cases:
            pages = {
                number
                for snippet in case.expected_evidence
                for number, text in enumerate(
                    PDFParser().extract_pages(
                        (SAMPLE_DOCS_DIR / case.document).read_bytes()
                    ),
                    start=1,
                )
                if contains_snippet(text, snippet)
            }
            assert len(pages) >= 2, (case.id, sorted(pages))


class TestVersionHistory:
    """A set is versioned, and the history it claims to keep is there."""

    def test_the_current_version_is_a_directory_of_its_own(self):
        """A new set is a new directory, so the one it replaces stays readable."""
        assert (DATASETS_DIR / CURRENT_DATASET).is_dir()

    def test_the_changelog_records_the_current_version(self):
        """The history says what changed, so two reports can be compared."""
        changelog = (DATASETS_DIR / "CHANGELOG.md").read_text(encoding="utf-8")
        assert CURRENT_DATASET in changelog

    def test_the_committed_file_is_what_the_loader_reads(self):
        """The version in the report is the version in the file."""
        payload = json.loads(
            (DATASETS_DIR / CURRENT_DATASET / "dataset.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["version"] == CURRENT_DATASET


class TestAbstention:
    """What a declared outcome says the abstention grader should decide."""

    def _case(self, **overrides):
        """Return a case carrying only what the overrides set."""
        declared: dict[str, Any] = {
            "id": "a",
            "document": "d",
            "split": TUNING,
            "category": EXACT_LOOKUP,
            "question": "q",
            "expected_outcome": ANSWERED,
        }
        declared.update(overrides)
        return Case(**declared)

    def test_an_answer_forbids_abstaining(self):
        """Abstaining on a question the evidence answers wastes the reader's time."""
        assert abstention_required(self._case()) is False

    def test_an_abstention_requires_one(self):
        """A question the evidence cannot reach has one honest answer."""
        assert abstention_required(self._case(expected_outcome="abstained")) is True

    def test_a_failure_decides_nothing_about_abstaining(self):
        """A provider that died did not abstain, and did not answer either."""
        assert (
            abstention_required(self._case(expected_outcome="provider_error")) is None
        )
        assert (
            abstention_required(self._case(expected_outcome="citation_error")) is None
        )

    def test_an_unanswerable_question_allows_either(self):
        """Abstaining and saying it does not know are both honest."""
        assert (
            abstention_required(
                self._case(
                    category=UNANSWERABLE,
                    expected_outcome=ANSWERED,
                    absent_terms=("spinster",),
                )
            )
            is None
        )
