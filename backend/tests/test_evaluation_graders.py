"""
The graders that decide an answer without asking a model.

A number a model produced is only worth as much as the model that produced it.
Everything in this module is decided by reading what the run recorded — the
stored answer, the claims it declared, the Passages the model was shown, and
the outcome the case ended as — so the same run always gets the same grades and
a disagreement can be traced to the exact claim that caused it.

Each grader answers pass, fail, or unknown. Unknown is a real answer, not a
missing one: a claim with nothing to check cannot be graded, and grading it as a
pass would inflate the metric with answers nobody read.
"""

import pytest

from evaluation.graders import (
    FAILED,
    PASSED,
    UNKNOWN,
    GradedCase,
    citation_precision,
    citation_recall,
    grade_claims_schema,
    grade_exact_evidence,
    grade_page_references,
    grade_provider_status,
    grade_required_abstention,
    grade_source_ids,
)

PASSAGE = "The identification algorithm counted 39 of the 41 program jumps."
OTHER_PASSAGE = "Rotation speeds reached up to 1,500 degrees per second."


def a_case(**overrides) -> GradedCase:
    """Return one answered case whose claims cite the first Passage."""
    fields = {
        "id": "skating-jump-counts",
        "outcome": "answered",
        "answer": "It counted 39 of the 41 program jumps.",
        "claims": ({"claim": "It counted 39 of the 41 jumps", "sources": ["S1"]},),
        "sources": (
            {
                "source_id": "S1",
                "content": PASSAGE,
                "page": 4,
                "rank": 1,
            },
            {"source_id": "S2", "content": OTHER_PASSAGE, "page": 5, "rank": 2},
        ),
        "expected_answer": "39 of the 41 program jumps.",
        "expected_evidence": ("counted 39 of the 41 program jumps",),
        "requires_abstention": False,
        "expected_outcome": "answered",
    }
    fields.update(overrides)
    return GradedCase(**fields)


class TestProviderStatus:
    """The outcome the run recorded is the outcome the case asked for."""

    def test_a_case_that_ended_as_asked_for_passes(self):
        """A case that ended the way it was meant to is not a failure."""
        assert grade_provider_status(a_case()).outcome == PASSED

    def test_a_provider_failure_is_a_failure_not_an_unanswered_question(self):
        """A dead provider is the worst result, so it is never counted as fine."""
        result = grade_provider_status(
            a_case(outcome="provider_error", expected_outcome="answered")
        )

        assert result.outcome == FAILED
        assert "provider_error" in result.detail

    def test_a_case_that_declares_no_expected_outcome_is_unknown(self):
        """A case set that says nothing about the outcome cannot complain about it."""
        result = grade_provider_status(a_case(expected_outcome=None))

        assert result.outcome == UNKNOWN


class TestRequiredAbstention:
    """A question the evidence does not reach has to be refused, not answered."""

    def test_a_case_that_requires_abstention_and_abstained_passes(self):
        """Refusing a question the evidence does not reach is the right answer to it."""
        result = grade_required_abstention(
            a_case(
                outcome="abstained",
                requires_abstention=True,
                expected_outcome="abstained",
            )
        )

        assert result.outcome == PASSED

    def test_answering_a_question_that_has_no_evidence_fails(self):
        """Answering anyway is a false answer, not a cautious one."""
        result = grade_required_abstention(a_case(requires_abstention=True))

        assert result.outcome == FAILED
        assert "abstention" in result.detail

    def test_a_failed_provider_does_not_count_as_a_correct_abstention(self):
        """A provider that died did not decide anything."""
        result = grade_required_abstention(
            a_case(outcome="provider_error", requires_abstention=True)
        )

        assert result.outcome == FAILED

    def test_an_answerable_case_must_actually_answer(self):
        """Abstaining on a question the evidence answers is a defect too."""
        result = grade_required_abstention(a_case(outcome="abstained"))

        assert result.outcome == FAILED
        assert "abstained" in result.detail

    def test_a_case_that_declares_nothing_to_abstain_about_is_unknown(self):
        """Nothing declared means nothing to grade."""
        assert (
            grade_required_abstention(a_case(requires_abstention=None)).outcome
            == UNKNOWN
        )


class TestExactEvidence:
    """The answer's citations have to point at the evidence that says it."""

    def test_a_citation_that_contains_the_expected_evidence_passes(self):
        """The citation the reader follows is the one that says the claim."""
        assert grade_exact_evidence(a_case()).outcome == PASSED

    def test_citing_the_wrong_passage_fails_and_names_the_missing_snippet(self):
        """A failure names what was missing, so a reader can go and look."""
        result = grade_exact_evidence(
            a_case(claims=({"claim": "It counted the jumps", "sources": ["S2"]},))
        )

        assert result.outcome == FAILED
        assert "counted 39 of the 41 program jumps" in result.detail

    def test_one_found_snippet_out_of_two_is_a_failure(self):
        """The evidence is either reachable through the citations or it is not."""
        result = grade_exact_evidence(
            a_case(
                expected_evidence=(
                    "counted 39 of the 41 program jumps",
                    "waist-mounted",
                )
            )
        )

        assert result.outcome == FAILED
        assert "waist-mounted" in result.detail

    def test_a_case_with_no_claims_has_no_citation_to_check(self):
        """Nothing to follow is unknown, not a pass."""
        result = grade_exact_evidence(a_case(claims=()))

        assert result.outcome == UNKNOWN

    def test_a_case_that_declares_no_expected_evidence_is_unknown(self):
        """A case with no expected evidence is not graded on reaching it."""
        result = grade_exact_evidence(a_case(expected_evidence=()))

        assert result.outcome == UNKNOWN


class TestClaimsSchema:
    """The claims block the model wrote has to be readable as claims."""

    def test_well_formed_claims_pass(self):
        """Claims the app can read back are claims the reader can follow."""
        assert grade_claims_schema(a_case()).outcome == PASSED

    def test_a_claim_with_no_text_fails(self):
        """A claim with nothing in it cannot be shown as a citation."""
        result = grade_claims_schema(
            a_case(claims=({"claim": "   ", "sources": ["S1"]},))
        )

        assert result.outcome == FAILED
        assert "text" in result.detail

    def test_a_claim_whose_sources_are_not_a_list_fails(self):
        """The sources value is what the app reads, so its shape matters."""
        result = grade_claims_schema(
            a_case(claims=({"claim": "It counted the jumps", "sources": "S1"},))
        )

        assert result.outcome == FAILED
        assert "sources" in result.detail

    def test_a_claim_repeating_one_passage_fails(self):
        """The same Passage twice says nothing a reader did not already have."""
        result = grade_claims_schema(
            a_case(claims=({"claim": "It counted the jumps", "sources": ["S1", "S1"]},))
        )

        assert result.outcome == FAILED
        assert "twice" in result.detail

    def test_an_answer_with_no_claims_at_all_is_unknown_not_a_pass(self):
        """An answer that cited nothing is not a well-formed set of claims."""
        assert grade_claims_schema(a_case(claims=())).outcome == UNKNOWN

    def test_a_case_that_reached_no_answer_has_no_claims_to_read(self):
        """A case that failed before the model wrote anything has no claims."""
        result = grade_claims_schema(
            a_case(outcome="provider_error", answer=None, claims=())
        )

        assert result.outcome == UNKNOWN


class TestSourceIds:
    """A claim has to name Passages the question was actually given."""

    def test_claims_naming_supplied_passages_pass(self):
        """Citations that resolve to a Passage the question was given are checkable."""
        """Citations that resolve to a Passage the question was given are checkable."""
        assert grade_source_ids(a_case()).outcome == PASSED

    def test_a_claim_naming_no_passage_fails(self):
        """A claim with an empty citation is a claim the reader takes on trust."""
        result = grade_source_ids(
            a_case(claims=({"claim": "It counted the jumps", "sources": []},))
        )

        assert result.outcome == FAILED
        assert "cited nothing" in result.detail

    def test_a_claim_naming_a_passage_that_was_not_supplied_fails(self):
        """An id that names nothing supplied is a citation to nowhere."""
        result = grade_source_ids(
            a_case(claims=({"claim": "It counted the jumps", "sources": ["S9"]},))
        )

        assert result.outcome == FAILED
        assert "S9" in result.detail

    def test_the_detail_lists_every_unsupported_claim_once(self):
        """One failure names every claim behind it, so nothing is hidden in a rerun."""
        result = grade_source_ids(
            a_case(
                claims=(
                    {"claim": "first", "sources": []},
                    {"claim": "second", "sources": ["S9"]},
                )
            )
        )

        assert result.outcome == FAILED
        assert "first" in result.detail and "second" in result.detail


class TestPageReferences:
    """A page a reader is pointed at has to be the page the Passage came from."""

    def test_a_page_the_cited_passage_really_has_passes(self):
        """A page promise the citation keeps is a promise kept."""
        """A page promise the citation keeps is a promise kept."""
        answer = "It counted 39 of the 41 jumps on page 4."
        assert grade_page_references(a_case(answer=answer)).outcome == PASSED

    def test_a_page_the_cited_passage_does_not_have_fails(self):
        """A page the Passage is not on sends the reader to the wrong place."""
        answer = "It counted 39 of the 41 jumps on page 9."
        result = grade_page_references(a_case(answer=answer))

        assert result.outcome == FAILED
        assert "9" in result.detail

    def test_a_page_named_in_a_claim_is_checked_against_that_claims_passages(self):
        """A claim's page is checked against the citations on that claim."""
        result = grade_page_references(
            a_case(
                answer="The jump monitor was worn at the waist.",
                claims=(
                    {"claim": "It is worn on page 5 at the waist", "sources": ["S1"]},
                ),
            )
        )

        assert result.outcome == FAILED
        assert "5" in result.detail

    def test_a_claim_cannot_borrow_another_claims_page(self):
        """
        The union of every citation is not enough.

            One claim citing the page-4 passage does not make "page 4" true for a
            second claim citing the page-5 passage, so a claim's page is checked
            against that claim's own citations.
        """
        result = grade_page_references(
            a_case(
                answer="Both facts are in the paper.",
                claims=(
                    {"claim": "It counted 39 of the 41 jumps", "sources": ["S1"]},
                    {"claim": "It is worn at the waist on page 4", "sources": ["S2"]},
                ),
            )
        )

        assert result.outcome == FAILED
        assert "page 4" in result.detail

    def test_a_cited_passage_with_no_page_fails_when_the_document_has_pages(self):
        """A document with page numbers has no excuse for a citation without one."""
        result = grade_page_references(
            a_case(
                claims=({"claim": "It counted the jumps", "sources": ["S3"]},),
                sources=(
                    {"source_id": "S1", "content": PASSAGE, "page": 4, "rank": 1},
                    {"source_id": "S3", "content": PASSAGE, "page": None, "rank": 2},
                ),
            )
        )

        assert result.outcome == FAILED
        assert "no page" in result.detail

    def test_a_document_with_no_pages_at_all_is_unknown(self):
        """Nothing to check is not a defect, so it is neither a pass nor a fail."""
        result = grade_page_references(
            a_case(
                sources=(
                    {"source_id": "S1", "content": PASSAGE, "page": None, "rank": 1},
                )
            )
        )

        assert result.outcome == UNKNOWN

    def test_a_year_in_the_answer_is_not_mistaken_for_a_page_reference(self):
        """Only a number that follows the word page is a page reference."""
        answer = "The 2018 study counted 39 of the 41 program jumps on page 4."
        assert grade_page_references(a_case(answer=answer)).outcome == PASSED


class TestCitationScores:
    """Precision and recall over the citations an answer actually made."""

    def test_a_citation_whose_passage_shares_the_claim_counts_as_precise(self):
        """A citation that repeats the claim's own words supports it."""
        """A citation that repeats the claim's own words supports it."""
        assert citation_precision(a_case()) == 1.0

    def test_a_citation_to_a_passage_that_says_something_else_is_not_precise(self):
        """A citation to a Passage about something else is a false pointer."""
        result = citation_precision(
            a_case(claims=({"claim": "39 jumps were counted", "sources": ["S2"]},))
        )

        assert result == 0.0

    def test_precision_counts_every_citation_pair(self):
        """One good and one bad citation is half precise, not wholly precise."""
        result = citation_precision(
            a_case(
                claims=(
                    {"claim": "It counted 39 of the 41 jumps", "sources": ["S1"]},
                    {
                        "claim": "Rotation speeds reached 1,500 degrees",
                        "sources": ["S2"],
                    },
                    {"claim": "It counted 39 of the 41 jumps", "sources": ["S2"]},
                )
            )
        )

        assert result == pytest.approx(2 / 3)

    def test_an_answer_with_no_citations_has_no_precision_to_report(self):
        """No citations is not zero precision; it is nothing to measure."""
        assert citation_precision(a_case(claims=())) is None

    def test_recall_is_the_share_of_claims_that_cite_something(self):
        """A claim with no citation is the missing citation recall counts."""
        result = citation_recall(
            a_case(
                claims=(
                    {"claim": "It counted 39 of the 41 jumps", "sources": ["S1"]},
                    {"claim": "The monitor was worn at the waist", "sources": []},
                )
            )
        )

        assert result == 0.5

    def test_an_answer_with_no_claims_has_no_recall_to_report(self):
        """Recall over no claims would be a division with nothing in it."""
        """Recall over no claims would be a division with nothing in it."""
        assert citation_recall(a_case(claims=())) is None

    def test_a_citation_to_a_passage_that_does_not_exist_cannot_be_precise(self):
        """A citation that resolves to nothing cannot support anything."""
        assert (
            citation_precision(
                a_case(
                    claims=(
                        {"claim": "It counted 39 of the 41 jumps", "sources": ["S9"]},
                    )
                )
            )
            == 0.0
        )
