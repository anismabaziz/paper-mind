"""The claim-level citation contract: source IDs, claims, and validation."""

import pytest

from services.citations import (
    CLAIMS_CLOSE,
    CLAIMS_OPEN,
    AnswerSplitter,
    assign_source_ids,
    claims_block,
    grounded_in_claims,
    parse_claims,
    prune_conflicting_claims,
    repair_instruction,
    validate_claims,
)
from services.llm.base import LLMProvider


def passage(content: str, page: int | None = 1) -> dict:
    """Return one retrieved Passage as the retrieval layer hands it over."""
    return {
        "content": content,
        "document": "doc.pdf",
        "chunk_index": 0,
        "score": 0.9,
        "page": page,
    }


def test_every_retrieved_passage_gets_a_stable_id_before_generation():
    """Do test every retrieved passage gets a stable id before generation."""
    sources = assign_source_ids([passage("first"), passage("second")])

    assert [source["source_id"] for source in sources] == ["S1", "S2"]
    # The rank is where the Passage landed in retrieval, not its score.
    assert [source["rank"] for source in sources] == [1, 2]


def test_a_passage_without_a_page_still_receives_an_id():
    """Do test a passage without a page still receives an id."""
    sources = assign_source_ids([passage("no page here", page=None)])

    assert sources[0]["source_id"] == "S1"
    assert sources[0]["page"] is None


def test_evidence_is_rendered_with_the_ids_the_model_may_cite():
    """Do test evidence is rendered with the ids the model may cite."""
    sources = assign_source_ids([passage("alpha"), passage("beta")])

    rendered = "\n\n".join(
        f"[{source['source_id']}] {source['content']}" for source in sources
    )

    assert rendered == "[S1] alpha\n\n[S2] beta"


def test_a_claims_block_is_split_off_the_answer_and_read():
    """Do test a claims block is split off the answer and read."""
    parsed = parse_claims(
        "The answer is 42.\n\n"
        f"{CLAIMS_OPEN}\n"
        '{"claim": "The answer is 42.", "sources": ["S1"]}\n'
        f"{CLAIMS_CLOSE}\n"
    )

    assert parsed.answer == "The answer is 42."
    assert [(claim.claim, claim.source_ids) for claim in parsed.claims] == [
        ("The answer is 42.", ("S1",))
    ]


def test_an_answer_without_a_claims_block_is_read_as_an_answer_with_no_claims():
    """Do test an answer without a claims block is read as an answer with no claims."""
    parsed = parse_claims("Just prose, no block.")

    assert parsed.answer == "Just prose, no block."
    assert parsed.claims == ()


def test_a_claim_the_model_never_wrote_comes_back_empty():
    """Do test a claim the model never wrote comes back empty."""
    parsed = parse_claims(f"{CLAIMS_OPEN}\nnot json at all\n{CLAIMS_CLOSE}\n")

    assert parsed.answer == ""
    assert parsed.claims == ()


def test_claims_keep_the_order_the_model_wrote_them_in():
    """Do test claims keep the order the model wrote them in."""
    parsed = parse_claims(
        claims_block(
            [
                {"claim": "first", "sources": ["S1"]},
                {"claim": "second", "sources": ["S2", "S1"]},
            ]
        )
    )

    assert [claim.claim for claim in parsed.claims] == ["first", "second"]


def test_citations_naming_a_source_that_was_not_supplied_are_rejected():
    """Do test citations naming a source that was not supplied are rejected."""
    parsed = parse_claims(claims_block([{"claim": "invented", "sources": ["S9"]}]))

    validated = validate_claims(parsed.claims, {"S1", "S2"})

    # The claim stays, because the reader should see that it was made; the
    # citation that named nothing is what goes.
    assert [claim.claim for claim in validated.claims] == ["invented"]
    assert [claim.source_ids for claim in validated.claims] == [()]
    assert validated.invalid_ids == ("S9",)
    assert validated.grounded is False


def test_a_repeated_citation_is_kept_once():
    """Do test a repeated citation is kept once."""
    parsed = parse_claims(claims_block([{"claim": "twice", "sources": ["S1", "S1"]}]))

    validated = validate_claims(parsed.claims, {"S1"})

    assert [claim.source_ids for claim in validated.claims] == [("S1",)]


def test_citations_are_matched_case_insensitively():
    """Do test citations are matched case insensitively."""
    parsed = parse_claims(claims_block([{"claim": "shouty", "sources": ["s1"]}]))

    validated = validate_claims(parsed.claims, {"S1"})

    assert [claim.source_ids for claim in validated.claims] == [("S1",)]


def test_a_claim_survives_with_the_citations_that_are_real():
    """Do test a claim survives with the citations that are real."""
    parsed = parse_claims(claims_block([{"claim": "half right", "sources": ["S1", "S7"]}]))

    validated = validate_claims(parsed.claims, {"S1", "S2"})

    assert [claim.source_ids for claim in validated.claims] == [("S1",)]
    assert validated.invalid_ids == ("S7",)


def test_an_answer_is_grounded_only_when_a_claim_names_a_real_source():
    """Do test an answer is grounded only when a claim names a real source."""
    parsed = parse_claims(claims_block([{"claim": "unsupported", "sources": []}]))

    validated = validate_claims(parsed.claims, {"S1"})

    assert [claim.claim for claim in validated.claims] == ["unsupported"]
    assert validated.grounded is False


def test_an_answer_with_no_claims_is_not_grounded():
    """Do test an answer with no claims is not grounded."""
    parsed = parse_claims("Prose only.")

    validated = validate_claims(parsed.claims, {"S1"})

    assert validated.grounded is False


def test_a_claim_without_any_citation_is_not_grounded():
    """Do test a claim without any citation is not grounded."""
    claims = parse_claims(claims_block([{"claim": "bare", "sources": []}])).claims

    validated = validate_claims(claims, {"S1"})

    assert validated.grounded is False
    assert grounded_in_claims(claims) is False
    assert grounded_in_claims(validate_claims(claims, {"S1"}).claims) is False


def test_claims_that_contradict_an_abstaining_answer_are_dropped():
    """Do test claims that contradict an abstaining answer are dropped."""
    claims = parse_claims(claims_block([{"claim": "but this", "sources": ["S1"]}])).claims

    assert prune_conflicting_claims(LLMProvider.FALLBACK_ANSWER, claims) == ()
    assert parse_claims(LLMProvider.FALLBACK_ANSWER).claims == ()


def test_claims_are_kept_when_the_answer_actually_answers():
    """Do test claims are kept when the answer actually answers."""
    claims = parse_claims(claims_block([{"claim": "it is 42", "sources": ["S1"]}])).claims

    assert prune_conflicting_claims("It is 42.", claims) == claims


def test_the_repair_instruction_names_the_claims_and_the_allowed_sources():
    """Do test the repair instruction names the claims and the allowed sources."""
    claims = parse_claims(claims_block([{"claim": "wrong page", "sources": ["S9"]}])).claims

    instruction = repair_instruction("It is 42.", claims, ("S1", "S2"))

    assert "wrong page" in instruction
    assert "S9" in instruction
    assert "S1" in instruction and "S2" in instruction
    # The reply is read by the same parser as the first one, so the block is
    # asked for in the same words.
    assert f"{CLAIMS_OPEN}" in instruction and f"{CLAIMS_CLOSE}" in instruction
    assert "It is 42." in instruction


def test_a_repaired_block_replaces_the_citations_that_were_wrong():
    """Do test a repaired block replaces the citations that were wrong."""
    original = parse_claims(claims_block([{"claim": "it is 42", "sources": ["S9"]}]))
    repaired = parse_claims(claims_block([{"claim": "it is 42", "sources": ["S2"]}]))

    validated = validate_claims(repaired.claims, {"S1", "S2"})

    assert validated.invalid_ids == ()
    assert validated.grounded is True
    assert original.claims[0].source_ids == ("S9",)


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_nothing_to_cite_reads_as_no_answer_and_no_claims(text):
    """Do test nothing to cite reads as no answer and no claims."""
    parsed = parse_claims(text)

    assert parsed.answer == ""
    assert parsed.claims == ()


def test_the_prose_a_model_streams_is_shown_as_it_arrives():
    """Do test the prose a model streams is shown as it arrives."""
    splitter = AnswerSplitter()

    assert splitter.feed("The answer ") == "The "
    assert splitter.finish() == "answer "


def test_the_claims_block_never_reaches_the_reader():
    """Do test the claims block never reaches the reader."""
    splitter = AnswerSplitter()

    shown = "".join(
        splitter.feed(part)
        for part in [
            "The answer is 42.\n\n",
            claims_block([{"claim": "42", "sources": ["S1"]}]),
        ]
    )
    shown += splitter.finish()

    assert shown.strip() == "The answer is 42."


def test_the_claims_marker_split_across_two_fragments_is_still_hidden():
    """Do test the claims marker split across two fragments is still hidden."""
    splitter = AnswerSplitter()

    shown = splitter.feed("The answer is 42.\n\n<cl")
    shown += splitter.feed('aims>{"claim": "42", "sources": ["S1"]}</claims>')
    shown += splitter.finish()

    assert shown.strip() == "The answer is 42."


def test_text_that_only_looks_like_the_marker_is_still_shown():
    """Do test text that only looks like the marker is still shown."""
    splitter = AnswerSplitter()

    shown = splitter.feed("The angle <cla") + splitter.feed("ss> is small.") + splitter.finish()

    assert shown == "The angle <class> is small."


def test_a_claims_block_with_no_prose_shows_nothing():
    """Do test a claims block with no prose shows nothing."""
    splitter = AnswerSplitter()

    shown = splitter.feed(claims_block([{"claim": "42", "sources": []}])) + splitter.finish()

    assert shown == ""
