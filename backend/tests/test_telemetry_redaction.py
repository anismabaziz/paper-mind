"""
What a trace is allowed to record, and for how long.

The rules a trace is trusted to hold to: private text is described rather than
repeated, credentials are removed whatever else is allowed, and the opt-in that
does write private text closes on its own.
"""

from services.telemetry.redaction import (
    REDACTED,
    Redactor,
    credential_free,
    fingerprint,
)


def test_a_question_is_described_rather_than_repeated():
    """The question itself is not in the trace; what can be said about it is."""
    scrubbed = Redactor().scrub({"query": "What is the retention policy?"})

    assert "retention policy" not in str(scrubbed)
    assert scrubbed["query"] == fingerprint("What is the retention policy?")


def test_two_traces_of_the_same_question_share_a_fingerprint():
    """A trace can still show that two spans were about the same text."""
    redactor = Redactor()

    first = redactor.scrub({"query": "What is the retention policy?"})["query"]
    again = redactor.scrub({"query": "What is the retention policy?"})["query"]
    other = redactor.scrub({"query": "How is data deleted?"})["query"]

    assert first["sha256"] == again["sha256"]
    assert first["sha256"] != other["sha256"]


def test_document_text_and_prompts_are_fingerprinted_by_name():
    """Every attribute that carries private text is covered, not just the question."""
    scrubbed = Redactor().scrub(
        {
            "context": "chunk about topic",
            "prompt": "answer from the context",
            "answer": "It is retained for 30 days.",
            "prior_turns": "User: earlier question",
        }
    )

    for name, value in scrubbed.items():
        assert value["redacted"] == REDACTED, name
    assert scrubbed["context"]["chars"] == len("chunk about topic")


def test_identifiers_ranks_and_scores_survive():
    """The things a trace exists to report are not text and are not removed."""
    scrubbed = Redactor().scrub(
        {
            "retrieval.method": "hybrid",
            "candidate_rank": 3,
            "score": 0.82,
            "index_generation": 7,
            "content_hash": "abc123",
        }
    )

    assert scrubbed == {
        "retrieval.method": "hybrid",
        "candidate_rank": 3,
        "score": 0.82,
        "index_generation": 7,
        "content_hash": "abc123",
    }


def test_nested_attributes_are_scrubbed_too():
    """A private value nested under a list or a mapping is still private."""
    scrubbed = Redactor().scrub(
        {
            "candidates": [{"source_id": "S1"}, {"content": "chunk about topic"}],
            "document": {"content": "Attention Is All You Need"},
        }
    )

    assert scrubbed["candidates"][0] == {"source_id": "S1"}
    assert scrubbed["candidates"][1]["content"]["redacted"] == REDACTED
    assert scrubbed["document"]["content"]["redacted"] == REDACTED


def test_a_credential_is_removed_from_any_value():
    """A key stored under an unexpected name is still removed."""
    scrubbed = Redactor().scrub(
        {
            "api_key": "placeholder-credential",
            "notes": "the key is placeholder-credential for this run",
            "header": "Bearer abcdefghijklmnop",
        }
    )

    assert scrubbed["api_key"] == REDACTED
    assert "placeholder-credential" not in scrubbed["notes"]
    assert scrubbed["header"] == REDACTED


def test_credential_free_leaves_ordinary_text_alone():
    """Stripping credentials must not damage the identifiers around them."""
    assert credential_free("hybrid 0.82 S1 7") == "hybrid 0.82 S1 7"


def test_capture_is_off_unless_it_was_asked_for():
    """Nobody opted in, so private text stays out."""
    assert Redactor().capturing is False
    assert Redactor().scrub({"query": "secret question"})["query"] != "secret question"


def test_opted_in_capture_records_the_text():
    """An explicit opt-in writes the text it was granted."""
    redactor = Redactor(capture_text=True, capture_window_seconds=900)

    assert redactor.scrub({"query": "secret question"})["query"] == "secret question"


def test_capture_window_closes_on_its_own():
    """A debugging session cannot become the way the application runs."""
    now = [1_000.0]
    redactor = Redactor(
        capture_text=True, capture_window_seconds=60, clock=lambda: now[0]
    )
    now[0] = 1_030.0
    assert redactor.capturing is True

    now[0] = 1_061.0

    assert redactor.capturing is False
    assert redactor.scrub({"query": "secret question"})["query"][
        "redacted"
    ] == REDACTED


def test_captured_text_never_carries_a_credential():
    """The opt-in grants text, not secrets."""
    redactor = Redactor(capture_text=True, capture_window_seconds=900)

    captured = redactor.scrub({"context": "the key is placeholder-credential here"})

    assert "placeholder-credential" not in captured["context"]


def test_a_fingerprint_of_empty_text_is_still_described():
    """An absent value is reported as an absence, not as a hash of nothing."""
    assert fingerprint(None) == {"redacted": REDACTED, "chars": 0, "sha256": fingerprint("")["sha256"]}
