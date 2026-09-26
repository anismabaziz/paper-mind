"""
The claim-level citation contract over HTTP: what an answer is allowed to cite.

Every test here reads the same three things a browser reads — the event stream,
the stored turn, and the reloaded transcript — because a citation that is only
validated in memory is a citation the user never sees checked.
"""

import pytest

from services.citations import PROMPT_VERSION, claims_block
from tests.chat_app import (
    app,
    client,
    fake_chat,
    indexed_document,
    repositories,
    session_factory,
    turns_of,
)
from tests.sse import parse_sse


def answer_with(*claims: dict) -> str:
    """Return what a model writes when it answers and cites its claims."""
    return f"The answer is 42.\n\n{claims_block(list(claims))}"


def ask(client, filename, query="What is the answer?"):
    """Ask one question and return the events the browser would receive."""
    response = client.post(
        "/response", json={"query": query, "filename": filename}
    )
    return parse_sse(response.get_data(as_text=True))


def done_of(events) -> dict:
    """Return the terminal completion event, or fail loudly without one."""
    for name, payload in events:
        if name == "done":
            return payload
    raise AssertionError(f"no done event in {events}")


def names_of(events) -> list[str]:
    """Return the event names in the order the stream sent them."""
    return [name for name, _payload in events]


@pytest.fixture
def document(client, app):
    """One indexed document to ask questions about."""
    return indexed_document(client, app)


def test_every_retrieved_passage_reaches_the_model_under_a_citable_id(
    client, fake_chat, document
):
    """Do test every retrieved passage reaches the model under a citable id."""
    events = ask(client, document)

    done = done_of(events)
    assert [source["source_id"] for source in done["sources"]] == ["S1"]
    assert [source["rank"] for source in done["sources"]] == [1]
    assert fake_chat.streamed[-1][1] == f"[S1] {done['sources'][0]['content']}"


def test_an_answer_reports_each_claim_with_the_sources_that_support_it(
    client, fake_chat, document
):
    """Do test an answer reports each claim with the sources that support it."""
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S1"]}))

    done = done_of(ask(client, document))

    assert done["claims"] == [{"claim": "The answer is 42.", "sources": ["S1"]}]
    assert done["grounded"] is True
    assert done["prompt_version"] == PROMPT_VERSION


def test_the_claims_block_never_reaches_the_reader_as_text(client, fake_chat, document):
    """Do test the claims block never reaches the reader as text."""
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S1"]}))

    events = ask(client, document)

    streamed = "".join(
        payload["text"] for name, payload in events if name == "token"
    )
    assert "<claims>" not in streamed
    assert '{"claim"' not in streamed


def test_an_invented_source_id_is_repaired_once_and_the_answer_completes(
    client, fake_chat, document
):
    """Do test an invented source id is repaired once and the answer completes."""
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S7"]}))
    fake_chat.complete_with(claims_block([{"claim": "The answer is 42.", "sources": ["S1"]}]))

    events = ask(client, document)

    assert names_of(events)[0] == "start"
    assert names_of(events)[-1] == "done"
    assert done_of(events)["claims"] == [{"claim": "The answer is 42.", "sources": ["S1"]}]
    assert done_of(events)["grounded"] is True
    # The repair is asked to re-map citations against the passages themselves,
    # and never to write the answer again.
    question, context, _prior = fake_chat.completed[0]
    assert "S1" in question and "S7" in question and "The answer is 42." in question
    assert context.startswith("[S1] ")


def test_a_citation_that_survives_the_repair_fails_the_turn_visibly(
    client, fake_chat, repositories, document
):
    """Do test a citation that survives the repair fails the turn visibly."""
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S7"]}))
    fake_chat.complete_with(claims_block([{"claim": "The answer is 42.", "sources": ["S8"]}]))

    events = ask(client, document)

    assert names_of(events)[0] == "start"
    assert names_of(events)[-1] == "citation_error"
    failure = events[-1][1]
    assert failure["category"] == "citations"
    turn = turns_of(repositories, document)[0]
    assert turn["status"] == "failed"
    assert turn["failure_reason"] == "invalid citations"
    assert turn["claims"] == []


def test_a_repair_the_model_cannot_answer_fails_the_turn_too(
    client, fake_chat, repositories, document
):
    """Do test a repair the model cannot answer fails the turn too."""
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S7"]}))
    fake_chat.complete_with("I don't know based on the given context.")

    events = ask(client, document)

    assert names_of(events)[-1] == "citation_error"
    assert turns_of(repositories, document)[0]["status"] == "failed"


def test_only_a_passage_the_model_was_shown_can_be_cited(client, fake_chat, document):
    """Do test only a passage the model was shown can be cited."""
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S1"]}))

    done = done_of(ask(client, document))

    # S1 is the only Passage the model was shown, and the only one stored.
    assert [source["source_id"] for source in done["sources"]] == ["S1"]
    assert done["claims"] == [{"claim": "The answer is 42.", "sources": ["S1"]}]


def test_a_claim_whose_only_citation_is_false_is_stored_without_it(
    client, fake_chat, repositories, document
):
    """Do test a claim whose only citation is false is stored without it."""
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S7"]}))
    fake_chat.complete_with(
        claims_block([{"claim": "The answer is 42.", "sources": []}])
    )

    done = done_of(ask(client, document))

    assert done["claims"] == [{"claim": "The answer is 42.", "sources": []}]
    assert done["grounded"] is False
    assert turns_of(repositories, document)[0]["status"] == "answered"


def test_an_answer_that_cites_nothing_is_not_reported_as_grounded(
    client, fake_chat, document
):
    """Do test an answer that cites nothing is not reported as grounded."""
    fake_chat.stream("The answer is 42.")

    done = done_of(ask(client, document))

    assert done["claims"] == []
    assert done["grounded"] is False


def test_claims_that_contradict_a_refusal_are_dropped(client, fake_chat, document):
    """Do test claims that contradict a refusal are dropped."""
    fake_chat.stream(
        "I don't know based on the given context.\n\n"
        + claims_block([{"claim": "The answer is 42.", "sources": ["S1"]}])
    )

    done = done_of(ask(client, document))

    assert done["claims"] == []
    assert done["grounded"] is False


def test_a_passage_with_no_page_is_still_cited(client, app, fake_chat, document):
    """Do test a passage with no page is still cited."""
    app.config["TEST_VECTORS"].script(
        {
            "content": "chunk about topic",
            "document": "doc.pdf",
            "chunk_index": 0,
            "score": 0.9,
            "page": None,
        }
    )
    fake_chat.stream(answer_with({"claim": "The answer is 42.", "sources": ["S1"]}))

    done = done_of(ask(client, document))

    assert done["sources"][0]["page"] is None
    assert done["sources"][0]["source_id"] == "S1"
    assert done["claims"] == [{"claim": "The answer is 42.", "sources": ["S1"]}]


def test_a_reloaded_answer_keeps_its_claim_order_source_ids_and_pages(
    client, app, fake_chat, document
):
    """Do test a reloaded answer keeps its claim order, source ids, and pages."""
    app.config["TEST_VECTORS"].script(
        {
            "content": "first passage",
            "document": "doc.pdf",
            "chunk_index": 0,
            "score": 0.9,
            "page": 3,
        },
        {
            "content": "second passage",
            "document": "doc.pdf",
            "chunk_index": 1,
            "score": 0.4,
            "page": 7,
        },
    )
    fake_chat.stream(
        answer_with(
            {"claim": "The second one is later.", "sources": ["S2"]},
            {"claim": "The first one is earlier.", "sources": ["S1", "S2"]},
        )
    )
    ask(client, document)

    messages = client.get("/messages", query_string={"filename": document}).get_json()[
        "messages"
    ]
    answer = messages[-1]
    assert answer["claims"] == [
        {"claim": "The second one is later.", "sources": ["S2"]},
        {"claim": "The first one is earlier.", "sources": ["S1", "S2"]},
    ]
    # The passages come back in the order the model was shown them, so the
    # citation numbers a reader saw still point at the same text.
    assert [source["source_id"] for source in answer["sources"]] == ["S1", "S2"]
    assert [source["rank"] for source in answer["sources"]] == [1, 2]
    assert [source["page"] for source in answer["sources"]] == [3, 7]
    assert [source["content"] for source in answer["sources"]] == [
        "first passage",
        "second passage",
    ]


def test_an_abstention_carries_no_claims(client, app, fake_chat, document):
    """Do test an abstention carries no claims."""
    app.config["TEST_VECTORS"].script()

    events = ask(client, document)

    assert names_of(events) == ["start", "abstained"]
    assert "claims" not in events[-1][1]
    assert turns_of(app.config["TEST_REPOSITORIES"], document)[0]["claims"] == []
