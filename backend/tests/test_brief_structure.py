"""
A structured Research Brief: what the final result promises a reader.

A brief that returns prose asks to be trusted. The structured result replaces
that trust with parts that can each be checked: a summary, ordered claims,
the Passages behind each claim kept apart from the Passages against it, the
gaps the Documents could not settle, and whether the brief abstained. These
tests read that result the way a browser does — off the terminal event — and
pin the contract: every claim names only evidence this run collected,
supporting and conflicting evidence stay distinct, and a claim with nothing
behind it is unresolved rather than completed from model knowledge.
"""

import json

from services.brief.prompts import BRIEF_PROMPT_VERSION
from tests.brief_app import (
    app,
    brief_turn,
    client,
    fake_brief,
    repositories,
    search_tool,
    session_factory,
    two_documents,
)
from tests.sse import parse_sse

__all__ = [
    "app",
    "client",
    "fake_brief",
    "repositories",
    "session_factory",
    "two_documents",
]


def brief(client, documents, question="where do they disagree?"):
    """Start a brief and return its parsed event stream."""
    response = client.post(
        "/research", json={"question": question, "documents": list(documents)}
    )
    assert response.mimetype == "text/event-stream", response.get_data(as_text=True)
    return parse_sse(response.get_data(as_text=True))


def done_of(stream):
    """Return the terminal completion event, or fail loudly without one."""
    for name, payload in stream:
        if name == "done":
            return payload
    raise AssertionError(f"no done event in {stream}")


def structured_answer(summary, claims, gaps=None, abstained=False):
    """Return what a model writes when it answers with a structured brief."""
    block = json.dumps(
        {
            "summary": summary,
            "claims": claims,
            "gaps": gaps or [],
            "abstained": abstained,
        }
    )
    return f"{summary}\n\n<brief>\n{block}\n</brief>"


class TestAgreement:
    """Claims both Documents support."""

    def test_agreement_keeps_each_claims_evidence(self, client, app, fake_brief):
        """Two claims over two Documents stay ordered with their own evidence."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            search_tool("c2", "B"),
            brief_turn(
                text=structured_answer(
                    "Both keep data for years.",
                    [
                        {"claim": "A keeps data.", "supports": ["E1"], "conflicts": []},
                        {"claim": "B keeps data.", "supports": ["E2"], "conflicts": []},
                    ],
                )
            ),
        )

        done = done_of(brief(client, documents))

        assert done["status"] == "complete"
        assert done["brief"]["summary"] == "Both keep data for years."
        assert [claim["order"] for claim in done["brief"]["claims"]] == [1, 2]
        assert done["brief"]["claims"][0]["supports"] == ["E1"]
        assert done["brief"]["claims"][1]["supports"] == ["E2"]
        assert all(claim["status"] == "supported" for claim in done["brief"]["claims"])


class TestDisagreement:
    """Claims the Documents settle differently."""

    def test_conflicting_evidence_stays_distinct(self, client, app, fake_brief):
        """A claim both papers speak to is contested, not averaged."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            search_tool("c2", "B"),
            brief_turn(
                text=structured_answer(
                    "They disagree on retention.",
                    [
                        {
                            "claim": "Retention is thirty months.",
                            "supports": ["E1"],
                            "conflicts": ["E2"],
                        }
                    ],
                    gaps=["Whether the policy changed since."],
                )
            ),
        )

        done = done_of(brief(client, documents))

        claim = done["brief"]["claims"][0]
        assert claim["supports"] == ["E1"]
        assert claim["conflicts"] == ["E2"]
        assert claim["status"] == "contested"
        assert done["brief"]["gaps"] == ["Whether the policy changed since."]


class TestExplicitUnresolved:
    """A claim the model itself marks unresolved."""

    def test_an_explicit_unresolved_stays_unresolved(self, client, app, fake_brief):
        """Abstaining on a claim with evidence is a signal, not a status to upgrade."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            brief_turn(
                text=structured_answer(
                    "A keeps data, but one point is left open.",
                    [
                        {
                            "claim": "The policy never changed.",
                            "supports": ["E1"],
                            "conflicts": [],
                            "status": "unresolved",
                        }
                    ],
                )
            ),
        )

        done = done_of(brief(client, documents))

        assert done["brief"]["claims"][0]["status"] == "unresolved"
        assert done["brief"]["claims"][0]["supports"] == ["E1"]


class TestMissingEvidence:
    """Claims nothing collected supports."""

    def test_a_claim_with_no_evidence_is_unresolved(self, client, app, fake_brief):
        """An unsupported claim is marked unresolved, not completed."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            brief_turn(
                text=structured_answer(
                    "One claim is grounded, one is not.",
                    [
                        {"claim": "A keeps data.", "supports": ["E1"], "conflicts": []},
                        {"claim": "B deletes everything.", "supports": [], "conflicts": []},
                    ],
                )
            ),
        )

        done = done_of(brief(client, documents))

        assert done["brief"]["claims"][0]["status"] == "supported"
        assert done["brief"]["claims"][1]["status"] == "unresolved"


class TestInvalidCitations:
    """Ids this run never collected."""

    def test_an_invented_id_is_stripped_and_reported(self, client, app, fake_brief):
        """A citation to nothing held is dropped, not shown as evidence."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            brief_turn(
                text=structured_answer(
                    "A keeps data.",
                    [{"claim": "A keeps data.", "supports": ["E9"], "conflicts": []}],
                )
            ),
        )

        done = done_of(brief(client, documents))

        assert done["brief"]["claims"][0]["supports"] == []
        assert done["brief"]["claims"][0]["status"] == "unresolved"
        assert done["invalid_citations"] == ["E9"]


class TestPartialCompletion:
    """A brief a limit stopped before it answered."""

    def test_a_partial_brief_carries_its_evidence_and_status(
        self, client, app, fake_brief, settings_obj, monkeypatch
    ):
        """A partial result is shown as partial, with what it collected."""
        monkeypatch.setattr(settings_obj.research, "max_turns", 1)
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"))

        done = done_of(brief(client, documents))

        assert done["status"] == "incomplete"
        assert done["stopped_by"] == "turns"
        assert done["brief"]["claims"] == []
        assert done["evidence"]
        assert done["prompt_version"] == BRIEF_PROMPT_VERSION


class TestReload:
    """What a saved brief needs to come back as itself."""

    def test_the_terminal_event_carries_everything_a_reload_needs(
        self, client, app, fake_brief
    ):
        """Brief, claims, evidence, status, prompt version, and model."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            brief_turn(
                text=structured_answer(
                    "A keeps data.",
                    [{"claim": "A keeps data.", "supports": ["E1"], "conflicts": []}],
                )
            ),
        )

        done = done_of(brief(client, documents))

        assert done["brief"]["summary"]
        assert done["claims"] == done["brief"]["claims"]
        assert done["evidence"]
        assert done["status"] in ("complete", "incomplete")
        assert done["prompt_version"] == BRIEF_PROMPT_VERSION
        assert done["model"]["model"] == "openai/gpt-oss-120b"
        assert done["model"]["provider"] == "groq"
        assert [item["label"] for item in done["documents"]] == ["A", "B"]


class TestPageNavigation:
    """What a citation needs to open its Page."""

    def test_evidence_carries_the_page_each_claim_points_at(
        self, client, app, fake_brief
    ):
        """Every claim's evidence names the Page the reader opens."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            brief_turn(
                text=structured_answer(
                    "A keeps data.",
                    [{"claim": "A keeps data.", "supports": ["E1"], "conflicts": []}],
                )
            ),
        )

        done = done_of(brief(client, documents))

        by_id = {item["evidence_id"]: item for item in done["evidence"]}
        for claim in done["brief"]["claims"]:
            for evidence_id in (*claim["supports"], *claim["conflicts"]):
                assert by_id[evidence_id]["page"] == 1
                assert by_id[evidence_id]["title"]
