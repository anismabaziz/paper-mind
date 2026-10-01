"""
A composed app for the Research Brief boundary, with every collaborator faked.

A brief is the one request in this app that calls a model more than once and
lets it choose what each call reads, so its tests need a provider whose turns
are scripted rather than whose single stream is scripted. That is what this
harness adds over :mod:`tests.chat_app`: the same real repositories over an
in-memory database and the same indexed Document, plus a factory whose next
turn is whatever the test wrote down.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import pytest

from services.llm.tools import ToolTurn

from tests.chat_app import (
    _ChatFactory,
    _ChatProvider,
    _Embeddings,
    _Parser,
    _Storage,
    _Vectors,
    in_memory_session_factory,
)

__all__ = [
    "BriefTurn",
    "brief_turn",
    "call_tool",
    "compare_tool",
    "read_page_tool",
    "read_tool",
    "search_tool",
    "app",
    "client",
    "fake_brief",
    "indexed_document",
    "repositories",
    "session_factory",
    "two_documents",
]


@dataclass
class BriefTurn:
    """
    Script one turn of a brief: what the model said, and what it asked for.

    ``calls`` is a list of ``(id, name, arguments)`` so a test can script a turn
    that searches and another that reads, which is the whole shape of a run.
    An ``error`` fails the turn instead, so a provider failure is reachable the
    same way a chat stream's is.
    """

    text: str = ""
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    finish_reason: str | None = "stop"
    error: BaseException | None = None
    input_tokens: int = 40
    output_tokens: int = 10

    def turn(self) -> ToolTurn:
        """Return this script as the turn the provider hands back."""
        from services.llm.tools import tool_call

        return ToolTurn(
            text=self.text,
            tool_calls=tuple(
                tool_call(call_id, name, arguments)
                for call_id, name, arguments in self.calls
            ),
            finish_reason=self.finish_reason or "stop",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


class BriefProvider(_ChatProvider):
    """
    Answers each brief turn with whatever the factory's next script says.

    It is a real provider, not a stub, so a brief's turns go through the same
    deadline, retry, and finish-reason handling the SDK providers get.
    """

    def _complete_with_tools(self, messages, tools, system_instruction):
        """Run one brief turn from the script, recording what it was asked."""
        self._factory.turns.append(
            (
                [message.role for message in messages],
                [tool.name for tool in tools],
                system_instruction,
                "\n".join(message.text for message in messages),
            )
        )
        plan = self._factory.plan_for(self._factory.attempts)
        self._factory.attempts += 1
        if plan.error is not None:
            raise plan.error
        self.last_finish_reason = plan.finish_reason
        return plan.turn()


class BriefFactory(_ChatFactory):
    """Builds a brief provider whose next turn follows its script."""

    def __init__(self):
        """Start with a script that answers immediately."""
        super().__init__()
        self.turns: list[tuple[list[str], list[str], str, str]] = []
        self.brief_plans: list[BriefTurn] = [BriefTurn(text="Both agree.")]
        self.attempts = 0

    def __call__(self, credentials):
        """Build a brief provider under the credentials the route resolved."""
        self.credentials = credentials
        return BriefProvider(self)

    def plan_for(self, attempt: int) -> BriefTurn:
        """Return what turn number N of this brief does."""
        return self.brief_plans[min(attempt, len(self.brief_plans) - 1)]

    def brief(self, *turns: BriefTurn) -> None:
        """
        Script successive turns of this brief.

        The last turn repeats, so a script of one turn describes a model that
        searches and then answers the same way every time it is asked again.
        """
        self.brief_plans = list(turns)


def brief_turn(text: str = "", *, error: BaseException | None = None) -> BriefTurn:
    """Return a turn that writes the brief, or fails it."""
    return BriefTurn(text=text, finish_reason="stop", error=error)


def call_tool(call_id: str, name: str, arguments: dict[str, Any]) -> BriefTurn:
    """Return a turn that calls any tool by name, whether it exists or not."""
    return BriefTurn(
        calls=[(call_id, name, arguments)],
        finish_reason="tool_calls",
    )


def search_tool(call_id: str, label: str, query: str = "retention") -> BriefTurn:
    """Return a turn that asks the brief to search one labelled Document."""
    return call_tool(call_id, "search_passages", {"label": label, "query": query})


def read_tool(call_id: str, *evidence_ids: str) -> BriefTurn:
    """Return a turn that asks the brief to read Passages it already holds."""
    return call_tool(call_id, "read_passages", {"evidence_ids": list(evidence_ids)})


def read_page_tool(call_id: str, label: str, page: int = 1) -> BriefTurn:
    """Return a turn that asks the brief to read one Page of a Document."""
    return call_tool(call_id, "read_page", {"label": label, "page": page})


def compare_tool(call_id: str, *evidence_ids: str) -> BriefTurn:
    """Return a turn that asks the brief to compare evidence it already holds."""
    return call_tool(call_id, "compare_evidence", {"evidence_ids": list(evidence_ids)})


@pytest.fixture
def session_factory():
    """Session factory over one in-memory database."""
    return in_memory_session_factory()


@pytest.fixture
def repositories(session_factory):
    """Build every repository over one in-memory database with saved settings."""
    from repositories import build_repositories
    from services.accounts.secrets_service import encrypt_api_key

    repositories = build_repositories(session_factory)
    repositories.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repositories


@pytest.fixture
def fake_brief():
    """Brief provider factory whose next turn the test scripts."""
    return BriefFactory()


@pytest.fixture
def app(repositories, fake_brief, settings_obj, tracer):
    """Compose the app with fakes over the in-memory database."""
    from app import create_app
    from composition import Services

    storage = _Storage()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=storage,
        parser=_Parser(),
        embedding_service=_Embeddings(),
        vector_service=_Vectors(),
        chat_provider_factory=fake_brief,
        tracer=tracer,
    )
    application = create_app(settings_obj, services=services)
    application.config.update(
        TEST_REPOSITORIES=repositories,
        TEST_STORAGE=storage,
        TEST_PARSER=services.parser,
        TEST_EMBEDDINGS=services.embedding_service,
        TEST_VECTORS=services.vector_service,
    )
    return application


@pytest.fixture
def client(app):
    """HTTP client for the composed app."""
    return app.test_client()


def indexed_document(client, app, name="doc.pdf") -> str:
    """Upload a Document and index it so a brief can select it."""
    from tests.chat_app import indexed_document as _indexed

    return _indexed(client, app, name)


def two_documents(client, app, first="doc.pdf", second="other.pdf") -> tuple[str, str]:
    """Return two indexed Documents a brief can be scoped to."""
    return indexed_document(client, app, first), indexed_document(client, app, second)
