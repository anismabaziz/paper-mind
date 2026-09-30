"""
How each provider reads a tool-using turn, in the shape the brief loop sees.

The brief loop is written against one vocabulary: what tools exist, what the
model said, and what it read. Both SDKs spell that differently, and both will
happily hand back a tool call whose arguments are a string on one path and a
dict on another. These tests pin the translations at the boundary, because a
mistake here is a search tool querying something it was never scoped to: the
arguments decide which Document is read, and neither provider is checking them.

They run without a network or an SDK client. Each provider is given a small
stand-in for its client's one method, so what is under test is the translation
and nothing else.
"""

from types import SimpleNamespace

import pytest

from services.llm.base import LLMProvider
from services.llm.google_provider import GoogleProvider
from services.llm.groq_provider import GroqProvider
from services.llm.tools import (
    ToolCallError,
    ToolSpec,
    ToolUseUnsupportedError,
    assistant_message,
    render_tool_result,
    tool_call,
    user_message,
)

SEARCH = ToolSpec(
    name="search_passages",
    description="Search one selected Document for passages.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


def google(result):
    """
    Return a Google provider whose client always returns this result.

    The request it was given is attached to the response and read back off the
    provider, so a test asserts on what was actually sent rather than on a
    separate spy that could drift from the call.
    """

    class _Client:
        class models:  # noqa: N801 - mirrors the SDK's attribute name
            @staticmethod
            def generate_content(**kwargs):
                result.config = kwargs["config"]
                result.contents = kwargs["contents"]
                return result

    provider = GoogleProvider(api_key="k", model="m", client=_Client())
    provider.result = result
    return provider


def groq(completion):
    """Return a Groq provider whose client always returns this completion."""

    class _Client:
        class chat:  # noqa: N801 - mirrors the SDK's attribute name
            class completions:  # noqa: N801 - mirrors the SDK's attribute name
                @staticmethod
                def create(**kwargs):
                    completion.request = kwargs
                    return completion

    provider = GroqProvider(api_key="k", model="m", client=_Client())
    provider.completion = completion
    return provider


def google_result(*, parts, finish_reason="STOP", text=None, usage=None):
    """Return a stand-in for a Gemini generate_content response."""
    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=list(parts)),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=usage,
        config=None,
        contents=None,
    )


def groq_completion(*, content=None, tool_calls=(), finish_reason="stop", usage=None):
    """Return a stand-in for a Groq chat completion."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=list(tool_calls)),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        request=None,
    )


def groq_call(name="search_passages", arguments='{"query": "x"}', call_id="c1"):
    """Return a stand-in for a Groq tool call."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


CONVERSATION = [
    user_message("compare the two papers"),
    assistant_message(
        "Let me look.", [tool_call("c1", "search_passages", '{"query": "x"}')]
    ),
    render_tool_result(
        tool_call("c1", "search_passages", '{"query": "x"}'), "[E1] a passage"
    ),
]


class TestGoogleTranslation:
    """What Gemini is asked for and what comes back."""

    def test_a_function_call_becomes_a_tool_call(self):
        """One declared function call reads back as one tool call."""
        provider = google(
            google_result(
                parts=[
                    SimpleNamespace(
                        function_call=SimpleNamespace(
                            id="c1", name="search_passages", args={"query": "x"}
                        )
                    )
                ],
                finish_reason="STOP",
            )
        )

        turn = provider.complete_with_tools([user_message("q")], [SEARCH])

        assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
            ("c1", "search_passages", {"query": "x"})
        ]
        assert turn.finish_reason == "stop"

    def test_the_declarations_carry_the_schema_the_brief_declared(self):
        """The schema the brief validated arguments against is the schema sent."""
        provider = google(google_result(parts=[]))

        provider.complete_with_tools([user_message("q")], [SEARCH])

        declared = provider.result.config.tools[0].function_declarations[0]
        assert declared.name == "search_passages"
        # The SDK coerces the mapping into its own Schema type; the required
        # list is read off that type, because a coercion that quietly dropped it
        # would leave the model free to call a tool with no arguments at all.
        assert list(declared.parameters.required) == ["query"]

    def test_a_tool_result_is_a_function_response_on_a_user_turn(self):
        """Gemini pairs a result by id, so the id has to survive the round trip."""
        provider = google(google_result(parts=[]))

        provider.complete_with_tools(CONVERSATION, [SEARCH])

        contents = provider.result.contents
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[1].parts[-1].function_call.name == "search_passages"
        assert contents[2].parts[0].function_response.id == "c1"
        assert contents[2].parts[0].function_response.response == {
            "result": "[E1] a passage"
        }

    def test_prose_before_a_call_is_kept(self):
        """A model that thinks out loud then searches is not truncated for it."""
        provider = google(
            google_result(
                parts=[
                    SimpleNamespace(text="Let me look."),
                    SimpleNamespace(
                        function_call=SimpleNamespace(
                            id="c1", name="search_passages", args={"query": "x"}
                        )
                    ),
                ]
            )
        )

        turn = provider.complete_with_tools([user_message("q")], [SEARCH])

        assert turn.text == "Let me look."
        assert [call.name for call in turn.tool_calls] == ["search_passages"]

    def test_a_call_with_arguments_that_are_not_an_object_is_not_dispatched(self):
        """A list of arguments is not a set of named ones, so it is dropped."""
        provider = google(
            google_result(
                parts=[
                    SimpleNamespace(
                        function_call=SimpleNamespace(
                            id="c1", name="search_passages", args=["x"]
                        )
                    )
                ]
            )
        )

        turn = provider.complete_with_tools([user_message("q")], [SEARCH])

        assert turn.tool_calls == ()

    def test_reported_usage_is_carried_and_absent_usage_is_zero(self):
        """A trace prices a turn from what the provider said it read and wrote."""
        counted = google(
            google_result(
                parts=[],
                usage=SimpleNamespace(
                    prompt_token_count=120, candidates_token_count=30
                ),
            )
        )
        uncounted = google(google_result(parts=[]))

        assert (
            counted.complete_with_tools([user_message("q")], [SEARCH]).input_tokens
            == 120
        )
        assert (
            uncounted.complete_with_tools([user_message("q")], [SEARCH]).input_tokens
            == 0
        )


class TestGroqTranslation:
    """What Groq is asked for and what comes back."""

    def test_a_tool_call_becomes_a_tool_call(self):
        """One declared tool call reads back as one tool call."""
        provider = groq(
            groq_completion(tool_calls=[groq_call()], finish_reason="tool_calls")
        )

        turn = provider.complete_with_tools([user_message("q")], [SEARCH])

        assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
            ("c1", "search_passages", {"query": "x"})
        ]
        assert turn.finish_reason == "tool_calls"

    def test_the_declarations_carry_the_schema_the_brief_declared(self):
        """The schema the brief validated arguments against is the schema sent."""
        provider = groq(groq_completion())

        provider.complete_with_tools([user_message("q")], [SEARCH])

        function = provider.completion.request["tools"][0]["function"]
        assert function["name"] == "search_passages"
        assert function["parameters"]["required"] == ["query"]

    def test_a_tool_result_is_its_own_message_addressed_by_call_id(self):
        """Groq pairs a result by id, so the id has to survive the round trip."""
        provider = groq(groq_completion())

        provider.complete_with_tools(CONVERSATION, [SEARCH])

        messages = provider.completion.request["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["tool_calls"][0]["id"] == "c1"
        assert json_args(messages[1]["tool_calls"][0]["function"]["arguments"]) == {
            "query": "x"
        }
        assert messages[2] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "[E1] a passage",
        }

    def test_prose_before_a_call_is_kept(self):
        """A model that thinks out loud then searches is not truncated for it."""
        provider = groq(
            groq_completion(content="Let me look.", tool_calls=[groq_call()])
        )

        turn = provider.complete_with_tools([user_message("q")], [SEARCH])

        assert turn.text == "Let me look."
        assert [call.name for call in turn.tool_calls] == ["search_passages"]

    def test_a_call_with_arguments_that_are_not_an_object_is_not_dispatched(self):
        """Malformed JSON is not a set of named arguments, so it is dropped."""
        provider = groq(groq_completion(tool_calls=[groq_call(arguments="{oops")]))

        turn = provider.complete_with_tools([user_message("q")], [SEARCH])

        assert turn.tool_calls == ()

    def test_reported_usage_is_carried(self):
        """A trace prices a turn from what the provider said it read and wrote."""
        provider = groq(
            groq_completion(
                usage=SimpleNamespace(prompt_tokens=90, completion_tokens=20)
            )
        )

        turn = provider.complete_with_tools([user_message("q")], [SEARCH])

        assert (turn.input_tokens, turn.output_tokens) == (90, 20)


def json_args(value):
    """Return one serialized tool-argument object as a mapping."""
    import json

    return json.loads(value)


class _NoToolsProvider(LLMProvider):
    """A provider whose catalog entry never declared tool use."""

    name = "no-tools"

    def _build_client(self):
        """No SDK client is needed: no tool call is ever attempted."""
        return None

    def verify(self) -> None:
        """Key verification is not exercised here."""
        raise NotImplementedError

    def _generate_response(self, query, context, prior_turns=""):
        """Answer a one-shot call, which this path never makes."""
        return ""

    def _stream_response(self, query, context, prior_turns=""):
        """Yield no fragments, which this path never asks for."""
        return iter(())


class TestUnsupportedProvider:
    """A provider that never implemented tool calls."""

    def test_the_default_raises_the_capability_error_not_a_crash(self):
        """A missing implementation is a capability the catalog can gate on."""
        provider = _NoToolsProvider(api_key="k", model="m")

        with pytest.raises(ToolUseUnsupportedError):
            provider.complete_with_tools([user_message("q")], [SEARCH])


class TestArgumentDecoding:
    """The one place arguments become a mapping."""

    def test_a_json_object_string_decodes_to_its_mapping(self):
        """Groq hands arguments over as a string, and that is still an object."""
        assert tool_call("c", "t", '{"query": "x", "n": 1}').arguments == {
            "query": "x",
            "n": 1,
        }

    def test_a_mapping_is_taken_as_it_arrived(self):
        """Gemini hands arguments over as a dict already."""
        assert tool_call("c", "t", {"query": "x"}).arguments == {"query": "x"}

    @pytest.mark.parametrize("arguments", ["[]", '"x"', "3", "{oops"])
    def test_anything_that_is_not_an_object_is_refused(self, arguments):
        """A bare scalar or a malformed string is not a set of named arguments."""
        with pytest.raises(ToolCallError):
            tool_call("c", "t", arguments)
