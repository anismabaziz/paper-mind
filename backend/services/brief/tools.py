"""
The two tools a Research Brief may call, and what happens to every other call.

A brief is allowed to look things up inside the two Documents it was given and
nowhere else. That is enforced by having exactly two tools and giving them no
argument that names anything outside the pair: there is no tool that opens a
site, reads a file the user did not select, changes App Settings, or deletes
anything, because there is no code here that would do it. A model that asks for
one gets a refusal it can read and reason about, and the refusal is recorded,
so a trace shows what the model reached for as well as what it was given.

Arguments are validated rather than trusted. A schema mismatch is not a hint to
be lenient about — the arguments decide which Document is read, and a search
with a missing ``label`` read under whichever default happened to be first
would be the model reading a Document it never named. So every argument is
checked against the declared schema, and a call that does not fit is refused and
recorded rather than repaired.

Neither tool writes anything. ``search_passages`` retrieves, ``read_passages``
returns text the brief already holds, and neither touches the store, the files,
or the settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from services.llm.tools import ToolSpec

SEARCH_PASSAGES = "search_passages"
READ_PASSAGES = "read_passages"

#: Why a call was refused. These are recorded and shown, never swallowed, so a
#: run that could not search says which of these it hit.
REFUSED_UNKNOWN_TOOL = "unknown_tool"
REFUSED_INVALID_ARGUMENTS = "invalid_arguments"
REFUSED_OUT_OF_SCOPE = "out_of_scope"
REFUSED_UNKNOWN_EVIDENCE = "unknown_evidence"
REFUSED_REPEATED = "repeated_call"
#: Not a bad call: the reader stopped the brief before this one ran.
REFUSED_CANCELLED = "cancelled"
#: Not a bad call either: a limit was reached before this one could run.
REFUSED_NOT_RUN = "not_run"

REFUSAL_MESSAGES: dict[str, str] = {
    REFUSED_UNKNOWN_TOOL: (
        "No such tool. A research brief can only call search_passages and "
        "read_passages."
    ),
    REFUSED_INVALID_ARGUMENTS: (
        "Those arguments do not match the tool's schema. Call the tool again "
        "with the arguments it declares."
    ),
    REFUSED_OUT_OF_SCOPE: (
        "That Document is not part of this brief. Search only the labelled "
        "Documents you were given."
    ),
    REFUSED_UNKNOWN_EVIDENCE: (
        "No evidence with that id was collected in this brief. Use an id from "
        "the search results you were given."
    ),
    REFUSED_REPEATED: (
        "That exact call has already been made. Search with a different query, "
        "or read evidence you have not read yet."
    ),
    REFUSED_CANCELLED: "The brief was stopped, so this call was not made.",
    REFUSED_NOT_RUN: (
        "The brief reached a limit before this call ran, so it was not made."
    ),
}

#: The schema both tools declare, written once so the two providers and the
#: validator below agree on what a valid call is.
SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # No enum here: the schema cannot know which Documents this run was
        # given, and a label outside the pair is an out-of-scope call rather
        # than a malformed one. The refusal says so, and names the scope.
        "label": {
            "type": "string",
            "minLength": 1,
            "maxLength": 8,
            "description": "The labelled Document to search: A or B.",
        },
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2000,
            "description": "What to look for in that Document.",
        },
    },
    "required": ["label", "query"],
    "additionalProperties": False,
}

READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^E[0-9]+$"},
            "minItems": 1,
            "maxItems": 10,
            "description": "Ids from the search results already returned.",
        },
    },
    "required": ["evidence_ids"],
    "additionalProperties": False,
}

SEARCH_TOOL = ToolSpec(
    name=SEARCH_PASSAGES,
    description=(
        "Search one of the two labelled Documents for passages answering a "
        "query. Returns the best passages with the ids you may cite. Read-only: "
        "it searches the Document you name and returns passages, nothing else."
    ),
    parameters=SEARCH_SCHEMA,
)

READ_TOOL = ToolSpec(
    name=READ_PASSAGES,
    description=(
        "Read the full text of passages already returned by search_passages, "
        "given their ids. Read-only: it returns text this brief already holds."
    ),
    parameters=READ_SCHEMA,
)

TOOL_SPECS = (SEARCH_TOOL, READ_TOOL)


@dataclass(frozen=True)
class ToolRefusal:
    """One call the brief would not make, with the reason it would not."""

    reason: str
    message: str

    def to_result(self) -> str:
        """Return the text the model reads back in place of a result."""
        return self.message


def refuse(reason: str) -> ToolRefusal:
    """Return the refusal for one reason, with the words the model reads."""
    return ToolRefusal(reason=reason, message=REFUSAL_MESSAGES[reason])


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> ToolRefusal | None:
    """
    Return the refusal when these arguments do not fit the schema, else None.

    This checks the subset of JSON Schema the two tools declare, which is
    deliberately small: object type, required keys, an exact key set, and the
    type, length, pattern, and enum each property names. A schema this module
    cannot fully check would be a schema a model could slip past, so the two
    tools declare nothing beyond what is checked here.
    """
    schema = spec.parameters
    if schema.get("type") != "object":
        return refuse(REFUSED_INVALID_ARGUMENTS)
    if not isinstance(arguments, dict):
        return refuse(REFUSED_INVALID_ARGUMENTS)
    properties: dict[str, Any] = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        return refuse(REFUSED_INVALID_ARGUMENTS)
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return refuse(REFUSED_INVALID_ARGUMENTS)
    for key, value in arguments.items():
        declared = properties.get(key)
        if declared is None or not _valid_value(value, declared):
            return refuse(REFUSED_INVALID_ARGUMENTS)
    return None


def _valid_value(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether one value satisfies the property schema it was declared with."""
    kind = schema.get("type")
    if kind == "string":
        if not isinstance(value, str):
            return False
        if len(value) < schema.get("minLength", 0):
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False
        if "enum" in schema and value not in schema["enum"]:
            return False
        pattern = schema.get("pattern")
        if pattern and not _matches(value, pattern):
            return False
        return True
    if kind == "array":
        if not isinstance(value, list):
            return False
        if len(value) < schema.get("minItems", 0):
            return False
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False
        item_schema = schema.get("items")
        if item_schema is None:
            return True
        return all(_valid_value(item, item_schema) for item in value)
    # A declared property this module cannot check would be a hole, so an
    # unrecognised kind is refused rather than passed through.
    return False


def _matches(value: str, pattern: str) -> bool:
    """Return whether one value matches one anchored pattern."""
    import re

    return re.fullmatch(pattern, value) is not None


def call_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """
    Return the key that decides whether two calls are the same call.

    Sorted keys, so an argument set is the same however the model happened to
    order it, and both names and arguments in the key, so a read of evidence
    named differently is not mistaken for a repeat of a search. A model stuck in
    a loop repeats a request rather than a shape, and this is what catches it.
    """
    return f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"


def render_result(payload: Any) -> str:
    """Return one tool result as the text the model reads."""
    return json.dumps(payload, default=str)
