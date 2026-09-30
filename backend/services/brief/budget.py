"""
The limits one Research Brief runs under.

A brief lets a model choose what to read next, which is exactly the property
that makes it able to spend without being asked. Nothing in that sentence is a
reason to leave it unbounded: every bound here is a decision about what the app
is willing to spend on one question, and each of them is checked at the moment
it could be crossed rather than after the fact.

Five bounds, because each catches a different runaway. Turns cap how many times
the model may think, so a model that alternates two tools forever stops. Tool
calls cap how many passages one brief can pull, which is what bounds retrieval
work and its latency. Repeated calls cap one identical call being made over and
over, which is what a stuck model actually does — it re-asks the same question
rather than looping between two different ones. Tokens cap what the loop may
bill. Seconds cap what a reader waits, checked between turns because a turn is
the smallest thing that can be abandoned without losing the transcript.

Nothing here is a target. A brief that finishes inside every bound has spent
less than it was allowed, and the trace reports what it actually spent rather
than the ceiling it was given.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

#: Why a brief stopped where it did. These are the words the interface, the
#: trace, and the terminal event all use, so one run tells three readers the
#: same thing.
BudgetStop = Literal["turns", "tool_calls", "repeated_call", "tokens", "time"]


@dataclass(frozen=True)
class BriefLimits:
    """
    What one Research Brief may consume.

    ``documents`` is the size of the pair, and it is not a dial: the brief is
    defined as a cross-Document question over two chosen Documents, and a limit
    that could be raised would make the trace's own field meaningless. The route
    passes it to the scope resolver, so the number on the record is the number
    that was enforced. The other five come from settings so a deployment can
    bound a brief against its own budget.
    """

    documents: int = 2
    max_turns: int = 6
    max_tool_calls: int = 8
    #: How many times one identical call may be made. The second is the last:
    #: a model that asks the same question twice has stopped exploring.
    max_repeated_calls: int = 2
    max_tokens: int = 120_000
    max_seconds: float = 120.0

    def to_dict(self) -> dict[str, int | float]:
        """Return the ceilings as the trace records them."""
        return {
            "documents": self.documents,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_repeated_calls": self.max_repeated_calls,
            "max_tokens": self.max_tokens,
            "max_seconds": self.max_seconds,
        }


@dataclass
class BriefBudget:
    """
    What one running brief has spent so far, and what it may still spend.

    The clock is injected so a test can run out a brief's wall-clock budget
    without waiting for it, which is the only honest way to assert that a
    timeout is enforced between turns: an unbounded test either takes the full
    budget or proves nothing.
    """

    limits: BriefLimits
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        """Open the clock at the moment the brief starts, not when it is built."""
        self._started_at = self.clock()
        self.turns = 0
        self.tool_calls = 0
        self.repeated_calls = 0
        self.tokens = 0
        #: Calls keyed by what was asked, so a repeat is recognised rather than
        #: counted twice under two spellings of the same request.
        self._fingerprints: dict[str, int] = {}

    @property
    def elapsed(self) -> float:
        """Return how long the brief has been running, in seconds."""
        return max(0.0, self.clock() - self._started_at)

    def stop_reason(self) -> BudgetStop | None:
        """
        Return why the brief must stop now, or None while it may continue.

        Wall-clock is checked before the counters because it is the one bound a
        model can cross without taking an action first: a single slow turn
        spends the reader's whole wait without ever exceeding a turn or a token
        count.
        """
        if self.elapsed >= self.limits.max_seconds:
            return "time"
        if self.turns >= self.limits.max_turns:
            return "turns"
        if self.tool_calls >= self.limits.max_tool_calls:
            return "tool_calls"
        if self.tokens >= self.limits.max_tokens:
            return "tokens"
        return None

    def repeats_call(self, fingerprint: str) -> int:
        """
        Return how many times this exact call has been made, before this one.

        The count is taken before the call is charged, so the first call to a
        fingerprint reads as no repeats and the second reads as one.
        """
        return self._fingerprints.get(fingerprint, 0)

    def charge_turn(self) -> None:
        """Record one model turn."""
        self.turns += 1

    def charge_tool_call(self, fingerprint: str) -> None:
        """Record one tool call, and whether it repeated an earlier one."""
        self.tool_calls += 1
        repeats = self.repeats_call(fingerprint)
        if repeats:
            self.repeated_calls += 1
        self._fingerprints[fingerprint] = repeats + 1

    def charge_tokens(self, tokens: int) -> None:
        """Record what one model turn read and wrote."""
        self.tokens += max(0, int(tokens))

    def usage(self) -> dict[str, int | float]:
        """
        Return what the brief spent, as the trace records it.

        The ceilings are recorded beside the spend so a run that stopped against
        a bound is legible from the record alone, without the reader having to
        know what the setting was on the day.
        """
        return {
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "repeated_calls": self.repeated_calls,
            "tokens": self.tokens,
            "elapsed_seconds": round(self.elapsed, 6),
            **self.limits.to_dict(),
        }
