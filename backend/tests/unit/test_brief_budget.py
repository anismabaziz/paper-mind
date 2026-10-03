"""Unit tests for brief spend budgets.

The clock is faked so timeouts are instant. Run with plain ``pytest``.
"""

import pytest

from services.brief.budget import BriefBudget, BriefLimits

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _budget(**limits):
    return BriefBudget(BriefLimits(**limits), clock=FakeClock())


class TestBriefLimits:
    def test_to_dict(self):
        assert BriefLimits().to_dict() == {
            "documents": 2,
            "max_turns": 6,
            "max_tool_calls": 8,
            "max_repeated_calls": 2,
            "max_tokens": 120_000,
            "max_seconds": 120.0,
        }


class TestStopReasons:
    def test_fresh_budget_runs(self):
        assert _budget().stop_reason() is None

    def test_turns_stop(self):
        budget = _budget(max_turns=2)
        budget.charge_turn()
        assert budget.stop_reason() is None
        budget.charge_turn()
        assert budget.stop_reason() == "turns"

    def test_tool_calls_stop(self):
        budget = _budget(max_tool_calls=1)
        budget.charge_tool_call("a")
        assert budget.stop_reason() == "tool_calls"

    def test_tokens_stop(self):
        budget = _budget(max_tokens=10)
        budget.charge_tokens(9)
        assert budget.stop_reason() is None
        budget.charge_tokens(1)
        assert budget.stop_reason() == "tokens"

    def test_time_checked_first(self):
        clock = FakeClock()
        budget = BriefBudget(BriefLimits(max_seconds=5.0), clock=clock)
        assert budget.stop_reason() is None
        clock.now += 10.0
        assert budget.stop_reason() == "time"
        assert budget.elapsed == pytest.approx(10.0)

    def test_repeated_calls_do_not_stop_alone(self):
        budget = _budget(max_repeated_calls=1, max_tool_calls=100)
        budget.charge_tool_call("same")
        budget.charge_tool_call("same")
        assert budget.stop_reason() is None
        assert budget.repeated_calls == 1


class TestRepeatTracking:
    def test_first_call_zero_repeats(self):
        budget = _budget()
        assert budget.repeats_call("fp") == 0
        budget.charge_tool_call("fp")
        assert budget.repeats_call("fp") == 1

    def test_distinct_calls_independent(self):
        budget = _budget()
        budget.charge_tool_call("a")
        assert budget.repeats_call("b") == 0

    def test_charge_tokens_ignores_negatives(self):
        budget = _budget()
        budget.charge_tokens(-50)
        assert budget.tokens == 0


class TestUsage:
    def test_usage_reports_spend_and_ceilings(self):
        budget = _budget()
        budget.charge_turn()
        budget.charge_tool_call("a")
        budget.charge_tokens(25)
        usage = budget.usage()
        assert usage["turns"] == 1
        assert usage["tool_calls"] == 1
        assert usage["tokens"] == 25
        assert usage["max_turns"] == 6
        assert usage["elapsed_seconds"] >= 0
