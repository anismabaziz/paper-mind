"""
The model-graded half of a run, against a versioned rubric.

Two things make a judge score mean something. The rubric it answers against is
versioned, so a number from one rubric is never silently compared with a number
from another. And the judge is told from the generator, so an answer graded by
the model that wrote it can be recognized as such.

The third outcome is Unknown. A judge that cannot read its own verdict has not
graded anything, and scoring that as unfaithful would make every unreadable
reply look like a hallucination — a claim about the system that the run did not
actually measure.
"""

import json

import pytest

from evaluation import cli, judge
from evaluation.calibration import (
    CALIBRATION_PATH,
    CalibrationCase,
    load_calibration,
    run_calibration,
)
from evaluation.judge import (
    CORRECTNESS_PROMPT,
    UNKNOWN,
    JudgeSettings,
    RUBRIC_VERSION,
    judge_correctness,
    judge_faithfulness,
    parse_verdict,
)

ANSWER = "The algorithm counted 39 of the 41 program jumps."
CONTEXT = "The identification algorithm counted 39 of the 41 program jumps correctly."


class TestVerdictParsing:
    """A verdict word, its score, and what an unreadable reply means."""

    def test_a_clean_verdict_is_read_as_its_score(self):
        """One word in, one score out."""
        assert parse_verdict("faithful") == ("faithful", 1.0)
        assert parse_verdict("partial") == ("partial", 0.5)
        assert parse_verdict("unfaithful") == ("unfaithful", 0.0)

    def test_a_verdict_inside_a_sentence_is_still_read(self):
        """A judge that explains itself has still given a verdict."""
        assert parse_verdict('The verdict is: "Unfaithful"') == ("unfaithful", 0.0)

    def test_a_reply_naming_unknown_is_read_as_unknown(self):
        """The judge saying it cannot decide is an answer, not a zero."""
        assert parse_verdict("unknown") == (UNKNOWN, None)
        assert parse_verdict("I cannot tell from this context. Unknown.") == (
            UNKNOWN,
            None,
        )

    def test_a_reply_that_is_not_a_verdict_is_unknown_rather_than_unfaithful(self):
        """A broken judge is not evidence of a hallucinating model."""
        verdict, score = parse_verdict("I think it is fine")

        assert verdict == UNKNOWN
        assert score is None

    def test_no_verdict_word_means_nothing_was_graded(self):
        """An empty reply grades nothing."""
        assert parse_verdict("") == (UNKNOWN, None)

    def test_unfaithful_is_not_read_as_faithful(self):
        """The longer word wins: 'unfaithful' is the opposite of 'faithful'."""
        assert parse_verdict("unfaithful")[0] == "unfaithful"

    def test_a_negated_verdict_is_not_the_verdict_it_names(self):
        """
        The failure this guards is a run reporting its best possible score.

            "The answer is not faithful" says the opposite of "faithful", and a
            reader scoring that 1.0 would be told the model never hallucinates.
        """
        assert parse_verdict("The answer is not faithful.") == (UNKNOWN, None)
        assert parse_verdict("I would not call it correct.") == (UNKNOWN, None)
        assert parse_verdict("It is never partial.") == (UNKNOWN, None)

    def test_a_verdict_the_judge_plainly_asserts_is_still_read(self):
        """A negation elsewhere in the sentence does not disarm the verdict."""
        assert parse_verdict("Not partial: this is faithful.") == ("faithful", 1.0)


class TestRubric:
    """The rubric is versioned, and the version travels with the run."""

    def test_the_rubric_version_is_recorded_with_the_judge(self):
        """The version is part of the number, not a comment beside it."""
        settings = JudgeSettings(provider="google", model="gemini-2.5-flash")

        assert settings.rubric_version == RUBRIC_VERSION
        assert settings.to_dict() == {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "rubric_version": RUBRIC_VERSION,
        }

    def test_the_faithfulness_prompt_names_the_verdicts_it_accepts(self):
        """A judge cannot return a word the prompt never offered it."""
        prompt = judge.FAITHFUL_PROMPT.format(
            question="Q", context="CTX", answer="A", rubric_version=RUBRIC_VERSION
        )

        assert "faithful" in prompt
        assert "unfaithful" in prompt
        assert UNKNOWN in prompt
        assert RUBRIC_VERSION in prompt

    def test_a_judge_call_that_failed_is_unknown_rather_than_a_crash(self, monkeypatch):
        """One throttled call must not throw away a run that is half done."""
        from services.llm.base import LLMProvider

        # The pause between attempts is exercised by the test below; this one
        # only asks what a judge that never answered becomes.
        monkeypatch.setattr(cli, "JUDGE_BACKOFF_SECONDS", ())

        class Throttled(LLMProvider):
            """A judge provider that is rate limited on every call."""

            name = "throttled"

            def _build_client(self):
                """No client is needed to fail the way the SDK does."""
                return None

            def verify(self):
                """Verification is not what this double exercises."""
                return None

            def _generate_response(self, query, context, history=""):
                """Fail the way an SDK reports a throttle with no body."""
                raise RuntimeError("RateLimitError")

            def _stream_response(self, query, context, history=""):
                """Refuse a stream: the judge grades one-shot, and this double never streams."""
                raise NotImplementedError

        original = cli._judge_provider
        cli._judge_provider = lambda provider, model, api_key: Throttled(api_key, model)
        built = cli.build_evaluation_judge("groq", "openai/gpt-oss-120b", "key")

        verdict, score = judge_faithfulness("Q", "A", "CTX", built)

        assert (verdict, score) == (UNKNOWN, None)

    def test_a_throttled_judge_is_asked_again_before_giving_up(self, monkeypatch):
        """A rate limit is the one failure a pause fixes, so it is retried."""
        import time as time_module

        from services.llm.base import ChatBudget, LLMProvider

        waits = []
        monkeypatch.setattr(time_module, "sleep", waits.append)
        # The run's own pacing is exercised elsewhere; this is the pause between
        # a failed call and the next attempt.
        monkeypatch.setattr(cli, "PROVIDER_PACE_SECONDS", 0.0)

        class ThrottledOnce(LLMProvider):
            """A judge provider that is throttled once and then answers."""

            name = "throttled-once"

            def __init__(self, api_key, model):
                """Start with one throttle still to come."""
                super().__init__(api_key, model, budget=ChatBudget())
                self.attempts = 0

            def _build_client(self):
                """No client is needed to fail and then answer."""
                return None

            def verify(self):
                """Verification is not what this double exercises."""
                return None

            def _generate_response(self, query, context, history=""):
                """Throttle the first call, then return the verdict."""
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("RateLimitError")
                return "faithful"

            def _stream_response(self, query, context, history=""):
                """Refuse a stream: the judge grades one-shot."""
                raise NotImplementedError

        monkeypatch.setattr(
            cli,
            "_judge_provider",
            lambda provider, model, api_key: ThrottledOnce(api_key, model),
        )
        built = cli.build_evaluation_judge("groq", "openai/gpt-oss-120b", "key")

        verdict, score = judge_faithfulness("Q", "A", "CTX", built)

        assert (verdict, score) == ("faithful", 1.0)
        assert waits == [cli.JUDGE_BACKOFF_SECONDS[0]]

    def test_the_correctness_prompt_carries_the_expected_answer(self):
        """Correctness is judged against the label, so the label is in the prompt."""
        captured = {}

        def spy(prompt):
            """Do spy."""
            captured["prompt"] = prompt
            return "correct"

        verdict, score = judge_correctness("Q", "39 of 41", "A", "CTX", spy)

        assert (verdict, score) == ("correct", 1.0)
        assert "39 of 41" in captured["prompt"]
        assert "A" in captured["prompt"]
        assert "CTX" in captured["prompt"]

    def test_a_case_rubric_reaches_the_correctness_prompt(self):
        """A reference answer is one wording of correct, and the rubric says so."""
        captured = {}

        def spy(prompt):
            """Do spy."""
            captured["prompt"] = prompt
            return "correct"

        judge_correctness(
            "Q",
            "I don't know based on the given context.",
            "The document does not cover this.",
            "CTX",
            spy,
            "A correct answer says the document does not cover this.",
        )

        prompt = captured["prompt"]
        assert "A correct answer says the document does not cover this." in prompt
        # The rubric is read before the instruction to answer, so it is part of
        # what the judge is asked rather than an aside after it.
        assert prompt.index("does not cover this") < prompt.index("Reply with exactly")

    def test_a_case_without_a_rubric_asks_the_original_question(self):
        """A run against a set that carries no rubric sends the prompt it always did."""
        captured = {}

        def spy(prompt):
            """Do spy."""
            captured["prompt"] = prompt
            return "correct"

        judge_correctness("Q", "E", "A", "CTX", spy)

        assert captured["prompt"] == CORRECTNESS_PROMPT.format(
            question="Q",
            context="CTX",
            answer="A",
            expected_answer="E",
            rubric_block="",
        )

    def test_an_unreadable_correctness_verdict_is_unknown(self):
        """Correctness reports the same three outcomes faithfulness does."""
        assert judge_correctness("Q", "39", "A", "CTX", lambda prompt: "maybe") == (
            UNKNOWN,
            None,
        )

    def test_the_two_graders_ask_different_questions(self):
        """Faithfulness is about the evidence; correctness is about the label."""
        captured = []

        def spy(prompt):
            """Do spy."""
            captured.append(prompt)
            return "faithful"

        judge_faithfulness("Q", ANSWER, CONTEXT, spy)
        judge_correctness("Q", "39 of 41", ANSWER, CONTEXT, spy)

        assert captured[0] != captured[1]
        assert "supported by" in captured[0]
        assert "expected answer" in captured[1]


class TestCalibrationSet:
    """Human labels, so a judge's disagreement with a person is measurable."""

    def test_the_committed_set_covers_every_outcome_a_judge_can_return(self):
        """A calibration set missing a verdict cannot check a judge that returns it."""
        cases = load_calibration()

        assert {case.human_verdict for case in cases} == {
            "faithful",
            "partial",
            "unfaithful",
            UNKNOWN,
        }

    def test_every_case_carries_the_question_the_context_and_the_answer(self):
        """A judge is asked about an answer in a context, so all three are in the file."""
        for case in load_calibration():
            assert case.question and case.context and case.answer
            assert case.human_verdict in judge.VERDICTS

    def test_the_set_is_versioned_with_the_rubric_it_was_written_against(self):
        """Labels written for another rubric are not labels for this one."""
        payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

        assert payload["rubric_version"] == RUBRIC_VERSION

    def test_a_judge_that_agrees_with_every_label_is_reported_as_agreement(self):
        """Agreement is measured over the cases the judge decided."""
        cases = _labelled()

        report = run_calibration(cases, lambda prompt: "faithful")

        assert report.agreement() == pytest.approx(0.25)
        assert report.agreed == 1
        assert report.disagreed == 3
        assert report.unknown == 0

    def test_a_judge_that_reads_nothing_is_reported_as_unknown(self):
        """A judge that grades nothing has not agreed with anybody."""
        report = run_calibration(_labelled(), lambda prompt: "I am not sure")

        assert report.agreed == 0
        assert report.disagreed == 0
        assert report.unknown == 4
        assert report.agreement() is None

    def test_the_report_lists_every_case_with_both_verdicts(self):
        """A disagreement is only useful if both verdicts are in the report."""
        report = run_calibration(_labelled(), lambda prompt: "faithful")

        assert [row["id"] for row in report.cases] == ["one", "two", "three", "four"]
        assert report.cases[1] == {
            "id": "two",
            "human_verdict": "partial",
            "judge_verdict": "faithful",
            "unknown": False,
            "agreed": False,
        }

    def test_a_case_the_judge_cannot_decide_is_not_agreement_and_not_disagreement(self):
        """Undecided is counted apart from both."""
        report = run_calibration(_labelled(), lambda prompt: "unknown")

        assert report.unknown == 4
        assert report.agreed == 0

    def test_the_report_serializes_with_its_rubric_version(self):
        """The calibration report travels with the run as JSON."""
        report = run_calibration(_labelled(), lambda prompt: "faithful")

        payload = report.to_dict()
        assert payload["rubric_version"] == RUBRIC_VERSION
        assert payload["cases"] == 4
        assert payload["agreement"] == pytest.approx(0.25)


def _labelled() -> tuple[CalibrationCase, ...]:
    """Return four calibration cases the scripted judge answers predictably."""
    return (
        CalibrationCase("one", "Q", CONTEXT, ANSWER, "faithful"),
        CalibrationCase("two", "Q", CONTEXT, ANSWER, "partial"),
        CalibrationCase("three", "Q", CONTEXT, ANSWER, "unfaithful"),
        CalibrationCase("four", "Q", CONTEXT, ANSWER, UNKNOWN),
    )
