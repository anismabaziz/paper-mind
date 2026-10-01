"""
Model-graded quality, against a rubric whose version is part of the number.

Two of a run's metrics cannot be decided by reading an answer: whether a model
said something the evidence does not contain, and whether what it said is the
answer the case set expected. Both are asked of a second model, and both are
only worth reading next to two facts — which rubric the judge answered against,
and which model answered. A judge graded by the model that wrote the answer is
still a judge, but it is not an independent one, and the run record has to say
so rather than leave it to be guessed.

The judge protocol is deliberately narrow so any provider, or a scripted stand-in
for one, can fill the role: given a rubric prompt, return one word. Turning that
word into a score lives here; calling the model lives behind the injected
callable, so grading is testable without one.

A fourth outcome exists besides faithful, partial, and unfaithful. A judge that
answers with no verdict has graded nothing, and the run reports that as
Unknown — never as unfaithful, which would turn a broken judge into a headline
quality number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

#: Bumped whenever the rubric's wording or its verdicts change. Recorded in the
#: run record beside every judged score, because a faithfulness number from one
#: rubric is not comparable with a number from another.
RUBRIC_VERSION = "faithfulness-rubric-v2"

#: The verdict a judge returns when it answered with none of the words.
UNKNOWN = "unknown"

#: Every word a judge may return, and the score each one earns. Unknown scores
#: nothing: it is excluded from the mean rather than counted as the worst mark,
#: because it says the judge did not answer, not that the answer was wrong.
VERDICTS: dict[str, float | None] = {
    "faithful": 1.0,
    "partial": 0.5,
    "unfaithful": 0.0,
    UNKNOWN: None,
}

CORRECTNESS_VERDICTS: dict[str, float | None] = {
    "correct": 1.0,
    "incorrect": 0.0,
    UNKNOWN: None,
}

_FAITHFULNESS_VERDICT = re.compile(
    r"\b(unfaithful|faithful|partial|unknown)\b", re.IGNORECASE
)
_CORRECTNESS_VERDICT = re.compile(r"\b(incorrect|correct|unknown)\b", re.IGNORECASE)

_VERDICT_NAMES = (
    'Reply with exactly one word: "faithful", "partial" (some claims '
    'unsupported or hedged beyond the context), "unfaithful" (the answer '
    'states things the context does not contain), or "unknown" (the context '
    "does not settle the question, so no verdict is possible)."
)

_CORRECTNESS_NAMES = (
    'Reply with exactly one word: "correct" (the answer says what the expected '
    'answer says, allowing its own wording), "incorrect" (it contradicts the '
    'expected answer or answers something else), or "unknown" (the expected '
    "answer and the answer cannot be compared)."
)

FAITHFUL_PROMPT = (
    f"You are grading the faithfulness of an answer. Rubric {RUBRIC_VERSION}.\n\n"
    "Question: {question}\n\n"
    "Context the answer was generated from:\n{context}\n\n"
    "Answer to grade: {answer}\n\n"
    "Is every claim in the answer supported by the context? " + _VERDICT_NAMES
)

#: The question the correctness rubric asks, kept apart so a case's own rubric
#: can be read before it rather than after the instruction to answer.
_CORRECTNESS_QUESTION = "Does the graded answer say what the expected answer says? "

CORRECTNESS_PROMPT = (
    f"You are grading an answer against the expected answer. Rubric "
    f"{RUBRIC_VERSION}.\n\n"
    "Question: {question}\n\n"
    "Context the answer was generated from:\n{context}\n\n"
    "Answer to grade: {answer}\n\n"
    f"Expected answer: {{expected_answer}}\n\n"
    "{rubric_block}" + _CORRECTNESS_QUESTION + _CORRECTNESS_NAMES
)

#: What a correct answer has to contain, in the reviewer's words. A reference
#: answer is one wording of the expectation, and a case whose rubric is prose
#: admits an answer that says the same thing differently.
RUBRIC_PROMPT = (
    "What a correct answer has to contain, written by the reviewer of this case: "
    "{rubric}\n\n"
    "Wording that differs from the expected answer is still correct when it "
    "meets this."
)


@dataclass(frozen=True)
class JudgeSettings:
    """
    Who graded the run, and against which rubric.

    Kept apart from the generator's own settings on purpose: the judge may be a
    different provider, a different model, and a different account. Recording
    all three is what makes a judged number comparable with another judged
    number, and what shows a run that graded itself.
    """

    provider: str
    model: str
    rubric_version: str = RUBRIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the judge's identity as a run record stores it."""
        return {
            "provider": self.provider,
            "model": self.model,
            "rubric_version": self.rubric_version,
        }


@dataclass(frozen=True)
class Judge:
    """
    A judge and the identity the run records for it.

    The callable and the settings travel together so a run cannot grade with one
    model and report another: a mismatched pair is the one error that would make
    every judged number in the report unattributable, and it is invisible in
    review because both halves look right on their own.
    """

    settings: JudgeSettings
    grade: Callable[[str], str]

    def __call__(self, prompt: str) -> str:
        """Send one rubric prompt to the judge's model and return its reply."""
        return self.grade(prompt)

    def to_dict(self) -> dict[str, Any]:
        """Return the judge's identity as a run record stores it."""
        return self.settings.to_dict()


#: Words that turn a verdict into its opposite. "The answer is not faithful"
#: says the opposite of "faithful", and a judge that writes a sentence instead
#: of the one word the rubric asked for has not returned a verdict at all.
_NEGATORS = frozenset({"not", "never", "no", "isn't", "wasn't", "hardly", "barely"})


def _parse(
    output: str, pattern: re.Pattern[str], scores: dict[str, float | None]
) -> tuple[str, float | None]:
    """
    Return the verdict word a reply names and the score that word earns.

    A negated mention is not a verdict, and is passed over rather than read:
    scoring "not faithful" as faithful would hand a model the best possible
    score for a reply that says the worst thing about it. The first word in the
    reply that is a verdict and is not negated wins, and a reply with no such
    word read as Unknown, because the judge answered in prose and prose is not
    one of the four words the rubric accepts.
    """
    reply = str(output or "")
    for match in pattern.finditer(reply):
        preceding = reply[: match.start()].lower().split()
        if preceding and preceding[-1] in _NEGATORS:
            continue
        verdict = match.group(1).lower()
        return verdict, scores[verdict]
    return UNKNOWN, scores[UNKNOWN]


def parse_verdict(judge_output: str) -> tuple[str, float | None]:
    """Extract the verdict word and its score from a judge reply."""
    return _parse(judge_output, _FAITHFULNESS_VERDICT, VERDICTS)


def parse_correctness(judge_output: str) -> tuple[str, float | None]:
    """Extract the correctness word and its score from a judge reply."""
    return _parse(judge_output, _CORRECTNESS_VERDICT, CORRECTNESS_VERDICTS)


def judge_faithfulness(
    question: str, answer: str, context: str, judge: Callable[[str], str]
) -> tuple[str, float | None]:
    """Grade one answer for support. ``judge`` maps a prompt to a model reply."""
    return parse_verdict(
        judge(FAITHFUL_PROMPT.format(question=question, context=context, answer=answer))
    )


def judge_correctness(
    question: str,
    expected_answer: str,
    answer: str,
    context: str,
    judge: Callable[[str], str],
    rubric: str = "",
) -> tuple[str, float | None]:
    """
    Grade one answer against what the case set expected it to say.

    A case's rubric joins the prompt when it has one, so a reference answer
    that admits only its own wording is not the whole expectation: a decline
    that says the document does not cover the question is the same answer as
    the reference decline, and the rubric is what says so.
    """
    block = f"{RUBRIC_PROMPT.format(rubric=rubric)}\n\n" if rubric else ""
    return parse_correctness(
        judge(
            CORRECTNESS_PROMPT.format(
                question=question,
                context=context,
                answer=answer,
                expected_answer=expected_answer,
                rubric_block=block,
            )
        )
    )
