"""
The versions of a configuration that get compared to each other.

A retrieval choice is only worth reporting as an improvement if it was chosen
somewhere other than the numbers being reported. Every experiment here is
therefore declared against the tuning split, and the loader here refuses an
experiment that claims to have been chosen on the validation split: a variant
picked by looking at validation is not an experiment, it is a second reading
of the same number.

An experiment changes one thing and says which thing. The family is the
allowance list — a ``retrieval-method`` experiment sets the method, a
``chunking`` experiment sets the chunk size and overlap — so a report cannot
label a comparison as something it did not actually change.

Settings-shaped knobs (chunk size, overlap, the reranker gate) are read by the
application, so changing them means an environment built from those settings.
Retrieval-shaped knobs (the method, the candidate depth, the reranker for one
call, whether the question is expanded) are arguments to the production
retrieval service, so they are measured against an index that already exists.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from evaluation.dataset import REPORTED, TUNING
from services.retrieval.base import RetrievalMethod
from settings import Settings

#: What a family is allowed to change. The key is the setting field, the value
#: the field names an experiment by.
_FAMILY_KNOBS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "retrieval-method": ("method",),
    "rerank": ("rerank",),
    "candidate-depth": ("candidate_depth",),
    "query-expansion": ("query_expansion",),
    "chunking": ("chunk_size_tokens", "chunk_overlap_tokens"),
}

#: The families a report may compare, the baseline aside.
FAMILIES = tuple(family for family in _FAMILY_KNOBS if family != "baseline")

#: The query expansion a retrieval experiment may ask for. The model rewriter
#: is not here: it costs a call, and a run that cannot make calls cannot
#: measure what it buys.
EXPANSIONS = ("none", "deterministic")

_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class Experiment:
    """
    One configuration, the question it answers, and the split that chose it.

    ``decided_on`` is the split a person looked at when they picked the value.
    It is the load-bearing field: everything else describes what changed, and
    this one says where the decision was allowed to be made.
    """

    id: str
    family: str
    question: str
    decided_on: str = TUNING
    method: RetrievalMethod | None = None
    rerank: bool | None = None
    candidate_depth: int | None = None
    query_expansion: str | None = None
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None

    def __post_init__(self) -> None:
        """Refuse an experiment whose family or split cannot be trusted."""
        if not _ID.fullmatch(self.id):
            raise ValueError(
                f"{self.id!r} is not an experiment id; expected lowercase "
                "hyphenated words"
            )
        if self.family not in _FAMILY_KNOBS:
            raise ValueError(
                f"{self.family!r} is not an experiment family; use one of "
                f"{', '.join(_FAMILY_KNOBS)}"
            )
        if self.decided_on != TUNING:
            raise ValueError(
                f"{self.id!r} was decided on the {self.decided_on} split. The "
                f"{REPORTED} split is the one a result is quoted from, so a "
                "choice made by reading it describes no improvement; choose the "
                f"value on the {TUNING} split instead."
            )
        if self.family == "baseline":
            return
        allowed = _FAMILY_KNOBS[self.family]
        outside = [
            name
            for name in self._knob_names()
            if getattr(self, name) is not None and name not in allowed
        ]
        if outside:
            raise ValueError(
                f"{self.id!r} is a {self.family} experiment, so it may only set "
                f"{', '.join(allowed) or 'nothing'}; it also sets "
                f"{', '.join(outside)}"
            )
        if not self.knobs():
            raise ValueError(f"{self.id!r} changes nothing, so it is the baseline")
        if self.query_expansion is not None and self.query_expansion not in EXPANSIONS:
            raise ValueError(
                f"{self.id!r} expands queries with {self.query_expansion!r}; use "
                f"one of {', '.join(EXPANSIONS)}"
            )
        if self.candidate_depth is not None and self.candidate_depth < 1:
            raise ValueError(f"{self.id!r} retrieves {self.candidate_depth} candidates")

    def _knob_names(self) -> tuple[str, ...]:
        """Return the names of the settings fields an experiment may set."""
        return (
            "method",
            "rerank",
            "candidate_depth",
            "query_expansion",
            "chunk_size_tokens",
            "chunk_overlap_tokens",
        )

    def knobs(self) -> dict[str, Any]:
        """Return what this experiment changes, for a report to publish."""
        return {
            name: getattr(self, name)
            for name in self._knob_names()
            if getattr(self, name) is not None
        }

    def retrieval(self) -> dict[str, Any]:
        """
        Return the retrieval this experiment asks for, unset knobs left to the app.

        ``None`` means the application decides, which is what the baseline
        reports and what every family compares against.
        """
        return {
            "method": self.method,
            "candidate_depth": self.candidate_depth,
            "rerank": self.rerank,
            "query_expansion": self.query_expansion,
        }

    @property
    def needs_fresh_index(self) -> bool:
        """Report whether this experiment has to index the documents again."""
        return (
            self.chunk_size_tokens is not None or self.chunk_overlap_tokens is not None
        )

    def settings(self, base: Settings) -> Settings:
        """
        Return the application settings this experiment is measured under.

        Only the settings that change what is stored need a different
        environment. The reranker gate is an argument to one retrieval call
        rather than a setting, because the application builds the reranker once
        and reads its gate from that call.
        """
        if self.needs_fresh_index:
            return base.model_copy(
                update={
                    "chunking": base.chunking.model_copy(
                        update={
                            key: value
                            for key, value in (
                                ("chunk_size_tokens", self.chunk_size_tokens),
                                ("chunk_overlap_tokens", self.chunk_overlap_tokens),
                            )
                            if value is not None
                        }
                    )
                }
            )
        return base

    def to_dict(self) -> dict[str, Any]:
        """Return the experiment as a report records it."""
        return {
            "id": self.id,
            "family": self.family,
            "question": self.question,
            "decided_on": self.decided_on,
            "changes": self.knobs(),
        }


#: The configuration the application ships, which every variant is compared to.
BASELINE = Experiment(
    id="baseline",
    family="baseline",
    question="What does the shipped configuration do?",
)


def _variant(
    id: str,
    family: str,
    question: str,
    **changes: Any,
) -> Experiment:
    """Return one declared experiment, written the way the set declares them."""
    return Experiment(id=id, family=family, question=question, **changes)


#: The experiments a report compares. Each value was chosen on the tuning
#: split; the questions say what a reader should look for in the columns.
REGISTRY: tuple[Experiment, ...] = (
    BASELINE,
    _variant(
        "dense-only",
        "retrieval-method",
        "What does dense retrieval alone find, and what does it miss?",
        method="dense",
    ),
    _variant(
        "sparse-only",
        "retrieval-method",
        "Do the rare terms and figures in a question carry the answer?",
        method="sparse",
    ),
    _variant(
        "hybrid-only",
        "retrieval-method",
        "Does combining both representations beat either one alone?",
        method="hybrid",
    ),
    _variant(
        "rerank-on",
        "rerank",
        "What does reranking the candidates buy, in seconds?",
        rerank=True,
    ),
    _variant(
        "depth-10",
        "candidate-depth",
        "Can the reranker and the context window live on fewer candidates?",
        candidate_depth=10,
    ),
    _variant(
        "depth-100",
        "candidate-depth",
        "Do more candidates than the fifty the application fetches bring in "
        "anything the reader would otherwise miss?",
        candidate_depth=100,
    ),
    _variant(
        "expansion-none",
        "query-expansion",
        "What does a follow-up retrieve on its own?",
        query_expansion="none",
    ),
    _variant(
        "expansion-deterministic",
        "query-expansion",
        "Does carrying the earlier questions forward find the passage?",
        query_expansion="deterministic",
    ),
    _variant(
        "chunk-256-25",
        "chunking",
        "Do smaller chunks read more precisely?",
        chunk_size_tokens=256,
        chunk_overlap_tokens=25,
    ),
    _variant(
        "chunk-1024-100",
        "chunking",
        "Do larger chunks carry whole tables and sections?",
        chunk_size_tokens=1024,
        chunk_overlap_tokens=100,
    ),
)

_BY_ID = {experiment.id: experiment for experiment in REGISTRY}


def experiment(id: str) -> Experiment:
    """Return one declared experiment by its id."""
    try:
        return _BY_ID[id]
    except KeyError:
        raise KeyError(f"{id!r} is not a declared experiment") from None


def validated(experiments: Iterable[Experiment]) -> tuple[Experiment, ...]:
    """
    Return the experiments as a report may compare them.

    A comparison is read as a table of columns, so two columns answering the
    same question under the same id is a table nobody can read. The first
    experiment is the baseline the others are compared against.
    """
    seen: list[Experiment] = []
    ids: set[str] = set()
    for candidate in experiments:
        if candidate.id in ids:
            raise ValueError(
                f"two experiments are both called {candidate.id!r}; a report "
                "cannot compare a variant with itself"
            )
        ids.add(candidate.id)
        seen.append(candidate)
    if not seen:
        raise ValueError("a report needs at least the baseline to compare against")
    return tuple(seen)


def describe(experiments: Sequence[Experiment] = REGISTRY) -> list[dict[str, Any]]:
    """Return the experiments as a report records them."""
    return [candidate.to_dict() for candidate in validated(experiments)]


__all__ = [
    "BASELINE",
    "EXPANSIONS",
    "FAMILIES",
    "REGISTRY",
    "Experiment",
    "describe",
    "experiment",
    "validated",
]
