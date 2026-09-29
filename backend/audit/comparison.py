r"""
Which changed files need an evaluation comparison attached to the change.

The automated tests measure contracts: a parser returns chunks, a manifest
reports staleness, a route answers. None of them measures whether retrieval got
better or worse, what an answer cost, or how long it took. A dependency bump, a
model revision, or a container digest can therefore move every published number
while the suite stays green — which is precisely the case where a reviewer
reading "tests pass" would be misled.

So the set of files that *can* move a measurement is named here, and a pull
request that touches one is asked for a comparison. This is deliberately a list
of paths rather than a diff analysis: it has to be reviewable, and a reviewer
reading the pull request is exactly the person who can tell whether the change
plausibly needs a run.

The comparison itself is produced by the evaluator, which already knows how to
diff two reports and name what moved:

    uv run python -m evaluation.cli --live --split validation \\
        --report reports/<new-name> --compare-report reports/<published-name>
"""

from __future__ import annotations

#: Files whose contents decide what the application does, rather than how it
#: looks. A change to any of them can move retrieval quality, latency, or cost
#: without failing a test, so it is asked to come with a comparison.
#:
#: - the lockfiles and manifests, because a resolved version is the code
#: - the model policy table, because it decides which weights load
#: - the compose files, because a digest decides which server answers
DEPENDENCY_AND_MODEL_FILES = frozenset(
    {
        "backend/uv.lock",
        "backend/pyproject.toml",
        "frontend/package-lock.json",
        "frontend/package.json",
        "backend/services/models.py",
        "backend/compose.yaml",
        "backend/compose.test.yaml",
    }
)


def changed_paths_needing_a_comparison() -> frozenset[str]:
    """Return the repository-relative paths that ask for an evaluation run."""
    return DEPENDENCY_AND_MODEL_FILES


def paths_needing_a_comparison(changed: set[str]) -> list[str]:
    """
    Return which of ``changed`` are paths that ask for an evaluation run.

    Sorted, so the message a reviewer reads is the same on every run.
    """
    return sorted(changed & DEPENDENCY_AND_MODEL_FILES)
