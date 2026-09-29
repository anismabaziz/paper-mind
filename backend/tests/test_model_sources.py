"""
Model source policy: which weights a local model loads, and whether it may.

A local model is named by a repository id that Hugging Face resolves to
whatever ``main`` currently points at. Left alone, an upstream push silently
changes what vectors an index holds and what order a reranker returns. The
policy module is the single place that answers two questions: which revision
is immutable for a given repository, and whether the model may execute code
from its own repository at all.

Tests here are pure: no weights are downloaded and ``sentence-transformers``
is never imported.
"""

import dataclasses

import pytest

from services.models import (
    MODEL_ID_ALIASES,
    PINNED_MODEL_REVISIONS,
    REVIEWED_REMOTE_CODE_MODELS,
    canonical_model_id,
    model_source,
)


@pytest.fixture
def reviewed_model(monkeypatch):
    """Add a reviewed model to the allowlist for the duration of a test."""
    monkeypatch.setitem(REVIEWED_REMOTE_CODE_MODELS, "acme/reviewed", "e" * 40)
    return "acme/reviewed"


class TestPinnedRevisions:
    """TestPinnedRevisions."""

    def test_every_pinned_revision_is_a_full_commit_sha(self):
        """A pin that is not a commit sha is not a pin."""
        for model_id, revision in PINNED_MODEL_REVISIONS.items():
            assert len(revision) == 40, model_id
            assert all(character in "0123456789abcdef" for character in revision)

    def test_the_two_shipped_models_are_pinned(self):
        """The default embedding and reranker load immutable weights."""
        assert "BAAI/bge-m3" in PINNED_MODEL_REVISIONS
        assert "cross-encoder/ms-marco-MiniLM-L6-v2" in PINNED_MODEL_REVISIONS

    def test_a_known_model_with_no_configured_revision_uses_the_pin(self):
        """The pin applies even when the environment variable is empty."""
        source = model_source("BAAI/bge-m3")

        assert source.revision == PINNED_MODEL_REVISIONS["BAAI/bge-m3"]

    def test_a_configured_revision_overrides_the_pinned_one(self):
        """The setting is the operator's escape hatch, and it wins."""
        assert model_source("BAAI/bge-m3", revision="b" * 40).revision == "b" * 40

    def test_an_unpinned_model_falls_back_to_its_configured_revision(self):
        """An operator can pin a repository the table does not know about."""
        assert model_source("acme/embedder", revision="a" * 40).revision == "a" * 40

    def test_an_unknown_model_with_no_revision_is_left_unpinned(self):
        """Nothing is invented for a repository nobody reviewed."""
        assert model_source("acme/unknown").revision == ""


class TestRemoteCode:
    """TestRemoteCode."""

    def test_no_shipped_model_may_run_repository_code(self):
        """Both models the app loads by default run without remote code."""
        assert "BAAI/bge-m3" not in REVIEWED_REMOTE_CODE_MODELS
        assert "cross-encoder/ms-marco-MiniLM-L6-v2" not in REVIEWED_REMOTE_CODE_MODELS

    def test_remote_code_is_off_by_default(self):
        """Do test remote code is off by default."""
        assert model_source("BAAI/bge-m3").trust_remote_code is False

    def test_requesting_remote_code_for_an_unreviewed_model_is_refused(self):
        """Arbitrary repository code is refused, naming the model."""
        with pytest.raises(ValueError, match="acme/hostile"):
            model_source("acme/hostile", trust_remote_code=True)

    def test_a_reviewed_revision_may_run_its_own_code(self, reviewed_model):
        """Do test a reviewed revision may run its own code."""
        assert model_source(reviewed_model, trust_remote_code=True).trust_remote_code

    def test_remote_code_at_a_different_revision_is_refused(self, reviewed_model):
        """The allowlist covers one commit, not the branch it was reviewed on."""
        with pytest.raises(ValueError, match="revision"):
            model_source(reviewed_model, revision="f" * 40, trust_remote_code=True)

    def test_a_reviewed_model_loads_its_reviewed_commit_by_default(
        self, reviewed_model
    ):
        """Reviewing a commit is also what pins it, so code never floats."""
        source = model_source(reviewed_model, trust_remote_code=True)

        assert source.revision == REVIEWED_REMOTE_CODE_MODELS[reviewed_model]

    def test_a_reviewed_model_loads_without_remote_code_by_default(
        self, reviewed_model
    ):
        """Being allowlisted does not switch the feature on by itself."""
        assert model_source(reviewed_model).trust_remote_code is False


class TestLoadKwargs:
    """TestLoadKwargs."""

    def test_a_pinned_model_passes_its_revision_and_disables_remote_code(self):
        """Do test a pinned model passes its revision and disables remote code."""
        kwargs = model_source("BAAI/bge-m3").load_kwargs()

        assert kwargs == {
            "revision": PINNED_MODEL_REVISIONS["BAAI/bge-m3"],
            "trust_remote_code": False,
        }

    def test_an_unpinned_model_passes_no_revision(self):
        """``revision=None`` would pin the load to nothing, so the key is absent."""
        assert model_source("acme/unknown").load_kwargs() == {
            "trust_remote_code": False
        }


class TestModelSource:
    """TestModelSource."""

    def test_it_is_frozen(self):
        """A loaded model's provenance cannot be rewritten in place."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            model_source("BAAI/bge-m3").revision = "c" * 40  # type: ignore[misc]

    def test_it_names_the_repository_it_will_load(self):
        """Do test it names the repository it will load."""
        assert model_source("acme/x", revision="d" * 40).model_id == "acme/x"


class TestRepositoryAliases:
    """Hugging Face answers to several spellings of one repository."""

    def test_an_alias_reaches_the_pin(self):
        """The default reranker setting uses a spelling the API redirects."""
        aliased = model_source("cross-encoder/ms-marco-MiniLM-L-6-v2")

        assert (
            aliased.revision
            == PINNED_MODEL_REVISIONS["cross-encoder/ms-marco-MiniLM-L6-v2"]
        )

    def test_the_canonical_spelling_reaches_the_same_pin(self):
        """Do test the canonical spelling reaches the same pin."""
        canonical = model_source("cross-encoder/ms-marco-MiniLM-L6-v2")

        assert (
            canonical.revision
            == model_source("cross-encoder/ms-marco-MiniLM-L-6-v2").revision
        )

    def test_the_library_is_given_the_id_the_operator_wrote(self):
        """So the cache directory stays the one an operator recognises."""
        assert (
            model_source("cross-encoder/ms-marco-MiniLM-L-6-v2").model_id
            == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def test_every_pinned_id_is_canonical(self):
        """A pin keyed by an alias could never be checked against the API."""
        for model_id in PINNED_MODEL_REVISIONS:
            assert canonical_model_id(model_id) == model_id

    def test_an_alias_cycle_terminates(self, monkeypatch):
        """Do test an alias cycle terminates."""
        monkeypatch.setitem(MODEL_ID_ALIASES, "acme/a", "acme/b")
        monkeypatch.setitem(MODEL_ID_ALIASES, "acme/b", "acme/a")

        assert canonical_model_id("acme/a") == "acme/b"
