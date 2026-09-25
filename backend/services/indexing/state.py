"""Index staleness: the one place that judges a Document's active index."""

from typing import Any

from services.indexing.manifest import (
    MISSING_MANIFEST_CHANGE,
    IndexManifest,
    change_details,
    change_labels,
    manifest_changes,
    manifest_from_json,
    runtime_manifest,
)
from settings import Settings

# States a Document's index can be in from the reader's point of view.
READY = "ready"
STALE = "stale"
PENDING = "pending"


class IndexState:
    """One Document's index, compared against the running configuration."""

    def __init__(
        self,
        state: str,
        stored: IndexManifest | None,
        changes: list[str],
        index_generation: int = 0,
    ):
        """Bind the computed state, the stored manifest, and what changed."""
        self.state = state
        self.manifest = stored
        self.changes = changes
        # Written in the same transaction as the manifest, so both agree on
        # which generation is active.
        self.index_generation = index_generation

    @property
    def is_stale(self) -> bool:
        """Return whether chat must be refused until a reindex succeeds."""
        return self.state == STALE

    def to_dict(self, settings: Settings) -> dict:
        """Return the state and both manifests for the interface."""
        return {
            "state": self.state,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            # The runtime manifest keeps the stored content hash and
            # generation so the interface can show both side by side, and
            # every difference it reports is one the configuration caused.
            "runtime_manifest": runtime_manifest(
                settings,
                content_hash_value=self.manifest.content_hash if self.manifest else "",
                index_generation=self.index_generation,
            ).to_dict(),
            "changes": list(self.changes),
            "change_details": change_details(self.manifest, self.changes, settings),
        }


def index_state(file_record: dict, settings: Settings) -> IndexState:
    """Compare a Document's stored manifest with the running configuration."""
    stored = manifest_from_json(file_record.get("index_manifest"))
    generation = int(file_record.get("index_generation") or 0)
    if not file_record.get("is_processed"):
        return IndexState(PENDING, stored, [], generation)
    if stored is None:
        # A processed document with no manifest predates manifest recording,
        # so nothing proves its vectors match what the app serves now.
        return IndexState(STALE, None, [MISSING_MANIFEST_CHANGE], generation)
    changes = manifest_changes(stored, runtime_manifest(settings))
    if changes:
        return IndexState(STALE, stored, changes, generation)
    return IndexState(READY, stored, [], generation)


def index_status(
    files_repository: Any, file_record: dict, settings: Settings
) -> IndexState:
    """Judge a Document's index and record the verdict for operators."""
    state = index_state(file_record, settings)
    reason = ", ".join(change_labels(state.changes)) if state.is_stale else None
    if (file_record.get("index_stale_reason") or None) != reason:
        files_repository.set_index_stale(file_record["filename"], reason)
    return state
