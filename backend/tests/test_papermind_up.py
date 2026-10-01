"""
Contract for the one-command local workspace.

The shell entry point is behavior-light by design: orchestration lives in
bash, decisions live in testable Python helpers. These tests pin the
contract — the command exists, its flags exist, it never deletes user data
on shutdown, and its output tells the operator how to recover.
"""

import pathlib
import stat

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "papermind.sh"


def _text() -> str:
    return SCRIPT.read_text()


def test_entry_point_exists_and_is_executable():
    """The repo root ships an executable entry point."""
    assert SCRIPT.exists(), "papermind.sh must exist at the repo root"
    assert SCRIPT.stat().st_mode & stat.S_IXUSR, "papermind.sh must be executable"


def test_supports_up_down_status_smoke_and_seed_flag():
    """The entry point supports its documented commands and flags."""
    text = _text()
    for token in ("up", "down", "status", "smoke", "--seed", "--help"):
        assert token in text, f"papermind.sh must support {token}"


def test_up_is_idempotent_and_preserves_user_data():
    """Up reuses secrets and shutdown never deletes volumes."""
    text = _text()
    # `up` must reuse existing infra/secrets rather than recreating them.
    assert "bootstrap-local.sh" in text
    # Shutdown must stop processes/containers without deleting volumes or files.
    assert "down -v" not in text, "shutdown must preserve volumes (no `down -v`)"
    assert "--volumes" not in text, "shutdown must preserve volumes"


def test_up_covers_the_full_stack():
    """Up wires infra, migrations, worker, seed, and smoke together."""
    text = _text()
    for token in (
        "docker compose",
        "alembic upgrade head",
        "worker.py",
        "seed_sample_docs",
        "smoke_setup",
    ):
        assert token in text, f"papermind.sh up must cover {token}"


def test_output_gives_recovery_guidance():
    """Every failure mode points at its fix."""
    text = _text()
    for token in ("missing tools", "occupied ports", "invalid secrets", "unavailable services"):
        assert token.lower() in text.lower(), f"script must guide recovery for {token}"
    assert "Fix:" in text, "recovery guidance must name the fix with `Fix:`"
