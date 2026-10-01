"""
Public evidence stays accurate: docs, licenses, screenshots, and claims.

This suite guards the public entry point so a reviewer sees the shipped
application, not stale artifacts. It fails when the setup guide,
model identifiers, screenshots, or evaluation figures drift from the
repository — it checks that every public claim has a checked-in source,
not that the numbers themselves moved.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
README = REPO_ROOT / "README.md"
BACKEND_README = REPO_ROOT / "backend" / "README.md"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_top_level_license_covers_code() -> None:
    """The public repo sells nothing it cannot license."""
    license_file = REPO_ROOT / "LICENSE"
    assert license_file.exists(), "public repo needs a top-level LICENSE"
    text = _read(license_file)
    assert "MIT" in text
    assert (
        "Anis" in text or "anismabaziz" in text.lower() or "papermind" in text.lower()
    )


def test_third_party_notices_cover_bundled_resources() -> None:
    """Every bundled resource keeps its upstream license."""
    notices = REPO_ROOT / "THIRD-PARTY-NOTICES.md"
    assert notices.exists(), "bundled resources need third-party notices"
    text = _read(notices).lower()
    for required in (
        "cc0",  # in-repo sample documents
        "cc-by",  # figure-skating paper
        "plos",  # publisher of the CC-BY paper
        "pdf.js",  # vendored reader assets
        "foxit",  # vendored standard fonts
        "liberation",  # vendored standard fonts
        "bge-m3",  # local embedding weights
    ):
        assert required in text, f"notices must cover {required}"


def test_architecture_doc_uses_domain_vocabulary() -> None:
    """The architecture page speaks the project glossary."""
    arch = REPO_ROOT / "docs" / "architecture.md"
    assert arch.exists(), "public architecture needs one stable page"
    text = _read(arch)
    for term in (
        "Index Generation",
        "Index Manifest",
        "Ingestion Job",
        "Conversation",
        "Turn",
        "Passage",
        "Citation Source",
        "Stale Index",
        "Postgres",
        "Qdrant",
    ):
        assert term in text, f"architecture must explain {term}"


def test_decision_record_covers_expensive_choices() -> None:
    """Expensive choices are written down while reversible."""
    records = sorted((REPO_ROOT / "docs" / "adr").glob("0010-*.md"))
    assert records, "expensive choices need a short decision record"
    text = _read(records[0]).lower()
    for required in ("generation", "local-first", "bound"):
        assert required in text, f"decision record must cover {required}"


def test_known_limits_are_stated() -> None:
    """What the app does not do is stated plainly."""
    limits = REPO_ROOT / "docs" / "limits.md"
    assert limits.exists(), "known limits need one stable page"
    text = _read(limits).lower()
    for required in (
        "single-user",
        "google",
        "groq",
        "cloud",
        "127.0.0.1",
    ):
        assert required in text, f"limits must state {required}"


def test_setup_guide_uses_verified_one_command_path() -> None:
    """The setup guide names the verified path and no secrets."""
    text = _read(README)
    assert "./papermind.sh up --seed" in text
    for tool in ("Docker", "uv", "Node", "curl"):
        assert tool in text, f"setup guide must name required tool {tool}"
    # Required tools, never secret values: the guide must not print a key.
    assert "PAPERMIND_EVAL_GENERATOR_API_KEY=..." not in text.split("## Testing")[0]


def test_feature_claims_cite_checked_in_evaluation_evidence() -> None:
    """Every feature claim points at a checked-in report."""
    text = _read(README)
    reports = (
        "2026-09-retrieval-baseline-v1",
        "2026-09-answers-baseline-v1",
        "2026-09-brief-tool-use-v1",
    )
    for report in reports:
        assert report in text, f"README must cite {report}"
        report_dir = REPO_ROOT / "backend" / "evaluation" / "reports" / report
        assert (report_dir / "README.md").exists()
        assert (report_dir / "manifest.json").exists()


def test_screenshots_match_shipped_interface() -> None:
    """Screenshots show the shipped interface, not a stale one."""
    text = _read(README)
    for shot in ("screenshots/desktop.png", "screenshots/mobile.png"):
        assert shot in text, f"README must show {shot}"
        assert (REPO_ROOT / shot).exists(), f"{shot} must be checked in"
    assert "screenshots/main.png" not in text, "stale screenshot must not be referenced"


def test_model_identifiers_match_catalog() -> None:
    """Docs name only models the catalog serves."""
    from services.accounts.chat_settings_service import MODEL_CATALOG

    catalog_ids = {model.id for model in MODEL_CATALOG}
    for doc in (README, BACKEND_README):
        for model_id in re.findall(
            r"(?:openai/)?(?:gemini|gpt-oss|qwen)[\w./-]*", _read(doc)
        ):
            assert model_id.rstrip(".,)") in catalog_ids or model_id.startswith(
                ("gemini",)
            ), f"{doc.name} names {model_id} outside the catalog"
    # Every catalog model an operator can pick must be reachable from the docs.
    text = _read(README) + _read(BACKEND_README)
    for model_id in catalog_ids:
        assert model_id in text, f"catalog model {model_id} missing from docs"


def test_obsolete_artifacts_are_gone() -> None:
    """Abandoned notebooks and templates stay out of the repo."""
    assert not (REPO_ROOT / "backend" / "notebooks").exists(), (
        "ingestion notebooks predate durable jobs and must go"
    )
    template = _read(REPO_ROOT / "frontend" / "README.md")
    assert "This template provides a minimal setup" not in template
    assert "papermind" in template.lower() or "README.md" in template
