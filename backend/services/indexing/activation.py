"""Index Generation activation: validate, retire the previous, switch."""

from typing import Any

from services.retrieval.base import VectorStoreConfigurationError
from services.retrieval.hybrid import SPARSE_METHOD, TOKENIZER_VERSION


def _sparse_setting_problems(
    label: str, found: list | None, expected: str
) -> list[str]:
    """
    Check the sparse settings a generation was indexed with.

    A generation missing the setting is as unusable as one built with another
    version, so an absent value fails instead of passing unchecked.
    """
    if found is None or not found:
        return [f"no {label} was recorded for the generation"]
    if list(found) != [expected]:
        return [f"indexed with {label} {found[0]!r} instead of {expected!r}"]
    return []


def _validate_generation_count(
    vector_store: Any, filename: str, generation: int, expected_count: int
):
    """Verify the point count when the store cannot report a generation."""
    counter = getattr(vector_store, "count", None)
    if not callable(counter):
        return None
    count = counter(filter={"pdf_name": filename, "index_generation": generation})
    if count != expected_count:
        raise VectorStoreConfigurationError(
            f"Index generation {generation} is incomplete: expected "
            f"{expected_count} vectors but found {count}."
        )
    return count


def _validate_generation(
    vector_store: Any,
    filename: str,
    generation: int,
    expected_count: int,
    page_count: int | None = None,
):
    """
    Check a built index generation before it may be activated.

    Activation is the moment chat starts reading a generation, so every
    stage the pipeline ran is verified here: the expected number of
    Passages, dense and sparse representations, payload metadata, Page
    provenance, and the sparse indexing settings the running configuration
    expects. A generation that fails any check is never activated and the
    previous one keeps serving.
    """
    reporter = vector_store.generation_report
    report = reporter(
        filter={"pdf_name": filename, "index_generation": generation},
        limit=expected_count,
        value_keys=("sparse_method", "sparse_tokenizer_version"),
    )
    if report is None:
        return _validate_generation_count(
            vector_store, filename, generation, expected_count
        )
    problems: list[str] = []
    total = int(report.get("total", 0))
    if total != expected_count:
        problems.append(f"expected {expected_count} vectors but found {total}")
    if report.get("truncated"):
        problems.append(
            f"the stored points were not fully inspected "
            f"({report.get('inspected', 0)} of {total})"
        )
    without_dense = int(report.get("without_dense", 0) or 0)
    if without_dense:
        problems.append(f"{without_dense} Passages without a dense vector")
    without_sparse = int(report.get("without_sparse", 0) or 0)
    if without_sparse:
        problems.append(f"{without_sparse} Passages without a sparse vector")
    for key, count in sorted((report.get("missing_payload") or {}).items()):
        if count:
            problems.append(f"{count} Passages missing payload for {key}")
    for key, count in sorted((report.get("empty_payload") or {}).items()):
        if count:
            problems.append(f"{count} Passages with blank {key}")
    pages = [int(page) for page in report.get("pages") or []]
    if expected_count and not pages:
        problems.append("Passages have no Page provenance")
    if page_count:
        outside = [page for page in pages if page < 1 or page > page_count]
        if outside:
            problems.append(
                f"Pages {sorted(outside)} fall outside the document's "
                f"{page_count} Pages"
            )
    distinct = report.get("distinct_values") or {}
    problems.extend(
        _sparse_setting_problems(
            "sparse method", distinct.get("sparse_method"), SPARSE_METHOD
        )
    )
    problems.extend(
        _sparse_setting_problems(
            "sparse tokenizer version",
            distinct.get("sparse_tokenizer_version"),
            TOKENIZER_VERSION,
        )
    )
    if problems:
        raise VectorStoreConfigurationError(
            f"Index generation {generation} of {filename} failed validation: "
            + "; ".join(problems)
            + "."
        )
    return report


def activate_generation(
    *,
    files: Any,
    jobs: Any,
    cleanups: Any,
    vector_store: Any,
    job: dict,
    worker_id: str,
    expected_count: int,
    page_count: int | None,
    manifest_json: str,
) -> dict | None:
    """
    Validate a built Index Generation and make it the active one.

    The order is fixed: validate through the store port, record the owed
    removal of the superseded generation, then switch with the generation
    and its manifest in one transaction. A failed replacement never unplugs
    the serving generation, and a restart between retiring and switching
    still retires it. The manifest arrives already built; numbering and
    queueing stay with the jobs repository.

    Returns the ready job row, or None when the claim was lost or the
    Document is gone. Validation failures raise and leave everything as is.
    """
    filename = job["filename"]
    generation = job["generation"]
    _validate_generation(vector_store, filename, generation, expected_count, page_count)
    fresh = files.get_file(filename)
    if fresh is None:
        return None
    # The removal of the generation this one replaces is written down
    # before activation, so a restart between the two still retires the
    # superseded vectors.
    cleanups.schedule(fresh["id"], filename, fresh.get("index_generation"))
    return jobs.mark_ready(
        job["id"],
        worker_id,
        index_generation=generation,
        index_manifest=manifest_json,
    )
