"""
Which weights a local model loads, and whether it may run its own code.

The embedding model and the reranker are named by repository id. Hugging Face
resolves a bare id to whatever the branch points at today, so an upstream push
silently changes what vectors an index holds and what order the reranker
returns — with nothing in the running application to show for it. This module
is the single place that closes that gap:

- :data:`PINNED_MODEL_REVISIONS` records the commit each shipped model is
  loaded at. Those are the identities every index manifest and every evaluation
  report already had a field for.
- :data:`REVIEWED_REMOTE_CODE_MODELS` records, per repository, the one commit
  whose code has been read. Nothing is in it: both shipped models load through
  ``transformers`` classes that already exist, so neither needs
  ``trust_remote_code``. The mapping exists so that the day a model genuinely
  does need it, the decision is a reviewed commit rather than a branch.

Both loaders (:mod:`services.embeddings.local_embeddings` and
:mod:`services.retrieval.reranker`) build their kwargs here, so neither can
quietly reintroduce an unpinned or remote-code load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

#: The commit each shipped model is loaded at. Changing a value makes every
#: stored vector or rerank order describe weights the application no longer
#: serves, so the index manifest marks the documents stale and asks for a
#: reindex. The commit is also what the evaluation report records, which is
#: what makes two evaluation runs comparable. Keys are canonical repository
#: ids, which is what the Hugging Face API answers for.
PINNED_MODEL_REVISIONS: dict[str, str] = {
    # Dense+sparse-capable multilingual encoder, 1024d Matryoshka-truncated.
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
    # The default gated reranker: a BERT cross-encoder, 22M parameters.
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    # The heavier reranker an operator can select instead.
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
}

#: Repository spellings that resolve to a canonical id rather than being one.
#: The default reranker setting has always used the hyphenated spelling, which
#: Hugging Face redirects; keeping the pin under the canonical id means the
#: spelling an operator typed cannot decide whether the weights are pinned.
MODEL_ID_ALIASES: dict[str, str] = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",
}

#: Repositories whose own modelling code has been read and accepted, mapped to
#: the one commit that review covered. Empty on purpose: the shipped models are
#: all standard ``transformers`` architectures, so running Python from the
#: model repository would be pure risk with no benefit.
REVIEWED_REMOTE_CODE_MODELS: dict[str, str] = {}


class _LoadKwargs(TypedDict, total=False):
    """
    The two arguments a model load takes, named.

    A typed mapping rather than a plain dict so that unpacking it into the
    loader is checked: adding a key here that the library does not accept is a
    type error, not a runtime surprise on the first embedding.
    """

    revision: str
    trust_remote_code: bool


@dataclass(frozen=True)
class ModelSource:
    """
    A repository, the exact commit to load, and its code-execution policy.

    Loaders pass ``revision`` and ``trust_remote_code`` through explicitly
    rather than splatting a dictionary: the two arguments are the whole
    contract, and a reader should not have to look up what a dict might
    contain.
    """

    model_id: str
    revision: str
    trust_remote_code: bool = False

    def load_kwargs(self) -> _LoadKwargs:
        """
        Return the load arguments, for a caller that forwards them as-is.

        ``revision`` is omitted rather than passed as ``None`` when the model is
        unpinned: the library treats an explicit ``None`` differently from an
        absent argument, and the point of the key is that it is never empty.
        """
        kwargs: _LoadKwargs = {"trust_remote_code": self.trust_remote_code}
        if self.revision:
            kwargs["revision"] = self.revision
        return kwargs


def canonical_model_id(model_id: str) -> str:
    """
    Return the repository's canonical id.

    Hugging Face answers to several spellings of the same repository, but only
    one of them is the id the API and the file listing use. Resolving here
    means an alias cannot decide whether the model is pinned.
    """
    seen = {model_id}
    current = model_id
    while current in MODEL_ID_ALIASES and MODEL_ID_ALIASES[current] not in seen:
        seen.add(current)
        current = MODEL_ID_ALIASES[current]
    return current


def model_source(
    model_id: str, revision: str = "", trust_remote_code: bool = False
) -> ModelSource:
    """
    Resolve a repository id to the exact weights the application will load.

    ``revision`` is the operator's setting and wins when set; otherwise a
    repository the table pins loads at that commit. A model that needs to
    execute code from its own repository is refused unless that exact commit
    was reviewed, because code on a branch is code nobody has read.
    """
    canonical = canonical_model_id(model_id)
    # A reviewed repository is pinned by definition: the review covered one
    # commit, so that commit is also the default.
    resolved = (
        revision
        or PINNED_MODEL_REVISIONS.get(canonical)
        or REVIEWED_REMOTE_CODE_MODELS.get(canonical, "")
    )
    if trust_remote_code:
        reviewed = REVIEWED_REMOTE_CODE_MODELS.get(canonical)
        if reviewed is None:
            raise ValueError(
                f"{model_id} is not reviewed to run code from its own repository. "
                "Read the repository's modelling code, then record its commit in "
                "REVIEWED_REMOTE_CODE_MODELS before enabling it."
            )
        if resolved != reviewed:
            raise ValueError(
                f"{model_id} is reviewed at revision {reviewed}, not {resolved or 'an unpinned branch'}. "
                "Load the reviewed revision, or review the new one and record it."
            )
    # The library is given the id the operator wrote, so the cache directory
    # stays the one an operator can recognise and delete.
    return ModelSource(
        model_id=model_id, revision=resolved, trust_remote_code=trust_remote_code
    )
