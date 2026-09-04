"""
    Retrieval and answer-quality evaluation against a ground-truth fixture.

    The evaluator is built on injected components — an embedder, a vector
    index, an answer generator, and a faithfulness judge — so the whole run
    can execute against deterministic fakes in tests. Only the CLI wires the
    real providers, and only behind an explicit ``--live`` flag.
"""
