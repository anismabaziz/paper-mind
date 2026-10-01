"""
Dependency and model-revision audits.

Two questions, both answered by a job rather than by a person noticing a
release note: are the packages this project installs carrying known
vulnerabilities, and do the model revisions it pins still exist upstream. The
policy that turns tool output into a pass or a failure lives in
:mod:`audit.policy`; the CLI in :mod:`audit.cli` is the entry point CI calls.
"""
