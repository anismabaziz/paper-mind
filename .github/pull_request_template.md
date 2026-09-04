<!--
Expected title : [Ticket ID] Short description
Example        : [PF-12] Settle the API response contract
-->

## Result

<!-- Two or three sentences: what behavior changes and for whom? -->

## References and responsibilities

<!--
The Lead is the issue assignee and owns the implementation.
The Reviewer must be distinct from the lead and receive the GitHub review request.
-->

- **Project ticket:** <!-- PF-12, T1, etc. -->
- **GitHub issue:** Closes #<!-- GitHub number, e.g. 15 -->
- **Source requirement:** <!-- link to spec, scope, decision, or requirement -->
- **Lead:** @
- **Reviewer:** @
- **Dependencies:** <!-- closed, or an exception explicitly approved by the PM -->

## Proof of execution (mandatory)

<!--
Give the exact commands and their results. Add depending on the change:
- backend/Python : automated API test and/or request/response trace ;
- frontend/TypeScript : screenshot or recording against the real API ;
- documentation/configuration : link, command, or verified behavior.
-->

```text
Commands executed and results:

```

<!-- Drop useful screenshots or recordings here. -->

## Common checks

- [ ] The lead matches the issue assignee and the reviewer is distinct from
      the lead.
- [ ] The ticket dependencies are closed or their exception is documented.
- [ ] The result matches the ticket and its source.
- [ ] No secret, real identifier, or sensitive data is added.
- [ ] No fictitious product data or new execution mock is added. Isolated test
      fixtures remain allowed.
- [ ] The empty, loading, error, and access-denied states concerned are
      handled.
- [ ] Documentation or shared demo data is updated if the behavior requires
      it.
- [ ] The diff contains no change unrelated to the ticket.

## Backend / Python (if relevant)

<!-- Run the commands from the backend directory. Adapt to your stack. -->

- [ ] Lint passes (`ruff`, `flake8`, or your configured tool), or each
      pre-existing failure is unchanged and linked to an issue; the modified
      files pass the targeted check.
- [ ] Tests pass (`pytest`, or your configured runner), or each pre-existing
      failure is identified and linked to an issue in the proof.
- [ ] Allowed **and** denied access is covered when permissions change.
- [ ] Migrations are provided and a dry-run check detects nothing missing.
- [ ] Not applicable (justification):

## Frontend / TypeScript (if relevant)

<!-- Run the commands from the frontend directory. Adapt to your stack. -->

- [ ] Lint passes, or each pre-existing failure is unchanged and linked to an
      issue; the modified files introduce no new error.
- [ ] Build passes with the same explicit treatment of any pre-existing
      failure.
- [ ] The screen was verified against the real API and shared demo data.
- [ ] Shared components and styles are reused.
- [ ] Not applicable (justification):

## Documentation or configuration (if relevant)

- [ ] The links, paths, and commands modified were verified.
- [ ] The examples match the current code and configuration.
- [ ] Not applicable (justification):

## Functional review

<!-- To be completed by the reviewer. -->

- [ ] I read the issue, its source, and its acceptance criteria.
- [ ] Functional change tested locally by the reviewer.
- [ ] The refused or boundary cases concerned were verified.
- [ ] All my blocking change requests are resolved.
- [ ] I approve this pull request, or my review status explains what still
      blocks approval.
- [ ] Not functional (documentation/configuration only) (justification):

## Points of attention for the reviewer

<!-- Known risk, choice to verify, or part that deserves a second look. -->

## Merge authorization

<!--
Reviewer approval does not automatically authorize merging. By default, the
project owner merges; anyone else must have received explicit delegation for
this pull request.
-->

- [ ] The required checks and review are done or their exception is
      documented.
- [ ] The project owner or the explicitly delegated person authorizes the
      merge.