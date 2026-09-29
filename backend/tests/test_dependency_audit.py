"""
Dependency audit: which known vulnerabilities the project is exposed to.

An advisory database entry is not automatically a problem for this project. A
critique in a package nothing here imports, or a DoS in a parser that only ever
sees documents a user chose to upload, are different decisions. What the audit
has to do is make that distinction visible and then hold the line: a direct
dependency with a new advisory fails, and anything else is reported with the
reason it is not blocking.

The tests feed recorded tool output rather than calling the tools, so the
parsing and the gating are both checkable without a network.
"""

import json
from pathlib import Path

import pytest

from audit.comparison import changed_paths_needing_a_comparison
from audit.policy import (
    Finding,
    decide,
    direct_python_packages,
    normalize_name,
    parse_npm_audit,
    parse_pip_audit,
)
from audit.report import format_report


#: A pip-audit report shape: a list of installed packages, each with the
#: advisories that name it. Recorded from a real run.
PIP_AUDIT_REPORT = {
    "dependencies": [
        {"name": "Flask", "version": "3.1.3", "vulns": []},
        {
            "name": "pymupdf",
            "version": "1.27.2.2",
            "vulns": [
                {
                    "id": "GHSA-8f5x-9w5v-2g3c",
                    "fix_versions": ["1.27.3"],
                    "description": "A heap overflow in page rendering.",
                }
            ],
        },
        {
            "name": "tiktoken",
            "version": "0.14.0",
            "vulns": [
                {
                    "id": "GHSA-aaaa-bbbb-cccc",
                    "fix_versions": [],
                    "description": "Unbounded allocation on crafted input.",
                }
            ],
        },
    ],
    "fixes": [],
}

#: An npm audit report shape. `via` carries a string for a package's own
#: advisory and an object for one inherited from a dependency.
NPM_AUDIT_REPORT = {
    "auditReportVersion": 2,
    "vulnerabilities": {
        "axios": {
            "name": "axios",
            "severity": "high",
            "isDirect": True,
            "via": [
                "A prototype pollution gadget in the config merge.",
                {
                    "source": 1234,
                    "name": "follow-redirects",
                    "dependency": "follow-redirects",
                    "title": "follow-redirects leaks auth headers on redirect",
                    "url": "https://github.com/advisories/GHSA-r4q5-vmmm-2653",
                    "severity": "moderate",
                },
            ],
            "fixAvailable": {"name": "axios", "version": "1.20.0"},
        },
        "brace-expansion": {
            "name": "brace-expansion",
            "severity": "high",
            "isDirect": False,
            "via": ["DoS via exponential-time expansion."],
            "fixAvailable": True,
        },
    },
    "metadata": {"vulnerabilities": {"total": 2, "high": 2}},
}


class TestNormalizeName:
    """TestNormalizeName."""

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("Flask", "flask"),
            ("sentence-transformers", "sentence-transformers"),
            ("sentence_transformers", "sentence-transformers"),
            ("Pillow", "pillow"),
            ("zope.interface", "zope-interface"),
        ],
    )
    def test_names_compare_the_way_pip_compares_them(self, spelling, expected):
        """Do test names compare the way pip compares them."""
        assert normalize_name(spelling) == expected


class TestParsePipAudit:
    """TestParsePipAudit."""

    def test_a_package_with_no_advisory_produces_no_finding(self):
        """Do test a package with no advisory produces no finding."""
        findings = parse_pip_audit(PIP_AUDIT_REPORT, direct=set())

        assert [f.package for f in findings] == ["pymupdf", "tiktoken"]

    def test_it_reports_the_advisory_id_and_the_installed_version(self):
        """Do test it reports the advisory id and the installed version."""
        findings = parse_pip_audit(PIP_AUDIT_REPORT, direct={"pymupdf"})

        assert findings[0] == Finding(
            ecosystem="python",
            package="pymupdf",
            version="1.27.2.2",
            advisory="GHSA-8f5x-9w5v-2g3c",
            severity="unknown",
            direct=True,
            fix_available=True,
        )

    def test_an_advisory_with_no_fixed_version_is_still_reported(self):
        """No upstream fix is exactly when a risk decision is worth writing down."""
        findings = parse_pip_audit(PIP_AUDIT_REPORT, direct={"tiktoken"})

        tiktoken = next(f for f in findings if f.package == "tiktoken")
        assert tiktoken.fix_available is False

    def test_it_marks_a_package_the_project_does_not_import_directly(self):
        """Do test it marks a package the project does not import directly."""
        findings = parse_pip_audit(PIP_AUDIT_REPORT, direct={"flask"})

        assert [(f.package, f.direct) for f in findings] == [
            ("pymupdf", False),
            ("tiktoken", False),
        ]

    def test_a_report_it_cannot_read_raises_rather_than_reporting_clean(self):
        """Do test a report it cannot read raises rather than reporting clean."""
        with pytest.raises(ValueError, match="pip-audit"):
            parse_pip_audit({"unexpected": True}, direct=set())


class TestParseNpmAudit:
    """TestParseNpmAudit."""

    def test_it_reports_a_direct_package_advisory(self):
        """Do test it reports a direct package advisory."""
        findings = parse_npm_audit(NPM_AUDIT_REPORT)

        own = [
            f
            for f in findings
            if f.package == "axios" and f.advisory.startswith("A proto")
        ]
        assert [(f.package, f.advisory, f.severity) for f in own] == [
            ("axios", "A prototype pollution gadget in the config merge.", "high")
        ]

    def test_it_reports_an_advisory_inherited_from_a_dependency(self):
        """The transitive one is real exposure and belongs in the report."""
        findings = parse_npm_audit(NPM_AUDIT_REPORT)

        inherited = [f for f in findings if f.advisory == "GHSA-r4q5-vmmm-2653"]
        assert [(f.package, f.severity) for f in inherited] == [("axios", "moderate")]
        # Attributed to the direct package, which is the one this project can
        # upgrade.
        assert inherited[0].direct is True

    def test_it_names_the_ghsa_id_when_the_inherited_advisory_has_one(self):
        """Do test it names the ghsa id when the inherited advisory has one."""
        findings = parse_npm_audit(NPM_AUDIT_REPORT)

        assert any(f.advisory == "GHSA-r4q5-vmmm-2653" for f in findings)

    def test_a_transitive_package_is_not_marked_direct(self):
        """Do test a transitive package is not marked direct."""
        findings = parse_npm_audit(NPM_AUDIT_REPORT)

        assert [f.direct for f in findings if f.package == "brace-expansion"] == [False]

    def test_a_report_it_cannot_read_raises_rather_than_reporting_clean(self):
        """Do test a report it cannot read raises rather than reporting clean."""
        with pytest.raises(ValueError, match="npm audit"):
            parse_npm_audit({"auditReportVersion": 2})


class TestDirectPythonPackages:
    """TestDirectPythonPackages."""

    def test_it_reads_the_runtime_optional_and_development_dependencies(self, tmp_path):
        """Do test it reads the runtime optional and development dependencies."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            "\n".join(
                [
                    "[project]",
                    'name = "backend"',
                    "dependencies = [",
                    '    "flask>=3.1.3",',
                    '    "psycopg[binary]>=3.2.0",',
                    "]",
                    "",
                    "[project.optional-dependencies]",
                    "docling = ['docling>=2.40.0']",
                    "",
                    "[dependency-groups]",
                    "dev = ['pytest>=8.3.0', 'mypy>=1.15.0']",
                ]
            )
        )

        assert direct_python_packages(pyproject) == {
            "flask",
            "psycopg",
            "docling",
            "pytest",
            "mypy",
        }

    def test_an_extras_marker_is_not_part_of_the_name(self, tmp_path):
        """Do test an extras marker is not part of the name."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "backend"\ndependencies = ["psycopg[binary]>=3.2.0"]\n'
        )

        assert direct_python_packages(pyproject) == {"psycopg"}

    def test_a_missing_file_raises_rather_than_auditing_nothing(self, tmp_path):
        """Do test a missing file raises rather than auditing nothing."""
        with pytest.raises(OSError):
            direct_python_packages(tmp_path / "absent.toml")


class TestDecide:
    """TestDecide."""

    def test_no_findings_is_a_pass(self):
        """Do test no findings is a pass."""
        verdict = decide(())

        assert verdict.ok is True
        assert verdict.blocking == ()

    def test_a_new_direct_finding_blocks(self):
        """Do test a new direct finding blocks."""
        verdict = decide(
            (
                Finding(
                    ecosystem="python",
                    package="flask",
                    version="3.1.3",
                    advisory="GHSA-1111-2222-3333",
                    severity="high",
                    direct=True,
                    fix_available=True,
                ),
            )
        )

        assert verdict.ok is False
        assert verdict.blocking[0].package == "flask"

    def test_a_transitive_finding_is_reported_but_not_blocking(self):
        """Do test a transitive finding is reported but not blocking."""
        verdict = decide(
            (
                Finding(
                    ecosystem="python",
                    package="idna",
                    version="3.10",
                    advisory="GHSA-4444-5555-6666",
                    severity="moderate",
                    direct=False,
                    fix_available=True,
                ),
            )
        )

        assert verdict.ok is True
        assert verdict.reported[0].package == "idna"

    def test_an_accepted_advisory_is_neither_blocking_nor_silent(self):
        """Do test an accepted advisory is neither blocking nor silent."""
        finding = Finding(
            ecosystem="python",
            package="tiktoken",
            version="0.14.0",
            advisory="GHSA-aaaa-bbbb-cccc",
            severity="moderate",
            direct=True,
            fix_available=False,
        )

        verdict = decide(
            (finding,), accepted={"GHSA-aaaa-bbbb-cccc": "no upstream fix"}
        )

        assert verdict.ok is True
        assert verdict.blocking == ()
        assert verdict.accepted == (finding,)


class TestFormatReport:
    """TestFormatReport."""

    def test_a_clean_run_names_both_audits(self):
        """Do test a clean run names both audits."""
        text = format_report(decide(()))

        assert "no known vulnerabilities" in text.lower()

    def test_it_names_a_blocking_finding_with_its_fix(self):
        """Do test it names a blocking finding with its fix."""
        verdict = decide(
            (
                Finding(
                    ecosystem="node",
                    package="axios",
                    version="1.8.4",
                    advisory="GHSA-q8qp-cvcw-x6jj",
                    severity="high",
                    direct=True,
                    fix_available=True,
                ),
            )
        )

        text = format_report(verdict)

        assert "axios" in text
        assert "GHSA-q8qp-cvcw-x6jj" in text

    def test_it_flags_an_acceptance_nothing_matches_any_more(self):
        """The package was upgraded; the decision outlived its advisory."""
        reasons = {"GHSA-1": "replaced on the next major"}

        text = format_report(decide(()), accepted=reasons)

        assert "no longer match" in text
        assert "GHSA-1" in text

    def test_it_names_the_reason_an_advisory_was_accepted(self):
        """Do test it names the reason an advisory was accepted."""
        finding = Finding(
            ecosystem="python",
            package="tiktoken",
            version="0.14.0",
            advisory="GHSA-aaaa-bbbb-cccc",
            severity="moderate",
            direct=True,
            fix_available=False,
        )
        reasons = {"GHSA-aaaa-bbbb-cccc": "no upstream fix; input is our own text"}

        text = format_report(decide((finding,), accepted=reasons), accepted=reasons)

        assert "GHSA-aaaa-bbbb-cccc" in text
        assert "no upstream fix; input is our own text" in text


class TestRecordShape:
    """TestRecordShape."""

    def test_a_tool_report_is_json_not_python_reprs(self, tmp_path):
        """The parsers take what the tool printed, so they take text."""
        findings = parse_pip_audit(
            json.loads(json.dumps(PIP_AUDIT_REPORT)), direct=set()
        )

        assert isinstance(findings, tuple)


class TestToolInvocation:
    """Running the tools, which report findings through their exit code."""

    def test_a_nonzero_exit_still_yields_a_report(self, tmp_path, monkeypatch):
        """
        A finding still produces a verdict.

        Both tools exit non-zero whenever they find anything, and npm exits
        non-zero on a clean tree too. Crashing on that would mean an accepted
        advisory and a clean report were both unreachable.
        """
        import audit.cli as cli

        monkeypatch.setattr(
            cli,
            "_run_json",
            lambda command, cwd, tool: json.loads(json.dumps(NPM_AUDIT_REPORT)),
        )

        verdict = cli.audit_node(tmp_path, {})

        assert [f.advisory for f in verdict.blocking]

    def test_a_tool_that_produced_nothing_is_an_error_not_a_pass(
        self, tmp_path, monkeypatch
    ):
        """An empty stdout cannot be a clean report."""
        import audit.cli as cli

        monkeypatch.setattr(
            cli,
            "_run_json",
            lambda command, cwd, tool: (_ for _ in ()).throw(
                ValueError(f"{tool} produced no report")
            ),
        )

        try:
            cli.audit_node(tmp_path, {})
        except ValueError as error:
            assert "npm audit" in str(error)
        else:
            raise AssertionError("an unreadable report must not read as clean")


class TestTheCommandItself:
    """What a job log shows when an audit cannot complete."""

    def test_a_tool_failure_is_reported_not_raised(self, tmp_path, monkeypatch, capsys):
        """A job that dies with a traceback is a job nobody can act on."""
        import audit.cli as cli

        monkeypatch.setattr(
            cli,
            "_run_json",
            lambda command, cwd, tool: (_ for _ in ()).throw(
                ValueError("npm audit produced no report (exit 1)")
            ),
        )

        exit_code = cli.main(["node", "--root", str(tmp_path)])

        assert exit_code == 1
        assert "Could not complete" in capsys.readouterr().out

    def test_it_returns_zero_when_nothing_blocks(self, tmp_path, monkeypatch, capsys):
        """Do test it returns zero when nothing blocks."""
        import audit.cli as cli

        monkeypatch.setattr(
            cli,
            "_run_json",
            lambda command, cwd, tool: {"dependencies": [], "fixes": []},
        )
        (tmp_path / "frontend").mkdir()
        (tmp_path / "backend").mkdir()
        (tmp_path / "backend" / "pyproject.toml").write_text(
            '[project]\nname = "backend"\ndependencies = []\n'
        )

        assert cli.main(["python", "--root", str(tmp_path)]) == 0
        capsys.readouterr()


class TestRecordedDecisions:
    """The file a reviewer edits to accept an advisory."""

    def test_it_reads_the_accepted_map(self, tmp_path):
        """Do test it reads the accepted map."""
        from audit.cli import load_accepted

        path = tmp_path / "accepted.json"
        path.write_text(
            json.dumps({"_comment": ["notes"], "accepted": {"GHSA-1": "why"}})
        )

        assert load_accepted(path) == {"GHSA-1": "why"}

    def test_a_file_without_the_map_raises_rather_than_guessing(self, tmp_path):
        """
        A file with no map is refused, not guessed at.

        A flat file would parse as an accepted map, and its prose keys would
        then silence whatever advisory happened to be named "notes".
        """
        from audit.cli import load_accepted

        path = tmp_path / "accepted.json"
        path.write_text(json.dumps({"_comment": ["notes"]}))

        try:
            load_accepted(path)
        except ValueError as error:
            assert "accepted" in str(error)
        else:
            raise AssertionError("a file with no map must not be read as one")

    def test_a_missing_file_is_no_decisions_yet(self, tmp_path):
        """Do test a missing file is no decisions yet."""
        from audit.cli import load_accepted

        assert load_accepted(tmp_path / "absent.json") == {}


class TestChangesNeedingAComparison:
    """
    Which changed files are the ones that can move a measurement silently.

    Tests measure contracts, not outcomes, so a dependency bump or a model
    revision passes every test while changing retrieval quality, latency, and
    cost. The set is what a pull request is checked against.
    """

    @pytest.mark.parametrize(
        "changed",
        [
            "backend/uv.lock",
            "backend/pyproject.toml",
            "frontend/package-lock.json",
            "frontend/package.json",
            "backend/services/models.py",
            "backend/compose.yaml",
        ],
    )
    def test_it_includes_what_can_move_the_numbers(self, changed):
        """Do test it includes what can move the numbers."""
        assert changed in changed_paths_needing_a_comparison()

    @pytest.mark.parametrize(
        "changed",
        ["README.md", "frontend/src/app/chat.tsx", "backend/routes/health.py"],
    )
    def test_it_excludes_what_cannot(self, changed):
        """A copy edit does not need an evaluation run attached to it."""
        assert changed not in changed_paths_needing_a_comparison()


class TestReportingThatAComparisonIsDue:
    """What the check says when a pull request touches one of those files."""

    def test_it_names_the_files_and_the_command(self, tmp_path, capsys):
        """Do test it names the files and the command."""
        import audit.cli as cli

        exit_code = cli.main(["comparison", "--changed", "backend/uv.lock,README.md"])

        assert exit_code == 0
        printed = capsys.readouterr().out
        assert "backend/uv.lock" in printed
        assert "evaluation.cli" in printed

    def test_a_change_that_cannot_move_a_measurement_passes_silently(self, capsys):
        """Do test a change that cannot move a measurement passes silently."""
        import audit.cli as cli

        assert cli.main(["comparison", "--changed", "README.md"]) == 0
        assert capsys.readouterr().out == ""

    def test_it_warns_but_does_not_block(self, capsys):
        """
        A missing comparison is a review conversation, not a red build.

        Failing here would mean every dependency bump needs a paid live run
        before it can be merged, including a security patch that has to land
        today. The check makes the omission visible and names the fix.
        """
        import audit.cli as cli

        assert cli.main(["comparison", "--changed", "backend/services/models.py"]) == 0
        assert "comparison" in capsys.readouterr().out.lower()


class TestThePathListIsReal:
    """The list names files that exist, or it is quietly wrong."""

    def test_every_listed_path_exists_in_the_repository(self):
        """Do test every listed path exists in the repository."""
        from audit.comparison import DEPENDENCY_AND_MODEL_FILES

        root = Path(__file__).resolve().parents[2]

        missing = [
            path
            for path in sorted(DEPENDENCY_AND_MODEL_FILES)
            if not (root / path).exists()
        ]

        assert missing == [], f"listed but absent from the repository: {missing}"

    def test_the_lockfiles_and_manifests_are_all_covered(self):
        """A new manifest that decides versions has to be added deliberately."""
        from audit.comparison import DEPENDENCY_AND_MODEL_FILES

        for expected in ("backend/uv.lock", "frontend/package-lock.json"):
            assert expected in DEPENDENCY_AND_MODEL_FILES
