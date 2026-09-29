"""Local services stay private by default (loopback, generated creds, gated remote)."""

import os
import pathlib
import re
import subprocess

import pytest

import providers
import settings as settings_module
from settings import Settings, url_is_loopback, validate

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = BACKEND_DIR / "compose.yaml"
COMPOSE_AUTH = BACKEND_DIR / "compose.qdrant-auth.yaml"
COMPOSE_TEST = BACKEND_DIR / "compose.test.yaml"

_PORT_RE = re.compile(r'"(?:(\d+\.\d+\.\d+\.\d+):)?(\d+):(\d+)"')
_SERVICE_PORT_RE = re.compile(r"^\s*-\s*\"(.*)\"\s*$")


def _published_ports(text: str) -> list[tuple[str | None, str, str]]:
    """Return (host_ip, host_port, container_port) for each published port."""
    return [(m.group(1), m.group(2), m.group(3)) for m in _PORT_RE.finditer(text)]


def _parse_publishing_entry(entry: str) -> tuple[str | None, str, str]:
    """Return the (host_ip, host_port, container_port) of one ports entry."""
    match = _PORT_RE.search(f'"{entry}"')
    assert match is not None, f"unreadable ports entry: {entry!r}"
    return (match.group(1), match.group(2), match.group(3))


def _ports_list(text: str) -> list[str]:
    """Return the literal entries of every ports list in a compose file."""
    entries: list[str] = []
    in_ports = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "ports:":
            in_ports = True
            continue
        if in_ports and stripped and not stripped.startswith("- "):
            in_ports = False
        if in_ports and (match := _SERVICE_PORT_RE.match(line)):
            entries.append(match.group(1))
    return entries


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Each test builds its own Settings instead of reusing a memoized one."""
    settings_module.set_settings(None)
    providers.reset_providers()
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/papermind")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    yield
    settings_module.set_settings(None)
    providers.reset_providers()


def test_compose_publishes_loopback_only():
    """Postgres, Qdrant HTTP, and Qdrant gRPC never bind beyond loopback."""
    for path in (COMPOSE, COMPOSE_TEST):
        published = _ports_list(path.read_text())
        assert published, f"{path.name} must publish service ports explicitly"
        for entry in published:
            # No interpolation: the rendered binding is the literal one below.
            assert "${" not in entry, f"{path.name} must not interpolate {entry}"
            host_ip, host_port, container_port = _parse_publishing_entry(entry)
            assert host_ip == "127.0.0.1", (
                f"{path.name} must bind {host_port}:{container_port} to 127.0.0.1"
            )


def test_effective_compose_config_binds_loopback():
    """`docker compose config` renders the same loopback-only bindings."""
    import shutil
    import subprocess

    if shutil.which("docker") is None or not (BACKEND_DIR / ".infra.env").exists():
        pytest.skip("needs docker and a bootstrapped backend/.infra.env")
    rendered = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config"],
        capture_output=True,
        text=True,
        cwd=BACKEND_DIR.parent,
        check=True,
    ).stdout
    assert "0.0.0.0" not in rendered
    assert rendered.count("host_ip: 127.0.0.1") >= 3


def test_compose_dev_exposes_expected_container_ports():
    """The loopback binding keeps the documented local addresses working."""
    container_ports = {
        container for _, _, container in _published_ports(COMPOSE.read_text())
    }
    assert {"5432", "6333", "6334"} <= container_ports


def test_compose_has_no_shared_default_password():
    """Local database credentials are not fixed defaults in source control."""
    for path in (COMPOSE, COMPOSE_TEST):
        text = path.read_text()
        assert "POSTGRES_PASSWORD: papermind" not in text
        assert "POSTGRES_PASSWORD:papermind" not in text.replace(" ", "")
        assert "papermind:papermind@" not in text


def test_dev_and_test_credentials_are_isolated():
    """Test infra never reads the development password variable."""
    dev = COMPOSE.read_text()
    test = COMPOSE_TEST.read_text()
    assert "POSTGRES_PASSWORD" in test
    # The test compose file uses its own variable names so a development
    # export cannot become the test password (and vice versa).
    assert "POSTGRES_TEST_PASSWORD" in test or "TEST_POSTGRES_PASSWORD" in test
    assert "POSTGRES_TEST_PASSWORD" not in dev and "TEST_POSTGRES_PASSWORD" not in dev


def test_compose_qdrant_supports_api_key():
    """Qdrant can require an API key before any non-loopback deployment."""
    dev = COMPOSE.read_text()
    # An empty QDRANT__SERVICE__API_KEY still switches Qdrant into
    # auth-required mode and locks out keyless local clients, so the base
    # file must not set the server-side variable at all.
    assert "QDRANT__SERVICE__API_KEY" not in dev

    overlay = COMPOSE_AUTH.read_text()
    assert "QDRANT__SERVICE__API_KEY" in overlay
    # The overlay refuses to render without a real key (no empty default).
    assert "${QDRANT_API_KEY:?" in overlay


def test_loopback_qdrant_variants_need_no_key():
    """Local Qdrant addresses validate without authentication."""
    for url in (
        "http://localhost:6333",
        "http://127.0.0.1:6333",
        "http://[::1]:6333",
    ):
        assert url_is_loopback(url) is True
        validate(
            Settings(
                database={"database_url": "postgresql://x"}, vector={"qdrant_url": url}
            )
        )


def test_non_loopback_qdrant_without_key_fails_clearly(capsys, monkeypatch):
    """A remote Qdrant URL without an API key is a configuration error."""
    monkeypatch.setenv("QDRANT_URL", "http://192.168.1.10:6333")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        validate()

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "QDRANT_API_KEY" in stderr
    assert "QDRANT_URL" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    "url",
    [
        "http://qdrant:6333",
        "http://0.0.0.0:6333",
        "http://192.168.1.10:6333",
        "http://[::]:6333",
        "https://vectors.example.com:6333",
        "not-a-url",
        "",
    ],
)
def test_non_loopback_and_unparseable_hosts_are_not_loopback(url):
    """Unknown hosts fail closed: they require an API key."""
    assert url_is_loopback(url) is False


def test_non_loopback_qdrant_with_key_passes(capsys, monkeypatch):
    """Setting the API key unlocks an explicitly remote Qdrant address."""
    monkeypatch.setenv("QDRANT_URL", "http://192.168.1.10:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")

    validate()

    assert "QDRANT_API_KEY" not in capsys.readouterr().err


def test_qdrant_client_receives_configured_api_key(monkeypatch):
    """The application sends the configured Qdrant API key to the client."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")
    seen = {}

    class FakeQdrantClient:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", FakeQdrantClient)

    providers.get_qdrant_client()

    assert seen.get("api_key") == "secret-key"


def test_qdrant_client_without_key_sends_none(monkeypatch):
    """Loopback development sends no API key rather than a placeholder."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    seen = {}

    class FakeQdrantClient:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", FakeQdrantClient)

    providers.get_qdrant_client()

    assert seen.get("api_key") in (None, "")


BOOTSTRAP = BACKEND_DIR / "scripts" / "bootstrap-local.sh"


def _run_bootstrap(backend: pathlib.Path) -> subprocess.CompletedProcess:
    """Run the bootstrap script against a throwaway backend directory."""
    scripts = backend / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    target = scripts / "bootstrap-local.sh"
    target.write_text(BOOTSTRAP.read_text())
    target.chmod(0o755)
    return subprocess.run(
        ["/bin/bash", str(target)], capture_output=True, text=True, check=True
    )


def _generated_password(backend: pathlib.Path) -> str:
    """Read the password the script generated."""
    infra = (backend / ".infra.env").read_text()
    return re.search(r"(?m)^POSTGRES_PASSWORD=(.+)$", infra).group(1)


def test_bootstrap_generates_a_password_outside_source_control(tmp_path):
    """A fresh clone gets a random password and a matching DATABASE_URL."""
    _run_bootstrap(tmp_path)

    password = _generated_password(tmp_path)
    assert password and password != "papermind"
    assert len(password) >= 32
    assert (
        f"postgresql+psycopg://papermind:{password}@localhost:5432/papermind"
        in (tmp_path / ".env").read_text()
    )


def test_bootstrap_never_commits_the_generated_files():
    """The generated files stay untracked in this repository."""
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "backend/.infra.env", "backend/.env"],
        capture_output=True,
        text=True,
        cwd=BACKEND_DIR.parent,
    )

    assert result.returncode == 0, "generated credentials must stay untracked"


def test_bootstrap_rotates_a_shared_default_password(tmp_path):
    """A documented default is replaced, not carried forward."""
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql+psycopg://papermind:papermind@localhost:5432/papermind\n"
    )

    _run_bootstrap(tmp_path)

    password = _generated_password(tmp_path)
    app_env = (tmp_path / ".env").read_text()
    assert password != "papermind"
    assert "papermind:papermind@" not in app_env
    assert password in app_env


def test_bootstrap_repairs_a_drifted_local_database_url(tmp_path):
    """A local URL that no longer matches the generated password is fixed."""
    (tmp_path / ".infra.env").write_text(
        "POSTGRES_USER=papermind\nPOSTGRES_PASSWORD=generated-secret\n"
    )
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql+psycopg://papermind:stale@localhost:5432/papermind\n"
    )

    _run_bootstrap(tmp_path)

    app_env = (tmp_path / ".env").read_text()
    assert "stale@" not in app_env
    assert "papermind:generated-secret@localhost" in app_env


def test_bootstrap_leaves_a_remote_database_url_alone(tmp_path):
    """A non-localhost DATABASE_URL is the operator's, not the script's."""
    remote = "postgresql+psycopg://papermind:remote@db.internal:5432/papermind"
    (tmp_path / ".env").write_text(f"DATABASE_URL={remote}\n")

    _run_bootstrap(tmp_path)

    assert (tmp_path / ".env").read_text().strip() == f"DATABASE_URL={remote}"


def test_bootstrap_keeps_an_unchanged_password(tmp_path):
    """Re-running never rotates a healthy password."""
    _run_bootstrap(tmp_path)
    first = (tmp_path / ".infra.env").read_text()

    _run_bootstrap(tmp_path)

    assert (tmp_path / ".infra.env").read_text() == first


def test_bootstrap_refuses_the_placeholder_password(tmp_path):
    """A copied example file is not a working credential."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    target = scripts / "bootstrap-local.sh"
    target.write_text(BOOTSTRAP.read_text())
    target.chmod(0o755)
    (tmp_path / ".infra.env").write_text(
        "POSTGRES_USER=papermind\nPOSTGRES_PASSWORD=replace-me-run-bootstrap-local-sh\n"
    )

    result = subprocess.run(
        ["/bin/bash", str(target)], capture_output=True, text=True, cwd=tmp_path
    )

    assert result.returncode == 1
    assert "no real POSTGRES_PASSWORD" in result.stderr
