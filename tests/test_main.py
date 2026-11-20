from __future__ import annotations

import json
import os
import signal
import threading
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from links2rag import __main__ as cli
from links2rag import __version__, pipeline
from links2rag.config import Settings
from links2rag.db import RunCounts
from links2rag.pipeline import RunResult

SettingsFactory = Callable[..., Settings]  # the make_settings fixture


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never read a developer's real .env during tests.
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


@pytest.fixture(autouse=True)
def migrations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    # Records the database URLs main() migrated instead of touching Postgres.
    calls: list[str] = []
    monkeypatch.setattr(cli, "migrate", calls.append)
    return calls


def _use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))


def _fake_run(
    monkeypatch: pytest.MonkeyPatch, result: RunResult | Exception, seen: list[Any] | None = None
) -> None:
    def run(
        settings: Settings, mode: str, stop: threading.Event, beat: Any = None, *, force: bool
    ) -> RunResult:
        if seen is not None:
            seen.append((mode, force))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(pipeline, "run", run)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"links2rag {__version__}\n"


def test_bad_config_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("LINKS_URL", "LINKS_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "LINKS_URL" in capsys.readouterr().err


def test_run_modes_and_force(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_settings(monkeypatch, make_settings())
    seen: list[Any] = []
    _fake_run(monkeypatch, RunResult("ok", RunCounts(seen=3, added=2)), seen)
    with caplog.at_level("INFO"):
        assert cli.main(["run"]) == 0
        assert cli.main(["run", "--mode", "reindex"]) == 0
        assert cli.main(["run", "--force"]) == 0
    assert seen == [("full", False), ("reindex", False), ("full", True)]
    assert "seen=3 added=2" in caplog.text


@pytest.mark.parametrize(
    ("result", "code"),
    [
        (RunResult("busy", RunCounts(), "locked"), 0),
        (RunResult("valve", RunCounts(), "too few"), 1),
        (RuntimeError("boom"), 1),
    ],
)
def test_run_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    result: RunResult | Exception,
    code: int,
) -> None:
    _use_settings(monkeypatch, make_settings())
    _fake_run(monkeypatch, result)
    assert cli.main(["run"]) == code


def test_stopped_run_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_settings(monkeypatch, make_settings())

    def run(settings: Settings, mode: str, stop: threading.Event, **kwargs: Any) -> RunResult:
        os.kill(os.getpid(), signal.SIGTERM)
        return RunResult("ok", RunCounts())

    monkeypatch.setattr(pipeline, "run", run)
    with caplog.at_level("INFO"):
        assert cli.main(["run"]) == 0
    assert "Stopped early" in caplog.text


def test_migrate_and_failure(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, migrations: list[str]
) -> None:
    _use_settings(monkeypatch, make_settings(database_url="postgresql://x"))
    assert cli.main(["migrate"]) == 0
    assert migrations == ["postgresql://x"]

    def broken(url: str) -> None:
        raise RuntimeError("no db")

    monkeypatch.setattr(cli, "migrate", broken)
    assert cli.main(["migrate"]) == 1


def test_status(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: SettingsFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use_settings(monkeypatch, make_settings())
    monkeypatch.setattr(pipeline, "status", lambda s: {"source": "links", "assets": 2})
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"source": "links", "assets": 2}

    def broken(s: Settings) -> dict[str, Any]:
        raise RuntimeError("no db")

    monkeypatch.setattr(pipeline, "status", broken)
    assert cli.main(["status"]) == 1


def test_healthcheck(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    beat = tmp_path / "beat"
    monkeypatch.setattr(cli, "HEARTBEAT_PATH", beat)
    assert cli.main(["healthcheck"]) == 1
    beat.touch()
    assert cli.main(["healthcheck"]) == 0


def test_serve_forces_only_the_first_scan(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "HEARTBEAT_PATH", tmp_path / "beat")
    _use_settings(monkeypatch, make_settings(force_reindex=True, scan_interval_minutes=5))
    seen: list[Any] = []
    _fake_run(monkeypatch, RunResult("ok", RunCounts()), seen)
    monkeypatch.setattr(pipeline, "last_full_run", lambda s: None)
    captured: dict[str, Any] = {}

    def fake_serve(**kwargs: Any) -> None:
        captured.update(kwargs)
        kwargs["scan"]("incremental")
        kwargs["scan"]("incremental")
        assert kwargs["last_full"]() is None

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert cli.main(["serve"]) == 0
    assert seen == [("full", True), ("incremental", False)]
    assert captured["interval"] == timedelta(minutes=5)
    assert captured["full_every"] == timedelta(hours=24)
