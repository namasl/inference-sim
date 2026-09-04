"""Tests for archwatch.cli — the three commands, their flags, and their exit codes.

Every test drives ``cli.main(argv)`` the way a shell would, so the wiring between the
flags and :mod:`archwatch.detector` is what is under test rather than a hand-built
Namespace. ``scan`` runs the real detector with :func:`detector.build_connectors`
monkeypatched to return fakes: that exercises CLI -> detector -> novelty -> emitter ->
run log end to end with no network in it.

The ``surface`` tests read the real ``support-surface/*.yaml``. That is deliberate — a
command whose only job is to say "the surface loaded and here is what is in it" is
worthless tested against a stub.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from archwatch import cli, detector, emitter
from archwatch.connectors.base import Signal

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
SURFACE_DIR = Path(__file__).resolve().parent.parent / "support-surface"

NOVEL_CONFIG = {
    "architectures": ["ArchwatchCliNetForCausalLM"],
    "model_type": "archwatch_clinet",
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "vocab_size": 128256,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 28672,
    "hidden_act": "silu",
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 131072,
    "archwatch_cli_novel_field": 128,
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def boom(*_a, **_kw):  # pragma: no cover - only fires on a regression
        raise AssertionError("test attempted a live network call")

    monkeypatch.setattr(socket, "socket", boom)
    import requests

    monkeypatch.setattr(requests, "Session", boom)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)


class FakeConnector:
    def __init__(self, name="hf", signals=(), *, error=None):
        self.name = name
        self.signals = list(signals)
        self.error = error
        self.polls = 0

    def poll(self, since):
        self.polls += 1
        if self.error:
            raise self.error
        return list(self.signals)


def novel_signal(arch: str = "ArchwatchCliNetForCausalLM") -> Signal:
    # The model id is derived from the architecture: join_signals unions signals that
    # share a normalized repo id, so two architectures under one repo id would collapse
    # into a single candidate and the cap test would have nothing to cap.
    model_id = "moonshotai/" + arch.removesuffix("ForCausalLM") + "-70B"
    config = dict(NOVEL_CONFIG, architectures=[arch])
    return Signal(
        source="hf",
        observed_at=NOW,
        arch_ids=[arch],
        model_type="archwatch_clinet",
        model_ids=[model_id],
        org="moonshotai",
        display_name=model_id,
        config=config,
        urls={"hf": f"https://huggingface.co/{model_id}"},
        evidence="created in window",
        raw_ref=model_id,
    )


@pytest.fixture
def fake_sources(monkeypatch):
    """Replace connector construction so ``scan`` runs offline against fakes."""
    holder: dict[str, list] = {"connectors": [FakeConnector("hf", [novel_signal()])]}

    def build(sources, cfg=None, **kwargs):
        holder["sources"] = list(sources)
        holder["cfg"] = cfg
        return list(holder["connectors"]), None

    monkeypatch.setattr(detector, "build_connectors", build)
    return holder


# ---------------------------------------------------------------------------
# parser plumbing
# ---------------------------------------------------------------------------


def test_a_command_is_required():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "archwatch" in capsys.readouterr().out


def test_scan_has_no_live_mode(capsys):
    """``--no-dry-run`` exists only to explain why it cannot work."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["scan", "--no-dry-run"])
    assert exc.value.code == 2
    assert "no live mode" in capsys.readouterr().err


def test_dry_run_defaults_to_true():
    args = cli.build_parser().parse_args(["scan"])
    assert args.dry_run is True


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------


def test_surface_prints_counts(capsys):
    assert cli.main(["surface"]) == 0
    out = capsys.readouterr().out
    assert "BLIS support surface" in out
    assert "parsed_fields" in out
    assert "known_architectures" in out
    assert "gaps" in out
    # A fresh checkout must not report an empty surface.
    assert "WARNING" not in out


def test_surface_json_is_machine_readable(capsys):
    assert cli.main(["surface", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["parsed_fields"] > 0
    assert data["known_architectures"] > 0
    assert isinstance(data["parsed_fields_by_role"], dict)


def test_surface_explicit_dir(capsys):
    assert cli.main(["surface", "--surface-dir", str(SURFACE_DIR), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["gaps"] > 0


def test_surface_reports_a_bad_directory_instead_of_traceback(tmp_path, capsys):
    assert cli.main(["surface", "--surface-dir", str(tmp_path / "nope")]) == 1
    assert "could not load the support surface" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def write_stub(out_dir: Path, arch: str = "ShownNetForCausalLM") -> Path:
    from archwatch.connectors.base import Candidate

    cand = Candidate(arch_id=arch, display_name=arch, signals=[novel_signal(arch)])
    cand.triggers = ["T1", "T3"]
    cand.significance = ["S1", "S2"]
    cand.unparsed_fields = ["archwatch_cli_novel_field"]
    cand.est_total_params = 70_553_706_496
    return emitter.write_issue(cand, out_dir).path


def test_show_on_an_empty_directory(tmp_path, capsys):
    assert cli.main(["show", "--out", str(tmp_path)]) == 0
    assert "no issue stubs yet" in capsys.readouterr().out


def test_show_missing_directory_is_an_error(tmp_path, capsys):
    assert cli.main(["show", "--out", str(tmp_path / "gone")]) == 1
    assert "no such directory" in capsys.readouterr().err


def test_show_lists_a_stub_from_its_front_matter(tmp_path, capsys):
    write_stub(tmp_path)
    assert cli.main(["show", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ShownNetForCausalLM" in out
    assert "T1,T3" in out
    assert "S1,S2" in out
    assert "70.6B" in out  # est_total_params, humanized
    assert "1 stub(s)" in out
    assert "stage 2: 0 done, 1 pending" in out


def test_show_json_carries_the_front_matter(tmp_path, capsys):
    write_stub(tmp_path)
    assert cli.main(["show", "--out", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    row = data["issues"][0]
    assert data["out_dir"] == str(tmp_path)
    assert row["arch_id"] == "ShownNetForCausalLM"
    assert row["triggers"] == ["T1", "T3"]
    assert row["sources"] == ["hf"]
    assert row["schema"] == emitter.SCHEMA
    assert row["stage2"] == "pending"
    assert "error" not in row


def test_show_detects_stage2_from_content_below_the_marker(tmp_path, capsys):
    """PLAN.md addendum 3: completion is content after the marker, not a flag.

    The front matter still says ``stage2: pending`` after the skill appends its
    analysis — it is regenerated from the candidate and the skill never edits it — so a
    reader that trusted that field would report every finished deep dive as pending.
    """
    path = write_stub(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8") + "\n## Mechanism\n\nBucket 3.\n",
        encoding="utf-8",
    )
    assert "stage2: pending" in path.read_text(encoding="utf-8")

    assert cli.main(["show", "--out", str(tmp_path), "--json"]) == 0
    row = json.loads(capsys.readouterr().out)["issues"][0]
    assert row["stage2"] == "done"
    assert row["stage2_chars"] > 0


def test_show_reports_a_malformed_stub_rather_than_hiding_it(tmp_path, capsys):
    (tmp_path / "Broken.md").write_text("no front matter here\n", encoding="utf-8")
    (tmp_path / "Bad.md").write_text("---\n: : not yaml : :\n---\nbody\n", encoding="utf-8")
    assert cli.main(["show", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no front matter" in out
    assert "Broken.md" in out and "Bad.md" in out


def test_show_sorts_by_detected_at_then_by_arch_id(tmp_path, capsys):
    write_stub(tmp_path, "AlphaNetForCausalLM")
    write_stub(tmp_path, "ZetaNetForCausalLM")
    assert cli.main(["show", "--out", str(tmp_path), "--sort", "arch_id", "--json"]) == 0
    ids = [r["arch_id"] for r in json.loads(capsys.readouterr().out)["issues"]]
    assert ids == ["AlphaNetForCausalLM", "ZetaNetForCausalLM"]


def test_show_ignores_non_markdown_files(tmp_path, capsys):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    assert cli.main(["show", "--out", str(tmp_path)]) == 0
    assert "no issue stubs yet" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def test_scan_end_to_end_writes_a_stub_and_a_run_log(tmp_path, fake_sources, capsys):
    out = tmp_path / "issues"
    runlog = tmp_path / ".runlog"
    code = cli.main(
        [
            "scan",
            "--sources", "hf",
            "--window-days", "1",
            "--out", str(out),
            "--runlog-dir", str(runlog),
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "(dry-run)" in printed
    assert "ArchwatchCliNetForCausalLM" in printed
    assert "run log" in printed

    stubs = list(out.glob("*.md"))
    assert [p.name for p in stubs] == ["ArchwatchCliNetForCausalLM.md"]

    logs = list(runlog.glob("*.json"))
    assert len(logs) == 1
    data = json.loads(logs[0].read_text(encoding="utf-8"))
    assert data["schema"] == detector.RUNLOG_SCHEMA
    assert data["window"]["days"] == 1
    assert data["out_dir"] == str(out)
    assert data["counts"]["passed"] == 1
    assert data["issues"][0]["arch_id"] == "ArchwatchCliNetForCausalLM"
    assert "T1" in data["issues"][0]["triggers"]
    assert fake_sources["sources"] == ["hf"]


def test_scan_json_output_is_the_run_log(tmp_path, fake_sources, capsys):
    code = cli.main(
        ["scan", "--sources", "hf", "--out", str(tmp_path / "i"), "--no-runlog", "--json"]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == detector.RUNLOG_SCHEMA
    assert data["counts"]["passed"] == 1
    assert not list(tmp_path.glob("**/*.json"))  # --no-runlog wrote nothing


def test_scan_no_runlog_skips_the_file(tmp_path, fake_sources, capsys):
    runlog = tmp_path / ".runlog"
    assert cli.main(
        ["scan", "--out", str(tmp_path / "i"), "--runlog-dir", str(runlog), "--no-runlog"]
    ) == 0
    assert not runlog.exists()
    assert "run log" not in capsys.readouterr().out


def test_scan_dedup_is_scoped_to_the_out_dir(tmp_path, fake_sources, capsys):
    """Two runs into the same --out: the second must dedup, not re-emit."""
    out = tmp_path / "issues"
    args = ["scan", "--sources", "hf", "--out", str(out), "--no-runlog"]
    assert cli.main(args) == 0
    fake_sources["connectors"] = [FakeConnector("hf", [novel_signal()])]
    capsys.readouterr()
    assert cli.main(args) == 0
    second = capsys.readouterr().out
    assert "already_reported=1" in second
    assert "passed     : 0" in second
    assert len(list(out.glob("*.md"))) == 1


def test_scan_survives_a_dead_source_and_still_exits_zero(tmp_path, fake_sources, capsys):
    fake_sources["connectors"] = [
        FakeConnector("hf", [novel_signal()]),
        FakeConnector("vllm", error=RuntimeError("GitHub 403")),
    ]
    code = cli.main(
        ["scan", "--sources", "hf,vllm", "--out", str(tmp_path / "i"), "--no-runlog"]
    )
    out = capsys.readouterr().out
    assert code == 0  # a partial scan is a real result
    assert "PARTIAL SCAN" in out
    assert "FAILED" in out
    assert (tmp_path / "i" / "ArchwatchCliNetForCausalLM.md").is_file()


def test_scan_exits_one_when_every_source_fails(tmp_path, fake_sources, capsys):
    fake_sources["connectors"] = [FakeConnector("hf", error=OSError("dns"))]
    code = cli.main(["scan", "--sources", "hf", "--out", str(tmp_path / "i"), "--no-runlog"])
    assert code == 1
    assert "NOTHING SCANNED" in capsys.readouterr().out


def test_scan_rejects_an_unknown_source_before_polling(tmp_path, fake_sources, capsys):
    code = cli.main(["scan", "--sources", "hf,inferncex", "--out", str(tmp_path)])
    assert code == 1
    assert "unknown source" in capsys.readouterr().err
    assert "sources" not in fake_sources  # never got as far as building connectors


def test_scan_rejects_a_non_positive_window(tmp_path, capsys):
    assert cli.main(["scan", "--window-days", "0", "--out", str(tmp_path)]) == 1
    assert "--window-days must be positive" in capsys.readouterr().err


def test_scan_max_issues_caps_what_is_written(tmp_path, fake_sources, capsys):
    fake_sources["connectors"] = [
        FakeConnector(
            "hf",
            [novel_signal("ArchwatchCliOneForCausalLM"), novel_signal("ArchwatchCliTwoForCausalLM")],
        )
    ]
    out = tmp_path / "i"
    code = cli.main(
        ["scan", "--out", str(out), "--max-issues", "1", "--no-runlog", "--json"]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"]["max_issues_per_run"] == 1
    assert data["counts"]["passed"] == 1
    assert data["counts"]["suppressed_by_reason"] == {"over_cap": 1}
    assert len(list(out.glob("*.md"))) == 1


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("--recheck-known", True),
        ("--no-recheck-known", False),
        (None, None),  # neither flag: whatever the calibrated default is
    ],
)
def test_scan_recheck_known_flags_reach_the_config(
    tmp_path, fake_sources, capsys, flag, expected
):
    """Both directions, and the untouched default.

    Asserting only ``--recheck-known -> True`` stopped proving anything the day the
    backtest flipped the default to True: the flag and the default agreed, so a broken
    flag would have looked identical. The no-flag case is compared against
    ``DetectorConfig()`` rather than a literal, so re-calibration cannot make this test
    lie either.
    """
    from archwatch.config import DetectorConfig as Cfg

    argv = ["scan", "--out", str(tmp_path / "i"), "--no-runlog", "--json"]
    if flag:
        argv.append(flag)
    assert cli.main(argv) == 0
    got = json.loads(capsys.readouterr().out)["config"]["recheck_known_architectures"]
    assert got is (Cfg().recheck_known_architectures if expected is None else expected)


def test_scan_recheck_flags_are_mutually_exclusive(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["scan", "--recheck-known", "--no-recheck-known"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_scan_no_trending_is_passed_through(tmp_path, monkeypatch, capsys):
    seen: dict[str, object] = {}
    real = detector.scan

    def spy(cfg=None, **kwargs):
        seen.update(kwargs)
        return real(cfg, **kwargs)

    monkeypatch.setattr(detector, "scan", spy)
    monkeypatch.setattr(
        detector, "build_connectors", lambda sources, cfg=None, **kw: ([], None)
    )
    assert cli.main(["scan", "--out", str(tmp_path), "--no-trending", "--no-runlog"]) == 0
    assert seen["trending"] is False
    assert seen["out_dir"] == str(tmp_path)
    assert seen["dry_run"] is True


def test_scan_reports_a_broken_surface_without_a_traceback(tmp_path, fake_sources, capsys):
    code = cli.main(
        [
            "scan",
            "--out", str(tmp_path / "i"),
            "--surface-dir", str(tmp_path / "missing"),
            "--no-runlog",
        ]
    )
    assert code == 1
    assert "archwatch scan failed" in capsys.readouterr().err


def test_scan_still_prints_the_summary_when_the_run_log_cannot_be_written(
    tmp_path, fake_sources, capsys, monkeypatch
):
    """A run log we cannot write must not throw away a scan we already did."""

    def boom(self, dir=None):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(detector.RunSummary, "write_run_log", boom)
    code = cli.main(["scan", "--out", str(tmp_path / "i"), "--no-trending"])
    captured = capsys.readouterr()
    assert code == 0
    assert "could not write run log" in captured.err
    assert "ArchwatchCliNetForCausalLM" in captured.out
    assert (tmp_path / "i" / "ArchwatchCliNetForCausalLM.md").is_file()


def test_scan_default_out_dir_is_the_emitters_issues_dir(tmp_path, monkeypatch, capsys):
    """Without --out, both the writer and the dedup use ``issues/``."""
    seen: dict[str, object] = {}

    def spy(cands, surface, cfg, *, issues_dir=None, already_reported=None):
        seen["issues_dir"] = issues_dir
        from archwatch.novelty import EvaluationReport

        return EvaluationReport()

    monkeypatch.setattr(detector, "evaluate_detailed", spy)
    monkeypatch.setattr(detector, "build_connectors", lambda s, cfg=None, **kw: ([], None))
    assert cli.main(["scan", "--no-runlog"]) == 0
    assert seen["issues_dir"] == emitter.default_issues_dir()


def test_read_stub_on_an_unreadable_path(tmp_path):
    row = cli.read_stub(tmp_path / "does-not-exist.md")
    assert "unreadable" in row["error"]
