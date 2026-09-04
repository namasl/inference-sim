"""``archwatch`` command line: ``scan``, ``show``, ``surface``.

Three commands, one of which does work:

``scan``
    Run the pipeline once over a window, write stubs, write a run log. **Dry-run is
    the only mode**: ``--dry-run`` defaults to true and exists so the invocation says
    out loud what it does. There is deliberately no ``--no-dry-run`` that works —
    passing it is an error with an explanation, not a hidden capability.
``show``
    List what is already in ``issues/``, read from each stub's YAML front matter.
    Whether stage 2 has run is decided by content after the emitter's marker
    (``emitter.split_stub``), never by a front-matter flag — PLAN.md addendum 3.
``surface``
    Print the loaded support surface's counts. A two-second sanity check that the
    YAML in ``support-surface/`` still parses and still contains what it should; run
    it first when a scan produces a suspiciously empty result.

Exit codes: ``0`` success (including a partial scan, which is a real result), ``1`` a
fatal error or a scan in which *every* source failed (nothing was looked at, so
"0 issues" would be a lie), ``2`` a usage error from argparse.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from archwatch import __version__, detector, emitter
from archwatch.config import DetectorConfig

log = logging.getLogger("archwatch.cli")

__all__ = ["build_parser", "main", "cmd_scan", "cmd_show", "cmd_surface", "read_stub"]

_FRONT_MATTER_FENCE = "---"


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for INFO logging, -vv for DEBUG (DEBUG prints one line per "
        "suppressed candidate — thousands on a real HuggingFace window)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON on stdout instead of the human summary",
    )
    return parser


class _RefuseLiveMode(argparse.Action):
    """``--no-dry-run`` exists only to explain why it does not exist.

    Someone will eventually try it — a flag that defaults to true invites its
    negation. Failing with the reason is better than argparse's bare "unrecognized
    argument", and far better than the flag quietly existing.
    """

    def __init__(self, option_strings: Any, dest: str, **kwargs: Any) -> None:
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("help", argparse.SUPPRESS)
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[no-untyped-def]
        parser.error(
            "archwatch has no live mode: it never posts to GitHub and needs no write "
            "scopes. The only output is markdown under --out (PLAN.md rule 2)."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archwatch",
        description=(
            "Early-warning pipeline for zero-day model architecture support in BLIS. "
            "Tracking only: nothing is ever posted to GitHub."
        ),
    )
    parser.add_argument("--version", action="version", version=f"archwatch {__version__}")
    subs = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # -- scan --------------------------------------------------------------
    scan = subs.add_parser(
        "scan",
        help="poll the sources, filter, and write issue stubs (dry-run only)",
        description=(
            "Poll each selected source once, join signals onto architectures, apply "
            "the novelty filter, and write one markdown stub per survivor. A source "
            "that fails degrades the run to a partial scan rather than aborting it."
        ),
    )
    _common(scan)
    scan.add_argument(
        "--window-days",
        type=int,
        default=None,
        metavar="N",
        help="how far back to look (default: DetectorConfig.window_days)",
    )
    scan.add_argument(
        "--sources",
        default="all",
        metavar="LIST",
        help="comma-separated: " + ", ".join(detector.SOURCE_NAMES) + ", or "
        + ", ".join(sorted(detector.SOURCE_ALIASES))
        + " (default: all)",
    )
    scan.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="where stubs are written AND where the dedup looks for existing ones "
        f"(default: {emitter.default_issues_dir()})",
    )
    scan.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="the only mode; defaults to true and exists for explicitness",
    )
    scan.add_argument("--no-dry-run", action=_RefuseLiveMode, dest="no_dry_run")
    scan.add_argument(
        "--max-issues",
        type=int,
        default=None,
        metavar="N",
        help="cap on stubs written this run (default: "
        f"{DetectorConfig().max_issues_per_run})",
    )
    scan.add_argument(
        "--recheck-known",
        action="store_true",
        help="also re-check architectures already in the seed set for unparsed config "
        "fields (T1-known-arch). Closes the filter's largest recall hole at a cost "
        "in noise; the backtest measures both settings.",
    )
    scan.add_argument(
        "--no-trending",
        action="store_true",
        help="skip the HuggingFace trending sweep that feeds S4",
    )
    scan.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite an existing stub whose stage-1 content has changed (any stage-2 "
        "analysis appended to it is preserved)",
    )
    scan.add_argument(
        "--runlog-dir",
        default=None,
        metavar="DIR",
        help=f"where the run log goes (default: {detector.default_runlog_dir()})",
    )
    scan.add_argument(
        "--no-runlog", action="store_true", help="do not write a run log"
    )
    scan.add_argument(
        "--surface-dir",
        default=None,
        metavar="DIR",
        help="load the support surface from somewhere other than support-surface/",
    )
    scan.set_defaults(func=cmd_scan)

    # -- show --------------------------------------------------------------
    show = subs.add_parser(
        "show",
        help="list the issue stubs already on disk",
        description=(
            "Read each stub's YAML front matter and list what stage 1 found. Stage-2 "
            "status comes from whether anything was appended below the emitter's "
            "marker, not from a flag."
        ),
    )
    _common(show)
    show.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help=f"directory to list (default: {emitter.default_issues_dir()})",
    )
    show.add_argument(
        "--sort",
        choices=("detected_at", "arch_id"),
        default="detected_at",
        help="ordering (default: detected_at, newest first)",
    )
    show.set_defaults(func=cmd_show)

    # -- surface -----------------------------------------------------------
    surface = subs.add_parser(
        "surface",
        help="print a summary of the loaded BLIS support surface",
        description=(
            "Counts of parsed fields, validators, known gaps and seeded architectures. "
            "Fast sanity check on a fresh checkout: if these are zero, every scan will "
            "silently fire T1 on everything."
        ),
    )
    _common(surface)
    surface.add_argument(
        "--surface-dir",
        default=None,
        metavar="DIR",
        help="load from somewhere other than support-surface/",
    )
    surface.set_defaults(func=cmd_surface)

    return parser


def _setup_logging(verbosity: int) -> None:
    """Route archwatch logging to stderr at the requested level.

    Only ``basicConfig`` — deliberately no ``setLevel`` on the ``archwatch`` logger.
    Pinning a level on a package logger is a process-wide mutation that outlives the
    call: it survives into anything else running in the same interpreter and silently
    filters records below it, which is exactly how this function broke another
    component's ``caplog`` test. ``basicConfig`` is a no-op when the root logger is
    already configured (as it is under pytest), so nothing leaks; child loggers stay at
    NOTSET and inherit the root level, which is all the verbosity flags need.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace, out: Any = None) -> int:
    if getattr(args, "dry_run", True) is not True:  # pragma: no cover - unreachable
        print("archwatch has no live mode.", file=sys.stderr)
        return 1

    try:
        sources = detector.parse_sources(args.sources)
    except ValueError as exc:
        print(f"archwatch scan: {exc}", file=sys.stderr)
        return 1

    cfg = DetectorConfig()
    if args.window_days is not None:
        if args.window_days <= 0:
            print("archwatch scan: --window-days must be positive", file=sys.stderr)
            return 1
        cfg = replace(cfg, window_days=args.window_days)
    if args.max_issues is not None:
        if args.max_issues < 0:
            print("archwatch scan: --max-issues must not be negative", file=sys.stderr)
            return 1
        cfg = replace(cfg, max_issues_per_run=args.max_issues)
    if args.recheck_known:
        cfg = replace(cfg, recheck_known_architectures=True)

    try:
        summary = detector.scan(
            cfg,
            sources=sources,
            out_dir=args.out,
            surface_dir=args.surface_dir,
            trending=not args.no_trending,
            overwrite=args.overwrite,
            dry_run=True,
        )
    except Exception as exc:
        # Only reached for a failure outside a connector — a missing/broken support
        # surface, or an unwritable --out. Connector failures are handled inside and
        # produce a partial scan instead.
        log.debug("scan failed", exc_info=True)
        print(f"archwatch scan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not args.no_runlog:
        try:
            summary.write_run_log(args.runlog_dir)
        except OSError as exc:
            # A run log we cannot write must not throw away a scan we already did.
            print(f"archwatch: could not write run log: {exc}", file=sys.stderr)

    if args.as_json:
        print(json.dumps(summary.as_dict(), indent=2, default=str), file=out)
    else:
        print(summary.text(), file=out)

    if summary.total_failure:
        return 1
    return 0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def read_stub(path: Path) -> dict[str, Any]:
    """Parse one stub into a row: front matter plus derived stage-2 status.

    Front matter is the machine-readable contract (``emitter.SCHEMA``), so this reads
    it rather than scraping prose. A stub with missing or unparseable front matter is
    reported as such instead of being skipped: a file in ``issues/`` that cannot be
    read is exactly the thing a human needs told about.
    """
    row: dict[str, Any] = {"path": str(path), "arch_id": path.stem}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        row["error"] = f"unreadable: {exc}"
        return row

    front: dict[str, Any] = {}
    if text.startswith(_FRONT_MATTER_FENCE):
        end = text.find(f"\n{_FRONT_MATTER_FENCE}", len(_FRONT_MATTER_FENCE))
        if end != -1:
            block = text[len(_FRONT_MATTER_FENCE) : end]
            try:
                import yaml

                loaded = yaml.safe_load(block)
            except Exception as exc:  # malformed YAML, not our problem to fix
                row["error"] = f"unparseable front matter: {exc}"
                loaded = None
            if isinstance(loaded, dict):
                front = loaded
            elif "error" not in row:
                row["error"] = "front matter is not a mapping"
        else:
            row["error"] = "unterminated front matter"
    else:
        row["error"] = "no front matter"

    row.update({str(k): v for k, v in front.items()})
    row["arch_id"] = str(front.get("arch_id") or path.stem)
    _stub, appendix = emitter.split_stub(text)
    row["stage2"] = "done" if appendix.strip() else "pending"
    row["stage2_chars"] = len(appendix.strip())
    return row


def _issue_files(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.glob("*.md") if p.is_file())


def cmd_show(args: argparse.Namespace, out: Any = None) -> int:
    out_dir = Path(args.out) if args.out else emitter.default_issues_dir()
    if not out_dir.is_dir():
        print(f"archwatch show: no such directory: {out_dir}", file=sys.stderr)
        return 1

    rows = [read_stub(p) for p in _issue_files(out_dir)]
    if args.sort == "arch_id":
        rows.sort(key=lambda r: str(r.get("arch_id", "")).lower())
    else:
        rows.sort(
            key=lambda r: (str(r.get("detected_at") or ""), str(r.get("arch_id", ""))),
            reverse=True,
        )

    if args.as_json:
        print(json.dumps({"out_dir": str(out_dir), "issues": rows}, indent=2, default=str), file=out)
        return 0

    if not rows:
        print(f"{out_dir}: no issue stubs yet", file=out)
        return 0

    header = ("ARCH_ID", "DETECTED", "SOURCES", "TRIGGERS", "SIGNIF", "BKT", "PARAMS", "STAGE2")
    table = [header]
    for row in rows:
        table.append(
            (
                str(row.get("arch_id", "?")),
                str(row.get("detected_at") or "—")[:19],
                ",".join(_as_list(row.get("sources"))) or "—",
                ",".join(_as_list(row.get("triggers"))) or "—",
                ",".join(_as_list(row.get("significance"))) or "—",
                "0" if row.get("bucket") == 0 else "?",
                _params(row.get("est_total_params")),
                str(row.get("stage2", "?")),
            )
        )
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for i, cells in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip(), file=out)
        if i == 0:
            print("  ".join("-" * w for w in widths), file=out)

    broken = [r for r in rows if r.get("error")]
    print(f"\n{len(rows)} stub(s) in {out_dir}", file=out)
    done = sum(1 for r in rows if r.get("stage2") == "done")
    print(f"stage 2: {done} done, {len(rows) - done} pending", file=out)
    for r in broken:
        print(f"  ! {r['path']}: {r['error']}", file=out)
    return 0


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _params(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "—"
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if n >= scale:
            return f"{n / scale:.3g}{unit}"
    return str(n)


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------


def cmd_surface(args: argparse.Namespace, out: Any = None) -> int:
    from archwatch.surface import load_surface

    try:
        surface = (
            load_surface(args.surface_dir)
            if args.surface_dir
            else load_surface()
        )
    except Exception as exc:
        log.debug("surface load failed", exc_info=True)
        print(
            f"archwatch surface: could not load the support surface: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    summary = surface.summary()
    if args.as_json:
        print(json.dumps(summary, indent=2, default=str), file=out)
        return 0

    print("BLIS support surface", file=out)
    for key, value in summary.items():
        if isinstance(value, dict):
            inner = ", ".join(f"{k}={v}" for k, v in value.items()) or "none"
            print(f"  {key:<32} {inner}", file=out)
        elif isinstance(value, list):
            if not value:
                print(f"  {key:<32} none", file=out)
            else:
                print(f"  {key:<32} {len(value)}", file=out)
                for item in value:
                    print(f"  {'':<32}   {item}", file=out)
        else:
            print(f"  {key:<32} {value}", file=out)

    if not surface.known_architectures:
        print(
            "\nWARNING: the seed set is empty — every architecture will look new.",
            file=out,
        )
    if not surface.parsed_field_names:
        print(
            "WARNING: no parsed fields loaded — T1 would fire on every config.",
            file=out,
        )
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _setup_logging(getattr(args, "verbose", 0))
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
