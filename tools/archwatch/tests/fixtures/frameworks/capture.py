#!/usr/bin/env python
"""Record real GitHub responses into the offline cassettes used by
``tests/test_frameworks.py``. Run by hand, never by the test suite.

    .venv/bin/python tests/fixtures/frameworks/capture.py            # re-record + rebuild
    .venv/bin/python tests/fixtures/frameworks/capture.py titles     # rebuild observed_titles.json only

Read-only GitHub calls (token from GH_TOKEN/GITHUB_TOKEN or ``gh auth token``).
Each cassette is a dict of request-key -> recorded response, where the key is
built by ``request_key()`` below -- the test's FixtureSession uses the identical
formula, so the cassette replays exactly the requests the connector makes.

Patch trimming: a real ``/pulls/{n}/files`` response for a model PR is megabytes
(whole new model files show up as one giant patch). For every PR except the ones
in ``KEEP_RAW_PATCHES`` the recorded ``patch`` is reduced to the lines the
extractor actually reads (hunk headers, ``+class`` / ``+EntryClass`` blocks,
``+"ArchKey":`` registry lines, HuggingFace repo mentions) and dropped entirely
for files outside the model-registry paths. ``_meta.patch_trimmed`` records this
per cassette. PR 55063 is kept verbatim so at least one test exercises a real
untouched patch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import requests  # noqa: E402

from archwatch.connectors import frameworks as F  # noqa: E402

HERE = Path(__file__).resolve().parent

# (name, since, until)
WINDOWS = [
    ("cassette_window_2026-08-28.json", "2026-08-28T00:00:00Z", "2026-09-04T00:00:00Z"),
    ("cassette_window_2026-07-27.json", "2026-07-27T00:00:00Z", "2026-08-01T00:00:00Z"),
]

KEEP_RAW_PATCHES = {"vllm-project/vllm:55063"}

_KEEP_LINE_RE = re.compile(
    r"^(?:@@|\+\s*class\s|\+\s*EntryClass\s*=|\+\s*\"[A-Za-z][A-Za-z0-9_]*\"\s*:)"
)
_KEEP_SUBSTR = ("huggingface.co", "_HfExamplesInfo")
_KEEP_REPO_ID_RE = re.compile(r'^\+.*"[A-Za-z0-9][\w.-]*/[\w.-]+"')


def request_key(url: str, params: dict | None) -> str:
    """MUST match tests/test_frameworks.py::_request_key."""
    items = sorted((str(k), str(v)) for k, v in (params or {}).items())
    return f"{url}?{urlencode(items)}" if items else url


def _trim_patch(patch: str | None) -> str | None:
    if not patch:
        return patch
    lines = patch.splitlines()
    keep: list[str] = []
    entry_budget = 0
    for ln in lines:
        wanted = (
            bool(_KEEP_LINE_RE.match(ln))
            or bool(_KEEP_REPO_ID_RE.match(ln))
            or any(s in ln for s in _KEEP_SUBSTR)
        )
        if ln.startswith("+") and re.match(r"^\+\s*EntryClass\s*=", ln):
            entry_budget = 12
        elif entry_budget and ln.startswith("+"):
            wanted = True
            entry_budget -= 1
            if "]" in ln:
                entry_budget = 0
        if wanted:
            keep.append(ln)
    if not keep:
        return None
    keep.append("# [archwatch fixture] patch trimmed to extractor-relevant lines")
    return "\n".join(keep)


def _trim_files(payload, repo: str, number: str) -> object:
    if not isinstance(payload, list) or f"{repo}:{number}" in KEEP_RAW_PATCHES:
        return payload
    spec = F.VLLM if repo.startswith("vllm-project") else F.SGLANG
    out = []
    for f in payload:
        if not isinstance(f, dict):
            out.append(f)
            continue
        g = dict(f)
        fn = g.get("filename") or ""
        relevant = F._is_registry_file(fn, spec) or F._is_model_path(fn, spec)
        g["patch"] = _trim_patch(g.get("patch")) if relevant else None
        if g["patch"] is None:
            g.pop("patch", None)
        # blob/raw urls carry commit shas and add noise; keep the useful fields
        for noisy in ("blob_url", "raw_url", "contents_url"):
            g.pop(noisy, None)
        out.append(g)
    return out


# Fields the connector actually reads. Everything else is dropped so the
# cassettes stay small; _meta.fields_pruned records that this happened.
_PR_FIELDS = ("number", "title", "body", "html_url", "merged_at", "closed_at", "state", "user")
_FILE_FIELDS = ("filename", "status", "additions", "deletions", "changes", "sha", "patch")
_COMMIT_TOP = ("sha", "html_url", "commit")


def _prune_user(u):
    return {"login": (u or {}).get("login")} if isinstance(u, dict) else None


def _clip(txt, n=4000):
    if isinstance(txt, str) and len(txt) > n:
        return txt[:n] + "\n[archwatch fixture] body truncated"
    return txt


def _prune_pr(pr):
    if not isinstance(pr, dict):
        return pr
    out = {k: pr.get(k) for k in _PR_FIELDS if k in pr}
    if "user" in out:
        out["user"] = _prune_user(out["user"])
    if "body" in out:
        out["body"] = _clip(out["body"])
    return out


def _prune_commit(c):
    if not isinstance(c, dict):
        return c
    out = {k: c.get(k) for k in _COMMIT_TOP if k in c}
    commit = out.get("commit")
    if isinstance(commit, dict):
        out["commit"] = {
            "message": commit.get("message"),
            "author": {"date": (commit.get("author") or {}).get("date")},
            "committer": {"date": (commit.get("committer") or {}).get("date")},
        }
    return out


def _prune(url: str, body):
    if url.endswith("/search/issues") and isinstance(body, dict):
        return {
            "total_count": body.get("total_count"),
            "incomplete_results": body.get("incomplete_results"),
            # search returns up to 100 items per page; clip their bodies harder
            "items": [
                {**_prune_pr(i), "body": _clip(i.get("body"), 1500)}
                for i in (body.get("items") or [])
            ],
        }
    if re.search(r"/repos/[^/]+/[^/]+/commits$", url) and isinstance(body, list):
        return [_prune_commit(c) for c in body]
    if re.search(r"/pulls/\d+$", url):
        return _prune_pr(body)
    if re.search(r"/pulls/\d+/files$", url) and isinstance(body, list):
        return [
            {k: f.get(k) for k in _FILE_FIELDS if k in f}
            for f in body
            if isinstance(f, dict)
        ]
    return body


class RecordingSession:
    def __init__(self, token: str | None) -> None:
        self.inner = requests.Session()
        self.token = token
        self.log: dict[str, dict] = {}

    def get(self, url, params=None, headers=None, timeout=None):
        resp = self.inner.get(url, params=params, headers=headers, timeout=timeout)
        key = request_key(url, params)
        body: object
        try:
            body = resp.json()
        except Exception:
            body = None
        m = re.search(r"/repos/([^/]+/[^/]+)/pulls/(\d+)/files$", url)
        if m:
            body = _trim_files(body, m.group(1), m.group(2))
        body = _prune(url.split("?")[0], body)
        self.log[key] = {
            "status": resp.status_code,
            "headers": {
                k: v
                for k, v in resp.headers.items()
                if k.lower().startswith("x-ratelimit")
            },
            "json": body,
        }
        print(f"  rec {resp.status_code} {key[:130]}", file=sys.stderr)
        return resp


# ---------------------------------------------------------------------------
# observed_titles.json — the real merged-PR title corpus
# ---------------------------------------------------------------------------

# Hand-verified labels. Every entry here was checked against the PR's actual
# diff (does it add a registry key / EntryClass export?), not against its title.
LABELLED: dict[tuple[str, int], tuple[bool, str]] = {
    ("vllm-project/vllm", 55063): (True, "adds registry key K2HorizonForCausalLM"),
    ("vllm-project/vllm", 53906): (True, "adds Glm5Next* registry keys"),
    ("vllm-project/vllm", 53896): (True, "adds Qwen4Exp* registry keys"),
    ("vllm-project/vllm", 54566): (True, "adds DeepseekV4ForConditionalGeneration"),
    ("vllm-project/vllm", 54160): (True, "adds HYV4ForCausalLM / HYV4MTPModel"),
    ("vllm-project/vllm", 50000): (True, "Kimi K3 umbrella PR"),
    ("vllm-project/vllm", 50089): (True, "Kimi K3 model files + kernels"),
    ("vllm-project/vllm", 50210): (True, "Qwen3.5 text-only dense and MoE"),
    ("sgl-project/sglang", 37654): (True, "adds EntryClass XllmForCausalLM/K2HorizonForCausalLM"),
    ("vllm-project/vllm", 54882): (False, "bugfix"),
    ("vllm-project/vllm", 54262): (False, "mypy"),
    ("vllm-project/vllm", 54753): (False, "CI sharding"),
    ("vllm-project/vllm", 49869): (False, "weight-loading fix"),
    ("vllm-project/vllm", 54380): (False, "[Model] tag but only a memory-profiling tweak"),
    ("vllm-project/vllm", 53608): (False, "removes architectures"),
    ("sgl-project/sglang", 37750): (False, "docs"),
    ("sgl-project/sglang", 37087): (False, "config refactor"),
    ("sgl-project/sglang", 37193): (False, "XPU weekly enablement, no new arch"),
    ("sgl-project/sglang", 34446): (False, "rotary kernel fix"),
    ("sgl-project/sglang", 37825): (False, "bugfix on an already-supported arch"),
}

# Real titles seen while probing outside the two recorded windows, plus one
# clearly-marked synthetic case for PLAN.md's idealized title.
EXTRA_TITLES = [
    ("sgl-project/sglang", 33561, "[Model] Support Ling-3.0-flash (BailingMoeV3) ", True,
     "adds class BailingMoeV3ForCausalLM"),
    ("sgl-project/sglang", 34859, "Qwen3.8-27B Model Support", True,
     "untagged '<name> Model Support' style"),
    ("sgl-project/sglang", 35963, "Add Spark3 Model", True, "untagged 'Add <name> Model' style"),
    ("sgl-project/sglang", 34262, "[Feature] Add Muse Glimmer model support", True,
     "model PR under a non-[Model] tag"),
    ("vllm-project/vllm", 53615,
     "[Model] Migrate FlexOlmo, Olmo3 and Hunyuan V1/VL to the Transformers modeling backend",
     False, "migration"),
    ("vllm-project/vllm", 23241, "[New Model] Add Seed-Oss model", True,
     "classic pre-2026 [New Model] style"),
    ("vllm-project/vllm", 14119, "[Model] New model support for Phi-4-multimodal-instruct", True,
     "classic 'New model support for X' style"),
    ("vllm-project/vllm", 99999, "[Model] Add KimiK3ForCausalLM", True,
     "SYNTHETIC: the idealized title PLAN.md assumes; no real PR looks like this"),
]


def build_observed_titles() -> None:
    """Rebuild observed_titles.json from whatever cassettes are on disk."""
    from urllib.parse import parse_qs, urlsplit

    titles: dict[tuple[str, int], str] = {}
    windows = []
    for cas in sorted(HERE.glob("cassette_window_*.json")):
        doc = json.loads(cas.read_text())
        meta = doc.get("_meta", {})
        windows.append(f"{meta.get('since')}..{meta.get('until')}")
        for key, resp in doc.get("requests", {}).items():
            if "/search/issues" not in key:
                continue
            q = parse_qs(urlsplit(key).query).get("q", [""])[0]
            m = re.search(r"repo:(\S+)", q)
            repo = m.group(1) if m else "?"
            for it in (resp.get("json") or {}).get("items") or []:
                if isinstance(it, dict) and isinstance(it.get("number"), int):
                    titles[(repo, it["number"])] = it.get("title") or ""

    cases = []
    for (repo, num), title in sorted(titles.items()):
        case = {"repo": repo, "number": num, "title": title}
        lab = LABELLED.get((repo, num))
        if lab:
            case["model_support"], case["note"] = lab
        cases.append(case)
    for repo, num, title, expected, note in EXTRA_TITLES:
        cases.append({
            "repo": repo, "number": num, "title": title,
            "model_support": expected, "note": note,
            "outside_recorded_window": True,
        })

    doc = {
        "_meta": {
            "what": (
                "Every merged-PR title returned by the recorded title searches, plus real "
                "titles seen outside those windows. `model_support` is present only where "
                "the classification was hand-verified against the PR diff."
            ),
            "captured_by": "tests/fixtures/frameworks/capture.py",
            "repos": [s.repo for s in F.DEFAULT_REPOS],
            "windows": windows,
            "n_titles": len(cases),
            "finding": (
                "No real merged vLLM or SGLang PR title in this corpus contains an "
                "architecture class name (XxxForCausalLM); a GitHub search for "
                "`ForCausalLM in:title` over all merged PRs in both repos returns 0 hits. "
                "Titles carry marketing names ('GLM-5.3-Flash', 'Kimi K3'); architecture "
                "names have to come from the diff."
            ),
        },
        "cases": cases,
    }
    out = HERE / "observed_titles.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} titles, "
          f"{sum(1 for c in cases if 'model_support' in c)} labelled)", file=sys.stderr)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "titles":
        build_observed_titles()
        return 0
    token = F.resolve_token()
    if not token:
        print("no GitHub token available", file=sys.stderr)
        return 1
    from datetime import datetime

    for name, since, until in WINDOWS:
        sess = RecordingSession(token)
        conn = F.FrameworkConnector(
            session=sess,
            token=token,
            until=datetime.fromisoformat(until.replace("Z", "+00:00")),
        )
        signals = conn.poll(datetime.fromisoformat(since.replace("Z", "+00:00")))
        cassette = {
            "_meta": {
                "captured_by": "tests/fixtures/frameworks/capture.py",
                "since": since,
                "until": until,
                "repos": [s.repo for s in F.DEFAULT_REPOS],
                "patch_trimmed": True,
                "patch_raw_for": sorted(KEEP_RAW_PATCHES),
                "fields_pruned": (
                    "responses keep only the fields the connector reads: "
                    f"PR {list(_PR_FIELDS)}, file {list(_FILE_FIELDS)}, "
                    "commit sha/message/dates; PR bodies clipped to 6000 chars"
                ),
                "http_calls": sess.inner and len(sess.log),
                "signals_recorded": len(signals),
            },
            "requests": sess.log,
        }
        out = HERE / name
        out.write_text(json.dumps(cassette, indent=1, sort_keys=True) + "\n")
        print(
            f"wrote {out} ({out.stat().st_size/1024:.0f} KiB, "
            f"{len(sess.log)} requests, {len(signals)} signals)",
            file=sys.stderr,
        )
        for s in signals:
            print(f"   {s.source} #{s.raw_ref} {s.arch_ids} {s.display_name!r}", file=sys.stderr)
    build_observed_titles()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
