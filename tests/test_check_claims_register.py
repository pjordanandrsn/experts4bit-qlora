"""scripts/check_claims_register.py: every rule of docs/claims-schema.md, failing where it should.

This file is identical in experts4bit-qlora (tests/) and grouped-nf4-gemm (kernel/), like the
checker and the schema it tests. Each case builds a throwaway git repository, writes a register
into it and breaks exactly one thing; the clean register must pass and every broken one must fail
naming what broke. The cases carry forward both repositories' earlier suites -- the 2026-09-05
audits found receipt logs that were never committed, annotated paths, superseded rows with no
successor, "pending" on measured rows and "best licensed" after a licence was withdrawn -- and add
one for every rule the 2026-09-23 convergence introduced. The last tests run the real check on
this repository's own register.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "check_claims_register.py"
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("check_claims_register", _SCRIPT)
ccr = importlib.util.module_from_spec(_spec)
sys.modules["check_claims_register"] = ccr
_spec.loader.exec_module(ccr)

RUNTIME, KERNELS = "experts4bit-qlora", "grouped-nf4-gemm"
URL = {RUNTIME: "https://github.com/o/runtime", KERNELS: "https://github.com/o/kernels"}
VOCAB = {s: s for s in ("verified", "confirmed", "measured", "measured-private", "projected", "retired",
                        "superseded", "open")}
README = "# T\n\n## What is measured\n\nline 5\n\n## What was retired\n\n```\n## not a heading\n```\n\n## Twice\n\n## Twice\n"
CHANGELOG = "# Changelog\n\n## 0.24.0 — 2026-08-31\n\nbody\n\n## Unreleased\n\nold\n"


def _git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                   cwd=cwd, check=True, capture_output=True)


def _repo(path: pathlib.Path, package: str = RUNTIME, git: bool = True) -> pathlib.Path:
    """A minimal repository of the system: pyproject, manifest, a README and CHANGELOG with headings,
    a receipt, a receipts directory and a gitignored log; committed when ``git``."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(f'[project]\nname = "{package}"\nversion = "0.0.0"\n', encoding="utf-8")
    (path / "docs").mkdir(exist_ok=True)
    manifest = {"packages": {"runtime": {"package": RUNTIME, "repository": URL[RUNTIME]},
                             "kernels": {"package": KERNELS, "repository": URL[KERNELS]}}}
    (path / "docs" / "system-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "README.md").write_text(README, encoding="utf-8")
    (path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (path / "bench" / "receipts").mkdir(parents=True, exist_ok=True)
    (path / "bench" / "RESULTS.md").write_text("# R\n\n## Result\n\nx\n", encoding="utf-8")
    (path / "bench" / "receipts" / "r1.json").write_text("{}\n", encoding="utf-8")
    (path / "bench" / "run.log").write_text("untracked\n", encoding="utf-8")
    (path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    if git:
        _git("init", "-q", cwd=path)
        _git("add", ".", cwd=path)
        _git("commit", "-q", "-m", "fixture", cwd=path)
    return path


def _row(cid="e4b.a", status="measured", **kw):
    c = {"id": cid, "status": status, "claim": "a sentence", "evidence": ["bench/RESULTS.md"],
         "measured_on": "2026-09-05"}
    c.update(kw)
    return {k: v for k, v in c.items() if v is not ...}


def _write(repo, rows, **top):
    doc = {"schema": "docs/claims-schema.md", "status_vocabulary": VOCAB, "claims": list(rows)}
    doc.update(top)
    (repo / "docs" / "claims.json").write_text(json.dumps(doc), encoding="utf-8")


def _findings(repo, *rows, sibling=None, **top):
    _write(repo, rows, **top)
    return ccr.check(repo, sibling=sibling).findings


@pytest.fixture
def repo(tmp_path):
    return _repo(tmp_path / "rt")


def _has(findings, *needles):
    return any(all(n in f for n in needles) for f in findings)


# ------------------------------------------------------------------------------ the clean case --

def test_a_clean_register_passes(repo):
    rows = [
        _row("e4b.a", evidence=["bench/RESULTS.md", "README.md#what-is-measured", "README.md#L5", "CHANGELOG.md#L3-L5",
                                {"url": URL[KERNELS] + "/issues/7"}, {"url": URL[RUNTIME] + "/pull/9"}],
             quoted_in=["README.md", "README.md#twice", "README.md#twice-1", "CHANGELOG.md#0240--2026-08-31"],
             supersedes=["e4b.old", "e4b.gone"]),
        _row("e4b.old", "superseded", superseded_by="e4b.a"),
        _row("e4b.gone", "retired", retired_reason="wrong box", superseded_by="e4b.a"),
        _row("e4b.never", "retired", retired_reason="no restatement exists"),
        {"id": "e4b.open.q", "status": "open", "claim": "a question", "evidence": [{"url": URL[RUNTIME] + "/issues/1"}]},
        _row("e4b.priv", "measured-private", evidence=[], evidence_private=["INT4B16/P1.md"]),
    ]
    assert _findings(repo, *rows) == []


# ---------------------------------------------------------------------------------- locations --

@pytest.mark.parametrize("loc, needle", [
    ("bench/run.log", "not in the git tree at HEAD"),                 # gitignored: not evidence until force-added
    ("bench/NOPE.md", "not in the git tree at HEAD"),
    ("bench/receipts", "is a directory"),
    ("bench/receipts/", "is a directory"),
    ("../elsewhere/RESULTS.md", "not a repository-relative path"),
    ("/abs/RESULTS.md", "not a repository-relative path"),
    ("bench/step.py (--ppl-fq)", "annotated or a glob"),
    ("bench/receipts/*.json", "annotated or a glob"),
    ("CHANGELOG.md 0.24.0", "annotated or a glob"),
    ("README.md#What was retired", "no heading whose GitHub anchor is"),   # a heading PREFIX is not an anchor
    ("README.md#not-a-heading", "no heading whose GitHub anchor is"),     # headings inside code fences do not count
    ("CHANGELOG.md#0.24.0", "no heading whose GitHub anchor is"),
    ("README.md#L400", "outside the file"),
    ("README.md#L5-L2", "outside the file"),
    ("bench/receipts/r1.json#result", "a non-Markdown file takes a line anchor"),
    ("README.md#", "empty anchor"),
])
def test_every_location_rule_fails_where_it_should(repo, loc, needle):
    f = _findings(repo, _row(evidence=[loc]))
    assert _has(f, "evidence[0]", needle), f
    f = _findings(repo, _row(quoted_in=[loc]))
    assert _has(f, "quoted_in[0]", needle), f


def test_a_force_added_log_becomes_evidence(repo):
    assert _has(_findings(repo, _row(evidence=["bench/run.log"])), "not in the git tree")
    _git("add", "-f", "bench/run.log", cwd=repo)
    assert _findings(repo, _row(evidence=["bench/run.log"])) == []


def test_outside_a_checkout_the_working_tree_stands_in_and_the_output_says_so(tmp_path, capsys):
    r = _repo(tmp_path / "plain", git=False)
    assert ccr.tracked_files(r) is None
    assert _findings(r, _row(evidence=["bench/RESULTS.md"])) == []
    assert _has(_findings(r, _row(evidence=["bench/NOPE.md"])), "does not exist in the working tree")
    assert _has(_findings(r, _row(evidence=["bench/receipts"])), "is a directory")
    _write(r, [_row()])
    assert ccr.main(["--root", str(r)]) == 0
    out = capsys.readouterr().out
    assert "NOTE:" in out and "not a git checkout" in out and "the working tree (not a checkout)" in out


# ----------------------------------------------------------------------------------- anchors --

@pytest.mark.parametrize("heading, anchor", [
    ("0.24.0 — 2026-08-31", "0240--2026-08-31"),
    ("10. Energy — measured ([bench/_upstream/bench_energy.py](../bench/_upstream/bench_energy.py), "
     "[bench/bench_energy_excluded.py](../bench/bench_energy_excluded.py))",
     "10-energy--measured-bench_upstreambench_energypy-benchbench_energy_excludedpy"),
    ("13.1 The routing-flip floor — why an MoE parity delta is not measured against zero",
     "131-the-routing-flip-floor--why-an-moe-parity-delta-is-not-measured-against-zero"),
    ("`code` and **bold** — x_y", "code-and-bold--x_y"),
    ("What changed — retired, superseded, corrected", "what-changed--retired-superseded-corrected"),
])
def test_the_anchor_is_the_one_github_renders(heading, anchor):
    # each pair was read off github.com's rendering of these repositories' own documents
    assert ccr.github_slug(heading) == anchor


def test_repeated_headings_are_numbered_and_fenced_ones_ignored():
    assert ccr.github_anchors(README) == {"t", "what-is-measured", "what-was-retired", "twice", "twice-1"}


# ---------------------------------------------------------------------------------- evidence --

@pytest.mark.parametrize("entry, needle", [
    ("https://github.com/o/runtime/issues/1", "is a URL -- write {\"url\""),
    ({"url": "https://example.com/x"}, "not a github.com issue or pull request"),
    ({"url": "https://github.com/o/runtime/tree/main"}, "not a github.com issue or pull request"),
    ({"url": "https://github.com/someone/else/issues/1"}, "not in one of this system's repositories"),
    ({"repository": "someone-else", "path": "x.md"}, "not a package of this system"),
    ({"repository": RUNTIME, "path": "bench/RESULTS.md"}, "is this repository -- cite the path directly"),
    ({"repository": KERNELS, "path": "../x.md"}, "not a repository-relative path"),
    ({"repository": KERNELS}, "an evidence object is"),
    ({"path": "CHANGELOG.md", "section": "0.24.0"}, "an evidence object is"),    # the pre-2026-09-23 kernel form
    ({"glob": "bench/receipts/*.json"}, "an evidence object is"),
    ({"url": URL[RUNTIME] + "/issues/1", "note": "x"}, "an evidence object is"),
    (42, "unsupported type"),
])
def test_each_evidence_form_outside_the_schema_is_named(repo, entry, needle):
    f = _findings(repo, _row(evidence=["bench/RESULTS.md", entry]))
    assert _has(f, "evidence[1]", needle), f


def test_a_public_run_needs_a_location_first_and_a_private_run_its_private_receipt(repo):
    assert _has(_findings(repo, _row(evidence=[])), "needs a public receipt")
    for st in ("measured", "confirmed", "verified"):
        f = _findings(repo, _row(status=st, evidence=[{"url": URL[RUNTIME] + "/issues/1"}, "bench/RESULTS.md"]))
        assert _has(f, "evidence[0] must be a location"), (st, f)
    assert _has(_findings(repo, _row(status="measured-private", evidence=[])), "needs evidence_private")
    assert _has(_findings(repo, _row(evidence_private=["ok", ""])), "evidence_private must be a list")
    assert _has(_findings(repo, _row(evidence="bench/RESULTS.md")), "evidence must be a list")


def test_cross_repository_evidence_is_skipped_without_a_sibling_and_resolved_in_one(tmp_path):
    rt, sib = _repo(tmp_path / "rt"), _repo(tmp_path / "k", package=KERNELS)
    row = _row(evidence=["bench/RESULTS.md", {"repository": KERNELS, "path": "bench/RESULTS.md#result"}])
    _write(rt, [row])
    r = ccr.check(rt)
    assert r.findings == [] and _has(r.skips, "not resolved (no --sibling checkout of grouped-nf4-gemm")
    r = ccr.check(rt, sibling=sib)
    assert r.findings == [] and r.skips == []
    bad = _row(evidence=["bench/RESULTS.md", {"repository": KERNELS, "path": "bench/MISSING.md"}])
    assert _has(_findings(rt, bad, sibling=sib), "in the sibling grouped-nf4-gemm", "not in the git tree")
    log = _row(evidence=["bench/RESULTS.md", {"repository": KERNELS, "path": "bench/run.log"}])
    assert _has(_findings(rt, log, sibling=sib), "in the sibling", "not in the git tree")   # the sibling's own tree


@pytest.mark.parametrize("make, needle", [
    (lambda t: _repo(t / "self"), "this repository's own package"),
    (lambda t: _repo(t / "stranger", package="some-other-package"), "not a package of this system"),
])
def test_a_sibling_that_is_not_the_other_package_is_exit_2_not_a_silent_skip(tmp_path, capsys, make, needle):
    rt = _repo(tmp_path / "rt")
    _write(rt, [_row()])
    assert ccr.main(["--root", str(rt), "--sibling", str(make(tmp_path))]) == 2
    out = capsys.readouterr().out
    assert needle in out and "OK:" not in out


def test_a_sibling_without_a_package_name_is_exit_2(tmp_path, capsys):
    rt = _repo(tmp_path / "rt")
    _write(rt, [_row()])
    sib = tmp_path / "sib"
    sib.mkdir()
    (sib / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    assert ccr.main(["--root", str(rt), "--sibling", str(sib)]) == 2
    assert "no pyproject project.name" in capsys.readouterr().out


# ------------------------------------------------------------------------------------- dates --

def test_dated_statuses_need_an_iso_calendar_date(repo):
    for st in ("measured", "measured-private", "confirmed", "verified"):
        extra = {"evidence_private": ["P"]} if st == "measured-private" else {}
        assert _has(_findings(repo, _row(status=st, measured_on=..., **extra)), "needs measured_on"), st
    assert _has(_findings(repo, _row(measured_on="2026-08")), "'2026-08' is not an ISO calendar date")
    assert _has(_findings(repo, _row(measured_on="2026-02-30")), "not an ISO calendar date")
    assert _has(_findings(repo, _row(measured_on=None)), "not an ISO calendar date", "omit the field")
    op = {"id": "e4b.open.x", "status": "open", "claim": "OPEN", "evidence": [{"url": URL[RUNTIME] + "/issues/1"}]}
    assert _findings(repo, op) == []
    assert _has(_findings(repo, dict(op, measured_on="soon")), "not an ISO calendar date")


# -------------------------------------------------------------------------------- successors --

def test_a_superseded_row_names_an_active_row_directly_and_is_named_back(repo):
    a = _row("e4b.a", supersedes=["e4b.old"])
    assert _has(_findings(repo, _row("e4b.old", "superseded")), "needs superseded_by")
    assert _has(_findings(repo, _row("e4b.old", "superseded", superseded_by="e4b.nope")), "'e4b.nope' is not in the register")
    assert _has(_findings(repo, _row("e4b.old", "superseded", superseded_by="e4b.a"), _row("e4b.a")),
                "does not list 'e4b.old' in its supersedes")
    assert _findings(repo, _row("e4b.old", "superseded", superseded_by="e4b.a"), a) == []
    chain = [_row("e4b.a", supersedes=["e4b.mid"]), _row("e4b.mid", "superseded", superseded_by="e4b.a", supersedes=["e4b.old"]),
             _row("e4b.old", "superseded", superseded_by="e4b.mid")]
    assert _has(_findings(repo, *chain), "e4b.old", "must name the ACTIVE row directly")
    cycle = [_row("e4b.x", "superseded", superseded_by="e4b.y", supersedes=["e4b.y"]),
             _row("e4b.y", "superseded", superseded_by="e4b.x", supersedes=["e4b.x"])]
    assert _has(_findings(repo, *cycle), "must name the ACTIVE row directly")


def test_retired_rows_may_name_a_restatement_and_nothing_else_carries_superseded_by(repo):
    a = _row("e4b.a", supersedes=["e4b.gone"])
    assert _findings(repo, a, _row("e4b.gone", "retired", retired_reason="r", superseded_by="e4b.a")) == []
    dead = [_row("e4b.gone", "retired", retired_reason="r", superseded_by="e4b.also"),
            _row("e4b.also", "retired", retired_reason="r", supersedes=["e4b.gone"])]
    assert _has(_findings(repo, *dead), "e4b.gone", "must name the ACTIVE row directly")
    assert _has(_findings(repo, _row("e4b.a", superseded_by="e4b.b"), _row("e4b.b")), "superseded_by on a 'measured' row")
    op = {"id": "e4b.open.x", "status": "open", "claim": "q", "superseded_by": "e4b.b"}
    assert _has(_findings(repo, op, _row("e4b.b")), "superseded_by on a 'open' row")


def test_every_supersedes_entry_exists_is_inactive_and_names_this_row_back(repo):
    assert _has(_findings(repo, _row("e4b.a", supersedes=["e4b.never"])), "supersedes 'e4b.never', which is not in the register")
    assert _has(_findings(repo, _row("e4b.a", supersedes=["e4b.b"]), _row("e4b.b")), "whose status is 'measured'")
    elsewhere = [_row("e4b.a", supersedes=["e4b.old"]), _row("e4b.b", supersedes=["e4b.old"]),
                 _row("e4b.old", "superseded", superseded_by="e4b.b")]
    assert _has(_findings(repo, *elsewhere), "e4b.a: supersedes 'e4b.old', whose superseded_by is 'e4b.b'")
    gone = [_row("e4b.a", supersedes=["e4b.gone"]), _row("e4b.gone", "retired", retired_reason="r")]
    assert _has(_findings(repo, *gone), "whose superseded_by is None")


def test_retired_needs_its_reason_and_the_reason_belongs_to_retired_rows_only(repo):
    assert _has(_findings(repo, _row("e4b.gone", "retired")), "needs retired_reason")
    assert _has(_findings(repo, _row("e4b.gone", "retired", retired_reason="  ")), "needs retired_reason")
    assert _findings(repo, _row("e4b.gone", "retired", retired_reason="K7 measured it flat")) == []
    assert _has(_findings(repo, _row("e4b.a", retired_reason="r")), "retired_reason on a 'measured' row")
    old = [_row("e4b.a", supersedes=["e4b.old"]), _row("e4b.old", "superseded", superseded_by="e4b.a", retired_reason="r")]
    assert _has(_findings(repo, *old), "retired_reason on a 'superseded' row")


# ------------------------------------------------------------------------------ placeholders --

def test_placeholder_words_are_refused_on_active_rows_only(repo):
    assert _has(_findings(repo, _row(notes="its number is pending on the validation lane")), "placeholder word 'pending'")
    assert _has(_findings(repo, _row(claim="TBD: a number")), "the claim", "'TBD'")
    assert _has(_findings(repo, _row(notes="TODO measure")), "'TODO'")
    assert _findings(repo, _row(notes="pendingly is not the word; independent is fine")) == []
    op = {"id": "e4b.open.x", "status": "open", "claim": "OPEN", "notes": "confirmation pending"}
    assert _findings(repo, op) == []


# ---------------------------------------------------------------------------------- licences --

def test_a_licence_label_needs_its_verdict_row(repo):
    assert _has(_findings(repo, _row(claim="best licensed configuration: 154.9 tok/s")), "asserts a licence", "no licensed_by")
    assert _findings(repo, _row(claim="fastest configuration -- measured, not licensed: 154.9 tok/s")) == []
    assert _findings(repo, _row(claim="NOT a licensed best: 284.6 tok/s")) == []
    gate = _row("e4b.gate", claim="LICENSED under the gate: -0.05 ppl", licensed_by="e4b.gate")
    assert _findings(repo, gate, _row("e4b.a", claim="the licensed stack: 304.9 tok/s", licensed_by="e4b.gate")) == []
    assert _has(_findings(repo, _row(claim="the licensed stack", licensed_by="e4b.nope")), "licensed_by 'e4b.nope' is not in")
    old = [_row("e4b.a", claim="the licensed stack", licensed_by="e4b.old", supersedes=["e4b.old"]),
           _row("e4b.old", "superseded", superseded_by="e4b.a")]
    assert _has(_findings(repo, *old), "a licence comes from an active verdict row")


def test_the_licence_rule_is_per_occurrence(repo):
    f = _findings(repo, _row(claim="the licensed stack: 304.9 tok/s (the unlicensed int4 arm, 487.8, beside)"))
    assert _has(f, "asserts a licence", "'the licensed'")
    f = _findings(repo, _row(claim="the pack read 11522 / 766, not the licensed 11512 / 776 -- unlicensed here"))
    assert _has(f, "asserts a licence"), "'not the licensed X' describes the licensed pack; it is not a disclaimer"
    for ok in ("measured, not licensed: 154.9 tok/s", "NOT a licensed best: 284.6 tok/s", "never licensed on this box",
               "no licensed arm exists for this family", "an un-licensed reading", "unlicensed, VOID stands"):
        assert _findings(repo, _row(claim=ok)) == [], ok
    assert ccr.asserts_licence("not licensed; the licensed stack") is not None
    assert ccr.asserts_licence("not licensed; never licensed") is None


def test_a_licence_citation_is_a_reference_not_an_assertion(repo):
    gate = _row("e4b.gate", claim="LICENSED under the gate: -0.05 ppl", licensed_by="e4b.gate")
    assert _findings(repo, gate, _row("e4b.a", claim="no ratio against the stack licensed by `e4b.gate`: 2.52")) == []
    assert _has(_findings(repo, gate, _row("e4b.a", claim="the stack licensed by `e4b.nope`: 2.52")),
                "claim cites `e4b.nope`", "not in the register")
    f = _findings(repo, gate, _row("e4b.b", claim="plain: 1.0"), _row("e4b.a", claim="the stack licensed by `e4b.b`: 2.52"))
    assert _has(f, "cites `e4b.b`", "no licensed_by")
    f = _findings(repo, gate, _row("e4b.a", claim="the stack licensed by `e4b.gate`; our licensed arm: 2.52"))
    assert _has(f, "asserts a licence", "our licensed")
    f = _findings(repo, gate, _row("e4b.v", claim="[VOID] not licensed here", notes="the pack licensed by `e4b.nope`"))
    assert _has(f, "notes cites `e4b.nope`")
    assert ccr.licence_citations("licensed by `e4b.x`, then licensed by `gnf4.y`") == ["e4b.x", "gnf4.y"]
    assert ccr.asserts_licence("licensed by `e4b.x`") is None


def test_pack_fingerprint_format_and_licence_match(repo):
    fp, other = "sha256:" + "b" * 64, "sha256:" + "c" * 64
    gate = dict(claim="LICENSED under the gate", licensed_by="e4b.gate")
    stack = dict(claim="the licensed stack", licensed_by="e4b.gate")
    assert _has(_findings(repo, _row(pack_fingerprint="not-a-hash")), "pack_fingerprint", "sha256")
    assert _findings(repo, _row(claim="measured, not licensed: 236.4 tok/s", pack_fingerprint="sha256:" + "a" * 64)) == []
    assert _findings(repo, _row("e4b.gate", **gate), _row("e4b.a", **stack)) == []
    assert _has(_findings(repo, _row("e4b.gate", pack_fingerprint=fp, **gate), _row("e4b.a", **stack)),
                "carries pack_fingerprint but this row does not")
    assert _has(_findings(repo, _row("e4b.gate", **gate), _row("e4b.a", pack_fingerprint=fp, **stack)),
                "licensed_by e4b.gate has none")
    assert _has(_findings(repo, _row("e4b.gate", pack_fingerprint=fp, **gate), _row("e4b.a", pack_fingerprint=other, **stack)),
                "!= licensed_by")
    assert _findings(repo, _row("e4b.gate", pack_fingerprint=fp, **gate), _row("e4b.a", pack_fingerprint=fp, **stack)) == []


# ------------------------------------------------------------------------------ the file itself --

def test_the_file_and_row_shapes_are_the_schemas(repo):
    assert _has(_findings(repo, _row(), extra=1), "top-level key 'extra' is not in the schema")
    (repo / "docs" / "claims.json").write_text(json.dumps(
        {"status_vocabulary": dict(VOCAB, bogus="b"), "claims": [_row()]}), encoding="utf-8")
    with pytest.raises(ccr.ContractError, match="bogus"):            # the check cannot run: exit 2, never a pass
        ccr.check(repo)
    assert ccr.main(["--root", str(repo)]) == 2
    _write(repo, [_row(status="gold")])
    with pytest.raises(ccr.ContractError):                           # a status the file does not define
        ccr.check(repo)
    assert _has(_findings(repo, _row("e4b.a"), _row("gnf4.b")), "more than one namespace")
    assert _has(_findings(repo, _row(superseeded_by="e4b.b")), "field 'superseeded_by' is not in the schema")
    for field, bad in (("area", "vibes"), ("tier", "gold"), ("validity", "OK"), ("row_status", "FINE"),
                       ("parity_verdict", "MAYBE")):
        assert _has(_findings(repo, _row(**{field: bad})), f"{field} {bad!r} is not one of"), field
    assert _findings(repo, _row(parity_verdict=None, area="serve", tier="measured", validity="VOID")) == []
    assert _has(_findings(repo, _row(package=KERNELS)), "is not this repository's")
    assert _findings(repo, _row(package=RUNTIME)) == []


# ----------------------------------------------------------------------- this repository itself --

def test_the_repository_register_passes():
    r = ccr.check(ROOT)
    assert r.findings == [], r.findings
    assert r.n_claims >= 30


def test_the_command_line_passes_on_this_repository():
    p = subprocess.run([sys.executable, str(_SCRIPT), "--root", str(ROOT)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "OK: docs/claims.json" in p.stdout
