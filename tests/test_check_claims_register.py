"""scripts/check_claims_register.py: the register's structure, pinned on fixtures.

The 2026-09-05 audit found every register check keyed on ``status`` alone:
three ``measured`` rows named receipt logs that were never committed,
annotated and scratch paths sat in ``evidence[]``, ``superseded`` rows had no
successor, ``notes`` said "pending" on rows measured days earlier, and a
sentence said "best licensed" after the register withdrew the licence. Each
of those is a finding here; the last test runs the real check on the
repository's own register.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_claims_register.py"
_spec = importlib.util.spec_from_file_location("check_claims_register", _SCRIPT)
ccr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccr)

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACKED = ccr.tracked_files(ROOT)


def _git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


def _claim(cid, status="measured", **kw):
    c = {"id": cid, "status": status, "claim": "a sentence", "evidence": ["README.md"], "measured_on": "2026-09-05"}
    c.update(kw)
    return c


def _findings(*claims, root=ROOT):
    return ccr.check_claims(root, list(claims), tracked=TRACKED)


def test_a_clean_register_passes():
    assert _findings(_claim("e4b.a"), _claim("e4b.old", "superseded", superseded_by="e4b.a"),
                     _claim("e4b.gone", "retired", retired_reason="wrong box")) == []


def test_evidence_must_exist_at_head_or_be_structured():
    f = _findings(_claim("e4b.a", evidence=["bench/nowhere/run.log"]))
    assert any("not in the git tree at HEAD" in x for x in f)
    f = _findings(_claim("e4b.a", evidence=["bench/hybrid-g9/step_decomp.py (--ppl-fq)"]))
    assert any("annotated or a glob" in x for x in f)
    f = _findings(_claim("e4b.a", evidence=["https://github.com/o/r/issues/1"]))
    assert any("is a URL" in x for x in f)
    assert _findings(_claim("e4b.a", evidence=["docs/METHODOLOGY.md#10", {"url": "https://github.com/o/r/issues/344"},
                                               {"repository": "o/other", "path": "kernel/RESULTS.md"}])) == []
    f = _findings(_claim("e4b.a", evidence=[{"repository": "o/other"}]))
    assert any("structured entry" in x for x in f)
    f = _findings(_claim("e4b.a", evidence=[{"url": "https://example.com/x"}]))
    assert any("not a github.com issue" in x for x in f)


def test_cross_repository_evidence_is_checked_against_the_sibling(tmp_path):
    sib = tmp_path / "sib"
    (sib / "kernel").mkdir(parents=True)
    (sib / "kernel" / "RESULTS.md").write_text("x")
    ok = [_claim("e4b.a", evidence=[{"repository": "o/gnf4", "path": "kernel/RESULTS.md"}])]
    assert ccr.check_claims(ROOT, ok, sibling=sib, sibling_slug="o/gnf4") == []
    bad = [_claim("e4b.a", evidence=[{"repository": "o/gnf4", "path": "kernel/MISSING.md"}])]
    assert any("is not a file in the sibling" in x for x in ccr.check_claims(ROOT, bad, sibling=sib, sibling_slug="o/gnf4"))
    up = [_claim("e4b.a", evidence=[{"repository": "o/gnf4", "path": "../kernel/RESULTS.md"}])]
    assert any("not a bare repository-relative path" in x for x in ccr.check_claims(ROOT, up, sibling=sib, sibling_slug="o/gnf4"))
    # another repository's file is not checked against this sibling
    assert ccr.check_claims(ROOT, bad, sibling=sib, sibling_slug="o/elsewhere") == []


def test_dated_statuses_need_an_iso_measured_on():
    for st in ("measured", "measured-private"):
        f = _findings(_claim("e4b.a", st, measured_on=None))
        assert any("needs measured_on as YYYY-MM-DD" in x for x in f)
        f = _findings(_claim("e4b.a", st, measured_on="2026-08"))
        assert any("needs measured_on as YYYY-MM-DD" in x for x in f)
    assert _findings(_claim("e4b.o", "open", measured_on=None, evidence=[])) == []


def test_superseded_rows_resolve_through_a_chain_to_an_active_row():
    f = _findings(_claim("e4b.old", "superseded"))
    assert any("superseded without superseded_by" in x for x in f)
    f = _findings(_claim("e4b.old", "superseded", superseded_by="e4b.nope"))
    assert any("not in the register" in x for x in f)
    chain = [_claim("e4b.a"), _claim("e4b.mid", "superseded", superseded_by="e4b.a"),
             _claim("e4b.old", "superseded", superseded_by="e4b.mid")]
    assert _findings(*chain) == []
    cycle = [_claim("e4b.x", "superseded", superseded_by="e4b.y"), _claim("e4b.y", "superseded", superseded_by="e4b.x")]
    assert any("cycle" in x for x in _findings(*cycle))
    dead = [_claim("e4b.old", "superseded", superseded_by="e4b.gone"), _claim("e4b.gone", "retired", retired_reason="r")]
    assert any("a successor must be active" in x for x in _findings(*dead))


def test_retired_needs_a_reason_and_supersedes_must_resolve():
    f = _findings(_claim("e4b.gone", "retired", retired_reason=""))
    assert any("retired without retired_reason" in x for x in f)
    f = _findings(_claim("e4b.a", supersedes=["e4b.never"]))
    assert any("supersedes 'e4b.never', which is not in the register" in x for x in f)


def test_quoted_in_paths_resolve():
    assert _findings(_claim("e4b.a", quoted_in=["README.md (opening)", "docs/METHODOLOGY.md#13 (corrected)", "CHANGELOG.md 0.28.0"])) == []
    f = _findings(_claim("e4b.a", quoted_in=["docs/NOPE.md#1"]))
    assert any("quoted_in[0]" in x and "does not exist" in x for x in f)


def test_active_rows_never_say_pending():
    f = _findings(_claim("e4b.a", notes="its number is pending on the validation lane"))
    assert any("says 'pending'" in x for x in f)
    f = _findings(_claim("e4b.a", claim="TBD: a number"))
    assert any("says 'TBD'" in x for x in f)
    assert _findings(_claim("e4b.o", "open", measured_on=None, evidence=[], notes="pending a run")) == []


def test_a_licence_label_needs_its_verdict_row():
    f = _findings(_claim("e4b.a", claim="best licensed configuration: 154.9 tok/s"))
    assert any("asserts a licence" in x and "no licensed_by" in x for x in f)
    # a disclaimer is not a licence
    assert _findings(_claim("e4b.a", claim="fastest configuration -- measured, not licensed: 154.9 tok/s")) == []
    assert _findings(_claim("e4b.a", claim="NOT a licensed best: 284.6 tok/s")) == []
    # licensed_by resolves to an active row (a row may name itself when its own receipt carries the verdict)
    ok = [_claim("e4b.gate", claim="LICENSED under the gate: -0.05 ppl", licensed_by="e4b.gate"),
          _claim("e4b.a", claim="the licensed stack: 304.9 tok/s", licensed_by="e4b.gate")]
    assert _findings(*ok) == []
    f = _findings(_claim("e4b.a", claim="the licensed stack", licensed_by="e4b.nope"))
    assert any("licensed_by 'e4b.nope' is not in the register" in x for x in f)
    f = _findings(_claim("e4b.a", claim="the licensed stack", licensed_by="e4b.old"),
                  _claim("e4b.old", "superseded", superseded_by="e4b.a"))
    assert any("a licence comes from an active verdict row" in x for x in f)


def test_the_repository_register_passes():
    p = subprocess.run([sys.executable, str(_SCRIPT), "--root", str(ROOT)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "OK: docs/claims.json" in p.stdout


# ------------------------------------------------- the review of #406: what the first version let through --

def test_evidence_is_a_file_in_the_git_tree_not_the_working_tree(tmp_path):
    """``*.log`` is gitignored: a receipt log is evidence once it is force-added and never before, so the rule
    reads ``git ls-files`` -- ``exists()`` passed an untracked log, a directory, an absolute path and ``..``."""
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "RESULTS.md").write_text("x")
    (tmp_path / "bench" / "run.log").write_text("untracked")
    (tmp_path / ".gitignore").write_text("*.log\n")
    _git("add", ".gitignore", "bench/RESULTS.md", cwd=tmp_path)
    _git("commit", "-q", "-m", "x", cwd=tmp_path)
    tracked = ccr.tracked_files(tmp_path)
    assert tracked == {".gitignore", "bench/RESULTS.md"}

    def f(ev):
        return ccr.check_claims(tmp_path, [_claim("e4b.a", evidence=ev)], tracked=tracked)

    assert f(["bench/RESULTS.md#L3"]) == []
    assert any("not in the git tree at HEAD" in x for x in f(["bench/run.log"]))
    assert any("is a directory" in x for x in f(["bench"]))
    assert any("not a repository-relative path" in x for x in f(["../elsewhere/RESULTS.md"]))
    assert any("not a repository-relative path" in x for x in f([str(tmp_path / "bench" / "RESULTS.md")]))
    _git("add", "-f", "bench/run.log", cwd=tmp_path)                       # force-added: now it is evidence
    assert ccr.check_claims(tmp_path, [_claim("e4b.a", evidence=["bench/run.log"])],
                            tracked=ccr.tracked_files(tmp_path)) == []


def test_outside_a_checkout_the_working_tree_stands_in_and_the_output_says_so(tmp_path):
    assert ccr.tracked_files(tmp_path) is None
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "R.md").write_text("x")

    def f(ev):
        return ccr.check_claims(tmp_path, [_claim("e4b.a", evidence=ev)], tracked=None)

    assert f(["docs/R.md"]) == []
    assert any("does not exist in the working tree" in x for x in f(["docs/NOPE.md"]))
    assert any("is a directory" in x for x in f(["docs"]))
    reg = {"status_vocabulary": {"measured": "m"},
           "claims": [{"id": "e4b.a", "status": "measured", "claim": "s", "evidence": ["docs/R.md"], "measured_on": "2026-09-05"}]}
    (tmp_path / "docs" / "claims.json").write_text(json.dumps(reg))
    p = subprocess.run([sys.executable, str(_SCRIPT), "--root", str(tmp_path)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "NOTE:" in p.stdout and "not a git checkout" in p.stdout and "the working tree (not a checkout)" in p.stdout


def test_a_sibling_whose_slug_cannot_be_resolved_is_exit_2_not_a_silent_skip(tmp_path):
    """With ``--sibling`` and no resolvable owner/name, every cross-repository entry used to be skipped while the
    summary said they were checked."""
    sib = tmp_path / "sib"
    sib.mkdir()
    (sib / "pyproject.toml").write_text('[project]\nname = "grouped-nf4-gemm"\nversion = "0.30.0"\n')   # no urls, no remote
    p = subprocess.run([sys.executable, str(_SCRIPT), "--root", str(ROOT), "--sibling", str(sib)], capture_output=True, text=True)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "cannot resolve the sibling's owner/name" in p.stdout and "OK:" not in p.stdout


def test_superseded_by_is_followed_on_any_row_and_never_sits_on_an_active_one():
    ok = [_claim("e4b.a"), _claim("e4b.gone", "retired", retired_reason="r", superseded_by="e4b.a")]
    assert _findings(*ok) == []
    dead = [_claim("e4b.gone", "retired", retired_reason="r", superseded_by="e4b.also-gone"),
            _claim("e4b.also-gone", "retired", retired_reason="r")]
    assert any("a successor must be active" in x for x in _findings(*dead))
    f = _findings(_claim("e4b.a", superseded_by="e4b.b"), _claim("e4b.b"))
    assert any("an active row carries superseded_by" in x for x in f)


def test_retired_reason_belongs_to_retired_rows_only():
    f = _findings(_claim("e4b.a"), _claim("e4b.old", "superseded", superseded_by="e4b.a", retired_reason="r"))
    assert any("retired_reason on a 'superseded' row" in x for x in f)
    assert any("retired_reason on a 'measured' row" in x for x in _findings(_claim("e4b.a", retired_reason="r")))


def test_the_licence_rule_is_per_occurrence():
    """A sentence-level negation let one "unlicensed" excuse a bare "licensed stack" beside it."""
    f = _findings(_claim("e4b.a", claim="the licensed stack: 304.9 tok/s (the unlicensed int4 arm, 487.8, beside)"))
    assert any("asserts a licence" in x and "'the licensed'" in x for x in f)
    f = _findings(_claim("e4b.a", claim="the pack read 11522 / 766, not the licensed 11512 / 776 -- unlicensed here"))
    assert any("asserts a licence" in x for x in f), "'not the licensed X' describes the licensed pack; it is not a disclaimer"
    for ok in ("measured, not licensed: 154.9 tok/s", "NOT a licensed best: 284.6 tok/s", "never licensed on this box",
               "no licensed arm exists for this family", "an un-licensed reading", "unlicensed, VOID stands"):
        assert _findings(_claim("e4b.a", claim=ok)) == [], ok
    assert ccr.asserts_licence("not licensed; the licensed stack") is not None
    assert ccr.asserts_licence("not licensed; never licensed") is None


def test_a_licence_citation_is_a_reference_not_an_assertion():
    """``licensed by `<id>` `` refers to another row's licence: the row needs no label, and the id must resolve to
    a row that itself carries licensed_by (a licensed row, or a verdict row, which names itself)."""
    gate = _claim("e4b.gate", claim="LICENSED under the gate: -0.05 ppl", licensed_by="e4b.gate")
    assert _findings(gate, _claim("e4b.a", claim="no ratio against the stack licensed by `e4b.gate`: 2.52")) == []
    f = _findings(gate, _claim("e4b.a", claim="the stack licensed by `e4b.nope`: 2.52"))
    assert any("claim cites `e4b.nope`" in x and "not in the register" in x for x in f)
    f = _findings(gate, _claim("e4b.b", claim="plain: 1.0"), _claim("e4b.a", claim="the stack licensed by `e4b.b`: 2.52"))
    assert any("cites `e4b.b`" in x and "no licensed_by" in x for x in f)
    # a citation never excuses a bare assertion beside it
    f = _findings(gate, _claim("e4b.a", claim="the stack licensed by `e4b.gate`; our licensed arm: 2.52"))
    assert any("asserts a licence" in x and "our licensed" in x for x in f)
    # citations in notes resolve too, on any row
    f = _findings(gate, _claim("e4b.v", claim="[VOID] not licensed here", notes="the pack licensed by `e4b.nope`"))
    assert any("notes cites `e4b.nope`" in x for x in f)
    assert ccr.licence_citations("licensed by `e4b.x`, then licensed by `e4b.y`") == ["e4b.x", "e4b.y"]
    assert ccr.asserts_licence("licensed by `e4b.x`") is None
