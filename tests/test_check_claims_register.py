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
import pathlib
import subprocess
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_claims_register.py"
_spec = importlib.util.spec_from_file_location("check_claims_register", _SCRIPT)
ccr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccr)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _claim(cid, status="measured", **kw):
    c = {"id": cid, "status": status, "claim": "a sentence", "evidence": ["README.md"], "measured_on": "2026-09-05"}
    c.update(kw)
    return c


def _findings(*claims, root=ROOT):
    return ccr.check_claims(root, list(claims))


def test_a_clean_register_passes():
    assert _findings(_claim("e4b.a"), _claim("e4b.old", "superseded", superseded_by="e4b.a"),
                     _claim("e4b.gone", "retired", retired_reason="wrong box")) == []


def test_evidence_must_exist_at_head_or_be_structured():
    f = _findings(_claim("e4b.a", evidence=["bench/nowhere/run.log"]))
    assert any("does not exist at HEAD" in x for x in f)
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
    assert any("does not exist in the sibling" in x for x in ccr.check_claims(ROOT, bad, sibling=sib, sibling_slug="o/gnf4"))
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
    assert any("asserts a licence and the row has no licensed_by" in x for x in f)
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
