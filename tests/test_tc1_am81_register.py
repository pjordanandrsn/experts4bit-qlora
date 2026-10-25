"""TC1 amendment 81's register and quote moves (the register schema's successor and notes rules, read against the register as it stood before the amendment).

- 13 rows whose CLAIM quoted a moved device figure get a successor ``<id>.am81``. The predecessor changes in exactly two fields: status
  ``superseded`` and ``superseded_by``. Its claim, figures and evidence are byte-identical.
- 3 rows whose moved figure sat only in their notes get one correction sentence in notes, naming the corrected value and the
  am81-rederive path. Every other field, the claim included, is identical.
- No other row changes.
- ``device_rederive --quotes`` run on this tree finds the old figures only where they are meant to stay. That means superseded
  predecessors, the notes-only rows, frozen or generated documents, and coincidences each named here with its reason.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AM81_BASE = "f372d65f629d29020eb2ad94683032e09ba752ed"        # main before amendment 81 (the machine-id pass's merge, #2143)
AM = "bench/tc1/am81-rederive/"
T = "e4b.train."

SUCCESSORS = {                                                 # predecessor (without e4b.train.) -> the runs its figures come from
    "h2h.unsloth.qwen3.5090.2026-10-02.host-variance": ["tc1-5090-16"],
    "h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile": ["tc1-5090-16", "tc1c-h100-2"],
    "env-gap.torch28.profile.qwen3.5090.2026-10-06": ["tc1-5090-110"],
    "ckpt-offload.field-torch28.5090.2026-10-07": ["tc1-5090-125"],
    "ckpt-flavour.default.5090.2026-10-07": ["tc1-5090-128"],
    "h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio": ["tc1-5090-136", "tc1-5090-133"],
    "h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio": ["tc1-5090-137"],
    "single-ladder.field.5090.2026-10-08": ["tc1-5090-138"],
    "single-ladder-auto.field.5090.2026-10-08": ["tc1-5090-139"],
    "single-ladder-auto.gpu-bound.5090.2026-10-08": ["tc1-5090-141"],
    "p129.fused-qkv.qwen3.5090.2026-10-10": ["tc1-5090-147", "tc1-5090-146"],
    "p129.fused-qkv.second-host.5090.2026-10-10": ["tc1-5090-150", "tc1-5090-147"],
    "h2h.unsloth.qwen3.h100.2026-10-04.route-v2": ["tc1c-h100-9"],
}
NOTES_ONLY = {"h2h.unsloth.qwen3.5090.2026-10-02": "tc1-5090-16", "h2h.unsloth.qwen3.h100.2026-10-04": "tc1c-h100-3",
              "h2h.unsloth.qwen3.h100.2026-10-04.moe-keep": "tc1c-h100-4"}

# old figures --quotes still finds on this tree, each with why it stays
ROW_KEEPS = {
    (T + "h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio.am81", "10152.2"):
        "named on purpose: the registered reducer's all-CUDA constant for amendment 66, beside its kernels-only 10125.2",
    (T + "p129.fused-qkv.qwen3.5090.2026-10-10.am81", "0.988"): "the successor's own new matched device ratio (tc1-5090-147); "
        "0.988 is also tc1-5090-150's OLD shipped ratio, which the shared family token places here",
    (T + "p129.fused-qkv.second-host.5090.2026-10-10.am81", "0.988"): "tc1-5090-150's matched device ratio, unchanged at 0.988 "
        "kernels-only; its old shipped ratio was also 0.988",
    (T + "single-ladder.field.5090.2026-10-08.am81", "0.330"): "the training peak's 0.33 GB, not a busy fraction",
}
DOC_KEEPS = {      # path -> (why, a regex the matched line must satisfy, so a real quote is never excused)
    "docs/STATUS-RECORD.md": ("frozen at 0.50.0: named in the amendment, never edited", r""),
    "llms-full.txt": ("generated from the documents checked here (build_llms_bundle --check)", r""),
    "docs/COMPARATORS.md": ("a version string, release-0.45.0", r"0\.45\.0"),
    "docs/capabilities.json": ("a version string, release-0.45.0", r"0\.45\.0"),
    "docs/solutions/qlora-fused-moe-experts.md": ("a version string, release-0.45.0", r"0\.45\.0"),
    "docs/REPRODUCE.md": ("a version string, e4b 0.49.0", r"0\.49\.0"),
}


def _base_register():
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{AM81_BASE}:docs/claims.json"], capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        if os.environ.get("CI") == "true":
            pytest.fail(f"the base commit {AM81_BASE[:8]} is not in this clone (fetch-depth): {r.stderr[:200]}")
        pytest.skip(f"no {AM81_BASE[:8]} in this clone")
    return {c["id"]: c for c in json.loads(r.stdout)["claims"]}


@pytest.fixture(scope="module")
def registers():
    head = {c["id"]: c for c in json.loads((REPO / "docs" / "claims.json").read_text(encoding="utf-8"))["claims"]}
    return _base_register(), head


def test_each_predecessor_changes_only_status_and_superseded_by(registers):
    base, head = registers
    for short in SUCCESSORS:
        old, now = base[T + short], head[T + short]
        assert now["status"] == "superseded" and now["superseded_by"] == T + short + ".am81", short
        assert {k: v for k, v in now.items() if k not in ("status", "superseded_by")} == \
               {k: v for k, v in old.items() if k != "status"}, short
        assert old["status"] == "measured", short


def test_each_successor_is_well_formed(registers):
    base, head = registers
    for short, runs in SUCCESSORS.items():
        old, new = base[T + short], head[T + short + ".am81"]
        assert new["status"] == "measured" and new["supersedes"] == [T + short] and "superseded_by" not in new, short
        assert new["measured_on"] == old["measured_on"] and new["evidence"][0] == old["evidence"][0], short
        for r in runs:
            assert AM + r + "/rederive.json" in new["evidence"] and (REPO / AM / r / "rederive.json").is_file(), (short, r)
            assert AM + r + "/" in new["notes"], (short, r)
        mids = re.compile(r"machine \d+")                                      # the predecessor's machine ids, as the machine-id pass left them
        assert mids.findall(json.dumps(new)) == mids.findall(json.dumps(old)), short
        assert old["claim"] != new["claim"], short


def test_the_two_headline_values_are_the_rederived_ratios(registers):
    _, head = registers
    for short, run, pid, value in (("h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio", "tc1-5090-137", "P205", 1.953),
                                   ("h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio", "tc1-5090-136", "P200", 1.161)):
        rec = json.loads((REPO / AM / run / "rederive.json").read_text(encoding="utf-8"))
        line = next(ln for ln in rec["diff"] if ln.startswith(f"+| {pid} |"))
        assert re.search(rf"= {value:.3f} vs in", line), line
        assert head[T + short + ".am81"]["value"] == value


def test_notes_only_rows_change_only_their_notes(registers):
    base, head = registers
    for short, run in NOTES_ONLY.items():
        old, now = base[T + short], head[T + short]
        assert {k: v for k, v in now.items() if k != "notes"} == {k: v for k, v in old.items() if k != "notes"}, short
        assert now["notes"].startswith(old["notes"].rstrip()) and len(now["notes"]) > len(old["notes"]), short
        added = now["notes"][len(old["notes"].rstrip()):]
        assert "TC1 amendment 81" in added and AM + run + "/" in added, short


def test_no_other_row_changes(registers):
    base, head = registers
    touched = {T + s for s in SUCCESSORS} | {T + s for s in NOTES_ONLY}
    added = {T + s + ".am81" for s in SUCCESSORS}
    assert set(head) == set(base) | added
    for rid in set(base) - touched:
        assert head[rid] == base[rid], rid


def _quotes():
    spec = importlib.util.spec_from_file_location("device_rederive_q", REPO / "bench" / "tc1" / "device_rederive.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["device_rederive_q"] = m
    spec.loader.exec_module(m)
    return m.quoted_figures(str(REPO), str(REPO / AM))


def test_every_old_figure_left_in_the_register_or_the_docs_is_meant_to_be_there(registers):
    _, head = registers
    q = _quotes()
    for rid, hits in q["rows"].items():
        if head[rid]["status"] == "superseded" or rid[len(T):] in NOTES_ONLY:
            continue
        for h in hits:
            assert (rid, h["old"]) in ROW_KEEPS, f"{rid} still quotes {h['old']} ({h['field']}, {h['run']} {h['where']}) -> {h['new']}"
    for d in q["docs"]:
        why, must = DOC_KEEPS.get(d["path"], (None, None))
        assert why is not None, f"{d['path']}:{d['line']} quotes {d['old']} -> {d['new']} as {d['form']!r}: {d['text'][:120]}"
        line = (REPO / d["path"]).read_text(encoding="utf-8").splitlines()[d["line"] - 1]
        assert re.search(must, line), f"{d['path']}:{d['line']} is excused as {why!r}, but the line does not show it"


def test_mutant_a_quote_left_behind_is_caught(registers, tmp_path):
    """Mutant: README.md:95 left at 1.92x beside the old id, and the predecessor row left active. In a tree holding run 137's
    committed cells and its am81 copies, --quotes must flag both: the row (cell 1693.9 per step and the reducer's 1.924) and the
    README line (1.92, 2 dp of 1.924, which moves to 1.95)."""
    base, _ = registers
    rec = json.loads((REPO / AM / "tc1-5090-137" / "rederive.json").read_text(encoding="utf-8"))
    src = REPO / rec["path"]
    for cell in rec["cells"]:
        rel = next(p.relative_to(REPO) for p in (src / d / cell for d in ("tc1", "public", "")) if p.exists())
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
        (tmp_path / "out" / "tc1-5090-137").mkdir(parents=True, exist_ok=True)
        (tmp_path / "out" / "tc1-5090-137" / cell).write_bytes((REPO / AM / "tc1-5090-137" / cell).read_bytes())
    (tmp_path / "out" / "tc1-5090-137" / "rederive.json").write_text(json.dumps(rec))
    rid = T + "h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "claims.json").write_text(json.dumps({"claims": [base[rid]]}), encoding="utf-8")
    (tmp_path / "README.md").write_text(f"| TC1 field recipe (`{rid}`) | Unsloth spends **1.92×** e4b's GPU time per step | measured |\n",
                                        encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    spec = importlib.util.spec_from_file_location("device_rederive_m", REPO / "bench" / "tc1" / "device_rederive.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["device_rederive_m"] = m
    spec.loader.exec_module(m)
    q = m.quoted_figures(str(tmp_path), str(tmp_path / "out"))
    olds = {h["old"] for h in q["rows"].get(rid, [])}
    assert {"1.924", "1693.9", "3259.1"} <= olds, olds
    assert {(d["path"], d["form"]) for d in q["docs"] if d["old"] == "1.924"} == {("README.md", "1.92")}, q["docs"]
