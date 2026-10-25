"""TC1 amendment 81: bench/tc1/device_rederive.py, the re-derivation tool, on a self-contained receipt and a stand-in reducer.

The stand-in reducer reads cells the way the registered tc1_reduce.py does for the figures this tool moves: a table row per arm with its
device ms and busy fraction and a VALID/VOID verdict cell, a prediction line with a bold verdict and a ``vs <=`` comparison, and a
by-family line computed as round(fraction * device_ms, 1) from the CELL (tc1_reduce.py ca6b8b29:1192). The 26 real runs are re-derived
with their own staged reducers in the amendment's report; this file pins the tool's rules and kills a mutant of each.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import shutil

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "bench" / "tc1" / "device_rederive.py"

STANDIN = r'''
import json, os, sys
work, md = sys.argv[1], sys.argv[3]
BAR = float(os.environ.get("STANDIN_BAR", "0.9"))
out = [f"# TC1 ({work})", "", "| arm | verdict | device ms | busy |", "|---|---|---|---|"]
fam_lines = []
for f in sorted(os.listdir(work)):
    if not f.endswith(".json") or f.endswith("_profile.json"):
        continue
    try:
        c = json.load(open(os.path.join(work, f)))
    except ValueError:
        continue                       # a tolerant reducer: the tool must not be

    pr = c.get("profile") or {}
    if pr.get("device_ms") is None:
        continue
    busy = pr["device_busy_fraction"]
    out.append(f"| {c['arm']} | {'VALID' if busy <= BAR else 'VOID'} | {pr['device_ms']} | {busy} |")
    fams = {k: round(v * pr["device_ms"], 1) for k, v in sorted(pr["device_by_family_fraction"].items())}
    fam_lines.append(f"- device ms by family, {c['arm']}: " + ", ".join(f"{k} {v}" for k, v in fams.items()))
    out.append(f"| P1 | {c['arm']} | **{'HELD' if busy <= BAR else 'FALSIFIED'}** | busy {busy} vs <= {BAR} |")
out += [""] + fam_lines
open(md, "w").write("\n".join(out) + "\n")
'''


def _load():
    spec = importlib.util.spec_from_file_location("device_rederive_t", TOOL)
    m = importlib.util.module_from_spec(spec)
    sys.modules["device_rederive_t"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def dr():
    return _load()


def _receipt(tmp_path, dr, busy_wall=10000.0, ann=60.0, extra_cells=()):
    """A receipt (tc1/ cells + profiles + its committed RESULTS) and the stand-in reducer. One arm, 3 profiled steps, device 9000 ms of
    which 60 ms is the Optimizer.step# annotation, in the optimizer family."""
    rc = tmp_path / "rcpt"
    (rc / "tc1").mkdir(parents=True)
    red = tmp_path / "standin_reduce.py"
    red.write_text(STANDIN)
    fam = {"matmul": 0.6, "optimizer": 0.1, "memcpy": 0.05, "other": 0.25}
    cells = [("arm_a", 9000.0)] + list(extra_cells)
    for arm, dev in cells:
        prof = {"summary": {"device_ms": dev, "profiled_steps": 3},
                "top_device": [{"name": "Optimizer.step#AdamW8bit.step", "family": "optimizer", "count": 3, "self_device_ms": ann},
                               {"name": "gemm_kernel", "family": "matmul", "count": 300, "self_device_ms": 5000.0}],
                "top_cpu": []}
        (rc / "tc1" / f"{arm}_profile.json").write_text(json.dumps(prof))
        cell = {"arm": arm, "profile": {"device_ms": dev, "wall_ms": busy_wall, "profiled_steps": 3,
                                        "device_busy_fraction": round(dev / busy_wall, 4), "device_by_family_fraction": dict(fam),
                                        "memcpy_ms": 450.0, "memcpy_fraction_of_device": round(450.0 / dev, 4),
                                        "path": f"{arm}_profile.json"}}
        (rc / "tc1" / f"{arm}.json").write_text(json.dumps(cell))
    md = dr.reduce_md(dr.assemble(str(rc), str(tmp_path / "pre" / "tc1")), str(red))
    (rc / "RESULTS-tc1.md").write_text(md)
    return rc, red


def test_a_clean_receipt_reproduces_and_moves_only_the_correction(dr, tmp_path):
    rc, red = _receipt(tmp_path, dr)
    r = dr.rederive(str(rc), str(red))
    assert r["refused"] is None and r["reproduced"], r
    m = r["cells"]["arm_a.json"]
    assert (m["recorded_ms"], m["kernels_only_ms"], m["annotation_ms"], m["annotation_families"]) == (9000.0, 8940.0, 60.0, ["optimizer"])
    assert r["flips"] == [] and r["unpaired"] == []
    assert r["share"] == round(60 / 9000, 4)


def test_only_the_annotation_family_moves_in_the_reducers_output(dr, tmp_path):
    """(c) exact re-attribution: in the reducer's own family line, every family but optimizer prints the same figure."""
    rc, red = _receipt(tmp_path, dr)
    r = dr.rederive(str(rc), str(red))
    fam_old = [ln for ln in r["diff"] if ln.startswith("-- device ms by family")]
    fam_new = [ln for ln in r["diff"] if ln.startswith("+- device ms by family")]
    assert len(fam_old) == len(fam_new) == 1
    old = dict(x.rsplit(" ", 1) for x in fam_old[0].split(": ", 1)[1].split(", "))
    new = dict(x.rsplit(" ", 1) for x in fam_new[0].split(": ", 1)[1].split(", "))
    assert {k for k in old if old[k] != new[k]} == {"optimizer"}
    assert float(old["optimizer"]) - float(new["optimizer"]) == pytest.approx(60.0, abs=0.11)


def test_an_unparseable_cell_refuses_the_receipt(dr, tmp_path):
    """(A) never `continue` past a cell that does not parse."""
    rc, red = _receipt(tmp_path, dr)
    (rc / "tc1" / "zz_broken.json").write_text("{not json")
    r = dr.rederive(str(rc), str(red))
    assert r["refused"] and "does not parse" in r["refused"]


@pytest.mark.parametrize("text,refused", [
    ('{"dist": "unsloth", "files": 2}\n{"dist": "unsloth_zoo", "files": 3}\n', False),      # amendment 79's release_proof_unsloth.json
    ('{"dist": "unsloth"}\n{"profile": {"device_ms": 1.0}}\n', True),                       # a record carrying a profile is not skipped
    ('{"dist": "unsloth"}\n{"dist": \n', True),                                             # a torn record file
])
def test_a_json_lines_record_file_is_not_a_cell_but_a_torn_one_refuses(dr, tmp_path, text, refused):
    rc, red = _receipt(tmp_path, dr)
    (rc / "tc1" / "release_proof_unsloth.json").write_text(text)
    r = dr.rederive(str(rc), str(red))
    assert bool(r["refused"]) is refused, r["refused"]
    if refused:
        assert "does not parse" in r["refused"]


def test_a_missing_profile_sidecar_refuses(dr, tmp_path):
    rc, red = _receipt(tmp_path, dr)
    (rc / "tc1" / "arm_a_profile.json").unlink()
    r = dr.rederive(str(rc), str(red))
    assert r["refused"] and "is not in the receipt" in r["refused"]


def test_a_cell_whose_device_ms_is_not_its_profiles_refuses(dr, tmp_path):
    rc, red = _receipt(tmp_path, dr)
    p = rc / "tc1" / "arm_a_profile.json"
    d = json.loads(p.read_text())
    d["summary"]["device_ms"] = 9100.0
    p.write_text(json.dumps(d))
    r = dr.rederive(str(rc), str(red))
    assert r["refused"] and "is not its profile's" in r["refused"]


@pytest.mark.parametrize("family", ["matmul", "nowhere"])
def test_an_annotation_row_outside_the_optimizer_family_refuses(dr, tmp_path, family):
    """An annotation row in any family but optimizer refuses the receipt: it needs a ruling, never a re-attribution here."""
    rc, red = _receipt(tmp_path, dr)
    p = rc / "tc1" / "arm_a_profile.json"
    d = json.loads(p.read_text())
    d["top_device"][0]["family"] = family
    p.write_text(json.dumps(d))
    r = dr.rederive(str(rc), str(red))
    assert r["refused"] and "not 'optimizer'" in r["refused"]


def test_a_second_annotation_row_is_recorded_and_subtracted(dr, tmp_path):
    """f15v2.ANNOTATION_RE also matches ProfilerStep#; each row is recorded with its name, family and ms, and all of them come out."""
    rc, red = _receipt(tmp_path, dr)
    p = rc / "tc1" / "arm_a_profile.json"
    d = json.loads(p.read_text())
    d["top_device"].append({"name": "ProfilerStep#7", "family": "optimizer", "count": 3, "self_device_ms": 30.0})
    p.write_text(json.dumps(d))
    r = dr.rederive(str(rc), str(red))
    assert r["refused"] is None, r["refused"]
    m = r["cells"]["arm_a.json"]
    assert [(a["name"], a["family"], a["self_device_ms"]) for a in m["annotation_rows"]] == [
        ("Optimizer.step#AdamW8bit.step", "optimizer", 60.0), ("ProfilerStep#7", "optimizer", 30.0)]
    assert m["kernels_only_ms"] == 8910.0


def test_an_annotation_family_the_cell_does_not_carry_refuses(dr, tmp_path):
    rc, red = _receipt(tmp_path, dr)
    p = rc / "tc1" / "arm_a.json"
    c = json.loads(p.read_text())
    fam = c["profile"]["device_by_family_fraction"]
    fam["other"] += fam.pop("optimizer")
    p.write_text(json.dumps(c))
    md = dr.reduce_md(dr.assemble(str(rc), str(tmp_path / "pre2" / "tc1")), str(red))
    (rc / "RESULTS-tc1.md").write_text(md)
    r = dr.rederive(str(rc), str(red))
    assert r["refused"] and "which the cell does not carry" in r["refused"]


def test_a_wrong_reducer_is_unreproduced_never_corrected(dr, tmp_path):
    rc, red = _receipt(tmp_path, dr)
    red.write_text(STANDIN.replace("busy {busy} vs", "busy={busy} vs"))
    r = dr.rederive(str(rc), str(red))
    assert not r["reproduced"] and r["cells"] == {}


@pytest.mark.parametrize("mutant", ["pro_rata", "no_memcpy", "extra_field", "busy_unchanged"])
def test_each_formula_mutant_is_killed_by_the_input_assertion(dr, tmp_path, mutant):
    """(a) + review condition 4: a correction that scales families pro rata, forgets memcpy_fraction, touches another field, or leaves
    the busy fraction refuses."""
    rc, red = _receipt(tmp_path, dr)
    good = dr.corrected_profile

    def bad(pr, full, f2):
        out, rec = good(pr, full, f2)
        if mutant == "pro_rata":
            out["device_by_family_fraction"] = dict(pr["device_by_family_fraction"])
        elif mutant == "no_memcpy":
            out["memcpy_fraction_of_device"] = pr["memcpy_fraction_of_device"]
        elif mutant == "extra_field":
            out["wall_ms"] = pr["wall_ms"] + 1
        else:
            out["device_busy_fraction"] = pr["device_busy_fraction"]
        return out, rec

    dr.corrected_profile = bad
    r = dr.rederive(str(rc), str(red))
    assert r["refused"], mutant


def test_a_flipped_table_cell_in_a_hunk_that_also_gains_a_line_is_a_flip(dr):
    """(B) + (d): pairing is per hunk by row key, so an added line cannot misalign the pair, and a table verdict cell counts."""
    before = "# TC1 (x)\n| arm | verdict | dev |\n| a1 | VALID | 100 |\n| a2 | VALID | 200 |\n"
    after = "# TC1 (x)\n| arm | verdict | dev |\n| a0 | VALID | 50 |\n| a1 | VALID | 99 |\n| a2 | VOID | 199 |\n"
    flips, unpaired = [], []
    for removed, added in dr.hunks(before, after):
        pairs, ur, ua = dr.pair(removed, added)
        flips += [(x, y) for x, y in pairs if dr.verdicts(x) != dr.verdicts(y)]
        unpaired += [ln for ln in ur + ua if dr.verdicts(ln)]
    assert flips == [("| a2 | VALID | 200 |", "| a2 | VOID | 199 |")]
    assert unpaired == ["| a0 | VALID | 50 |"]


def test_mutant_global_zip_pairing_would_miss_that_flip(dr):
    """The old pairing (zip of all removed and added lines) pairs a2 with a1's new row and reads no flip: why pairing is per hunk."""
    before = "# TC1 (x)\n| a1 | VALID | 100 |\n| a2 | VALID | 200 |\n"
    after = "# TC1 (x)\n| a0 | VALID | 50 |\n| a1 | VALID | 99 |\n| a2 | VOID | 199 |\n"
    minus = [ln for r, _ in dr.hunks(before, after) for ln in r]
    plus = [ln for _, a in dr.hunks(before, after) for ln in a]
    zipped = [(x, y) for x, y in zip(minus, plus) if dr.BOLD.findall(x) != dr.BOLD.findall(y)]
    assert zipped == []            # bold-only, zip-paired: blind to the flip the rule above catches


def test_a_bold_verdict_flip_is_read(dr, tmp_path):
    rc, red = _receipt(tmp_path, dr, busy_wall=9990.0)       # recorded busy 0.9009 > 0.9 -> VOID / FALSIFIED; corrected 0.8949 -> flips
    r = dr.rederive(str(rc), str(red))
    words = [(dr.verdicts(x), dr.verdicts(y)) for x, y in r["flips"]]
    assert (["VOID"], ["VALID"]) in words
    assert any("FALSIFIED" in x and "HELD" in y for x, y in words)


def test_near_bar_lists_a_margin_below_the_share_and_unparsed_lines(dr):
    assert dr.near_bar("| P1 | a | **HELD** | busy 0.8995 vs <= 0.9 |", share=0.01) == [("0.8995", "<=", "0.9", 0.0006)]
    assert dr.near_bar("| P1 | a | **HELD** | busy 0.5 vs <= 0.9 |", share=0.01) == []
    assert dr.near_bar("device_busy_fraction 0.295 (>= 0.5 predicted)", share=0.5) == [("0.295", ">=", "0.5", 0.41)]


@pytest.mark.parametrize("line,want", [
    # the real corrected lines whose forms the first parser missed (amendment 81's 26-run report)
    ("| P213 | qwen3sladder | **HELD** | device ms per profiled step l1 / l0: m: 1670.0 -> 1746.8 ms = 1.046; shipped: 1294.4 -> 1336.5 ms = "
     "1.033 (each <= 1.05) |", [("1.046", "<=", "1.05", 0.0038), ("1.033", "<=", "1.05", 0.0162)]),
    ("| P194 | qwen3cbk | **HELD** | m: k1 / k0 0.973 (device ms per profiled step k0 9102.5 / 9132.5, k1 8866.3 / 8875.0); shipped: "
     "k1 / k0 0.978 (device ms per profiled step k0 7164.3 / 7179.6, k1 7012.8 / 7015.6) (each arm <= 1.01) |",
     [("0.973", "<=", "1.01", 0.0366), ("0.978", "<=", "1.01", 0.0317)]),
    ("| P200 | qwen3pos68 | **HELD** | device ms per profiled step Unsloth 11613.2 / e4b 10006.7 = 1.161 vs in [1.06, 1.24]; s/step "
     "Unsloth / e4b 1.472 on this host (descriptive) |", [("1.161", "in", "[1.06, 1.24]", 0.0637)]),
    ("| P174 | qwen3ckptoff28 | **HELD** | m: g1 / g0 0.994 [0.993, 0.995 over 4 cross-draw ratios] vs <= 1.01; s/step g0 2.474 / 2.470, "
     "g1 2.457 / 2.458; g0 device busy vs the timed step 0.764 (premise <= 0.9) |",
     [("0.994", "<=", "1.01", 0.0158), ("0.764", "<=", "0.9", 0.1511)]),
    ("| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1779.2 -> 1878.3 ms = 1.056 (<= 1.06); shipped: 1383.6 -> "
     "1383.0 ms = 1.000 (in [0.98, 1.02]) |", [("1.000", "in", "[0.98, 1.02]", 0.0196), ("1.056", "<=", "1.06", 0.0038)]),
])
def test_every_comparison_form_on_the_real_corrected_lines_is_parsed(dr, line, want):
    assert sorted(dr.comparisons(line)) == sorted(want)


def test_a_branch_that_moves_only_words_is_reworded(dr):
    x = "| P215 | qwen3slauto | **HELD** | la / l0 1.031 vs <= 1.05; matched l0 busy_t 0.884: a GPU-bound box |"
    assert not dr.reworded(x, x.replace("0.884", "0.871"))
    assert dr.reworded(x, x.replace("0.884: a GPU-bound", "0.845: a host-bound"))
    assert dr.verdicts(x) == dr.verdicts(x.replace("GPU-bound", "host-bound"))       # no verdict word moves: only REWORDED sees it


def test_a_line_with_no_printed_bar_parses_nothing(dr):
    """A branch bar that lives only in the reducer (P215's host-bound 0.85) is not on the line: it stays UNPARSED, read by hand."""
    assert dr.comparisons("| P215 | qwen3slauto | **HELD** | la / l0: matched l0 busy_t 0.871: a GPU-bound box |") == []


def _repo(tmp_path, dr):
    """A checkout with the receipt committed at bench/x/receipts/run-1 and a second, untouched receipt beside it; the MAP for run-1."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rc, red = _receipt(tmp_path, dr)
    repo = tmp_path / "repo"
    shutil.copytree(rc, repo / "bench" / "x" / "receipts" / "run-1")
    shutil.copytree(rc, repo / "bench" / "y" / "receipts" / "run-2")
    m = tmp_path / "map.txt"
    m.write_text(f"run-1 {red.name} {hashlib.sha256(red.read_bytes()).hexdigest()[:12]}\n")
    return repo, red, m


def _tree(p):
    return {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(p).rglob("*")) if f.is_file()}


def test_batch_refuses_a_reducer_that_is_not_its_registered_sha(dr, tmp_path):
    repo, red, m = _repo(tmp_path, dr)
    m.write_text(f"run-1 {red.name} {'0' * 12}\n")
    rows = dr.batch(str(m), str(tmp_path), str(repo))
    assert rows[0]["refused"] and "is missing or not" in rows[0]["refused"]


def test_the_committed_receipts_and_the_reducer_are_byte_identical_after_a_persisted_run(dr, tmp_path):
    """The bench rule (a committed receipt is never edited): hash every receipt file and the reducer before and after a run with --out."""
    repo, red, m = _repo(tmp_path, dr)
    before = {**_tree(repo / "bench"), "reducer": hashlib.sha256(red.read_bytes()).hexdigest()}
    out = tmp_path / "am81-rederive"
    assert dr.main(["--batch", str(m), "--store", str(tmp_path), "--repo", str(repo), "--out", str(out)]) == 0
    assert {**_tree(repo / "bench"), "reducer": hashlib.sha256(red.read_bytes()).hexdigest()} == before
    assert sorted(_tree(out)) == ["run-1/arm_a.json", "run-1/rederive.json"]
    assert b"\r" not in b"".join(f.read_bytes() for f in out.rglob("*") if f.is_file())
    cell = json.loads((out / "run-1" / "arm_a.json").read_text())
    assert cell["profile"]["device_ms"] == 8940.0 and cell["profile"]["device_ms_method"] == "kernels_only_rederived"


def test_check_re_derives_a_persisted_run_from_its_copies_and_catches_a_tampered_copy(dr, tmp_path):
    repo, red, m = _repo(tmp_path, dr)
    out = tmp_path / "am81-rederive"
    assert dr.main(["--batch", str(m), "--store", str(tmp_path), "--repo", str(repo), "--out", str(out)]) == 0
    assert dr.main(["--batch", str(m), "--store", str(tmp_path), "--repo", str(repo), "--check", str(out)]) == 0
    rj = out / "run-1" / "rederive.json"
    good = rj.read_text()
    rec = json.loads(good)
    rec["corrected_results_sha256"] = "0" * 64
    rj.write_text(json.dumps(rec))
    assert dr.main(["--batch", str(m), "--store", str(tmp_path), "--repo", str(repo), "--check", str(out)]) == 1
    rj.write_text(good)
    p = out / "run-1" / "arm_a.json"
    c = json.loads(p.read_text())
    c["profile"]["device_busy_fraction"] = 0.5
    p.write_text(json.dumps(c))
    assert dr.main(["--batch", str(m), "--store", str(tmp_path), "--repo", str(repo), "--check", str(out)]) == 1


def test_mutant_correcting_in_place_refuses_and_persists_nothing(dr, tmp_path):
    """Mutant: a correction that writes into the committed receipt. The guard refuses the run and nothing is written."""
    real = dr.assemble
    calls = []

    def in_place(receipt_dir, work):
        calls.append(work)
        return real(receipt_dir, work) if len(calls) % 2 else str(Path(receipt_dir) / "tc1")

    dr.assemble = in_place
    repo, red, m = _repo(tmp_path / "one", dr)
    rows = dr.batch(str(m), str(tmp_path / "one"), str(repo))
    assert rows[0]["refused"] and "never edited" in rows[0]["refused"]
    repo, red, m = _repo(tmp_path / "two", dr)
    out = tmp_path / "am81-rederive"
    assert dr.main(["--batch", str(m), "--store", str(tmp_path / "two"), "--repo", str(repo), "--out", str(out)]) == 1
    assert not out.exists()


def test_mutant_touching_another_committed_receipt_during_the_batch_refuses_every_row(dr, tmp_path):
    """The batch guard covers every file under bench/*/receipts, not only the receipts it reads."""
    repo, red, m = _repo(tmp_path, dr)
    real = dr.reduce_md

    def touching(work, reducer):
        (repo / "bench" / "y" / "receipts" / "run-2" / "tc1" / "arm_a.json").write_text("{}")
        return real(work, reducer)

    dr.reduce_md = touching
    rows = dr.batch(str(m), str(tmp_path), str(repo))
    assert rows[0]["refused"] and "changed during the batch" in rows[0]["refused"] and "persist" not in rows[0]
