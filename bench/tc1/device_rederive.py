"""TC1 amendment 81's $0 re-derivation: a TC1 receipt read twice by the registered reducer, once as recorded and once with every profiled
cell's device time kernels-only, and the line-level difference between the two RESULTS. That difference is the amendment's scope: every
figure that moves and every verdict that flips, computed rather than recalled.

    python bench/tc1/device_rederive.py RECEIPT_DIR --reducer TC1_REDUCE_PY [--out DIR] [--json OUT.json] [--md OUT.md]
    python bench/tc1/device_rederive.py --batch MAP --store STORE --repo REPO [--out DIR] [--json OUT.json] [--md OUT.md]
    python bench/tc1/device_rederive.py --batch MAP --store STORE --repo REPO --check DIR

MAP lines are ``<run> <store-relative reducer path> <sha256 prefix>``; the receipt is found under REPO by run id, and each reducer's sha256
must start with its prefix (else the run REFUSES: a wrong reducer never reads a receipt).

The reducer is the run's OWN: the byte-exact ``tc1_reduce.py`` its box staged (in the private receipt store, beside its ``tc1/``), never
this checkout's, because the published RESULTS came from that version (bench/common rule 6). A PROOF STEP comes first. The as-recorded
pass must reproduce the receipt's committed ``RESULTS-tc1.md`` byte for byte below its title line (older reducers print the box's work
path there); a run that doesn't is reported UNREPRODUCED and is not corrected, because what can't be reproduced can't be corrected. That
also makes the reducer choice self-checking: a wrong reducer fails the proof.

Why: the arm's summarize_profile summed every CUDA-typed row into device_ms, the ``Optimizer.step#…`` user-annotation span included.
That span covers the optimizer kernels, which are rows of their own (e4b#2129). Here each cell's profile becomes kernels-only:
- ``device_ms`` = recorded minus the profile's annotation row(s), read from its full ``_profile.json``
  (bench/tc1/f15v2.kernels_only_device_ms: exactly one ``Optimizer.step#`` row, else the receipt REFUSES);
- ``device_busy_fraction`` = the new ``device_ms`` over the same wall (wall is never touched);
- ``device_by_family_fraction`` is RE-ATTRIBUTED EXACTLY: each annotation row's time comes out of ITS OWN family (the row's ``family``
  field in the profile, never re-classified) and every other family keeps its absolute ms. The new fractions are left unrounded, so the
  reducer's ``round(fraction * device_ms, 1)`` reproduces every other family's figure exactly. Every annotation row must sit in the
  ``optimizer`` family: one that does not REFUSES the receipt (a ruling for the maintainer, never a wider rule here);
- ``memcpy_fraction_of_device`` = ``memcpy_ms`` over the new ``device_ms`` (memcpy rows are kernels, so their time is unchanged);
- ``device_ms_method`` = ``kernels_only_rederived``.
THE INPUT ASSERTION: both passes are assembled from the same bytes, and the corrected pass may differ from the recorded one ONLY in those
five fields of profiled cells, each recomputed here from its formula (families in ms, to float epsilon, and their total conserved). Any
other difference REFUSES the receipt.

A COMMITTED RECEIPT IS NEVER EDITED. Both passes run on temporary copies. With ``--out``, each run's corrected cells and its record
(the RESULTS diff, and the corrected RESULTS' sha256 below its title) are written to ``OUT/<run>/``, and only after every guard holds. The guard hashes what the re-derivation reads,
before and after: the receipt and its reducer per run, and with ``--batch`` every file under ``REPO/bench/*/receipts`` too. Any change
REFUSES and nothing is written. ``--check`` re-derives each persisted run from those copies: the committed receipt with the persisted
cells laid over it, read by its reducer, must print the corrected RESULTS whose sha256 below the title line the record holds.

FAIL CLOSED: a receipt with an unparseable cell, a profiled cell whose ``_profile.json`` is missing, a cell whose device_ms disagrees with
its profile's, or an annotation row in a family the cell does not carry (or outside ``optimizer``) REFUSES as a whole. Nothing partly
corrected is ever reported as corrected.

FLIPS are read per diff hunk. Within a hunk, table rows are paired by their key (the first cell) and other lines by position; a verdict is
a bold word (``**HELD**``) or a table cell that is one upper-case word (``| VOID |``). A changed verdict on a paired line is a FLIP; a
verdict on a line that has no partner is listed as an UNPAIRED verdict line. Both are printed first and loud. A paired line whose words
changed with no verdict word moving (its numbers masked) is listed as REWORDED: a reducer branch whose bar is not printed on the line
(P215's host-bound 0.85) can move only words.

NEAR-BAR: every comparison on a changed line (``0.596 vs >= 0.5``, ``0.295 (>= 0.5 …``, ``1.043 <= 1.06``) whose corrected margin to its
bar, relative to the bar, is smaller than the run's annotation share (the largest annotation ms / recorded device ms over its cells) is
listed, flip or not. A changed prediction line with no comparison this can parse is listed as UNPARSED, to be read by hand.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD = "kernels_only_rederived"
ANNOTATION_FAMILY = "optimizer"
ALLOWED = frozenset({"device_ms", "device_busy_fraction", "device_ms_method", "device_by_family_fraction", "memcpy_fraction_of_device"})


class Refused(Exception):
    """The receipt cannot be corrected as a whole."""


def _f15v2():
    spec = importlib.util.spec_from_file_location("f15v2_rederive", os.path.join(HERE, "f15v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(text)


def assemble(receipt_dir, work):
    """The receipt as the reducer reads it, in ``work`` (named tc1/, since the reducer titles its report with the directory's name)."""
    tc1 = os.path.join(receipt_dir, "tc1")
    shutil.copytree(tc1 if os.path.isdir(tc1) else receipt_dir, work)
    pub = os.path.join(receipt_dir, "public")
    if os.path.isdir(pub):
        for f in os.listdir(pub):
            if f.endswith(".json") or f in ("box.json", "summary.txt"):
                shutil.copy(os.path.join(pub, f), os.path.join(work, f))
    return work


def tree_hashes(work):
    out = {}
    for root, _, files in os.walk(work):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, work)] = _sha(p)
    return out


def guarded_hashes(paths):
    """{path: sha256} over every file under each path (or the file itself): what the re-derivation READS and must never write."""
    out = {}
    for p in paths:
        if os.path.isdir(p):
            out.update({os.path.join(p, k): v for k, v in tree_hashes(p).items()})
        elif os.path.exists(p):
            out[p] = _sha(p)
    return out


def guard_changes(paths, before):
    now = guarded_hashes(paths)
    return sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k))


def _json_lines(text):
    """The records of a JSON Lines file (every non-empty line one JSON value), else None."""
    try:
        recs = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    except ValueError:
        return None
    return recs or None


def _cells(work):
    """[(file, cell)] for every receipt JSON that is not a kernel-table sidecar. A JSON Lines record file (amendment 79's
    release_proof_unsloth.json, which its reducer reads line by line) is not a cell, unless a record carries a profile. Anything else
    that does not parse REFUSES the receipt."""
    out = []
    for f in sorted(os.listdir(work)):
        if not f.endswith(".json") or f.endswith("_profile.json"):
            continue
        with open(os.path.join(work, f), encoding="utf-8") as fh:
            text = fh.read()
        try:
            out.append((f, json.loads(text)))
        except ValueError as e:
            recs = _json_lines(text)
            if recs is None or any(isinstance(r, dict) and "profile" in r for r in recs):
                raise Refused(f"{f} does not parse ({e}): a receipt with an unreadable cell is never reported as corrected")
    return out


def corrected_profile(pr, full, f2):
    """(the corrected profile dict, a record of what moved) for one cell's profile and its full _profile.json."""
    try:
        k_ms, a_ms, names = f2.kernels_only_device_ms(full)
    except f2.AnnotationError as e:
        raise Refused(str(e))
    n = int(pr["profiled_steps"])
    old = float(pr["device_ms"])
    if abs(float(full["summary"]["device_ms"]) - old) > 0.05:
        raise Refused(f"the cell's device_ms {old} is not its profile's {full['summary']['device_ms']}")
    new = round(k_ms * n, 1)
    ann_rows = [{"name": r["name"], "family": r.get("family"), "self_device_ms": float(r["self_device_ms"])}
                for r in full["top_device"] if f2.ANNOTATION_RE.match(r["name"])]
    for r in ann_rows:
        if r["family"] != ANNOTATION_FAMILY:
            raise Refused(f"the annotation row {r['name']!r} is in family {r['family']!r}, not {ANNOTATION_FAMILY!r}: a ruling for the "
                          "maintainer, never re-attributed here")
    out = dict(pr)
    out["device_ms"] = new
    if pr.get("wall_ms"):
        out["device_busy_fraction"] = round(new / pr["wall_ms"], 4)
    fam = pr.get("device_by_family_fraction")
    if fam:
        absolute = {f: v * old for f, v in fam.items()}
        for r in ann_rows:
            if r["family"] not in absolute:
                raise Refused(f"the annotation row {r['name']!r} is in family {r['family']!r}, which the cell does not carry")
            absolute[r["family"]] -= r["self_device_ms"]
        out["device_by_family_fraction"] = {f: v / new for f, v in absolute.items()}       # unrounded: round(frac * new, 1) is exact
    if pr.get("memcpy_fraction_of_device") is not None and pr.get("memcpy_ms") is not None:
        out["memcpy_fraction_of_device"] = round(pr["memcpy_ms"] / new, 4)
    out["device_ms_method"] = METHOD
    return out, {"recorded_ms": old, "kernels_only_ms": new, "annotation_ms": round(a_ms * n, 1),
                 "annotation_families": sorted({r["family"] for r in ann_rows}), "annotation_rows": ann_rows, "annotations": names}


def correct(work):
    """Rewrite each profiled cell kernels-only, in ``work`` (a temporary copy). Returns {cell file: record}."""
    f2 = _f15v2()
    moved = {}
    for f, cell in _cells(work):
        pr = cell.get("profile") if isinstance(cell, dict) else None
        if not isinstance(pr, dict) or pr.get("device_ms") is None:
            continue
        side = os.path.join(work, pr.get("path") or (f[:-5] + "_profile.json"))
        if not os.path.exists(side):
            raise Refused(f"{f} carries a profile summary but its {os.path.basename(side)} is not in the receipt")
        with open(side, encoding="utf-8") as fh:
            cell["profile"], rec = corrected_profile(pr, json.load(fh), f2)
        _write(os.path.join(work, f), json.dumps(cell, indent=1) + "\n")
        moved[f] = rec
    return moved


def assert_only_the_correction(a, b, moved):
    """The INPUT ASSERTION: the corrected tree differs from the recorded one only in ALLOWED fields of the moved cells' profiles."""
    ha, hb = tree_hashes(a), tree_hashes(b)
    if set(ha) != set(hb):
        raise Refused(f"the two passes do not hold the same files: {sorted(set(ha) ^ set(hb))}")
    for rel in sorted(ha):
        if ha[rel] == hb[rel]:
            continue
        if rel not in moved:
            raise Refused(f"{rel} changed, but no correction touched it")
        with open(os.path.join(a, rel), encoding="utf-8") as fa, open(os.path.join(b, rel), encoding="utf-8") as fb:
            x, y = json.load(fa), json.load(fb)
        px, py = x.pop("profile", None), y.pop("profile", None)
        if x != y:
            raise Refused(f"{rel}: a field outside the profile changed")
        changed = {k for k in set(px) | set(py) if px.get(k) != py.get(k)}
        if not changed <= ALLOWED:
            raise Refused(f"{rel}: profile fields outside the correction changed: {sorted(changed - ALLOWED)}")
        check_formulas(rel, px, py, moved[rel])
    return True


def check_formulas(rel, old, new, rec, eps=1e-9):
    """Each changed field equals its formula, recomputed here from the recorded profile and the annotation rows (independently of
    corrected_profile). Families are checked in ms, to float epsilon: every family keeps old_frac * old_dev minus its OWN annotation rows'
    ms (zero for every family but optimizer), and the total is conserved: sum(new) * new_dev == sum(old) * old_dev - annotation ms."""
    od, nd = float(old["device_ms"]), float(new["device_ms"])
    ann_f = {}
    for r in rec["annotation_rows"]:
        if r["family"] != ANNOTATION_FAMILY:
            raise Refused(f"{rel}: the annotation row {r['name']!r} is in family {r['family']!r}, not {ANNOTATION_FAMILY!r}")
        ann_f[r["family"]] = ann_f.get(r["family"], 0.0) + r["self_device_ms"]
    ann = sum(ann_f.values())
    if new.get("device_ms_method") != METHOD:
        raise Refused(f"{rel}: device_ms_method is not {METHOD!r}")
    if abs(nd - round(od - ann, 1)) > 0.11:
        raise Refused(f"{rel}: device_ms {nd} is not recorded {od} minus the annotation {ann}")
    if old.get("wall_ms") and new.get("device_busy_fraction") != round(nd / old["wall_ms"], 4):
        raise Refused(f"{rel}: device_busy_fraction is not device_ms / wall")
    if old.get("memcpy_fraction_of_device") is not None and old.get("memcpy_ms") is not None:
        if new.get("memcpy_fraction_of_device") != round(old["memcpy_ms"] / nd, 4):
            raise Refused(f"{rel}: memcpy_fraction_of_device is not memcpy_ms / device_ms")
    of, nf = old.get("device_by_family_fraction") or {}, new.get("device_by_family_fraction") or {}
    if set(of) != set(nf):
        raise Refused(f"{rel}: the family set changed")
    tol = eps * max(1.0, od)
    for f in of:
        want = of[f] * od - ann_f.get(f, 0.0)
        if abs(nf[f] * nd - want) > tol:
            raise Refused(f"{rel}: family {f!r} is {nf[f] * nd:.6f} ms, not {want:.6f} (old minus its own annotation rows)")
    if of and abs(sum(nf.values()) * nd - (sum(of.values()) * od - ann)) > tol * len(of):
        raise Refused(f"{rel}: the family total is not conserved (old minus the annotation)")


def reduce_md(work, reducer):
    md = work + ".md"
    r = subprocess.run([sys.executable, reducer, work, "--md", md], capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"the reducer exited {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
    with open(md, encoding="utf-8") as fh:
        return fh.read()


def committed_results(receipt_dir):
    for p in (os.path.join(receipt_dir, "RESULTS-tc1.md"), os.path.join(receipt_dir, "public", "RESULTS-tc1.md")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return fh.read()
    return None


def body_sha256(md):
    """sha256 of a RESULTS below its title line (the title may carry the work path)."""
    return hashlib.sha256(md.split("\n", 1)[-1].encode("utf-8")).hexdigest()


def same_below_title(got, want):
    """Byte-equal below the report's title line. Older reducers print the box's work path in the title (``(/root/tc1)``), which a re-run
    elsewhere cannot reproduce. Every other byte must match."""
    if want is None:
        return False
    g, w = got.split("\n", 1), want.split("\n", 1)
    return len(g) == len(w) == 2 and g[0].startswith("# TC1") and w[0].startswith("# TC1") and g[1] == w[1]


BOLD = re.compile(r"\*\*([A-Z][A-Z_ -]*?)\*\*")
CELL_VERDICT = re.compile(r"[A-Z][A-Z0-9_]*(?:[ -][A-Z0-9_]+)*")
_NUM = r"[-+]?\d+(?:\.\d+)?"
_OP = r"<=|>=|<|>|≤|≥"
_CI = r"(?:\[[^\]]*\]\s*)?"                     # a value's interval between it and its bar: ``0.994 [0.993, 0.995 over 4 ...] vs <= 1.01``
COMPARISON = re.compile(rf"(?P<v>{_NUM})\s*{_CI}(?:vs\.?\s*)?\(?\s*(?:premise\s+)?(?P<op>{_OP})\s*(?P<b>{_NUM})")
RANGE = re.compile(rf"(?P<v>{_NUM})\s*{_CI}(?:vs\.?\s*|\(\s*)in\s*\[\s*(?P<lo>{_NUM})\s*,\s*(?P<hi>{_NUM})\s*\]")
EACH = re.compile(rf"\(each(?: arm)?\s*(?P<op>{_OP})\s*(?P<b>{_NUM})\)")    # one bar for every ratio on the line: ``... = 1.046 ... (each <= 1.05)``
EACH_VALUE = re.compile(rf"(?:=\s*|/\s*[A-Za-z]\w*\s+)(?P<v>{_NUM})(?![\d.]|\s*(?:->|/|ms|GB))")


def _cells_of(line):
    return [c.strip() for c in line.strip().strip("|").split("|")] if line.lstrip().startswith("|") else None


def verdicts(line):
    """The verdict words on a line: bold words, and table cells that are one upper-case word (VALID, VOID, HELD, ...)."""
    out = BOLD.findall(line)
    cells = _cells_of(line)
    if cells:
        out += [c.strip("*") for c in cells if CELL_VERDICT.fullmatch(c.strip("*"))]
    return out


def _key(line):
    cells = _cells_of(line)
    return cells[0] if cells else None


def hunks(before, after):
    """[(removed lines, added lines)] per changed hunk of the two RESULTS (title line excluded)."""
    a, b = before.splitlines()[1:], after.splitlines()[1:]
    return [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if op != "equal"]


def pair(removed, added):
    """Pairs within one hunk: table rows by key, the rest by position. Returns (pairs, unpaired removed, unpaired added)."""
    pairs, ra, aa = [], list(removed), list(added)
    by_key = {}
    for ln in aa:
        k = _key(ln)
        if k is not None:
            by_key.setdefault(k, []).append(ln)
    for ln in list(ra):
        k = _key(ln)
        if k is not None and by_key.get(k):
            m = by_key[k].pop(0)
            pairs.append((ln, m))
            ra.remove(ln)
            aa.remove(m)
    while ra and aa:
        pairs.append((ra.pop(0), aa.pop(0)))
    return pairs, ra, aa


def _margin(v, b):
    return abs(float(v) - float(b)) / max(abs(float(b)), 1e-9)


def comparisons(line):
    """[(value, op, bar, relative margin)] for every comparison this can parse on a line: ``V [CI] vs OP B``, ``V (OP B``,
    ``V (premise OP B)``, ``V [CI] vs in [LO, HI]`` / ``V (in [LO, HI])`` (margin to the nearer end), and ``(each OP B)`` applied to
    every ``= V`` or ``X / Y V`` ratio on the line."""
    out = []
    for m in RANGE.finditer(line):
        out.append((m.group("v"), "in", f"[{m.group('lo')}, {m.group('hi')}]",
                    round(min(_margin(m.group("v"), m.group("lo")), _margin(m.group("v"), m.group("hi"))), 4)))
    for m in COMPARISON.finditer(line):
        out.append((m.group("v"), m.group("op"), m.group("b"), round(_margin(m.group("v"), m.group("b")), 4)))
    for e in EACH.finditer(line):
        for m in EACH_VALUE.finditer(line[:e.start()]):
            out.append((m.group("v"), e.group("op"), e.group("b"), round(_margin(m.group("v"), e.group("b")), 4)))
    return out


def near_bar(line, share):
    """[(value, op, bar, margin)] on a corrected line whose relative margin to the bar is below ``share``."""
    return [c for c in comparisons(line) if c[3] < share]


_ANY_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def reworded(x, y):
    """True when a paired line changed in more than its numbers: a reducer branch whose bar is not printed (P215's host-bound
    0.85, say) can move a line's words without a verdict word, and is read by hand."""
    return _ANY_NUM.sub("#", x) != _ANY_NUM.sub("#", y)


def _is_prediction(line):
    return bool(BOLD.search(line)) or bool(re.match(r"\|\s*P\d+", line))


def rederive(receipt_dir, reducer):
    """{receipt, reproduced, refused, cells, diff, flips, unpaired, near_bar, unparsed, share} for one receipt, plus ``persist`` (the
    corrected cells' text and the corrected RESULTS) when it was corrected. The receipt and the reducer are hashed before and after: a
    change to either REFUSES the run."""
    name = os.path.basename(os.path.normpath(receipt_dir))
    rec = {"receipt": name, "reproduced": False, "refused": None, "cells": {}, "diff": [], "flips": [], "unpaired": [],
           "near_bar": [], "unparsed": [], "reworded": [], "share": None}
    guard_paths = [receipt_dir, reducer]
    guard = guarded_hashes(guard_paths)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            a = assemble(receipt_dir, os.path.join(tmp, "a", "tc1"))
            b = assemble(receipt_dir, os.path.join(tmp, "b", "tc1"))
            if tree_hashes(a) != tree_hashes(b):
                raise Refused("the two passes were not assembled from the same bytes")
            before = reduce_md(a, reducer)
            if same_below_title(before, committed_results(receipt_dir)):
                rec["reproduced"] = True
                moved = correct(b)
                assert_only_the_correction(a, b, moved)
                after = reduce_md(b, reducer)
                files = {}
                for f in moved:
                    with open(os.path.join(b, f), encoding="utf-8") as fh:
                        files[f] = fh.read()
        changed = guard_changes(guard_paths, guard)
        if changed:
            raise Refused(f"a committed input changed during the re-derivation (a receipt is never edited): {changed[:5]}")
    except Refused as e:
        rec["refused"] = str(e)
        return rec
    if not rec["reproduced"]:
        return rec
    rec["cells"] = moved
    rec["share"] = round(max((m["annotation_ms"] / m["recorded_ms"] for m in moved.values() if m["recorded_ms"]), default=0.0), 4)
    for removed, added in hunks(before, after):
        rec["diff"] += ["-" + ln for ln in removed] + ["+" + ln for ln in added]
        pairs, ur, ua = pair(removed, added)
        for x, y in pairs:
            if verdicts(x) != verdicts(y):
                rec["flips"].append((x, y))
            elif reworded(x, y):
                rec["reworded"].append((x, y))
        for ln in ur + ua:
            if verdicts(ln):
                rec["unpaired"].append(ln)
        for _, y in pairs:
            nb = near_bar(y, rec["share"])
            if nb:
                rec["near_bar"].append((y, nb))
            elif _is_prediction(y) and not comparisons(y):
                rec["unparsed"].append(y)
    rec["corrected_results_sha256"] = body_sha256(after)
    rec["persist"] = files
    return rec


def persist(rows, out):
    """Write each corrected run to ``out/<run>/``: its corrected cells and rederive.json (the record, with the corrected RESULTS' diff
    and its sha256 below the title; the full corrected RESULTS is not copied, since --check prints it again from the cells)."""
    for r in rows:
        files = r.pop("persist", None)
        if not files:
            continue
        d = os.path.join(out, r["receipt"])
        for f, text in files.items():
            _write(os.path.join(d, f), text)
        _write(os.path.join(d, "rederive.json"), json.dumps(r, indent=1) + "\n")


def check(out_run, receipt_dir, reducer):
    """Re-derive a persisted run from its copies: the committed receipt with the persisted corrected cells laid over it must pass the
    input assertion against the receipt, and its reducer must print the corrected RESULTS whose sha256 (below the title) is recorded."""
    with open(os.path.join(out_run, "rederive.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    cells = sorted(f for f in os.listdir(out_run) if f.endswith(".json") and f != "rederive.json")
    if cells != sorted(rec["cells"]):
        raise Refused(f"{out_run}: the persisted cells {cells} are not the record's {sorted(rec['cells'])}")
    with tempfile.TemporaryDirectory() as tmp:
        a = assemble(receipt_dir, os.path.join(tmp, "a", "tc1"))
        b = assemble(receipt_dir, os.path.join(tmp, "b", "tc1"))
        for f in cells:
            shutil.copy(os.path.join(out_run, f), os.path.join(b, f))
        assert_only_the_correction(a, b, rec["cells"])
        got = reduce_md(b, reducer)
    if body_sha256(got) != rec["corrected_results_sha256"]:
        raise Refused(f"{out_run}: the reducer on the persisted copies does not print the recorded corrected RESULTS")
    return True


def _runs(map_path, store, repo):
    """[(run, reducer path, sha prefix, [committed receipt dirs])] for each MAP line."""
    out = []
    with open(map_path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith("#"):
                continue
            run, rel, prefix = ln.split()
            hits = [os.path.join(r, d) for r, ds, _ in os.walk(os.path.join(repo, "bench")) for d in ds
                    if d == run and os.path.basename(r) == "receipts"]
            out.append((run, os.path.join(store, rel), prefix, hits))
    return out


def _resolve(run, reducer, prefix, hits):
    if not os.path.exists(reducer) or not _sha(reducer).startswith(prefix):
        raise Refused(f"the reducer {reducer} is missing or not {prefix}")
    if len(hits) != 1:
        raise Refused(f"{len(hits)} committed receipt copies of {run}, not one")
    return hits[0]


def batch_guard_paths(repo, runs):
    return sorted(glob.glob(os.path.join(repo, "bench", "*", "receipts"))) + [red for _, red, _, _ in runs]


def batch(map_path, store, repo):
    """Every run of a MAP, each with its own reducer (sha-checked) and its committed receipt copy under ``repo``. Every file under
    ``repo/bench/*/receipts`` and every reducer read is hashed before and after: any change REFUSES every row (nothing is persisted)."""
    runs = _runs(map_path, store, repo)
    paths = batch_guard_paths(repo, runs)
    guard = guarded_hashes(paths)
    rows = []
    for run, reducer, prefix, hits in runs:
        try:
            receipt = _resolve(run, reducer, prefix, hits)
        except Refused as e:
            rows.append({"receipt": run, "reproduced": False, "refused": str(e)})
            continue
        r = rederive(receipt, reducer)
        r["path"] = os.path.relpath(receipt, repo).replace(os.sep, "/")
        r["reducer_sha256"] = _sha(reducer)
        rows.append(r)
    changed = guard_changes(paths, guard)
    if changed:
        for r in rows:
            r.pop("persist", None)
            r["refused"] = f"a committed input changed during the batch (a receipt is never edited): {changed[:5]}"
    return rows


def check_batch(map_path, store, repo, out):
    """[(run, None or the refusal)] for every MAP run persisted under ``out``; a corrected run with nothing persisted is a refusal."""
    res = []
    for run, reducer, prefix, hits in _runs(map_path, store, repo):
        try:
            check(os.path.join(out, run), _resolve(run, reducer, prefix, hits), reducer)
            res.append((run, None))
        except (Refused, OSError) as e:
            res.append((run, str(e)))
    return res


QUOTE_DOCS = ("README.md", "docs", "llms.txt", "llms-full.txt")
QUOTE_SKIP = ("docs/claims.json", ".sha256")
QUOTE_FIELDS = ("claim", "value", "unit", "notes", "conditions")
QUOTE_WORDS = re.compile(r"TC1|tc1|Unsloth|unsloth|busy|device|GPU time|qwen3|\bP\d{1,3}\b")
_TOKEN = re.compile(r"(?<![\d.,])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\d.,])[-+]?\d+(?:\.\d+)?(?:\s?%)?")
GENERIC_FAMILIES = frozenset({"qwen3"})          # a family token too common to place a figure: such runs need their run id


def _committed_cell(repo, rec, cell):
    for d in ("tc1", "public", ""):
        p = os.path.join(repo, rec["path"], d, cell)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    raise Refused(f"{rec['receipt']}: the committed cell {cell} is not under {rec['path']}")


def moved_figures(repo, out):
    """Every figure the correction moves, by two methods, as {run: {"families": set, "figures": [(source, where, old, new)]}}:
    ``reducer``, each number that changes on a paired line of the run's RESULTS diff; ``cell``, each cell's busy fraction (3 dp), device
    ms (total and per profiled step, 1 dp) and memcpy fraction (3 dp), committed vs the persisted copy. The second finds figures a row
    quotes straight from a cell that the reducer never prints."""
    runs = {}
    for run in sorted(os.listdir(out)):
        p = os.path.join(out, run, "rederive.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        figs = []
        minus = [ln[1:] for ln in rec["diff"] if ln.startswith("-")]
        plus = [ln[1:] for ln in rec["diff"] if ln.startswith("+")]
        for x, y in pair(minus, plus)[0]:
            a, b = _ANY_NUM.findall(x), _ANY_NUM.findall(y)
            if len(a) != len(b):
                raise Refused(f"{run}: a paired line changed its number count, so its figures cannot be matched: {x[:120]}")
            figs += [("reducer", (_key(x) or x[:40]).strip()[:40], u, v) for u, v in zip(a, b) if u != v and "." in u]
        for cell in sorted(rec["cells"]):
            old = _committed_cell(repo, rec, cell)["profile"]
            with open(os.path.join(out, run, cell), encoding="utf-8") as fh:
                new = json.load(fh)["profile"]
            n = int(old["profiled_steps"])
            for field, a, b, dp in (("device_busy_fraction", old["device_busy_fraction"], new["device_busy_fraction"], 3),
                                    ("device_ms", old["device_ms"], new["device_ms"], 1),
                                    ("device_ms per profiled step", old["device_ms"] / n, new["device_ms"] / n, 1),
                                    ("memcpy_fraction_of_device", old.get("memcpy_fraction_of_device"),
                                     new.get("memcpy_fraction_of_device"), 3)):
                if a is not None and f"{a:.{dp}f}" != f"{b:.{dp}f}":
                    figs.append(("cell", f"{cell[:-5]} {field}", f"{a:.{dp}f}", f"{b:.{dp}f}"))
        fams = {re.split(r"_(?:e4b|unsloth|axolotl|hf)_", c)[0] for c in rec["cells"]} - GENERIC_FAMILIES
        runs[run] = {"families": fams, "figures": figs}
    return runs


def _register_forms(old):
    """The ways a register row writes a figure: as printed, without trailing zeros, comma-grouped, unsigned."""
    out = {old, old.lstrip("+")}
    if "." in old:
        out.add(old.rstrip("0").rstrip("."))
    i, _, f = old.lstrip("+-").partition(".")
    if len(i) >= 4:
        out.add(f"{int(i):,}" + (f".{f}" if f else ""))
    return {x for x in out if len(x.lstrip("+-")) >= 4}


def _doc_forms(old, new):
    """The register forms plus the rounded ones (2 dp, 1 dp, a percent) that a document may print, each kept only when the rounded
    old and new differ (a quote whose rounding does not move is still right)."""
    a, b = float(old), float(new)
    out = set(_register_forms(old))
    for dp in (2, 1):
        if f"{a:.{dp}f}" != f"{b:.{dp}f}" and abs(a) < 1000:
            out.add(f"{a:.{dp}f}")
    if abs(a) < 1.5 and f"{a * 100:.0f}" != f"{b * 100:.0f}":
        out |= {f"{a * 100:.0f} %", f"{a * 100:.0f}%"}
    return {x for x in out if len(x.replace(" ", "").replace("%", "").lstrip("+-")) >= 3}


def _tokens(text):
    found = _TOKEN.findall(text)
    return set(found) | {t.lstrip("+-") for t in found}


def _placed(text, run, fams):
    return run in text or any(re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", text) for f in fams)


def quoted_figures(repo, out):
    """{"rows": {row id: [hit]}, "docs": [hit]}: every register row and document line that quotes a moved figure.
    A register hit is a figure, in a register form, in one of the row's QUOTE_FIELDS, the row placing the run (its id or a family token
    anywhere in the row). A document hit is a figure in a document form on a line with a TC1 word, placed by the run id, a family token
    or the id of a row with a register hit for that run, within three lines. Coincidences are possible and are read by hand; a figure
    no rule places is not listed, which is why both methods run."""
    runs = moved_figures(repo, out)
    with open(os.path.join(repo, "docs", "claims.json"), encoding="utf-8") as fh:
        claims = json.load(fh)["claims"]
    rows, row_runs = {}, {}
    for c in claims:
        whole = json.dumps(c)
        for run, r in runs.items():
            if not _placed(whole, run, r["families"]):
                continue
            for field in QUOTE_FIELDS:
                toks = _tokens(str(c.get(field) or ""))
                for source, where, u, v in r["figures"]:
                    for form in _register_forms(u) & toks:
                        rows.setdefault(c["id"], []).append({"field": field, "run": run, "source": source, "where": where,
                                                             "old": u, "new": v, "form": form})
                        row_runs.setdefault(c["id"], set()).add(run)
    files = subprocess.run(["git", "-C", repo, "ls-files", *QUOTE_DOCS], capture_output=True, text=True).stdout.split()
    names = {rid: [rid] + [f"`.{m.group(1)}`" for m in [re.search(r"\.\d{4}-\d{2}-\d{2}\.(.+)$", rid)] if m] for rid in row_runs}
    docs = []
    forms = {run: [(f, s, w, u, v) for s, w, u, v in r["figures"] for f in _doc_forms(u, v)] for run, r in runs.items()}
    for path in files:
        if path.startswith(QUOTE_SKIP[0]) or path.endswith(QUOTE_SKIP[1:]):
            continue
        try:
            with open(os.path.join(repo, path), encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, ln in enumerate(lines):
            if not QUOTE_WORDS.search(ln):
                continue
            toks = _tokens(ln)
            lo, hi = i, i                                   # the line's paragraph (to blank lines), at most six lines each side
            while lo > max(0, i - 6) and lines[lo - 1].strip():
                lo -= 1
            while hi + 1 < min(len(lines), i + 7) and lines[hi + 1].strip():
                hi += 1
            near = "\n".join(lines[lo:hi + 1])
            for run, fl in forms.items():
                hit = [(f, s, w, u, v) for f, s, w, u, v in fl if f in toks]
                if hit and (_placed(near, run, runs[run]["families"])
                            or any(n in near for rid, rr in row_runs.items() if run in rr for n in names[rid])):
                    for f, s, w, u, v in hit:
                        docs.append({"path": path, "line": i + 1, "run": run, "source": s, "where": w, "old": u, "new": v,
                                     "form": f, "text": ln.strip()[:200]})
    return {"rows": rows, "docs": docs}


def quotes_md(q, claims_by_id):
    out = ["# TC1 amendment 81: where the moved figures are quoted", "",
           f"{len(q['rows'])} register rows; {len({(d['path'], d['line']) for d in q['docs']})} document lines.", "",
           "| register row | status | field | figure (old -> new) | run | method | where |", "|---|---|---|---|---|---|---|"]
    for rid in sorted(q["rows"]):
        seen = set()
        for h in sorted(q["rows"][rid], key=lambda h: (h["field"], h["run"], h["old"])):
            k = (h["field"], h["old"], h["new"], h["run"], h["source"])
            if k in seen:
                continue
            seen.add(k)
            out.append(f"| `{rid}` | {claims_by_id[rid]['status']} | {h['field']} | {h['old']} -> {h['new']} (as `{h['form']}`) | "
                       f"{h['run']} | {h['source']} | {h['where']} |")
    out += ["", "| document line | figure (old -> new) | run | method | text |", "|---|---|---|---|---|"]
    seen = set()
    for d in sorted(q["docs"], key=lambda d: (d["path"], d["line"], d["old"])):
        k = (d["path"], d["line"], d["form"], d["old"], d["new"])
        if k in seen:
            continue
        seen.add(k)
        out.append(f"| {d['path']}:{d['line']} | {d['old']} -> {d['new']} (as `{d['form']}`) | {d['run']} | {d['source']} | "
                   f"{d['text'][:120].replace('|', '/')} |")
    return "\n".join(out) + "\n"


def report_md(rows):
    flips = sum(len(r.get("flips", [])) + len(r.get("unpaired", [])) for r in rows)
    fams = {}
    for r in rows:
        for m in (r.get("cells") or {}).values():
            for a in m["annotation_rows"]:
                k = f"{a['name'].split('#')[0]}# in {a['family']}"
                fams[k] = fams.get(k, 0) + 1
    out = ["# TC1 amendment 81: the device-time re-derivation", "",
           f"{len(rows)} runs; reproduced {sum(1 for r in rows if r.get('reproduced') and not r.get('refused'))}; "
           f"refused {sum(1 for r in rows if r.get('refused'))}; VERDICT FLIPS (incl. unpaired verdict lines): {flips}; "
           f"REWORDED lines: {sum(len(r.get('reworded', [])) for r in rows)}.",
           f"Annotation rows removed, by name and family, over all corrected cells: {dict(sorted(fams.items()))}.", ""]
    for r in rows:
        if r.get("refused"):
            out += [f"## {r['receipt']}: REFUSED: {r['refused']}", ""]
            continue
        if not r.get("reproduced"):
            out += [f"## {r['receipt']}: UNREPRODUCED: the as-recorded pass is not its committed RESULTS-tc1.md, so it is not corrected", ""]
            continue
        out.append(f"## {r['receipt']} ({r.get('path', '')}): REPRODUCED, {len(r['diff'])} changed lines, {len(r['flips'])} flips, "
                   f"{len(r['unpaired'])} unpaired verdict lines, annotation share {r['share']}")
        for x, y in r["flips"]:
            out += [f"- **FLIP** `{x[:200]}` -> `{y[:200]}`"]
        for ln in r["unpaired"]:
            out += [f"- **UNPAIRED VERDICT LINE** `{ln[:200]}`"]
        for c, m in sorted(r["cells"].items()):
            out.append(f"- cell {c}: device ms (all profiled steps) {m['recorded_ms']} -> {m['kernels_only_ms']} "
                       f"(annotation {m['annotation_ms']} in {', '.join(m['annotation_families'])})")
        for y, nb in r["near_bar"]:
            out.append(f"- NEAR BAR (margin below the share {r['share']}): {', '.join(f'{v} {op} {b} (margin {mg})' for v, op, b, mg in nb)}"
                       f" :: `{y[:160]}`")
        for y in r["unparsed"]:
            out.append(f"- UNPARSED prediction line, read by hand: `{y[:200]}`")
        for x, y in r.get("reworded", []):
            out += [f"- **REWORDED** (a branch may have moved) `{x[:200]}` -> `{y[:200]}`"]
        out += ["```diff", *r["diff"], "```", ""]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("receipt", nargs="?")
    ap.add_argument("--reducer", help="the run's own staged tc1_reduce.py")
    ap.add_argument("--batch", help="a MAP of runs to reducers")
    ap.add_argument("--store", help="with --batch: the receipt store's experts4bit-qlora directory")
    ap.add_argument("--repo", help="with --batch: the e4b checkout holding the committed receipt copies")
    ap.add_argument("--out", help="write each corrected run to OUT/<run>/ (never the receipt)")
    ap.add_argument("--check", help="with --batch: re-derive every run persisted under CHECK from its copies")
    ap.add_argument("--quotes", help="with --repo: list every register row and document line quoting a figure moved in QUOTES (an "
                                     "--out directory), by both methods")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    if a.quotes:
        if not a.repo:
            ap.error("--quotes needs --repo")
        q = quoted_figures(a.repo, a.quotes)
        with open(os.path.join(a.repo, "docs", "claims.json"), encoding="utf-8") as fh:
            by_id = {c["id"]: c for c in json.load(fh)["claims"]}
        md = quotes_md(q, by_id)
        if a.md:
            _write(a.md, md)
        if a.json:
            _write(a.json, json.dumps(q, indent=1, sort_keys=True) + "\n")
        print(md.split("\n", 3)[2])
        return 0
    if a.check:
        if not (a.batch and a.store and a.repo):
            ap.error("--check needs --batch, --store and --repo")
        res = check_batch(a.batch, a.store, a.repo, a.check)
        for run, why in res:
            print(f"{run}: {'OK' if why is None else 'FAIL: ' + why}")
        return 1 if any(why for _, why in res) else 0
    if a.batch:
        if not (a.store and a.repo):
            ap.error("--batch needs --store and --repo")
        rows = batch(a.batch, a.store, a.repo)
    else:
        if not (a.receipt and a.reducer):
            ap.error("give RECEIPT --reducer R, or --batch MAP --store S --repo R")
        rows = [rederive(a.receipt, a.reducer)]
    bad = [r for r in rows if r.get("refused") or not r.get("reproduced")]
    if a.out and not bad:
        persist(rows, a.out)
    for r in rows:
        r.pop("persist", None)
    md = report_md(rows)
    if a.md:
        _write(a.md, md)
    if a.json:
        _write(a.json, json.dumps(rows, indent=1) + "\n")
    print(md if not a.md else md.split("\n\n", 1)[0] + "\n" + md.split("\n", 3)[2])
    if a.out and bad:
        print(f"nothing written to {a.out}: {len(bad)} run(s) refused or unreproduced")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
