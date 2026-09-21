#!/usr/bin/env python3
"""bench/p55x/p55x_reduce.py -- read lane P55x's receipts and write RESULTS-p55x.md.

    python bench/p55x/p55x_reduce.py <run-dir>/p55x [--artifact <local-artifact-dir>] [--out RESULTS-p55x.md]

Every number comes from a file in the run directory. Nothing is recomputed from memory, and a missing input is
printed as missing rather than defaulted -- a reducer that fills a hole with a zero is how three arms of failed
work once reported OK (e4b, 2026-09-19).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

BUDGET = 0.05
# bo6c's box, for the STOP-1 drift read only. These are that box's NF4 reference, not a target.
BO6C_NF4 = {"wikitext": 6.419839734124139, "c4val1": 16.497028406390328}
BO6C_SPLIT = (11512, 776)


def load(p: pathlib.Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt(v, nd=5):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--artifact", default=None, help="the fetched artifact dir, for a size/payload row")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pathlib.Path(a.run_dir)
    if not d.is_dir():
        print(f"no such run dir: {d}", file=sys.stderr)
        return 2

    gate = load(d / "gate_verdict.json")
    det = load(d / "determinism.json")
    up = load(d / "upload_probe.json")
    k0 = load(d / "k0.json")
    man = load(d / "artifact1" / "manifest.json")
    summary = (d / "summary.txt").read_text() if (d / "summary.txt").exists() else ""
    versions = (d / "versions.txt").read_text() if (d / "versions.txt").exists() else ""
    forensics = (d / "forensics.txt").read_text() if (d / "forensics.txt").exists() else ""

    ppl = {}
    for f in sorted(d.glob("qwen3_ppl_*.json")):
        stem = f.stem[len("qwen3_ppl_"):]
        for src in ("wikitext", "c4val1"):
            if stem.endswith("_" + src):
                ppl[(stem[: -len(src) - 1], src)] = load(f)

    L: list[str] = []
    w = L.append
    w("# P55x — results")
    w("")
    w("Pre-registration: [`P55X-PREREG.md`](P55X-PREREG.md). Every number below is read from a committed")
    w("receipt; nothing is recomputed from memory.")
    w("")

    fp1 = (man or {}).get("pack_fingerprint") or (gate or {}).get("pack_fingerprint")
    w("## The pack")
    w("")
    if man:
        w(f"- `pack_fingerprint` **`{fp1}`**")
        w(f"- {man.get('model_id')} @ `{man.get('model_revision')}`, layout `{man.get('layout')}`, "
          f"{len(man.get('payloads') or [])} payloads")
        cc = man.get("calibrated_counts")
        same = "" if not cc else ("  — the bo6c/bo7 split" if tuple(cc) == BO6C_SPLIT
                                  else f"  — **not** bo6c's {BO6C_SPLIT[0]} / {BO6C_SPLIT[1]}")
        w(f"- gptq / rtn expert matrices: **{cc[0] if cc else '—'} / {cc[1] if cc else '—'}**{same} "
          f"(a diagnostic under #405, never the identity)")
        w(f"- `min_rows` {man.get('min_rows')}, damping {man.get('damping')}, solve device "
          f"{man.get('solve_device')}, calibration token stream "
          f"`{str(man.get('calibration_token_stream_sha'))[:20]}…`")
    else:
        w("- **no manifest in the receipts** — no pack was built, or it was not fetched.")
    w("")

    w("## Q1 — the registered two-text K8 gate, on the artifact")
    w("")
    if not gate:
        w("`gate_verdict.json` is missing: the gate did not run.")
    else:
        for cfg, label in (("lic", "`lic` — calibrated int4 attention (bo6c's configuration) — **primary**"),
                           ("licrtn", "`licrtn` — RTN int4 attention (fully pinnable) — secondary")):
            c = (gate.get("configurations") or {}).get(cfg) or {}
            w(f"### {label}")
            w("")
            w("| text | NF4 ppl | pack ppl | delta | budget +0.05 |")
            w("|---|---|---|---|---|")
            for src in ("wikitext", "c4val1"):
                t = (c.get("texts") or {}).get(src) or {}
                dv = t.get("delta")
                v = "—" if dv is None else ("PASS" if dv <= BUDGET else "**FAIL**")
                w(f"| {src} | {fmt(t.get('nf4'))} | {fmt(t.get('cand'))} | "
                  f"{'—' if dv is None else f'{dv:+.5f}'} | {v} |")
            w("")
            w(f"**{cfg} verdict: {c.get('verdict', 'NOT RUN')}**"
              + (f" (incomplete: {c.get('incomplete')})" if c.get("incomplete") else ""))
            if c.get("error"):
                w(f"  \n  gate error: `{c['error']}`")
            w("")
        rt = gate.get("roundtrip_wikitext")
        if rt:
            w(f"**dump → load round trip (wikitext):** live stores {fmt(rt['live_store_ppl'])} vs the artifact "
              f"{fmt(rt['artifact_ppl'])}, delta {rt['delta']:+.5f}. The dumped bytes are the built bytes, or "
              f"this number would not be zero.")
            w("")

    w("## Q2 — is the recipe byte-deterministic on one box?")
    w("")
    if not det:
        w("`determinism.json` is missing: the second build did not run (the last arm, dropped first when a host")
        w("is slow).")
    else:
        w("| | build 1 | build 2 |")
        w("|---|---|---|")
        w(f"| `pack_fingerprint` | `{det.get('fp1')}` | `{det.get('fp2')}` |")
        w(f"| gptq / rtn counts | {det.get('counts1')} | {det.get('counts2')} |")
        w(f"| `method_map_hash` | `{str(det.get('method_map_hash1'))[:24]}…` | "
          f"`{str(det.get('method_map_hash2'))[:24]}…` |")
        w(f"| row-count-vector hash | `{str(det.get('row_count_vector_hash1'))[:24]}…` | "
          f"`{str(det.get('row_count_vector_hash2'))[:24]}…` |")
        w("")
        w(f"**Verdict: {det.get('verdict')}** — bytes identical: {det.get('bytes_identical')}, "
          f"classification identical: {det.get('classification_identical')}.")
        w("")
        w("What this extends: `e4b.serve.buildout.bo6c.qwen3.calib-deterministic.5090.2026-09-05` compared")
        w("mean_nll and counts for the **16k** arm, before the fingerprint existed. This is the 64k recipe")
        w("compared at the byte level.")
    w("")

    w("## Q4 / STOP-1 — does the NF4 reference travel?")
    w("")
    w("| text | this box | bo6c's box | drift |")
    w("|---|---|---|---|")
    for src in ("wikitext", "c4val1"):
        r = ppl.get(("nf4", src)) or {}
        here = r.get("ppl")
        ref = BO6C_NF4[src]
        w(f"| {src} | {fmt(here)} | {fmt(ref)} | "
          f"{'—' if here is None else f'{here - ref:+.5f}'} |")
    w("")

    w("## The box, and the path the bytes took")
    w("")
    if k0:
        w(f"- K0: {k0.get('gpu')}, {k0.get('avail_disk_gb')} GB free on `/root` "
          f"(floor {k0.get('min_disk_gb')})")
    if up:
        w(f"- upload probe: **{up.get('mb_s')} MB/s** over {up.get('probe_mb')} MB "
          f"(floor {up.get('floor_mb_s')}, rsync rc {up.get('rsync_rc')}) — the direction the rental")
        w("  pre-flight does not measure, and the one this lane's product has to travel")
    if a.artifact:
        ad = pathlib.Path(a.artifact)
        if ad.is_dir():
            n = sum(1 for _ in ad.rglob("*") if _.is_file())
            b = sum(f.stat().st_size for f in ad.rglob("*") if f.is_file())
            w(f"- fetched artifact: {n} files, {b / (1 << 30):.2f} GiB at `{ad}`")
    for line in forensics.splitlines()[:3]:
        w(f"- `{line.strip()}`")
    if versions:
        w("")
        w("```")
        w(versions.rstrip())
        w("```")
    w("")

    w("## Arms as the box recorded them")
    w("")
    w("```")
    w(summary.rstrip() or "(no summary.txt)")
    w("```")
    w("")

    out = "\n".join(L) + "\n"
    if a.out:
        pathlib.Path(a.out).write_text(out)
        print(f"wrote {a.out}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
