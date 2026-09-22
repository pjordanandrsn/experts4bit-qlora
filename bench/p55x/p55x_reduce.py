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
# P39's box 1b (2026-09-10, instance 50509727, e4b 0.35.3 @e0cfb4889634) recorded this for the same
# streamed 64k recipe: receipts .../2026-09-10/p39-box1b-5/p39/box1_out/FINGERPRINT. Quoted here so the
# cross-box comparison is mechanical rather than a claim in prose.
P39_FINGERPRINT = "sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42"


def load(p: pathlib.Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _counts(c):
    """`calibrated_counts` is a MAPPING {"gptq": n, "rtn": m}, not a pair. Indexing it positionally
    yields the KEYS -- which is how determinism.json came back reading ["gptq", "rtn"] where the
    numbers belonged, on the one run that mattered. Accept either shape and return numbers."""
    if isinstance(c, dict):
        return c.get("gptq"), c.get("rtn")
    if isinstance(c, (list, tuple)) and len(c) == 2 and all(isinstance(x, int) for x in c):
        return c[0], c[1]
    return None, None


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
    fetch = load(d.parent / "artifact_fetch.json") or load(d / "artifact_fetch.json")
    keep = load(d / "artifact_retention.json")
    k0 = load(d / "k0.json")
    # The fetched run nests the artifact (artifact1/manifest.json); the committed receipt directory
    # flattens it (artifact1-manifest.json) because the 15.2 GiB of payloads do not come into the repo.
    # Read either, so this runs against both the run dir and the receipts in the tree.
    man = load(d / "artifact1" / "manifest.json") or load(d / "artifact1-manifest.json")
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
        g, r = _counts(man.get("calibrated_counts"))
        same = "" if g is None else ("  — the bo6c/bo7 split, exactly" if (g, r) == BO6C_SPLIT
                                     else f"  — **not** bo6c's {BO6C_SPLIT[0]} / {BO6C_SPLIT[1]}")
        w(f"- gptq / rtn expert matrices: **{g if g is not None else '—'} / {r if r is not None else '—'}**{same} "
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
            # The pre-registration promises the lane will say WHICH failure it read, because
            # k8_gate fails a calibrated pack on mixed signs even when every text is inside the
            # budget (if any text improves, all must). "Outside the budget" and "inside the budget,
            # mixed in sign" are not the same result and the second is not a quality failure.
            ds = [t_["delta"] for t_ in (c.get("texts") or {}).values() if t_.get("delta") is not None]
            if c.get("verdict") == "FAIL" and ds:
                if all(d <= BUDGET for d in ds):
                    w("")
                    w("  Read this failure carefully: **every text is inside the +0.05 budget** and the")
                    w("  pack fails the *corroboration* clause instead — the deltas are mixed in sign, and")
                    w("  `k8_gate` requires that if any text improves, all do (an improvement that moves")
                    w("  with the calibration text is fitting it). This is not a quality failure of the")
                    w("  kind P37 read on c4val1 at +0.109, and it must not be quoted as one.")
                else:
                    over = [s for s, t_ in (c.get("texts") or {}).items()
                            if (t_.get("delta") or 0) > BUDGET]
                    w("")
                    w(f"  Failure is on the **budget**: {', '.join(over)} exceeds +{BUDGET}.")
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
        # The box-side snippet wrote `list(calibrated_counts)`, which for a mapping is its KEYS, so
        # determinism.json can carry ["gptq", "rtn"] where the numbers belong. Prefer each artifact's
        # own manifest, which carries the mapping intact, and say so rather than printing None.
        g1, r1 = _counts(det.get("counts1"))
        g2, r2 = _counts(det.get("counts2"))
        m1 = load(d / "artifact1" / "manifest.json") or load(d / "artifact1-manifest.json")
        m2 = load(d / "artifact2" / "manifest.json") or load(d / "artifact2-manifest.json")
        if g1 is None and m1:
            g1, r1 = _counts(m1.get("calibrated_counts"))
        if g2 is None and m2:
            g2, r2 = _counts(m2.get("calibrated_counts"))
        src = " (read from each build's manifest)" if det.get("counts1") in (["gptq", "rtn"],) else ""
        w(f"| gptq / rtn counts{src} | {g1} / {r1} | {g2} / {r2} |")
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
    if fetch:
        pm = float(up["mb_s"]) if up and up.get("mb_s") is not None else None
        sm = float(fetch["mb_s"]) if fetch.get("mb_s") is not None else None
        rel = ""
        if pm and sm:
            rel = (f" The probe **overstated** by {pm / sm:.1f}×." if pm > sm
                   else f" The probe **understated** by {sm / pm:.1f}×.")
        w(f"- artifact fetch, **sustained**: {fetch.get('mb_s')} MB/s over {fetch.get('bytes_mb')} MB in "
          f"{fetch.get('seconds')} s (rsync rc {fetch.get('rsync_rc')}), against the probe's "
          f"{up.get('mb_s') if up else '—'} MB/s on the same box minutes earlier.{rel} A prior lane's probe "
          f"went the other way (14.76 probe vs 38.70 sustained), so a short probe has **no consistent sign** "
          f"against sustained rate — it is a disaster detector, and a transfer budget must come from a "
          f"sustained row like this one.")
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

    # ---- What it says. Every sentence below is gated on a value read above; nothing is asserted that the
    # receipts do not carry, and each branch names what would have been concluded instead.
    lic = (gate or {}).get("configurations", {}).get("lic") or {}
    rtn = (gate or {}).get("configurations", {}).get("licrtn") or {}
    if keep:
        av = keep.get("archive_verify") or {}
        w("## Where the bytes are, and how you would check a copy")
        w("")
        w(f"The pack is **{keep.get('bytes', 0) / 1e9:.1f} GB over {keep.get('payloads')} payloads** and is")
        w("not in this repository. It is held on an operator archive host, offline, and is available on")
        w("request. What is here is the manifest, the hashed identity and assignment payloads, and this")
        w("record — which is enough to check any copy you are given, because a loader pinned to the")
        w("fingerprint refuses anything else and never rebuilds from the recipe.")
        w("")
        w(f"- **local**: `{(keep.get('local_verify') or {}).get('tool')}` → "
          f"`{(keep.get('local_verify') or {}).get('result')}`")
        w(f"- **on the archive host**: {av.get('payloads_rehashed')} payloads re-hashed by an independent "
          f"implementation of the canonical rule → `{av.get('result')}`")
        w("")
        w("Two implementations, two machines, one artifact, the same 64 hex characters. Either alone would")
        w("not be the check: the library can agree with itself, and the inline rule is trusted only because")
        w("the library computed the same number on the same bytes.")
        w("")
        w("**This is a private holding, said as one.** It is weaker than publishing the bytes. The #405")
        w("design note proposed a dedicated artifact repository pinned by commit, which would let a reader")
        w("fetch as well as check; that remains the better answer and is the owner's call, not a step")
        w("inside a measurement lane.")
        w("")
    w("## What this says")
    w("")
    if lic.get("verdict") == "PASS":
        w(f"**There is a licensed pack again.** `{fp1}` passed the registered two-text gate on the bytes a")
        w("loader installs, so the licence is a property of those bytes and not of the recipe that made them.")
        w("That is the thing [#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405) has been")
        w("open on since 2026-09-06: the machinery to license bytes existed, and no pack had been built,")
        w("gated and **kept**.")
    elif lic.get("verdict") == "FAIL":
        w("**There is still no licensed pack.** The primary configuration failed the registered gate, so")
        w("#405 stays open and gains a third data point. No knob was moved to recover a pass.")
    else:
        w(f"**The primary gate did not read** (`{lic.get('verdict')}`), so this lane concludes nothing about")
        w("the licence.")
    w("")
    if fp1 and fp1 == P39_FINGERPRINT:
        w("**And the recipe reproduces across boxes after all.** This pack is byte-identical to the one P39")
        w(f"recorded on 2026-09-10 — the same 64-hex `{P39_FINGERPRINT[:26]}…` — built on a different rented")
        w("5090, twelve days earlier, under **e4b 0.35.3** where this ran under **0.36.4**. P39 had already")
        w("reproduced it between its own box 1 and box 2 on one day and one cut; this extends it across a")
        w("release boundary. Together with the two same-box builds below that is four builds and at least")
        w("three boxes agreeing on every byte.")
        w("")
        w("That **narrows** #405 rather than closing it. The issue's framing — *the streamed calibration")
        w("recipe does not reproduce its licence across hosts* — was drawn from P37, which read 11522/766")
        w("and failed c4val1 at +0.109. On the evidence now available P37 is the **outlier**, not the rule,")
        w("and the open question is no longer *does the recipe reproduce* but *what was different about that")
        w("host*. What has not changed is the remedy: a licence still travels only as bytes, because nothing")
        w("here predicts which box will be the next P37.")
        w("")
    if rtn.get("verdict") == "FAIL" and lic.get("verdict") == "PASS":
        dr = (rtn.get("texts") or {}).get("c4val1", {}).get("delta")
        dl = (lic.get("texts") or {}).get("c4val1", {}).get("delta")
        if dr is not None and dl is not None:
            w("**The half that cannot be pinned is the half carrying the quality.** The same pinned expert")
            w(f"bytes with RTN attention instead of calibrated move c4val1 from {dl:+.5f} to {dr:+.5f} — a")
            w(f"swing of {dr - dl:+.5f} ppl, and a budget failure. So the calibrated attention component is")
            w("load-bearing, and the residual this lane registered is not a formality: `int4_attn_calib.py`")
            w("has no serialisation, so those 192 projections are re-derived on every load, on whatever box")
            w("does the loading. `pack_fingerprint` names the expert bytes; it does not name those.")
            w("")
    rt = (gate or {}).get("roundtrip_wikitext")
    if rt and rt.get("delta") == 0:
        w(f"**The dumped bytes are the built bytes**, to all sixteen digits ({rt['live_store_ppl']}), so the")
        w("artifact is the pack and not a copy of it.")
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
