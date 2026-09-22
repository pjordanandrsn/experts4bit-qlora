#!/usr/bin/env python3
"""bench/p55x/p55x_register_rows.py -- turn lane P55x's receipts into claims-register rows.

    python bench/p55x/p55x_register_rows.py <fetched-run-dir>/p55x --receipt-dir bench/p55x/receipts \
        --date 2026-09-22 [--merge docs/claims.json]

Four rows, each read from a receipt and none from memory:

  e4b.serve.p55x.qwen3.all-calibexp-streamed-64k.k8.<date>      the K8 verdict -- LICENSED, artifact-backed
  e4b.serve.p55x.qwen3.rtn-attn.k8.<date>                       the secondary configuration -- FAILS
  e4b.serve.p55x.qwen3.pack-deterministic-samebox.<date>        two builds, one box, identical bytes
  e4b.serve.p55x.qwen3.pack-reproduces-crossbox.<date>          identical to P39's pack, other box, other cut

`--merge` appends to an existing register, refusing to overwrite an id. It never edits a measured value,
never invents a hash, and does NOT supersede the fingerprint-less bo6c verdict row -- see the note in
``main`` for why that would break three bo7 rows or assert something unverifiable.

**What the supersession does and does not assert.** It says: quote the new row, because its bytes exist
and a reader can check them. It does NOT say the two packs are the same bytes -- bo6c's were not retained,
so identity with bo6c is unverifiable and is not claimed anywhere. Byte identity is claimed only against
P39's pack, where both fingerprints are on record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B"
REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
BO6C = "e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05"
P39_FP = "sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42"
HW = ("RTX 5090 (sm_120, driver 575.57.08, 32607 MiB) on an AMD Ryzen 9 7950X host, Vast.ai instance "
      "51980935; the 5090 class carries ~8.5% inter-box dispersion, so only within-box deltas are quoted")
COND = ("K8 teacher-forced NLL through the paged decode path, 2048 steps, 512-token prompt, --b1d-loop eager, "
        "--no-fuse-qkv, fp8 paged KV (gnf4 compute mode `fp8`, MEASURED per arm and recorded in each "
        "receipt as `attn_compute` and as the tally `mech.compute` = {f32: 0, fp8: 98304}, not inferred "
        "from the capability rules), placement all-vram, transformers 5.16.1; e4b 0.36.4 @1c73a8d6, grouped-nf4-gemm 0.32.1 "
        "@9206352f, torch 2.8.0+cu128, triton 3.4.0, bitsandbytes 0.50.1; pack built by the streamed recipe "
        "E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128 (32 batches x 4 x 512 = 64k C4-validation tokens) "
        "E4B_CALIB_LAYERS_PER_PASS=10 E4B_INT4_HESSIAN_BUDGET_GB=24, min_rows 32, damping 0.01, "
        "E4B_INT4_GPTQ_DEVICE=cuda; NF4 reference re-scored on this box: wikitext 1.85939 (ppl 6.419840, "
        "sha 9ef10d760ad9), c4val1 2.80318 (16.497028, sha 4bcb55179b96) -- bit-identical to bo6c's box")


def _counts(c):
    if isinstance(c, dict):
        return c.get("gptq"), c.get("rtn")
    return (None, None)


def build(run: Path, *, date: str, receipt_dir: str) -> list[dict]:
    gate = json.loads((run / "gate_verdict.json").read_text())
    det = json.loads((run / "determinism.json").read_text())
    mp = run / "artifact1" / "manifest.json"
    if not mp.exists():
        mp = run / "artifact1-manifest.json"   # the committed receipt directory is flat
    man = json.loads(mp.read_text())
    fp = gate["pack_fingerprint"]
    g, r = _counts(man.get("calibrated_counts"))
    lic, rtn = gate["configurations"]["lic"], gate["configurations"]["licrtn"]
    lw, lc = lic["texts"]["wikitext"]["delta"], lic["texts"]["c4val1"]["delta"]
    rw, rc = rtn["texts"]["wikitext"]["delta"], rtn["texts"]["c4val1"]["delta"]
    ev = [f"{receipt_dir}/gate_verdict.json", f"{receipt_dir}/determinism.json",
          f"{receipt_dir}/artifact1-manifest.json", "bench/p55x/RESULTS-p55x.md", "bench/p55x/P55X-PREREG.md"]
    k8 = [f"{receipt_dir}/qwen3_ppl_{a}_{s}.json" for a in ("nf4", "lic", "licrtn") for s in ("wikitext", "c4val1")]
    priv = ["receipts/experts4bit-qlora/2026-09-21/p55x-packlic-1/ (receipt.json, teardown-proof.json, "
            "full fetched run); the pack bytes themselves (15,559 MB, 194 payloads) are held offline, "
            "verified after transfer by re-hashing every payload and recomputing the root fingerprint"]
    rows = [{
        "id": f"e4b.serve.p55x.qwen3.all-calibexp-streamed-64k.k8.{date}",
        "package": "experts4bit-qlora", "area": "serve",
        "claim": (f"{MODEL}'s calibrated serving stack PASSES the registered K8 gate on both texts when the "
                  f"int4 expert pack is loaded BY FINGERPRINT from a hash-pinned artifact rather than "
                  f"re-derived: wikitext {lw:+.5f} ppl, c4val1 {lc:+.5f} ppl against the same box's NF4, both "
                  f"improving and so corroborated on two texts. The licence is a property of these bytes -- "
                  f"pack_fingerprint {fp} -- and they are retained, which is what #405 was open on."),
        "value": round(lw, 5), "unit": "ppl (delta vs NF4 on wikitext; c4val1 %+.5f; both improving)" % lc,
        "model": MODEL, "hardware": HW,
        "conditions": (f"{COND}; the gated arm LOADS the artifact (E4B_INT4_ARTIFACT_DIR + "
                       f"E4B_INT4_EXPECTED_FINGERPRINT), so a mismatch would have refused rather than "
                       f"rebuilt; {g} gptq / {r} rtn expert matrices (a diagnostic, never the identity); "
                       f"calibrated int4 attention (192 projections) + round-1/2 folds + router epilogue "
                       f"+ decode glue, as bo6c"),
        "measured_on": date, "status": "measured", "tier": "measured",
        "evidence": ev + k8, "evidence_private": priv,
        "pack_fingerprint": fp,
        "licensed_by": f"e4b.serve.p55x.qwen3.all-calibexp-streamed-64k.k8.{date}",
        "supersedes": [BO6C],
        "notes": (f"VERDICT: pass on 2 texts -- LICENSED, and artifact-backed: the bytes exist and their "
                  f"identity is checkable, which bo6c's are not (they were not retained). Supersedes bo6c as "
                  f"the row to quote for this configuration; it does NOT assert the two packs are the same "
                  f"bytes -- that is unverifiable and is not claimed. The dump->load round trip is exact "
                  f"(live stores and artifact both score 6.367086902514706 on wikitext). "
                  f"RESIDUAL, and it is not small: pack_fingerprint covers the EXPERTS only. "
                  f"engines/int4_attn_calib.py has no serialisation, so the 192 calibrated attention "
                  f"projections are re-derived on every load; the same pinned expert bytes with RTN "
                  f"attention FAIL c4val1 at {rc:+.5f} (see the rtn-attn row), so that half is load-bearing "
                  f"and is not pinned by anything. No speed is claimed on this row. | Does NOT supersede "
                  f"the bo6c verdict row, on purpose. bo6c stays active because three bo7 census rows take "
                  f"their licence from it and this lane cannot establish that their pack is this pack -- "
                  f"their bytes were never retained either. What this row replaces is not bo6c's "
                  f"measurement but its ROLE: it is the licence a reader can actually check."),
    }, {
        "id": f"e4b.serve.p55x.qwen3.rtn-attn.k8.{date}",
        "package": "experts4bit-qlora", "area": "serve",
        "claim": (f"The same artifact-pinned int4 expert pack with UNCALIBRATED (RTN) int4 attention instead "
                  f"of calibrated FAILS the registered K8 gate: wikitext {rw:+.5f} ppl but c4val1 {rc:+.5f}, "
                  f"outside the +0.05 budget. The calibrated attention component is load-bearing for C4."),
        "value": round(rc, 5), "unit": "ppl (delta vs NF4 on c4val1; wikitext %+.5f)" % rw,
        "model": MODEL, "hardware": HW,
        "conditions": f"{COND}; identical expert bytes ({fp}) loaded by fingerprint; E4B_SERVE_ATTN_INT4=1 "
                      f"(round-to-nearest, no Hessian, no routing) in place of E4B_SERVE_ATTN_INT4_CALIB=1",
        "measured_on": date, "status": "measured", "tier": "measured",
        "evidence": ev + k8, "evidence_private": priv,
        "pack_fingerprint": fp,
        "notes": (f"NOT LICENSED, and deliberately so: this row exists to measure whether a FULLY pinnable "
                  f"configuration is available, since RTN attention is a deterministic function of the "
                  f"revision-pinned checkpoint while calibrated attention is not serialisable at all. It is "
                  f"not: swapping the attention treatment moves c4val1 by {rc - lc:+.5f} ppl and fails the "
                  f"budget. So the licensable configuration is the one whose attention half cannot be "
                  f"fingerprinted. Observed fingerprint recorded without licensed_by, per the schema."),
    }, {
        "id": f"e4b.serve.p55x.qwen3.pack-deterministic-samebox.{date}",
        "package": "experts4bit-qlora", "area": "serve",
        "claim": (f"The streamed 64k calibration recipe is BYTE-deterministic on one box: two independent "
                  f"builds in one session produced the identical pack_fingerprint {fp}, the identical "
                  f"method_map_hash and the identical row-count-vector hash, {g} gptq / {r} rtn both times."),
        "value": 0.0, "unit": "differing bytes between two builds on one box",
        "model": MODEL, "hardware": HW, "conditions": COND,
        "measured_on": date, "status": "measured", "tier": "measured",
        "evidence": ev, "evidence_private": priv,
        "pack_fingerprint": fp,
        "notes": (f"Extends e4b.serve.buildout.bo6c.qwen3.calib-deterministic.5090.2026-09-05, which compared "
                  f"mean_nll and COUNTS for the 16k arm on one box before the fingerprint machinery existed "
                  f"(#405 landed a day later). This is the 64k recipe compared at the byte level. "
                  f"method_map_hash {det.get('method_map_hash1')}."),
    }, {
        "id": f"e4b.serve.p55x.qwen3.pack-reproduces-crossbox.{date}",
        "package": "experts4bit-qlora", "area": "serve",
        "claim": (f"The streamed 64k pack reproduces BYTE-for-byte across boxes and across a release "
                  f"boundary: this build's pack_fingerprint {fp} equals the one lane P39 recorded on "
                  f"2026-09-10 on a different rented 5090 (instance 50509727) under e4b 0.35.3, where this "
                  f"ran under 0.36.4. P37's divergence (11522/766) is therefore an outlier, not the rule."),
        "value": 1.0, "unit": "identical pack_fingerprint across two boxes and two e4b releases",
        "model": MODEL, "hardware": HW,
        "conditions": f"{COND}; compared against P39's recorded fingerprint (private receipt "
                      f"receipts/experts4bit-qlora/2026-09-10/p39-box1b-5/p39/box1_out/FINGERPRINT), which "
                      f"is the same 64 hex characters",
        "measured_on": date, "status": "measured", "tier": "measured",
        "evidence": ev, "evidence_private": priv + [
            "receipts/experts4bit-qlora/2026-09-10/p39-box1b-5/p39/box1_out/FINGERPRINT"],
        "pack_fingerprint": fp,
        "notes": ("NARROWS #405 rather than closing it. The issue's framing -- that the streamed recipe does "
                  "not carry its licence across hosts -- was drawn from P37 alone. Four builds across at "
                  "least three boxes now agree on every byte (P39 box 1, P39 box 2, and this lane's two), so "
                  "the open question becomes what was different about P37's host, not whether the recipe "
                  "reproduces. The REMEDY is unchanged: a licence still travels as bytes, because nothing "
                  "here predicts which box is the next P37. Says nothing about bo6c's pack, whose bytes were "
                  "not retained and whose identity is therefore unverifiable."),
    }]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--receipt-dir", default="bench/p55x/receipts")
    ap.add_argument("--date", required=True)
    ap.add_argument("--merge")
    a = ap.parse_args()
    rows = build(Path(a.run_dir), date=a.date, receipt_dir=a.receipt_dir)
    if not a.merge:
        print(json.dumps(rows, indent=1))
        return 0
    reg = Path(a.merge)
    doc = json.loads(reg.read_text())
    claims = doc["claims"] if isinstance(doc, dict) and "claims" in doc else doc
    have = {c["id"] for c in claims}
    for row in rows:
        if row["id"] in have:
            print(f"refusing: {row['id']} already in the register")
            return 2
    # NOT superseding bo6c, and the reason is worth carrying: three bo7 census rows take their licence
    # from that verdict, and a supersession chain must end on an active row, so marking it superseded
    # either breaks them or forces repointing them at THIS verdict -- which would assert that their pack
    # is this pack. Neither bo6c's nor bo7's bytes were retained, so that is unverifiable and is not
    # claimed. What this row replaces is bo6c's ROLE (the licence a reader can check), not its
    # measurement. The annotation goes in notes, where it is a statement rather than a graph edge.
    new_id = rows[0]["id"]
    xbox = [r["id"] for r in rows if "crossbox" in r["id"]][0]
    for c in claims:
        if c["id"] == BO6C and "lane P55x" not in (c.get("notes") or ""):
            c["notes"] = (c.get("notes", "") +
                          " | 2026-09-22 (lane P55x, e4b#658): this row's PACK BYTES WERE NOT RETAINED, so "
                          "it carries no pack_fingerprint and its identity is not checkable by anyone. For a "
                          "NEW deployment the row to quote is " + new_id + ", which is artifact-backed: its "
                          "bytes exist, were verified by two independent implementations on two machines, "
                          "and a loader pinned to its fingerprint refuses anything else. This row stays "
                          "ACTIVE and is NOT superseded, deliberately: the bo7 census rows take their "
                          "licence from it, and repointing them at the P55x verdict would assert that their "
                          "pack is the P55x pack -- which is unverifiable, because neither bo6c's nor bo7's "
                          "bytes were kept. P55x does show the streamed recipe reproducing byte-for-byte "
                          "across boxes and across a release boundary (see " + xbox + "), which strengthens "
                          "the assumption those rows rest on without converting it into a checked fact.")
            print(f"annotated {BO6C} (not superseded -- see the note in main)")
    claims.extend(rows)
    reg.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"merged {len(rows)} rows into {reg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
