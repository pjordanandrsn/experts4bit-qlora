# P85 — results: **CONFIRMED**. grouped-nf4-gemm#413 (the fused fp8 KV append's IEEE-rounded quotient) is the whole step that moved the recipe's fp32 K8 from 6.36709 to 6.36396

Registration: `bench/p85/PREREG-p85.md` (#797, `38f076b`), with amendment 1 (#798, `cbd33ea`). Issue: #674.

**Verdict by `p85_reduce.py`: CONFIRMED**, the stated expectation. `p85-5090-4` ran on one RTX 5090 on the Ryzen
7950X host that read P55x's, P70's and P84's floats (the same GPU). Every reading used P70's harness, the fp32 router
and e4b 0.37.4:
- **The control held.** P70's build read O's known mean NLL, `1.8511420498367808`, bit for bit.
- **F reads N's float bit for bit.** This is the same stack with `E4B_FUSED_KV_APPEND=0`, so every KV append goes
  through `quantize_kv_fp8`. It reads `1.8506507749113845`, the float P82's and P84's builds on grouped-nf4-gemm 0.33.7
  read.
- **S reads O's float bit for bit.** This is grouped-nf4-gemm 0.33.6 (every release of the cut but #413's) with the
  fused append on.
- Every pair repeated itself.

So the P84 KERNEL step is exactly #413. P84 inferred it from the code; this reading measures it.

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p85-prove-1` | **PROVED**, lane rc 0 on an AMD EPYC 9655: both stacks installed with a pre-#413 `fp8_kv`; the append stamps read on by default and off under the knob | $0.0573 |
| `p85-5090-1` | **refused**, rc 16, before any install: an Intel Core Ultra 9 285K (machine 56343). No data | $0.0251 |
| `p85-5090-2` | **refused**, rc 16, before any install: an Intel Xeon E5-2698 v4 (machine 96642), after the cheapest AMD offer was taken ahead of the launch. No data | $0.0148 |
| `p85-5090-3` | **not run**: the launcher's deadline guard did not arm, because Vast answered its auth probe with HTTP 429 (rate-limited). The box (machine 37958) was destroyed before the command ran | $0.0004 |
| `p85-5090-4` | **the reading: CONFIRMED** | $0.6386 |
| **total** | | **$0.7362** of the $4.00 ceiling |

Every teardown is proven (`vast-destroy`, HTTP 200, instance absent). The ledger prices each run as the offer's rate ×
its runtime, not from an invoice.

The two refusals are why amendment 1 exists: the launcher cannot exclude a machine on a lane's rc 16. Between them,
amendment 1 raised the first-chunk watchdog to 1,500 s, so a Zen 2 host (the cheapest offer at the time) could build.
This reading's host needed 360 s. The attempt after the 429 waited three minutes for the rate window and checked the
offer list once.

## The reading (`p85-5090-4`)

- **Host.** One RTX 5090 (driver 575.57.08, 600 W) on an **AMD Ryzen 9 7950X** (machine 37958;
  [`forensics.txt`](receipts/p85-5090-4/forensics.txt)). Its GPU UUID is `7a6634c6…`, the card in P55x's, P70's and
  P84's (`p84-5090-3`) forensics.
- **Timeline (UTC, from the run's log):**

  | step | window |
  |---|---|
  | install O, fetch the model | 21:40–21:52 |
  | bake, then `O_build` | 21:52–22:24 (first calibration chunk 360 s) |
  | `O_rep` | 22:24–22:29 |
  | `F_1`, then `F_2` | 22:29–22:40 |
  | install S, bake | 22:40–22:41 |
  | `S_1`, then `S_2` | 22:41–22:51 |

  The lane exited rc 0 at 22:51:45.

| reading | stack | fused append | K8 mean NLL | ppl |
|---|---|---|---|---|
| `O_build` (control) | e4b 0.37.4 + gnf4 0.33.0 | on | `1.8511420498367808` = O's known float | 6.36709 |
| `O_rep` | same | on | `1.8511420498367808` | 6.36709 |
| `F_1` | same | **off** | `1.8506507749113845` = **N's known float** | 6.36396 |
| `F_2` | same | **off** | `1.8506507749113845` | 6.36396 |
| `S_1` | e4b 0.37.4 + gnf4 **0.33.6** | on | `1.8511420498367808` = **O's known float** | 6.36709 |
| `S_2` | same | on | `1.8511420498367808` | 6.36709 |

- **Nats.** F − O = −0.000491, F − N = 0.0, and S − O = 0.0, all exact.
- **Stamps.** Every K8 process was stamped with:
  - the fp32 router;
  - its stack's versions;
  - a pre-#413 `fp8_kv`;
  - the fused append resolved as its row says (`O_*` and `S_*` on with the knob unset; `F_*` off under `"0"`).
- **Pack.** `O_build` dumped the licensed expert pack, `0c9955a9…`. Every lic-arm reading loaded it by fingerprint.
- **The predictions, scored.**
  - The stated expectation, CONFIRMED, held exactly.
  - The failure mode the registration flagged (a single scale byte differing between Triton's `/` and torch's divide)
    did not occur on this window: F is N's float to the last bit.

## What this means for #674

The recipe's two fp32 K8 values differ by exactly one thing: how the fp8 KV cache's scored-token entries are
quantized.
- **6.36709** (grouped-nf4-gemm 0.33.0, P70's software, and 0.33.6, both read here): the fused Triton append divides by
  the scale with Triton's approximate `/`. Lane B771 measured it writing a different byte from `quantize_kv_fp8` in
  about one of 25 million values.
- **6.36396** (grouped-nf4-gemm 0.33.7, read by P82 and P84; or 0.33.0 with the fused append turned off, read here):
  the reference quantizer's bytes.

Turning the old append off on P70's own stack gives the new number exactly, and every other change in the cut leaves
the old number exactly. The e4b release and the harness moved nothing (P84).

- **The newer number is the reference-exact one.** The difference, −0.00049 nats, is a function change far inside the
  0.0095 floor.
- **The licensed row.** It reads 6.36709, the software it was read on. Whether it should name that software (the
  pre-#413 append), or be re-read on the current release, is the owner's decision. This read does not change the row.
- **The P81/P82 cast pair** (grouped-nf4-gemm 0.33.5 vs 0.33.7; 6.33015 vs 6.31811) straddles #413 too. This lane
  measured the fp32 router only, so #413 is the leading explanation for that pair, not a measured one.
- **The Intel-host attention calibration** (P84 reading 1) is a separate, still-open question. This lane ran on AMD
  hosts only.

## Receipts

- **In this repo** (every file byte-identical to the store's, checked file by file):
  - [`receipts/p85-5090-4/`](receipts/p85-5090-4/), 23 files: the six K8 receipts and their stamps,
    `stamp_{O,S}{,_append_off}.json`, `versions_{O,S}.txt`, `verdict.json`, `packs.json`, `summary.txt`,
    `forensics.txt`, the expert pack's manifest, and `SHA256SUMS`.
  - [`receipts/p85-prove-1/`](receipts/p85-prove-1/): the proof's summary, forensics, versions and stamps.
  - [`receipts/p85-5090-1/`](receipts/p85-5090-1/) and [`receipts/p85-5090-2/`](receipts/p85-5090-2/): each refusal's
    summary, forensics and `REFUSAL`.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-30/p85-{prove-1,5090-1,5090-2,5090-3,5090-4}/`
  (commits `bd9a968`, `3ed6f17`, `254463c`, `1dd6b21`, `cf5f81e`). `p85-5090-4` also holds the run's logs, the harness
  copy, `outer.log`, `receipt.json` and `teardown-proof.json`.
