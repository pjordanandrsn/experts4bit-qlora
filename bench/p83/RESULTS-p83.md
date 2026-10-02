# P83 — results: on one box, P70's build and P82's build read different K8 (**DIFFERENT**, −0.00049 nats), and each reading is bit-identical to its own earlier readings on other machines, so K8 is box-invariant and the software moved it (this corrects P82's read)

*(Qualified 2026-09-30, lane P84: "across machines" below means the AMD hosts it covered: Ryzen 7950X, EPYC 7C13,
7R13 and 9755. On an Intel Core i9-14900K, P82's own build calibrated a different attention pack (`45b4cc5a…`, not
`d7cfa1f4…`) and read K8 6.36276, so the recipe's calibrated attention, and the K8 with it, does not reproduce there.
See `bench/p84/RESULTS-p84.md`.)*

Read 2026-09-29 from `p83-5090-1`. Registration: `bench/p83/PREREG-p83.md` (#786, `4019b96`). Issue: #674. Verdict by
`p83_reduce.py` from the four K8 receipts and their router stamps: **DIFFERENT**. Both stated expectations held:
DIFFERENT, and stack O reading the licensed 6.36709.

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p83-prove-1` | **PROVED**, lane rc 0: both stacks installed and tripwired, both router stamps fp32 | $0.0508 |
| `p83-5090-1` | **the reading**: lane rc 0, four K8 receipts, verdict | $0.7671 |
| **lane** | | **$0.8179** |

Both teardowns are proven (`vast-destroy`, HTTP 200, instance absent). The lane stayed under its $4.00 ceiling.

## The reading (`p83-5090-1`)

One RTX 5090 (sm_120, driver 580.119.02, 575 W), AMD EPYC 9755
([`forensics.txt`](receipts/p83-5090-1/forensics.txt)). Timeline (UTC):
- Stack O: install 17:44, fetch 17:44–17:48, bake 17:48, build 17:50–18:20 (first calibration chunk 340 s), repeat
  18:20–18:25.
- Stack N: install 18:25, bake 18:26, build 18:28–18:57 (first chunk 360 s), repeat 18:57–18:59.

| reading | stack | K8 | mean NLL |
|---|---|---:|---|
| `O_build` | P70's build: e4b 0.37.4 @`c77aab6`, grouped-nf4-gemm 0.33.0 | **6.36709** | `1.8511420498367808` |
| `O_rep` | P55x's "lic" arm: O's expert pack loaded, attention calibrated live | 6.36709 | `1.8511420498367808` |
| `N_build` | P82's build: e4b 0.37.8 @`4019b96`, grouped-nf4-gemm 0.33.7 | **6.36396** | `1.8506507749113845` |
| `N_rep` | P82's K32: both of N's packs loaded by fingerprint | 6.36396 | `1.8506507749113845` |

- **Validity held.** Every reading is on window `9ef10d760ad9` at 2,048 steps. Every one is stamped fp32
  (`router_epi_cast_weights false`) with its own stack's e4b and gnf4 versions. **Each stack repeats itself bit for
  bit.**
- **DIFFERENT.** N minus O is **−0.00049 nats**, which is ln(6.36396 / 6.36709). That is well under the family's 0.0095
  K8 floor. On one box, the software between P70's build and P82's build moves the fp32 K8.
- **The packs.** Both builds produced expert pack `0c9955a9…`, the licensed one: the two manifests are identical,
  payload list included. N's attention pack is `d7cfa1f4…`, now on a third box. O's attention half cannot be dumped on
  0.37.4, so its bytes are unknown.

## K8 is bit-reproducible across machines; the software moved it

Each stack's mean NLL is **the same float** that its software gave on other machines:

| software | this box (EPYC 9755, 580.119.02) | earlier readings, bit-identical |
|---|---|---|
| P70's (0.37.4 / 0.33.0) | `1.8511420498367808` | P70: Ryzen 7950X, 575.57.08. P64: EPYC 7C13, 595.71.05 |
| P82's (0.37.8 / 0.33.7) | `1.8506507749113845` | P82: EPYC 7R13, 595.71.05 |

- On this recipe, for a fixed software stack, the fp32 K8 does not depend on the machine *(on the AMD hosts tested; not on an
  Intel host, P84)*: three machines for one stack,
  two for the other, three drivers.
- Why the licensed 6.36709 stopped reproducing is the software. P70's software gives it anywhere; P82's gives 6.36396
  anywhere.

**Correction to P82's read** (made in place in `RESULTS-p82.md`, its register row's notes, STATUS and on #674).
- **What P82 claimed.** It said that "K8 does not reproduce across boxes on this stack" and that the fp32 residual
  "cannot be attributed to software".
- **Why both are wrong for fp32.** They rested on one pair: P81's cast-router build read 6.33015 on P81's box, and P82's
  cast arm K16 read 6.31811 on P82's box, with byte-identical packs. From that one pair, P82 inferred a box effect.
- **What stays unexplained.** That cast pair itself. P83 did not run the cast. Given that the fp32 reading is
  bit-stable across machines, the first suspects are a difference between the two runs that P82's diff reading missed,
  or the cast path specifically, before the box. *(2026-09-30, lane P84: the diff reading missed one. P81 ran
  grouped-nf4-gemm 0.33.5 and P82 ran 0.33.7, and 0.33.7's fused fp8 KV append change (#413) is on K8's path; not
  measured for this pair. See `bench/p84/RESULTS-p84.md`.)*

## What moved it: not isolated

*(Answered 2026-09-30 by lane P84, KERNEL: on one box, grouped-nf4-gemm 0.33.0 → 0.33.7 is the whole −0.00049 nats; e4b 0.37.4 → 0.37.8 and the harness move nothing. The one change in that cut on K8's path, read from the code, is #413's fused fp8 KV append. See `bench/p84/RESULTS-p84.md`.)*

O and N differ in the package (e4b 0.37.4 → 0.37.8), the harness (`bench/p39/step_decomp.py` → `bench/p81`'s copy,
hook v6 → v7) and the kernels (grouped-nf4-gemm 0.33.0 → 0.33.7).
- **Two places it could be.**
  - **The attention calibration.** O's attention bytes were never dumped, so they may differ from `d7cfa1f4…`.
  - **The decode path**, given identical bytes.
- **The bisect this reading calls for** (PREREG-p83, "What follows") runs on one box. It needs one K8 per step across
  e4b 0.37.5, 0.37.6 and 0.37.7 and the harness change. From 0.37.6 on, each build can dump its attention, so the two
  places separate.
- **What is at stake.** The K8 moved by 0.00049 nats. That is a function change, not a quality regression: it is far
  inside the floor, and both readings sit below NF4's.

## What this establishes

- The calibrated int4 recipe's fp32-router wikitext K8 is **bit-reproducible across machines** *(the AMD hosts tested; P84)* for a fixed software
  stack. P70's reads 6.36709 on three; P82's reads 6.36396 on two.
- On one box, the software between those two stacks moves it by −0.00049 nats. The licensed 6.36709 is a property of
  P70's software.
- The expert pack reproduces with the licensed bytes under both stacks. N's attention pack reproduces on a third box.

**Not established.**
- Which change moves K8.
- Whether O's attention bytes differ from `d7cfa1f4…`.
- Why P81's and P82's cast readings differ.
- Anything about c4val1 or other texts.

## Receipts

- **In this repo:**
  - [`receipts/p83-5090-1/`](receipts/p83-5090-1/): the four K8 receipts and their router stamps; both stacks' install
    stamps and `versions_*.txt`; `verdict.json`, `packs.json`, `summary.txt` and `forensics.txt`; the three pack
    manifests; and `SHA256SUMS`. All 19 files are byte-identical to the store's, and re-running `p83_reduce.py` over
    them gives the run's `verdict.json` exactly.
  - [`receipts/p83-prove-1/`](receipts/p83-prove-1/): the proof's summary, versions, stamps and forensics.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-29/p83-{prove-1,5090-1}/` (commits `ed6045f`,
  `16da612`), with each run's `receipt.json` and `teardown-proof.json`, and the fetched runs with `logs/`.
