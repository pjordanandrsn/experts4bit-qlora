# P146 results — the CLI trainer with LAYER_MAJOR=1 under expert offload, RTX 4090: LICENSED

**Verdict: LICENSED, S = 1.455** (`p146-4090-6`, 2026-10-18). `python -m experts4bit_qlora.train` at its own default
recipe, on Qwen3-30B-A3B with its experts offloaded, steps 1.455× faster with `OFFLOAD_EXPERTS=1 LAYER_MAJOR=1` than
with `OFFLOAD_EXPERTS=1 FAST_TRAIN=1`, on an RTX 4090. The bar was S ≥ 1.10. The maintainer re-derived the verdict
independently and got a byte-identical `verdict.json` (sha256 `44f8621e…`).

The library-call licences (P137 2.365×, P140 2.311×, P144 2.118×) ran the lane's own loop at P137's recipe. This is the
first measured number for the CLI trainer's own recipe, which `docs/STATUS.md` called unread.

Registration: [PREREG-p146.md](PREREG-p146.md) (#1770), with spare run ids (#1773) and Amendment A1, the host-quiet
guard (#1775).

## What was run

- **Subject:** e4b `4917506a` (0.55.0, the registered library pin, with #1744's `FAST_TRAIN` in `train.py` and #1752's
  thread cap); grouped-nf4-gemm 0.46.0 `6b7c2e74`; torch 2.8.0+cu128; transformers 5.18.0; bitsandbytes 0.50.2.
- **Lane source:** e4b `2da27b68` (Amendment A1's merge).
- **The trainer, unchanged:** each arm is one fresh process running `train.py`'s own `main()` at its defaults:
  - SEQ 192, `TOKEN_BUDGET` 2048, `GRAD_ACCUM` 4;
  - r 8 / α 16, attention and expert LoRA (321,257,472 trainable parameters);
  - torch AdamW at lr 2e-4 with cosine warm-up;
  - alpaca `train[:2000]` (the pinned parquet, sha256 `06391b65…`), `SEED` 0.
- **The lane set only** `STEPS=34` (2 warm + 32 timed), `DO_GEN=0` and its two instruments: `train.py`'s step clock
  and the staging counter.
- **Arms:** A0 is `OFFLOAD_EXPERTS=1 FAST_TRAIN=1`; L1 is `OFFLOAD_EXPERTS=1 LAYER_MAJOR=1`. ABBA, A0a L1a L1b A0b.
- **Box:** one rented RTX 4090 (sm_89, 24 GB, driver 595.71.05) on a Vast.ai verified/secure host: machine 8024, AMD
  EPYC 7B13 (128 CPUs, a 30.72-CPU container quota, threads capped 64 → 31). Pinned host-to-device ceiling
  26.60 GB/s. Loads took 47–55 s per process. Cost $0.819.

## Numbers

| arm | median s/step (32 timed) | timed range | stagings, step 0 / steady | peak allocated | held-out (AFTER) |
|---|---|---|---|---|---|
| A0a (`FAST_TRAIN=1`) | 11.596 | 11.04–11.90 | 381 / 380 | 9.10 GiB | 0.9935 |
| L1a (`LAYER_MAJOR=1`) | 7.956 | 7.42–8.28 | 336 / 95 | 9.44 GiB | 0.9929 |
| L1b (`LAYER_MAJOR=1`) | 7.968 | 7.42–8.29 | 336 / 95 | 9.44 GiB | 0.9954 |
| A0b (`FAST_TRAIN=1`) | 11.595 | 11.05–11.93 | 381 / 380 | 9.10 GiB | 0.9956 |

- **The gain.** S = min(A0a/L1a, A0b/L1b) = min(1.4575, 1.4552) = **1.455**.
- **Self-pairs** (across fresh processes): 1.0000 (A0) and 1.0015 (L1), against [0.97, 1.03].
- **STRICT rungs, all clean:**
  - the adapter sha256 at the optimizer, the BEFORE held-out and step 0's loss are bit-identical in all four
    processes;
  - step 0 is cold: 381 for A0, and 336 for both L1 processes, where the drift guard runs once;
  - steady stagings are exactly 380 and 95;
  - e4b's reentrant checkpoint is on 48/48 layers in every arm;
  - the token budget stayed 2048 (no OOM back-off), and the alpaca rows were requested as `train.py` requests them;
  - 32 timed steps each;
  - L1's peak is 0.33 GiB above A0's, inside the 4.25 GiB term.
- **The host-quiet guard (Amendment A1) held:**
  - the throttled share of CFS periods was 0 and steal was 0 in every arm;
  - CPU pressure ("some", the cgroup's own `cpu.pressure`) was 0.02–0.04 % of every window, with both self-pairs
    within 0.003 points;
  - the five pinned host-to-device probes read 24.29, 26.41, 25.60, 26.41 and 26.48 GB/s, none below the first.

## Predictions, scored

- **Q1 HELD.** Every arm fit in 24 GB at `TOKEN_BUDGET` 2048 without the back-off.
- **Q2 HELD.** Same adapters, a bit-identical BEFORE held-out and step 0, and stagings exactly 380 and 95.
- **Q3 MISSED LOW (REPORTING).** S = 1.455 is under the a-priori band [1.5, 2.6]. The band is not re-drawn.
- **The host-link model:**
  - A0 = C + 4T and L1 = C + T fit at T = 1.211 s and C = 6.751 s.
  - T gives an effective rate of 32.3 GB / T = 26.7 GB/s, at the box's measured 26.6 GB/s pinned ceiling and inside
    the predicted 1.29–1.90 s.
  - C is above the predicted 1.7–6 s. The CLI's micro-batches pack up to 2048 padded tokens, and they carry more
    compute per micro-batch than the prediction allowed. At this C the copy is a smaller share of the step, which is
    what kept S under the band.
- **Q4 HELD.** L1's peak is within the 4.25 GiB term.
- **Q5 HELD (REPORTING).** The AFTER held-out losses are L1 0.9942 and A0 0.9945, |Δ| 0.0004 against 0.01.
- **Q6 HELD on this box.** The self-pairs held across fresh processes on a quiet host.
- **Q7:** LICENSED, as predicted at about 80 %.

## The earlier box, disclosed

`p146-4090-5` ($0.489) VOIDed on its cross-process self-pairs alone, while every other STRICT rung held. Its step
walls showed the host changing partway through the run. Its numbers are a VOID's and are not quoted as a result. It
led to Amendment A1, the host-quiet guard, which `p146-4090-6` ran under.

Before that box, run ids `-1` to `-4` were spent at $0.001 in total, with no box-started run: refusals before the
create, one listing gone before the create, and one pre-flight NOT_RUN. Lane total: $1.309.

## Scope of the licence

The licence covers exactly this configuration:
- the CLI trainer, `python -m experts4bit_qlora.train`, at its default recipe, comparing `OFFLOAD_EXPERTS=1
  LAYER_MAJOR=1` against `OFFLOAD_EXPERTS=1 FAST_TRAIN=1`;
- Qwen3-30B-A3B;
- the RTX 4090 class (sm_89).

No default changes.

## Receipts

[`receipts/p146-4090-6/`](receipts/p146-4090-6/SHA256SUMS) (the reading) and
[`receipts/p146-4090-5/`](receipts/p146-4090-5/SHA256SUMS) (the VOID) each hold:
- the four arm records and `train.py`'s per-step census for each arm;
- `verdict.json` and `summary.txt`;
- the logs and the launcher's teardown proof;
- the link probes (`-6`).

All are listed in `SHA256SUMS`. The alpaca parquet is identified by its sha256 and not committed.
