# P102 — results: **`DEFAULT=k19`**. On an RTX 5090, Qwen3-30B-A3B at `max_seqs` 1: TTFT-4096 7.21 s → 1.37 s (5.25×) and TTFT-512 0.85 s → 0.11 s, with the prefill-shaped NLL within +0.011 / +0.007 ppl of the loop on 12 fresh windows. `batched` is bit-identical but only 1.24× on this host; `mtile` passes at 4.36×

Registration: `bench/p102/PREREG-p102.md` (#922 `04616bf`; A1 #923 `0a36f54`; A2 #930 `bae71e4`). Issue: #916.
Code under test: `E4B_INT4_PREFILL` (#921). e4b at `bae71e4`; grouped-nf4-gemm at `34da93d`.

**Verdict by `p102_reduce.py`: `DEFAULT=k19`**, on the registered redraw `p102-5090-6`. No VOID reason fired. Every
route ran as registered in its 512-token dispatch census:

| route | grouped calls | loop calls | device-grouped | reference decodes | K19 launches | M-tile launches | device kernels |
|---|---:|---:|---:|---:|---:|---:|---:|
| loop | 48 | 48 | 0 | 8,390 | 0 | 0 | 132,216 |
| batched | 48 | 48 | 0 | 0 | 0 | 0 | 38,919 |
| k19 | 48 | 0 | 48 | 0 | 96 | 0 | 15,383 |
| mtile | 48 | 0 | 48 | 0 | 0 | 96 | 15,479 |

## Speed (`p102-5090-6`: RTX 5090, AMD Ryzen 9 7900, 24 CPUs)

One engine, the route switched between requests. Medians of three interleaved rounds, `max_new_tokens=1`, chunk 512:

| route | TTFT-512 | TTFT-4096 | vs loop at 4096 |
|---|---:|---:|---:|
| loop | 0.851 s | 7.207 s | 1.00× |
| batched | 0.672 s | 5.820 s | 1.24× |
| **k19** | **0.113 s** | **1.373 s** | **5.25×** |
| mtile | 0.147 s | 1.652 s | 4.36× |

## Quality (the calibrated K8 rule on a prefill-shaped NLL)

`step_decomp --ppl-oracle eager`, on one model, over 12 fresh windows (P96's layout), with every MoE call at T ≥ 256:

| route | c4val1 (W = 8): mean Δppl (SD), windows over 0.05 | wikitext (W = 4): mean Δppl (SD), windows over 0.05 | rule |
|---|---|---|---|
| batched | 0 (bit-identical NLL on every window) | 0 (bit-identical) | **PASS** (and every first token identical) |
| k19 | +0.0115 (0.043), 1 of 8 | +0.0070 (0.040), 0 of 4 | **PASS** |
| mtile | +0.0080 (0.022), 0 of 8 | +0.0190 (0.029), 0 of 4 | **PASS** |

- Window-mean ppl under loop: 13.979 (c4val1) and 8.951 (wikitext). `k8_gate.verdict` (uncalibrated regime, budget
  0.05) passed both device routes on both texts.
- Every route's first token equalled loop's, in every draw at both lengths.

**The default rule.**
- All three routes passed, and k19 had the lowest TTFT-4096.
- mtile was 20 % slower, so it was not within the 10 % tie band, and batched was 4.2× slower.
- k19's 1.373 s is under half of loop's 7.207 s, so the verdict is `DEFAULT=k19`.

## The first full draw, VOID (reported; not the reading)

`p102-5090-5` (Intel Xeon E5-2698 v4, 40 CPUs, $0.3466) was voided by the reducer's own stale threshold. It required
≥ 9,600 reference decodes from `loop`, a uniform-routing guess that P100 had already measured as wrong; the draw
recorded 8,390. A2 (#930) made the check structural and the lane was redrawn. Its numbers, descriptively:

| route | TTFT-512 | TTFT-4096 |
|---|---:|---:|
| loop | 3.813 s | 30.111 s |
| batched | 1.317 s | 10.620 s |
| k19 | 0.461 s | 3.445 s |
| mtile | 0.473 s | 3.590 s |

- **The NLL deltas reproduced to every digit between the two draws**, on two different hosts and cards: k19 +0.01146 /
  +0.00701 and mtile +0.00797 / +0.01898. This is the same reproducibility K8 shows across boxes.
- **The host matters most to the loop.** The same configuration's TTFT-4096 under loop:
  - 30.1 s on this Xeon;
  - 16.6 s on SC1 box B's host;
  - 7.2 s on the Ryzen 9 7900 here;
  - 4.8 s on P100's Ryzen 9 9950X3D (P100's TTFT-512 at chunk 512 was 0.55 s).

  `batched` helps only on slow hosts (2.84× on the Xeon, 1.24× here). The loop there is bound by host launches, and
  `batched` cuts them about 3.4-fold.
- **k19's lead over loop holds on both hosts** (8.7× on the Xeon, 5.25× here). k19 itself is far less host-sensitive
  (2.5× between the hosts, against loop's 4.2×).

## Where a k19 prefill spends its time (descriptive; one 4096-token request under `torch.profiler`)

On `p102-5090-6`, 924 ms of device time sits in a 1.51 s wall:

| item | device ms | launches |
|---|---:|---:|
| K19 (the experts, both projections, 8 chunks × 48 layers) | 143 | 768 |
| fp32 SIMT SGEMMs (`cutlass_80_simt_sgemm_128x32`, `tn` + `nn`) | 231 | 768 |
| fp32 elementwise around them (`where`, `add`, `isneginf`) | 182 | 1,152 |
| softmax (fp32) | ≥ 31 | 96+ |
| bf16 copies | 52 | 2,712 |
| device-to-device memcpy | 40 | 50,501 |

- The SGEMM pair runs once per layer per chunk, with masking and softmax around it. That is the shape of prefill
  attention computed in fp32 without tensor cores, and with the loop gone it is the largest item (~45 % of device time).
- **This is the next lead for TTFT.** It is not this lane's question, and not attributed beyond the kernel names.

## Box and cost

| run | outcome | cost (`cost_usd.actual`) | receipt |
|---|---|---:|---|
| `p102-5090-1` | HARNESS_ERROR rc 25: the premise could not run (no pytest; my defect, fixed by A1) | $0.0303 | `139544a` |
| `p102-5090-2` | NOT_RUN: instance stuck `loading` 600 s (provider) | $0.0907 | `4046f11` |
| `p102-5090-3` | NOT_RUN: bandwidth 3.1 MB/s (provider) | $0.0344 | `9e75424` |
| `p102-5090-4` | NOT_RUN: bandwidth 10.1 MB/s (provider) | $0.0223 | `21bd374` |
| `p102-5090-5` | OK, **VOID** (my stale threshold, fixed by A2) | $0.3466 | `796486c` |
| `p102-5090-6` | OK, **`DEFAULT=k19`**, 2026-10-03T04:30:45Z → 04:54:07Z | $0.2085 | `695579b` |

- **Lane total: $0.7328** (ceiling $1.50).
- Copies of both full draws' records, logs and verdicts are under `bench/p102/receipts/` (`SHA256SUMS` in each).

**The registered consequence:** a PR makes `k19` `E4B_INT4_PREFILL`'s default, with every route kept selectable,
`loop` included. It carries a CHANGELOG entry and a `docs/SERVING.md` note citing this read, and the flip ships in the
next release.
