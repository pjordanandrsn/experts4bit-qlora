# `p133-4090-2`: the bandwidth GEMV on an RTX 4090 (LICENSED)

One RTX 4090 (cc 8.9, 128 SMs, a 450 W power limit) on an AMD EPYC 7742 host (256 CPUs, 1,007 GiB RAM, 200 GB disk),
machine 51612, 2026-10-18T00:32:25Z to 01:55:43Z. Cost **$0.919**.
- **Code:** e4b `c8d7e629` (the box's pin, main at #1691), grouped-nf4-gemm 0.46.0 at `6b7c2e74`. The harness is
  `bench/p133/` at the registration's reviewed main (#1695 with Amendments 1, #1704, and 2, #1711).
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2; Qwen/Qwen3-30B-A3B @ `ad44e777`.
- **The run id before it:** `p133-4090-1` ($0.50) was ABORTED before its quality phase: the registered time-left check
  showed the OFF pass could not start in time, so it would have exited with no verdict. It is not a reading.

## Before the arms

- **The premise:** grouped-nf4-gemm's K33 kernel contract gave 33 passed with the two registered sm_90+ skips, and the
  served route 18 passed.
- **The fit:** the default server built at `max_seqs` 16, buckets 1-16, graphs on, so the control is W16.
- **Engagement:** each B0 build dispatched 480 calls to the scalar GEMV, and each B1 build 480 to `bw_prmt32`, none
  elsewhere.

## The arms

| process | `GNF4_GEMV_BW` | W1 tok/s | W16 tok/s | W1 ms a step |
|---|---|---|---|---|
| B0a | unset (auto: the scalar GEMV at 128 SMs) | 117.98 | 573.39 | 8.4759 |
| B1a | 1 | 178.68 | 573.28 | 5.5967 |
| B1b | 1 | 178.99 | 571.07 | 5.5870 |
| B0b | unset | 118.11 | 570.99 | 8.4664 |

## The verdict

**LICENSED** (`verdict.json`): g1 = **1.5145** (the smaller of B1a/B0a 1.5145 and B1b/B0b 1.5155) >= 1.03, and
gW = **0.9998** at W16 (0.9998, 1.0001) >= 0.99.
- **Self-pairs:** W1 1.0011 (B0) and 1.0017 (B1), W16 0.9958 and 0.9961, inside the registered bands.
- **Tokens:** B1 decoded B0's tokens on every row, at W1 and W16 and both lengths.
- **Quality** (P110's teacher-forced instrument, 24 windows a text, T == 1):
  - wikitext: bias +0.00325 (bar 0.01387), spread 0.01173 (bar 0.02640), K8 Δppl +0.0283 (gated, budget 0.05);
  - c4val1: bias -0.00208 (bar 0.01053), spread 0.00874 (bar 0.03229), K8 -0.0354 (reported).
  - The scale mutant failed the bar on both texts, as it must.

## Files

- **Here:** the four arm records, `fit.json`, `prompts.json`, `quality_off.json`, `quality_on.json`, `verdict.json`,
  `summary.txt`, `versions.txt`, and the bake, clone, pip and premise logs. `SHA256SUMS` covers each file here.
- **Only in the private receipt store,** byte-exact, listed by sha256 in `STORE-ONLY.sha256`:
  - `forensics.txt` and the runner's console log (the GPU's UUID);
  - the premise suites' junit reports (the container's host name);
  - the logs that carry carriage returns from progress output;
  - the instance record and teardown proof (the provider's instance).
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.

The verdict re-derives byte for byte from these records with the registered reducer, run beside its two staged
siblings (both in the repository at their staged bytes), under Python 3.10 (as checked). Under 3.12 or later
only the last digits of a few floats move, never a figure or the verdict:

    mkdir -p /tmp/rd && cp bench/p133/p133_reduce.py bench/p115/p115_reduce.py bench/p110/p110_box.py /tmp/rd/
    python /tmp/rd/p133_reduce.py --dir bench/p133/receipts/p133-4090-2 --out verdict.json \
        --e4b-sha c8d7e629e4c8431fc86970cf7195d8e67f72e22a
