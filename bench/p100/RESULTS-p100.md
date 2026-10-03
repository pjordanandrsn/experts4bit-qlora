# P100 — results: **REFUTED** by the registered rule. The per-expert loop runs exactly as the trace said, once per chunk per layer, and is 73 % of a 512-token chunk on this host. But it costs 0.42 s per chunk here, not 2.07 s, and a second cost that grows with chunk size makes TTFT-4096 2.25 s at chunk 2048 rather than 4.2 s. On another RTX 5090 host, SC1 box B's TTFT ran 3.5–3.8× faster

Registration: `bench/p100/PREREG-p100.md` (#917, `b676a8e`). Issue: #916. e4b at `b676a8e`; grouped-nf4-gemm at
`34da93d`.

**Verdict by `p100_reduce.py`: `REFUTED`.**
- **SCALING is INDETERMINATE.** rho = T(chunk 512) / T(chunk 2048) = 4.79 / 2.25 = 2.13, under the registered 3.0
  (the per-chunk prediction was ~4). T(chunk 2048) = 2.25 s is above 3 × TTFT-512 = 1.65 s.
- **MECHANISM is NOT_CONFIRMED, on its size threshold.** A 512-token chunk made 8,390 `dequant_int4_ref` calls, under
  the registered 9,600. Every other mechanism clause held (below).
- Under the rule, either failure is enough for REFUTED. No reason voided the run: every record was on its registered
  shape, and the prompts carried SC1 box B's digests (`a8e6ea1`, `cd70a14`), so these are the token ids SC1 read.

## What the box measured (`p100-5090-2`: RTX 5090, AMD Ryzen 9 9950X3D, 32 CPUs, 125 GB)

| arm | chunks | TTFT median (3 reps) | SC1 box B (`sc1b-5090-1`) |
|---|---:|---:|---:|
| 512 tokens, chunk 512 | 1 | **0.550 s** | 2.078 s |
| 4096 tokens, chunk 512 | 8 | **4.790 s** | 16.597 s |
| 4096 tokens, chunk 1024 | 4 | **3.065 s** | — |
| 4096 tokens, chunk 2048 | 2 | **2.255 s** | — |

- The three reps of each arm agree to within 0.016 s, and the engine's own `Request.ttft` sits beside each wall.
- The configuration is SC1's (the same registered TTFT script, stack, revision and token ids), on another host. It
  ran **3.8× faster at 512 tokens and 3.5× faster at 4096**.

**The dispatch census** (one more request per arm, after the timed ones) confirms the trace exactly:

| arm | loop calls | = 48 × chunks | rows per call | distinct experts per call (mean) | `dequant_int4_ref` calls | device-grouped calls |
|---|---:|---|---:|---:|---:|---:|
| 512 / chunk 512 | 48 | yes | 4,096 | 87.4 | 8,390 | 0 |
| 4096 / chunk 512 | 384 | yes | 4,096 | 86.9 | 66,704 | 0 |
| 4096 / chunk 1024 | 192 | yes | 8,192 | 93.7 | 35,968 | 0 |
| 4096 / chunk 2048 | 96 | yes | 16,384 | 100.5 | 19,288 | 0 |

- Every T > 1 MoE call took the host-grouped int4 loop. There were exactly two reference decodes per distinct expert
  per call, and none outside a loop call.
- The loop is paid once per chunk per layer: 384 calls for one 8-chunk request.
- **The kernel count** for one 512-token request was 117,247 device kernels and 14,969 copies. The top names are the
  reference decoder's own elementwise ops at 8,390–17,214 launches each (the int16 casts, `& 0xF`, `- 8`, `>> 4`,
  the stack, the fp32 multiply, the bf16 cast), then the per-expert bf16 GEMMs, in cuBLAS's 16×16 and 32×32 WMMA tiles.
- **The cProfile** (`step_decomp.py --cprofile-out`, `--batch 1`, int4 experts, one 512-token chunk, then 31 decode
  steps):
  - `dequant_int4_ref` has `ncalls` 8,390, equal to the census's window count;
  - the 31 decode steps' 1,488 T == 1 calls made 0 reference decodes;
  - the prefill took 0.619 s under the profiler, and the int4 branch's `_mm` took **0.454 s of it (73 %)**, of which
    `dequant_int4_ref` was 0.311 s.

## Which predictions failed, and why

1. **Distinct experts: 87 per layer per 512-token chunk, not 125–128.** Real routing concentrates. The 9,600 threshold
   (≥ 100 distinct experts) came from a near-uniform assumption. Each call's count is still exactly 2 × its distinct
   experts.
2. **rho = 2.13, not ~4.** The cost per chunk is not constant. The census requests took 0.605, 0.778 and 1.139 s per
   chunk at chunk 512, 1024 and 2048. That fits about **0.43 s fixed per chunk, plus about 0.35 ms per token in the
   chunk**. The registered fit over the three 4096-token arms gives alpha 0.42 s per chunk and beta 1.39 s.
3. **TTFT-512 = 0.55 s, not 2.1 s.** This is the host. The same configuration and token ids ran 3.5–3.8× faster here.

## Interpretation (reported; not a reading of this lane's rule)

- **The loop is the largest cost of a chunk, on this host too.** It is 73 % of a 512-token chunk in the profile. At
  4096 tokens and chunk 512, the fixed part alone is about 8 × 0.43 = 3.4 s of 4.8 s.
- **Why the fixed part may not be launch-bound here (an estimate, not measured).** The reference decoder materialises
  int16 and fp32 copies of each weight: about 110 MB of memory traffic per gate_up decode and 55 MB per down decode. At
  87 experts per layer, that is about 14 GB per layer per chunk, or about 0.4 s per chunk at the card's bandwidth,
  close to the measured 0.43 s. On SC1 box B the same work took 2.07 s per chunk, which looks host-bound (launches);
  P100 did not measure that host.
- **The part that grows with chunk size (0.35 ms per token) is not attributed by this lane.** Prefill attention was
  21 ms of device time for the 512-token chunk. The loop's per-expert bf16 GEMMs are the obvious candidate: cuBLAS
  chose 16×16 and 32×32 WMMA tiles for 47–163 rows per expert on average.
- **The documentation errors are confirmed by the counts:**
  - the branch comment's "paid once per request" (it is paid per chunk);
  - `int4_experts.py`'s "T > 1 with DEVICE_GROUPING off -- the NF4 grouped path" (it is the host-grouped int4 loop).
- **For lane SC1 (not touched):** e4b's TTFT is host-sensitive by about 3.5–3.8× in this configuration. That affects
  how SC1's TTFT row (an upstream engine at 0.010× e4b's TTFT-4096 on box B) generalises. SC1's registered runs
  measure the engine as it is, on their host; this is a note for its read, not a change to it.

## The registered consequence, and what happens next

(Lane numbers: P100's registration names the A/B "P101"; P101 was taken by the hybrid-graphs lane, #919, while P100 ran, so the A/B is **P102**.)

The registration said: **REFUTED** → "no fix lane against the loop. #916 records the reading. The census and the
scaling say where the time is instead, and the next lane is written against that."

On this data those two sentences disagree. The census and the profile put the largest share of the time IN the loop:
73 % of a chunk, and its fixed part is 3.4 s of TTFT-4096 here. The rule read REFUTED because the hypothesis's
QUANTITIES were wrong (a constant 2.07 s per chunk, ~128 distinct experts), not because the loop is elsewhere.

**Decision:** made by the executing agent under the owner's delegation of owner decisions, stated here so it can be
reversed. The next lane, P102, is written against where the time was measured to be:
- the loop's per-chunk cost;
- the part that grows with tokens.

It is an A/B of prefill routes with a fresh rule and with predictions taken from these numbers, not from #916's. It
does not assume the confirmation P100 did not give. **P100's verdict stays REFUTED.**

## Box and cost

- **`p100-5090-1`:** REFUSED before any rental, $0. The launch script did not arm the live provider (`E4B_RENT_LIVE`).
  Receipt: adertha-receipts `2006be6`.
- **`p100-5090-2`:** OK, `cost_usd.actual` **$0.163**. Started 2026-10-03T01:45:53Z; teardown 02:04:38Z (19 min, from
  `receipt.json` and `teardown-proof.json`); instance 53949139 destroyed and absent. Receipt: adertha-receipts
  `2fb24ad`.
- **Lane total $0.163** (ceiling $1.50).
- Copies of the records, censuses, cProfile listing, logs and verdict are under `bench/p100/receipts/p100-5090-2/`
  (`SHA256SUMS`).
