# P106 — results: **NEUTRAL**. Against transformers' torch path, flash-linear-attention and causal-conv1d cost Qwen3.6-35B-A3B nothing measurable in nats: KL 5.7e-3 on prefill and 4.9e-3 on decode, d_nll within ±0.001. They cut prefill TTFT by only 10–11 % (1.10–1.12×)

Registration: `bench/p106/PREREG-p106.md` (#947, `6e63088`). Issue: #944. Code under test: e4b at `6e63088`,
grouped-nf4-gemm at `34da93d`, `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`, torch 2.8.0+cu128 held.

**Verdict by `p106_reduce.py`: `NEUTRAL`.** On both phases (prompt positions and decode steps):
- mean KL(torch ‖ kernels) ≤ 0.05;
- argmax agreement ≥ 0.85;
- d_nll ≤ +0.01.

None of the VOID conditions fired:
- the toggle check passed on the card;
- the engagement was as registered;
- the loaded commit is `995ad96`;
- the switch was not inert (no compared position was bit-identical);
- the null pair was bit-identical;
- the mutant failed the bar.

## The reading (`p106-5090-1`)

One RTX 5090 (sm_120, driver 595.91.07) on an Intel Core Ultra 9 285K, Vast machine 151350 (pre-flight 94.7 MB/s).
$0.2559. The lane started at 09:10:45Z and reported `TP_DONE` at 09:33:25Z. The box was destroyed and proven absent
at 2026-10-03T09:34:04Z.
- **Premise fc:** 9 passed.
- **Toggle check on this card:** the torch path read 0 differing logits against the kernel-free process.
- **Bake:** OK.
- **The box:** build 25 s, quality 98 s, TTFT 32 s, peak GPU memory 23.25 GB.

**Quality, the two paths in lockstep on fp32 log-probs** (8 wikitext windows, each 2,048 prompt tokens in 512-token
chunks plus 64 teacher-forced decode steps):

| | prompt positions (chunk rule) | decode steps (recurrent rule) |
|---|---:|---:|
| positions | 16,384 | 512 |
| mean KL(torch ‖ kernels), nats | **5.69e-3** | **4.86e-3** |
| median / p90 / p99 KL | 2.05e-3 / 1.07e-2 / 5.37e-2 | 2.39e-3 / 1.14e-2 / 3.62e-2 |
| max KL | 2.80 | 0.062 |
| argmax agreement | **0.969** | **0.973** |
| mean NLL, torch → kernels | 1.80880 → 1.80883 | 1.70058 → 1.69974 |
| mean d_nll (kernels − torch) | **+3.3e-5** | **−8.4e-4** |
| per-window mean KL | 3.9e-3 – 8.3e-3 | 3.1e-3 – 8.0e-3 |
| per-window d_nll | −0.0071 – +0.0030 | −0.0153 – +0.0223 |

- **The kernels are not worse in nats.** Over the 16,384 prompt positions, d_nll is +3.3e-5 with a per-position
  standard error of 9.6e-4. Over the decode steps it is −8.4e-4. The window means scatter on both sides of zero.
- **The KL is the size of the fp8 KV's own effect.** P97 measured the paged path's fp8 KV against transformers' bf16
  cache on this model at 4.43e-3 nats, over 512 + 256 windows. The kernels' 5.7e-3 / 4.9e-3 is 1.1–1.3× that.
- **The tail is argmax near-ties.** 70 prompt positions (0.4 %) exceed 0.1 nats, 27 of them where the argmax flips,
  and 3 exceed 1 nat. No decode step exceeds 0.1. The 506 prompt-position flips (3.1 %) are the near-ties that let
  free-running greedy decode diverge, as P105 saw (41–57 % positional agreement).
- **The null pair** (the torch path against itself, windows 0–1, all 4,096 prompt positions and 128 decode steps)
  was bit-identical: the engine is deterministic, and the instrument's floor is zero.
- **The mutant** (fla's delta rules with `use_qk_l2norm_in_kernel=False`, window 0) diverged to NaN logits on every
  position: KL and d_nll are NaN, and argmax agreement is 0.0 on both phases. The bar rejects it on agreement, which
  needs no NaN comparison.

**TTFT, one request at a time, max_new_tokens 1** (median of 5 rounds, alternated):

| prompt | torch path | kernels | torch ÷ kernels | saved |
|---:|---:|---:|---:|---:|
| 512 | 169.8 ms | 151.1 ms | **1.124×** | 18.7 ms |
| 2,048 | 688.3 ms | 622.5 ms | **1.106×** | 65.8 ms |
| 4,096 | 1,427.4 ms | 1,296.5 ms | **1.101×** | 130.9 ms |

- **The saving is about 32 µs per prompt token** at every length, roughly 10 % of prefill. With the kernels,
  prefill runs about 3.2–3.4 k tokens/s at one request.
- **The Gated DeltaNet layers are a small part of prefill on this stack.** The MoE experts and the rest dominate. The
  torch chunk rule's large per-call deficit on the A2000 (4–10×) does not carry to the whole forward on a 5090.
- The first timed torch-path round at 2,048 tokens read 1,327.7 ms against 680–695 ms for the other four (a
  first-shape cost the untimed warm-up did not absorb). The median excludes it.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| NEUTRAL | **yes** |
| mean KL between 1e-3 and 2e-2 nats, both phases | **yes** (5.69e-3, 4.86e-3) |
| argmax agreement ≥ 0.95, both phases | **yes** (0.969, 0.973) |
| \|mean d_nll\| ≤ 0.005 nats | **yes** (3.3e-5, 8.4e-4) |
| the null pair bit-identical | **yes** |
| the mutant's mean KL above 1 nat, both phases | **no, as stated**: its logits went NaN (agreement 0.0) |
| TTFT at 4,096 tokens between 1.3× and 4× | **no: 1.101×** |
| TTFT at 512 tokens between 1.1× and 3× | **yes** (1.124×) |
| the TTFT ratio grows with length | **no**: 1.124×, 1.106×, 1.101× (flat, slightly falling) |
| peak GPU memory at or below 28 GB | **yes** (23.25 GB) |

**The TTFT prediction made P105's mistake again.** P105's W16 decode band also came from the kernels' per-call gain on
the A2000, and it also overestimated. A per-call kernel ratio says how much faster the Gated DeltaNet layers get, not
what share of the step they are. On this stack that share is about 10 % of prefill and 12–13 % of a graph decode step.

## The registered consequence (NEUTRAL)

- `docs/SERVING.md`'s hybrid section replaces "quality against the torch path in nats is not measured" with the
  measured KL, agreement and d_nll, and adds the TTFT ratios. **The recommendation stands.**
- Register row: `e4b.serve.p106.qwen36-gdn-kernel-quality.5090.2026-10-03`.
- #944 closes.

## Box and cost

| run | status | machine | cost | note |
|---|---|---:|---:|---|
| `p106-prove-1` | OK, PROVED | 40549 | $0.0913 | premise 9; toggle check exact on sm_120 |
| `p106-5090-1` | **OK, NEUTRAL** | 151350 | $0.2559 | the reading |

- **The lane: $0.3472.**
- **The Gated DeltaNet kernel question across its four lanes: $0.9609** (P103 $0.1330, P104 $0.0527, P105 $0.4280,
  P106 $0.3472).

Receipts (`receipts/p106-5090-1/`), with `SHA256SUMS` over every file copied:
- `box.json` (every compared position's KL, NLLs, agreement and identity), `verdict.json`, `kernels_fc.json`,
  `toggle_check.json`, `summary.txt`, `forensics.txt`, `versions.txt`;
- the install, engagement, premise, toggle, fetch, bake and box logs;
- `work/bake.json` and the teardown proof.

The toggle check's reference tensor (`work/toggle_ref.pt`) stayed on the box. The launcher's receipts and ledger rows
are in the receipt store (adertha-receipts `19d98d7`, `66c69d5`).
