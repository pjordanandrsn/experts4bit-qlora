# SPEC — the 40 ms step (400+ tok/s aggregate at B=16, Qwen3-30B-A3B)

Owner directive 2026-08-24: "we're chasing 400+". At B=16 that is a
**40 ms decode step**. Baseline, frozen from the kvappend receipts
(EPYC 9655 + RTX 5090, the reference class): step 131.3 ms = 122 tok/s.

## The budget (every number a receipt)

| bucket | now (ms) | physics floor | mechanism |
|---|---|---|---|
| other-submission | 76.4 | ~5–15 | eager launch path: norms, qkv/o projections, router — 48 layers of per-op python; the hostbill cProfile attributes it |
| attention host+device | 40.2 | ~3–6 | KV bytes ≈ 0.65 MB/layer/step; the 28 ms device residue is kernel shape-inefficiency, ~12 ms host is per-seq block writes |
| GPU expert kernels | 29.9 (device) | ~3–5 | ~5.4 GB/step VRAM weight reads ÷ 1.8 TB/s ≈ 3 ms — 10× off the floor, launch-bound |
| CPU dram experts | 14.0 | ~0 exposed | independent per-layer contribution; overlappable under GPU work |
| sched/drain | 0.7 | 0.7 | already negligible |

Floors sum to ~15–25 ms (640–1000 tok/s). 400+ requires wins on at
least the first three fronts; no single bucket suffices.

## Gate ladder (each its own prereg, measurement-first where the
mechanism is unattributed; every gate A/B/A with a void gate and
receipts; bars frozen per gate)

* **T1 — launch-path kill** (other-submission 76.4 → ≤ 25):
  torch.compile / piecewise CUDA graphs over the non-MoE, non-paged-
  attention layer body. Void gate: token-identical greedy continuation
  (64 tokens); if compile's kernel fusion jitters logits, max-abs
  logit delta ≤ 1e-2 reported and the divergence step recorded — a
  DIVERGENT continuation refutes the arm regardless of speed.
* **T2 — paged-attention kernel** (device ~28 → ≤ 8): shape-tuned
  fp8_paged_decode_attention (gnf4 side). Bit-identical void gate
  (same kernel math, better launch/occupancy) unless the kernel
  changes reduction order — then tolerance-gated and disclosed.
* **T3 — GPU expert launch efficiency** (29.9 → ≤ 8): per-layer
  grouped-launch batching in the VRAM expert path (gnf4 kernels).
  Locked-tree bit-exactness rules from the phase-2 program apply.
* **T4 — tier overlap** (expose ≤ 5 of the 14 ms CPU wall): submit
  GPU work before the synchronous CPU kernel per layer. Subject to the
  G1c DRAM-budget law — overlap is NOT free bandwidth; the A/B must
  watch for the κ mixing tax the cold-engine receipts measured.

Order is by measured size × attributed confidence: T1 first (largest,
already attributed), then T2/T3 (device kernels, measurement-first
probes), T4 last (smallest, law-constrained).

## Discipline

Unchanged from the program: prereg before measurement, committed
instruments and verdict arithmetic, Bugbot-reviewed merges, one scored
box per gate, receipts beside verdicts, hard stops honored. The spec's
target (40 ms) is the CAMPAIGN goal; each gate certifies its own bar
and the ladder re-freezes the running baseline after every certified
gate.
