# Routing-informed hot sets + CPU-affinity effects — 2026-07-20

Follow-up to `RESULTS-gptoss-hybrid-ab.md` (PR #27), which left two open
threads: the Gemma-4 K-curve was flat with naive hot sets, and the weak-CPU
end of the regime map was unmeasured. Driver: `bench/bench_gptoss_hybrid.py`
`HOT_MODE=informed` (calibration = one uncounted greedy pass of the same
workload with routing hooks, then pin each layer's K most-selected experts —
an **oracle upper bound** by construction: calibration tokens == served
tokens, greedy-deterministic). Pod driver: `bench/informed_hotsets_pod.sh`.

## Informed vs naive — decode scales with coverage, on every model tried

**gpt-oss-20b** (RunPod A5000, dual Xeon Gold 6342; receipts
`bench/receipts-informed-20260720/`):

| cell | coverage of routed selections | decode tok/s | vs cold floor |
|---|---|---|---|
| K=0 (all-cold) | — | 0.515 | — |
| K=4 naive (ids 0..3) | 12.8 % | 0.514 | ±0 % |
| K=4 **informed** | 42.9 % | 0.805 | **+56 %** |
| K=8 **informed** | 66.7 % | 1.135 | **+120 %** |

**Gemma-4-26B-A4B** (RunPod L40S, EPYC 9354, same-box pair; receipts
`bench/receipts-gemma-informed-20260720/`):

| cell | coverage | decode tok/s |
|---|---|---|
| K=8 naive (ids 0..7) | 5.6 % | 0.661 |
| K=8 **informed** | 49.7 % | **0.951 (+44 %)** |

**OLMoE-1B-7B** (home A2000, free validation): naive K=4 3.448 → informed
K=4 **4.099 tok/s (+19 %)**, coverage 7.3 % → 29.1 %.

Reading: PR #27's flat Gemma K-curve (K=8 == K=0) was a **naive-hot-set
artifact**, not a residency limit. At E=128/k=8, eight informed experts are
6 % of the pool yet capture **half of all routed selections** — MoE routing
is skewed enough that small informed hot sets buy large cold-traffic
reductions, and the decode gain tracks measured coverage across all three
architectures (E=32/64/128). Peak VRAM is unchanged vs naive at equal K
(hot-stack size is K by construction).

Caveat: same-workload calibration is the oracle ceiling. Production hot sets
come from routing histograms over a broader corpus (the Phase-1 kernel-lane
captures exist for exactly this); cross-workload generalization is not
measured here.

## CPU affinity dominates cold-stream decode on multi-socket hosts

Incidental finding from the weak-CPU cells, dual-socket Xeon Gold 6342 box
(96 threads):

| cell | decode tok/s |
|---|---|
| ours K=0, unrestricted | 0.515 |
| ours K=0, `taskset -c 0-3` | **3.548 (6.9×)** |
| llama.cpp `--n-cpu-moe 24`, all cores | 6.14 |
| llama.cpp `--n-cpu-moe 24`, `taskset -c 0-3 -t 4` | **19.41 (3.2×)** |

Cold-stream decode is transfer-latency-bound, and the CPU-MoE path is
memory-locality-bound: letting the scheduler migrate the process across
sockets wrecks both. **Pinning affinity is worth more than any hot-set
choice on this class of host** — it should be the first knob in any
constrained-serving runbook, and the ours-vs-llama comparison is only
meaningful with both arms pinned (pinned: llama 19.41 vs ours 3.55 — llama
still wins ~5.5× on these server cores; four pinned Xeon-Gold cores with
server memory channels are NOT a proxy for a weak edge CPU, so the ≤4-core
edge-box rows from the home-lab E-axis data remain the honest weak-CPU
evidence).

## Ops notes

- Three pods this arc (~$1.1): the informed/weak-CPU run (A5000, all seven
  cells except Gemma), one **disk-full void** (Gemma's 49 GB did not fit an
  80 GB container disk already holding gpt-oss + GGUF + llama build —
  `containerDiskInGb` raised to 120 for model-download pods), and the Gemma
  same-box pair (L40S). All torn down on evidence-complete, 404-verified.
- `pipefail` (PR #27's Bugbot fix) did its job: the disk-full Gemma cell
  reported `FAILED` instead of sailing to a fake success.
