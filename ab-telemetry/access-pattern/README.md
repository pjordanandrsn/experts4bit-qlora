# Phase 0 — access-pattern precharacterization (Claims 4/5, the router axis)

Counts distinct experts gathered per layer per forward as effective tokens grow. **Counting, not
timing** — the access-pattern axis is router behavior, identical wherever the bytes live, so it is
characterized on owned hardware for $0. Scopes Session 4 from a 2D grid to a few vertical slices:
the eff_tokens where measured read-fraction crosses ~0.5 and ~0.9.

## Files

| file | what |
|---|---|
| `access_counter.py` | model-agnostic counter; hooks the router gate `Linear` (out_features == n_experts) — no e4b/offload needed |
| `test_access_counter.py` | **$0 validation** — reproduces the occupancy null with simulated routing (5/5 pass, no model) |
| `access_predictions.json` | pre-registration: the occupancy null + the reframed grid + the deviation hypothesis |
| `run_access_sweep.py` | the real-model forward-only sweep → `access_pattern.jsonl` |
| `reduce_access.py` | measured curve vs null + the 0.5 / 0.9 crossing eff_tokens |

## The finding that reframed the grid (analysis, $0, pre-run)

Occupancy null (uniform routing): `read_fraction(n) = 1 − (1−k/E)^n`, k=8, E=128.

| eff_tokens | 1 | 8 | 16 | 32 | 64 | 128 | ≥512 |
|---|---|---|---|---|---|---|---|
| read_fraction (null) | 0.06 | 0.40 | 0.64 | 0.87 | 0.98 | 1.00 | 1.00 |

Crosses **0.5 at ~11 tokens, 0.9 at ~36 tokens.** So the handoff's default grid (batch 1–64 ×
seq 1024–4096, all ≥ 1024 eff_tokens) is **saturated in every cell** under the null — it would
measure nothing about the crossover. The crossover lives at **eff_tokens ∈ [1, ~128]**, so the
grid is reframed to a token-count sweep (`run_access_sweep.py CELLS`), max one 2048-token forward.

## What the measurement tests

Whether **real** routing deviates from the null: token correlation within a sequence and learned
load-imbalance both make the real union grow *slower*, pushing the 0.9 crossing to higher
eff_tokens and *widening* the sparse-read window the SSD-tier thesis needs. If real ≈ null, the
useful regime is decode-scale — a sharp scoping result that makes D2 (a wider-expert-count second
curve) load-bearing.

## Run (real model — forward only, counting)

```bash
# GPU (4-bit, fits a 24GB card resident, ~10 min):
MODEL=Qwen/Qwen3-30B-A3B DEVICE=cuda OUT=access_pattern.jsonl python run_access_sweep.py
# CPU (bf16, ~61GB host RAM; minutes for the big cell):
MODEL=Qwen/Qwen3-30B-A3B DEVICE=cpu  OUT=access_pattern.jsonl python run_access_sweep.py
python reduce_access.py access_pattern.jsonl
```

**Compute note:** the reframed grid is small (biggest forward ≈ 2048 tokens), so this is cheap on
either substrate — but it still needs the 30B model *loaded* (≈19 GB 4-bit on GPU, ≈61 GB bf16 on
CPU). Owned-hardware options: the QNAP (128 GB RAM, CPU-feasible if free RAM allows) or the A2000
via 4-bit. A ~$0.10 spot GPU also finishes it in minutes; recorded as a decision, not taken
unilaterally, per the "$0 no-approval" gate on this phase.
