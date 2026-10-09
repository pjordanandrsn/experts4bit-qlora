# The training estimate's loss branch under the chunked LM loss (RTX A2000, granite-3.1-3b-a800m)

**Question.** `estimate_qlora_footprint` priced the loss branch of its activation item as whole stock logits
(`T × V × LOGITS_LOSS_BYTES`) even where `enable_fast_train` runs the chunked LM loss. How far off is that, and is
pricing the branch with `chunked_loss_bytes` where the run chunks right?

**Runs.** These are memory only, with no timing. granite-3.1-3b-a800m-instruct, `grouped_nf4`, resident, r8 bf16
adapters, AdamW, `E4B_ABSMAX_DQ=0`, loggetta 0.4.0 + experts4bit-qlora 0.51.0 + grouped-nf4-gemm 0.44.0, torch
2.11.0+cu128, transformers 5.19.0 (`freeze.txt`), one RTX A2000 12GB, about 10 minutes of GPU in all.

| run | tokens per micro-batch | `E4B_CHUNKED_LM_LOSS` | loss | allocated peak, MiB | receipt |
|---|---|---|---|---|---|
| G2 | 1024 (512 × 2) | unset (auto: 0.19 GiB of stock fp32 logits, under the gate) | stock | 3131.5 | loggetta `evidence/2026-10-09-a2000-granite-train-residual/runs/G2/` |
| P1 | 1024 (512 × 2) | `1` (every training forward chunks, 512-token chunks) | chunked | 2984.2 | `runs/P1/` |
| S6 | 6144 (3072 × 2) | `0` | stock | 6522.7 | `runs/S6/` |
| A6 | 6144 (3072 × 2) | `auto` (1.13 GiB of stock fp32 logits ≥ the 1 GiB gate) | chunked | 4105.3 | `runs/A6/` |

P1 came from `run.sh`, and S6 and A6 from `ab.sh`. The receipts do not record `E4B_CHUNKED_LM_LOSS`, so each run's
setting is in the script that ran it.

## The estimate against them

`eval_estimate.py` prices each point through loggetta's planner (main, which adds its `grouped_nf4` backward line).
The before column uses experts4bit-qlora main (`d14bcb10`) and the after column this change. MiB; estimate minus
allocated peak.

| run | before | after |
|---|---|---|
| G2, T = 1024, stock | +72.5 | +72.5 (unchanged) |
| S6, T = 6144, stock | +41.5 | +41.5 (unchanged) |
| A6, T = 6144, auto, chunked | **+2458.9** | **+58.7** |
| P1, T = 1024, forced chunks | +219.8 | **−36.2** |

- **The default regime (A6).** The old estimate charged 3.38 GiB of whole logits that the run never materializes, so a
  long-sequence plan was over-priced by 2.4 GiB. With the chunked branch it is 58.7 MiB over. The step's largest
  activation there is the `grouped_nf4` MoE backward (loggetta's line), not the loss.
- **Forced chunking at a small T (P1) is now 36.2 MiB under.** `E4B_CHUNKED_LM_LOSS=1` is opt-in, and `auto` does not
  chunk at T = 1024. The chunk's own coefficient, `CHUNK_BYTES_PER_LOGIT` (10), is a stated formula. The 12 B per logit
  of whole logits was measured by allocator replay, but the chunk's coefficient was not.
- **Why the chunk is not attributed.** The allocator recorder (`torch.cuda.memory._record_memory_history` with Python
  stacks, as loggetta's `bench/train_residual.py` runs it) fails inside the chunk's checkpoint recompute:
  `SystemError: <built-in function cross_entropy_loss> returned NULL without setting an exception`. The same run
  without the recorder (P1) trains normally. That is `run.sh`'s C1 attempt; its C2 stopped on the Alpaca demo data's
  empty row 284 at 6 steps, and the A/B (`ab.sh`) used 2 steps.
- **The coefficient is unchanged here.** `chunked_loss_bytes` also prices loggetta's dense chunked-loss workspace, and
  a change there needs its own attribution.

## The plans against the driver's peak

The estimate is one line of a plan. `replan.py` plans each run the way loggetta's `bench/plan_vs_driver.py --replan`
does: the receipt's own hardware, workload and setup, with `E4B_CHUNKED_LM_LOSS` set as the run ran, and this branch's
estimate. Each run is planned twice, with no receipts on file and with loggetta's committed RTX A2000 training receipts
on file. GiB:

| run | `E4B_CHUNKED_LM_LOSS` | driver peak | receipts on file | plan total | driver / plan | reserve line | context line |
|---|---|---|---|---|---|---|---|
| P1 | `1` | 3.219 | none | 3.955 | **0.814** | inferred 0.576 | inferred 0.500 |
| P1 | `1` | 3.219 | A2000 training receipts | 3.702 | **0.869** | measured 0.657 | measured 0.166 |
| A6 | `auto` | 4.451 | none | 5.380 | **0.827** | inferred 0.813 | inferred 0.500 |
| A6 | `auto` | 4.451 | A2000 training receipts | 5.160 | **0.863** | measured 0.928 | measured 0.166 |
| S6 | `0` | 6.775 | none | 8.192 | 0.827 | inferred 1.282 | inferred 0.500 |
| S6 | `0` | 6.775 | A2000 training receipts | 8.039 | 0.843 | measured 1.463 | measured 0.166 |

- **Every plan covers its driver peak.**
- **The forced point's 36.2 MiB allocator shortfall** sits inside its plan's reserve (0.58–0.66 GiB), so no margin is
  added on the forced path.
- **The A2000 receipts on file include granite-3.1's own,** so its reserve line is its own model's (loggetta#49).

**In sample.** These are four points on one model and one card. No committed loggetta MoE training receipt reaches
the chunked regime: all run at T = 1024, and Qwen3-30B-A3B's stock fp32 logits there are 0.59 GiB, under the gate.

**Update (#1504).** The chunk's own coefficient was later measured without the recorder, as peak allocated around
`chunked_causal_lm_loss` at T = 1024, 2048 and 4096 with 512-token chunks: 12.0–12.07 B per chunk logit after the
hidden-row term. `CHUNK_BYTES_PER_LOGIT` now equals `LOGITS_LOSS_BYTES`, 12. It does not move the four points above: at
P1 the estimate's `grouped_nf4` backward branch exceeds the chunk at either coefficient. P1's remaining 36.2 MiB is
not attributed. A candidate is granite's `logits_scaling` transform, an extra bf16 copy per chunk, which the
identity-transform measurement does not include.
