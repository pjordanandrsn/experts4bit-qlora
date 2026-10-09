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

**In sample.** These are four points on one model and one card. No committed loggetta MoE training receipt reaches
the chunked regime: all run at T = 1024, and Qwen3-30B-A3B's stock fp32 logits there are 0.59 GiB, under the gate.
