### SV3 registered (#1224): the hybrid state pool and the decode-bucket cap beside the estimate on Qwen3.6-35B-A3B, + gpt-oss-20b (bench and prereg only)

- `bench/sv3/`: `SV3-PREREG.md`, `sv3_run.sh` (box side, `tc1_drive.sh`'s `TC1_RUNNER` contract) and `sv3_measure.py`.
  The measure script records the per-slot linear-attention state pool's own `nbytes()` beside its price.
- One RTX 5090. Arms:
  - Qwen3.6: `q36_e16` (the anchor), `q36_g16`, `q36_g1_default`, `q36_g1_capped`;
  - gpt-oss: `gptoss_g16`.
  - Both arenas are baked on the box through `bench/p98/p98_bake.py`, outside the fetched tree.
- Readings, registered with the estimate's numbers computed before the box:
  - W1, W3, W5: the estimate within ±5% of each peak;
  - W2: the pool's `nbytes()` exactly its price;
  - W4: the bucket cap's saving, priced 931 MiB, measured within −32 / +256 MiB of it.
- Spend is capped at $10 by #1224, within the owner's $50 approval.
