# SV3 read — `sv3-5090-1`

Registered in `bench/sv3/SV3-PREREG.md` (#1225) before the box; work item #1224 (owner-authorized, $10 cap within an
owner-approved $50).

**The run.**
- One RTX 5090: Vast instance 54421432, driver 580.178.04, AMD Ryzen 9 7900, 125 GB RAM, 320 GB disk.
- **$0.514**; teardown proven (destroy HTTP 200, instance absent from the list after).
- experts4bit-qlora `c500651` (#1225's head, 0.48.0, including #1219), grouped-nf4-gemm 0.41.0 (`dc8f94a`),
  torch 2.8.0+cu128, transformers 5.18.0.
- Both arenas were baked on the box through the loader (`p98_bake.py`):
  - Qwen3.6-35B-A3B: 40 layers × 256 experts, 12 s;
  - gpt-oss-20b: 24 × 32, 10 s.
- The lane finished `TC1_SUCCESS` with five receipts, and the final fetch carried no arena.

Receipts: `receipts/sv3-5090-1/*.json` (`sv3-arm/1`), `summary.txt`, `forensics.txt`.

| arm | model | seqs | decode graphs | estimate | allocator peak | vs estimate | state pool `nbytes()` = price |
|---|---|---|---|---|---|---|---|
| `q36_e16` | Qwen3.6 | 16 | off | 23.217 | 23.408 | +196 MiB (+0.8%) | 1,038,090,240 B (16 slots), equal |
| `q36_g16` | Qwen3.6 | 16 | buckets 1–16, all captured | 24.187 | 24.423 | +242 MiB (+1.0%) | 2,076,180,480 B (32 slots), equal |
| `q36_g1_default` | Qwen3.6 | 1 | buckets 1–16: **only 1 captured** | 22.629 | 22.893 | +270 MiB (+1.2%) | 1,102,970,880 B (17 slots), equal |
| `q36_g1_capped` | Qwen3.6 | 1 | bucket 1, captured | 21.720 | 21.948 | +233 MiB (+1.0%) | 129,761,280 B (2 slots), equal |
| `gptoss_g16` | gpt-oss-20b | 16 | buckets 1–16, all captured | 15.391 | 15.673 | +289 MiB (+1.8%) | — |

GiB unless marked. "Estimate" is `estimate_serve_footprint`'s device total for the arm's `ServeSetup`, read against the
allocator peak.

**Readings.**
- **W1 (Qwen3.6, eager): HELD.** The peak is 0.8% over the estimate, inside ±5%.
- **W2 (the pool, exact): HELD.**
  - In all four Qwen3.6 arms the pool's own `nbytes()` at the end equalled the estimate's
    "linear-attention state pool" item **to the byte**.
  - The slot counts were 16, 32, 17 and 2 (`max_seqs` + the largest bucket with graphs), and the layer count 30
    every time.
  - That is #1219's arithmetic on the full model: a bf16 conv window and an fp32 recurrent state, ~2.06 MiB per
    layer per slot.
- **W3 (Qwen3.6, graphs): HELD.** +1.0%.
- **W4 (the bucket cap): ALARM; integrity failed, so the reading is not used.**
  - In `q36_g1_default` (one sequence, buckets 1–16) bucket 1 captured. **Buckets 2, 4, 8 and 16 failed to
    capture** ("AcceleratorError: CUDA error: operation failed due to a previous error during capture") and ran
    eagerly.
  - Their graph pools were never allocated, so the arm's peak understates what the default buckets cost when
    capture works.
  - Recorded, not read: 22.893 − 21.948 = 0.945 GiB (968 MiB), against 931 MiB priced. That is a floor on the
    saving.
  - The failure is itself a finding. On a hybrid model served for one sequence, the buckets above `max_seqs` do
    not even capture. Capping the buckets at the sequences avoids them altogether.
- **W5 (gpt-oss-20b, a fourth family): HELD.** +1.8%. Per-expert biases ride beside the arena at all-VRAM, and the
  estimate prices them in the dense weights.
- **Integrity:**
  - every arm finished all its requests;
  - every decode bucket in `q36_g16` and `gptoss_g16` reported `graph`;
  - `q36_g1_default` failed integrity (W4 above).

**Also recorded, not registered.**
- Every arm peaked 196–289 MiB over its estimate, the same direction and size as the allocator residual earlier
  serve runs measured (0.15–0.21 GiB). The planner learns it per model from these receipts.
- tok/s, one draw each: 34.8 / 93.9 / 49.2 / 47.8 / 109.0.
- CUDA context: 0.58–0.71 GiB. Allocator slack: 0.4–4.9%.
