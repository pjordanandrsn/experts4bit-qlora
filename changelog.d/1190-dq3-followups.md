### DQ3 follow-ups (#1190): `dense_offload_report` records the late-bound route; the DQ3 timing peaks start from a clean cache

- `dense_offload_report(handles)["late_bound_4bit"]` counts the offloaded bnb `Linear4bit` projections whose grad-mode
  matmul is late-bound. 0 means stock bnb, for example after a source mismatch. DQ3's arm receipt records it.
- `bench/dq3/dq3_arm.py` empties the CUDA cache before resetting the peak stats for the timing pass.
- Doc fixes: `dq3_run.sh`'s header cites Amendments 0–3, and the bnb-mirror comment notes the inference-only CPU
  AVX-512 branch.
- **Observation** (rehearsal, 4 layers at Qwen3-32B width): streaming cuts peak allocated (5.82 → 5.37 GiB) but raises
  peak reserved (6.45 → 6.69 GiB), most likely through caching-allocator fragmentation. A capacity claim needs real
  OOM boundaries.
