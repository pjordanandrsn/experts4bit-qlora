### P237 registered: e4b's CPU expert offload with layer-major against Unsloth's offload_layers, one RTX 4090, two VRAM budgets

- **The registration, `bench/p237/PREREG-p237.md`.** E (experts4bit-qlora 0.58.1, `--offload 1 --layer-major 1`) against Unsloth
  2026.10.8's `offload_layers` (`auto` as UA, tuned as UT) on Qwen3-30B-A3B, TC1's field recipe, at a 12 GiB cap and at the full
  card. The verdict is per budget: quality first (COMPARABLE at 0.05 nats), then the wall ratio U*/E. No data has been read.
- **The observing hook, `bench/p237/p237_arm.py`** (`TC1_ARM_HOOK=p237_arm:make`, e4b#2146's seam). It counts per interval between
  step edges and launches no device work:
  - e4b's real stagings (`_OffloadStats.record_copy`) and its `layer_major_step` calls;
  - Unsloth's real block fetches and evictions (`block_swap._Block.prefetch` / `evict`, counted only on a residency change), plus
    a residency-versus-tensors check at every edge;
  - the shared Unsloth counter (e4b#2132) and gnf4's route.
- **The kit:** the runner (`p237_run.sh`), the driver (`p237_drive.sh`, on `bench/common/drive_tail.sh`), the schedule, the host
  decisions (`p237_host.py`: CPU class, the full-card cap, N_auto, the probe outcomes, N_t), a download probe, the reducer
  (`p237_reduce.py`, with a self-test that removes each piece of evidence in turn) and the staged pin. Tests:
  `tests/test_p237.py` and `tests/test_p237_arm.py`.
