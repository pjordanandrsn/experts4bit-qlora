# The conversion-mapping reference on real Qwen3 weights (guard a)

**When and where.** 2026-10-10, 20:06–20:58 UTC, on the host CPU beside the RTX A2000 (6 intra-op threads, nice 19), in the
calibration's environment: torch 2.11.0+cu128, transformers 5.19.0. The model is Qwen/Qwen3-30B-A3B at revision ad44e777 (as
verified in `../calibration-30b/`). The tokens are the calibration's (sha256 d077cc63…).

**What it checks.** `layer_stream_ref.py` now reads a checkpoint through transformers' own conversion mapping. Two checks against
the original Qwen3-only assembly and the calibration it produced:

| check | result |
|---|---|
| every layer's state dict, new vs original `layer_state`, key for key and bit for bit | **48 of 48 identical** |
| 6-layer CPU reference, resident, vs `ref6_resident_cpu`: per-layer sha256 and loss | **identical** (loss 12.781570434570312) |
| the same, streamed | **identical** |

Verdict **PASS**: the 30B calibration's references and gate 1's stay valid under the new code. Files: `guard_a.py`,
`run_guard_a.sh`, `guard_a.log`, and `guard_a.json`, which holds every number. The CUDA references and the 4-layer 235B slice
are a bonus, run when the A2000 is free.
