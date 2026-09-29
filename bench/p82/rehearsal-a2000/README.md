# P82 rehearsal on the home A2000 (2026-09-29, not a reading)

`P82_PROVE=1` in a throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container on the QNAP's RTX A2000 (sm_86),
with the card class and disk floor lifted by the runner's own knobs (`P82_GPU_CLASS=A2000`, `P82_MIN_DISK_GB=1`, so
the summary says REHEARSAL). The staged files are the registered ones, byte-for-byte, except in M1.

| arm | pins | change | rc | refused by |
|---|---|---|---:|---|
| R | e4b `619c719` (0.37.8), gnf4 `9407d49` (0.33.7) | none | **0** | proved: tripwire OK, `router_epi_cast_weights False`, 21-case self-test |
| M1 | as R | the runner's `export E4B_ROUTER_EPI_CAST=0` deleted (its own sums) | **9** | `the router is not at the licensed fp32 weights (env None, CAST_WEIGHTS None)` |
| M2 | e4b `2264847` (before #777), gnf4 `9407d49` | none | **9** | `e4b cut lacks #777 (a bound one-row bucket appends to its own slot)` |
| M3 | e4b `619c719`, gnf4 `fb15cf5` (0.33.5, before #413) | none | **9** | `gnf4 cut lacks #413's append (quantize_kv_fp8's bytes; >= 0.33.7)` |

Each gate the registration adds fails on the change it guards against. The A2000 cannot run the fp8 paged KV, so no
arm, build or K8 was rehearsed here. `rehearse_log.txt` and the per-arm `outer_*_log.txt` are the container's logs, renamed from `.log` (ignored here).
