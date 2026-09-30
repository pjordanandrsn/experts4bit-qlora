# P84 rehearsal on the home A2000 (2026-09-30, not a reading)

`P84_PROVE=1` in throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` containers on the QNAP's RTX A2000 (sm_86).
The card class and disk floor are lifted by the runner's own knobs, so the summary says REHEARSAL. Stack B was
installed at `a83937a`, the `main` of the day.

| arm | change | rc | result |
|---|---|---:|---|
| R | none | **0** | Stack B (e4b 0.37.8 + grouped-nf4-gemm 0.33.7) installed with the dependencies. P70's harness (`o/`) and P82's (`n/`) both import under it. Stack A (e4b 0.37.4 @`c77aab6` over B, the same gnf4 0.33.7) installed with `--no-deps`, and P70's harness imports under it. Every tripwire passed: pinned commits from pip's `direct_url.json`, versions, and the router at fp32. Both router stamps read `CAST_WEIGHTS False` (`R_stamp_{B,A}.json`). The 15-case self-test passed. No pip retry was needed. |
| M1 | the runner's `export E4B_ROUTER_EPI_CAST=0` deleted (its own sums) | **9** | Refused by stack B's tripwire: `B: the router is not at fp32 (env None, CAST_WEIGHTS None)`. 0.37.8's default is the cast (None), so without the export the lane would have measured the wrong router. |

No build or K8 can run here: the A2000 has 12 GB of memory, and the fp8 paged KV needs sm_89+.
