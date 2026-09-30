# P85 rehearsal on the home A2000 (2026-09-30, not a reading)

`P85_PROVE=1` in throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` containers on the QNAP's RTX A2000 (sm_86,
driver 575.64.05, Intel Xeon W-1250). The card class, the CPU vendor and the disk floor are lifted by the runner's own
knobs, so the summary says REHEARSAL. The card was idle before the first arm (0 %, 1,374 MiB, no compute apps). Each
mutation arm ran with its own `staged.sha256`, so the staged-bytes check passed and the refusal came from the check
under test.

| arm | change | rc | result |
|---|---|---:|---|
| R | none | **0** | Stack O (e4b 0.37.4 @`c77aab6` + grouped-nf4-gemm 0.33.0 @`5ca1897`) installed with the dependencies. Stack S (0.33.6 @`f879761`) installed over it with `--no-deps`. Every tripwire passed for both stacks: pinned commits from pip's `direct_url.json`, versions, the router at fp32, a pre-#413 `fp8_kv` with `fp8_kv_append_t1` present, the resolver on by default and off under the knob, and P70's harness importing. Each stack's stamp read `fused_kv_append_resolved: true` with the knob unset and `false` under `E4B_FUSED_KV_APPEND=0` (`R_stamp_*.json`). The 20-case self-test passed; the egress probe read 54.1 MB/s. |
| M1 | S pinned to v0.33.7 (`9407d49`), with the version check moved to 0.33.7 so that only the #413 check could catch it | **9** | Refused by S's tripwire: `S: fp8_kv carries #413 (_quantize_kv_fp32) -- not a pre-0.33.7 kernel` |
| M2 | `E4B_FUSED_KV_APPEND=0` exported to every process | **9** | Refused by O's stamp check (`STAMP FAIL (O)`): the default stamp read the append resolved off. Without the check, O's control and S would have run with the append off. |
| V | the vendor knob left at its default | **16** | `REFUSED: host CPU vendor is 'GenuineIntel', the lane registers AuthenticAMD (P84 amendment 1)`, before any install (the log has no install line) |

The arms ran the runner as of `c6c6e31`. The one later runner change (`5afe9c3`: O_build's pack is verified before the
control gate) moves steps after the proving run's exit, which `P85_PROVE=1` never reaches.

No build or K8 can run here: the A2000 has 12 GB of memory, and the fp8 paged KV needs sm_89+.
