# P88 rehearsal on the NAS RTX A2000 — NOT a reading

`rehearse.sh` staged the lane's files exactly as `p88_drive.sh` would, in a throwaway
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container. It then ran `p88_run.sh` twice on 2026-10-01, with e4b
installed from GitHub at this registration's branch head (`fe5cbaa`) and gnf4 at the pin (`7b7e6b1`).

| run | knobs | result |
|---|---|---|
| A | `P88_PROVE=1 P88_GPU_CLASS=A2000 P88_MIN_DISK_GB=20` | **rc 18**, `REFUSED: host CPU vendor is 'GenuineIntel'`, before any install ([`refusal_runA.txt`](refusal_runA.txt)) |
| B | as A, plus `P88_CPU_VENDOR=GenuineIntel` | **rc 0, `PROVED`**: the tripwire passed, including K19's default plan (32, 256); premise 3 passed; K19 and K16 contracts compiled, 22 passed, including `test_plans_are_bit_identical_compiled`; egress 54.4 MB/s ([`summary_runB.txt`](summary_runB.txt), [`k19_contract_runB_log.txt`](k19_contract_runB_log.txt)) |

What it cannot show: anything on sm_120 or about Qwen3-30B-A3B's decode loops. The A2000 cannot run the fp8 paged-KV
kernel. The proving rental runs path B on a 5090 on an AMD host.
