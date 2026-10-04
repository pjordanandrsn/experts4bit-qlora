# K29 rehearsal, RTX A2000 (QNAP), cgroup v1, $0

`k29_run.sh` and `pinned_charge_probe.py` were byte-identical to their pins (`91f3d41a…`, `760bcfd0…`). They ran in a
throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container on 2026-10-04, 15:18:06–15:20:23Z by the run's own
log, and exited rc 0.

STOP-1 fired as registered: the host is cgroup v1, so the reducer's verdict is VOID. That is the path this rehearsal
exists to prove. The rows are still evidence for v1. All 18 pinned rows read r = charged / pow2ceil(N) in
[1.0043, 1.005], over ten sizes and two repetitions, and the pageable slope is 1.002 (R² 1.0).
