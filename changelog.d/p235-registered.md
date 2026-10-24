### P235 registered: P58's same-box vLLM head-to-head, re-read at the release pins (e4b 0.58.0, grouped-nf4-gemm 0.51.0)

A served registration, no box yet (`bench/p235/PREREG-p235.md`). P58's reviewed harness is re-read at the release
pins, against the latest vLLM (0.32.0 today) and against 0.30.0, P58's primary. It is quoted beside P58's reading
(1.087 at B=1, 1.396 at B=16), never instead of it.
- `bench/p235/p235_run.sh` and `p235_reduce.py`: P58's runner and reducer by named edits (lane names, transformers
  5.17.0, the secondary vLLM build). The runner's tripwire refuses (rc 9) any e4b other than 0.58.0 and any
  grouped-nf4-gemm other than 0.51.0, and the runner no longer collects the GPU's UUID. The 18 arms, the alarms, the
  STOP rules and the verdict code are P58's.
- `bench/p235/p235_drive.sh`: the pod-launch drive form (lane liveness by launch identity, the clean HEAD as the launch
  commit, the `--no-links` fetch), staging P58's harness pieces at P58's digests.
- `tests/test_p235.py`: the staged pins against P58's, every changed runner line a named edit, the reducer's output on
  P58's receipts unchanged but for its title, the version asserts, no UUID, and the drive's refusals and dry run.
