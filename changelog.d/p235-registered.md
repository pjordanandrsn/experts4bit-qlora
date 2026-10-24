### P235 registered: P58's same-box vLLM head-to-head, re-read at the release pins (e4b 0.58.0, grouped-nf4-gemm 0.51.0)

A served registration, no box yet (`bench/p235/PREREG-p235.md`). P58's reviewed harness is re-read at the release
pins against vLLM 0.32.0 and 0.30.0 (P58's primary). It is quoted beside P58's reading (1.087 at B=1, 1.396 at B=16),
never instead of it. Every distribution is a PyPI release with its provenance checked: a version string is not a
release.
- `bench/p235/provenance.py` (new): the registered pins as constants (experts4bit-qlora 0.58.0, grouped-nf4-gemm
  0.51.0, vLLM 0.32.0 and 0.30.0). For each distribution it checks the installed version and its provenance: a PyPI
  install, or a git install at a tag commit an amendment names. Editable, `file://`, other-host and other-commit
  installs are refused. It writes `provenance.json` into the receipts.
- `bench/p235/p235_run.sh` and `p235_reduce.py`: P58's runner and reducer by named edits. The runner installs
  experts4bit-qlora, grouped-nf4-gemm and each vLLM build from PyPI wheels, with transformers 5.17.0 and constant vLLM
  versions. It runs the provenance check before the tripwire and every arm (rc 9) and in each vLLM venv (a failure
  refuses that build's rows), and it no longer collects the GPU's UUID. The 18 arms, the alarms, the STOP rules and
  the verdict code are P58's.
- `bench/p235/p235_drive.sh`: the pod-launch drive form (lane liveness by launch identity, the clean HEAD as the
  harness's launch commit, the `--no-links` fetch), staging P58's harness pieces at P58's digests.
- `tests/test_p235.py`: the provenance rules on fake distributions, the constants against the PREREG and the runner,
  the check's place in the runner, the staged pins against P58's, every changed runner line a named edit, the
  reducer's output on P58's receipts, and the drive's refusals and dry run.
