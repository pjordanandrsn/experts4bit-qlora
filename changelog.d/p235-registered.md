### P235 registered: P58's same-box vLLM head-to-head, re-read at the release pins (e4b 0.58.0, grouped-nf4-gemm 0.51.0)

A served registration, no box yet (`bench/p235/PREREG-p235.md`). P58's reviewed harness is re-read at the release
pins against vLLM 0.32.0 and 0.30.0 (P58's primary). It is quoted beside P58's reading (1.087 at B=1, 1.396 at B=16),
never instead of it. Every distribution is a PyPI release with its provenance checked: a version string is not a
release.
- `bench/p235/provenance.py` (new): the registered pins as constants (experts4bit-qlora 0.58.0, grouped-nf4-gemm
  0.51.0, vLLM 0.32.0 and 0.30.0). For each distribution it checks the installed version and its provenance: a PyPI
  index install only. Editable, `file://`, other-host and every git install (whatever its URL or commit) are refused.
  It writes `provenance.json` into the receipts.
- `bench/p235/release_proof.py` (new): loggetta bench/release_proof.py at f09e5676 (#126's merge), staged byte-identical. It proves each
  installed distribution IS the published wheel's bytes (wheel digest, every RECORD file equal on disk, every
  top-level name importing from the proven install). It runs as
  the last step before each build's first arm, in that build's import environment, and writes
  `release_proof*.jsonl` into the receipts. A fast-check copy also runs right after each build's install, so a
  wrong wheel refuses in minutes; only the binding proof is ruled. A proof whose lines are not all `pycache_fresh` true refuses. Every
  python uses a fresh `PYTHONPYCACHEPREFIX` (with `RELEASE_PROOF_FRESH_PYCACHE` set to the same value), and every
  install uses `--no-compile`.
- `bench/p235/p235_run.sh` and `p235_reduce.py`: P58's runner and reducer by named edits. The runner installs
  experts4bit-qlora, grouped-nf4-gemm and each vLLM build from PyPI wheels, with transformers 5.17.0 and constant vLLM
  versions. It runs the provenance check before the tripwire and every arm (rc 9) and in each vLLM venv (a failure
  refuses that build's rows), and it no longer collects the GPU's UUID. The 18 arms, the alarms, the STOP rules and
  the verdict code are P58's.
- The runner checks the guard fit before its first install (`bench/common/README.md`, "Guard fit"): FIT = 6,060 s
  from P58's record and stated arithmetic, x 1.25, refused with rc 41 (not a strike) when the box's time left is short.
- `bench/p235/p235_drive.sh`: the pod-launch drive form (lane liveness by launch identity, the clean HEAD as the
  harness's launch commit, the `--no-links` fetch), staging P58's harness pieces at P58's digests.
- `tests/test_p235.py`: the provenance rules on fake distributions, the constants against the PREREG and the runner,
  the check's place in the runner, the staged pins against P58's, every changed runner line a named edit, the
  reducer's output on P58's receipts, and the drive's refusals and dry run.
