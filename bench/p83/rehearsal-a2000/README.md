# P83 rehearsal on the home A2000 (2026-09-29, not a reading)

`P83_PROVE=1` in throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` containers on the QNAP's RTX A2000 (sm_86).
The card class and disk floor are lifted by the runner's own knobs (`P83_GPU_CLASS=A2000`, `P83_MIN_DISK_GB=1`), so the
summary says REHEARSAL. Stack N was installed at `0f9713c`, the `main` of the day.

| arm | change | rc | result |
|---|---|---:|---|
| R | none | **0** | Both stacks installed at their pins, O then N over it (`R_versions_{O,N}.txt`). Both tripwires passed: the pinned commits from pip's `direct_url.json`, the versions, the router at fp32, and each stack's hook loading as `usercustomize` from its own directory. Both router stamps read `CAST_WEIGHTS False` (`R_stamp_{O,N}.json`), on 0.37.4's router module, which has no `_cast_for`, and on 0.37.8's. The 13-case self-test passed. |
| M1 | the runner's `export E4B_ROUTER_EPI_CAST=0` deleted (its own sums) | **9** | refused by stack O's tripwire: `the router is not at fp32 (env None, CAST_WEIGHTS False)`. 0.37.4 defaults to fp32, but the gate requires the explicit export. |

**First attempt: a transient failure.** The first R attempt (17:21Z) failed installing stack O: pip's `git` exited
128, and the runner kept only pip's last four lines. The same pin then installed cleanly in M1 minutes later, and in a
diagnostic container with pip's full output (`diag_pip_log.txt`: "Successfully installed … experts4bit-qlora-0.37.4",
and a plain `git clone` + `checkout c77aab6` with rc 0). So the runner now gives each pip install **one retry after
20 s**, and keeps both attempts' output. R above is the rerun on the final bytes, which needed no retry.

No build or K8 can run here: the A2000 has 12 GB of memory, and the fp8 paged KV needs sm_89+.
