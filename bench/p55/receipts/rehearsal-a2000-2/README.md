# P55 rehearsal 2, RTX A2000 (QNAP), $0: evidence for Amendment 2

The whole box script, `p55_run.sh` (sha256 `1940ed7d…`, byte-identical to the pinned file), ran in a throwaway container
from the lane's own image (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`). It used the lane's own override:
`P55_MODEL=ibm-granite/granite-3.1-3b-a800m-instruct`, `P55_REVISION=a02780686e08a03fe0d2679a293b5c74a90efa89`,
`E4B_SHA=a3bca427a8c3`. It ran 2026-10-04 13:38:06–13:41:51Z (the run's own log) with rc=0.

This is not a P55 reading. STOP-1 fired by design: 125.7 GiB effective, and the model is not the subject. It proves
the harness runs end to end: git, install, tripwire, fetch, three arms, trace, and markers.
