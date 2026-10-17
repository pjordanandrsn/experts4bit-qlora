# DQ11 census flow binding correction

This corrects the backward binding rule in
[DQ11-L-CENSUS-INSTRUMENT.md](DQ11-L-CENSUS-INSTRUMENT.md).
Kineto's `Fwd thread id` identifies a profiler thread; the Chrome export's
CPU `tid` identifies an OS thread. Their numeric values need not coincide.
Earlier synthetic fixtures assigned both the value `1`, so they did not
exercise this distinction. This does not establish that CPU exports always
use equal identities or that the distinction is specific to one GPU.

The normalizer now requires one explicit `fwdbwd` start/finish flow pair
for each sequenced backward creator node. The finish timestamp, process and
OS thread must anchor the named inner backward operation inside its engine
node. The start must anchor the single registered forward creator on its
exported process/thread with the same sequence number, before the backward
node. Nested wrappers and checkpoint recomputation do not substitute for
that creator. Missing, duplicate, incomplete, ambiguous or mismatched pairs
refuse; no thread alias is inferred. Parameter `AccumulateGrad` hooks retain
their direct class scope because they have no forward creator.

Synthetic regressions cover equal and different thread identities, nested
creators/recomputation, and corrupted flow pairs. Flow arrows never contribute
to device durations. The seven classes, external-id kernel correlation,
interval unions, science source, thresholds and launch limits are unchanged.
This correction authorizes no rental and establishes no scientific result.

The callers are the proof path in `dq11_l_census_run.py` and the proof/capture
CLI paths in `dq11_l_census_reduce.py`. Both use the same normalizer.
