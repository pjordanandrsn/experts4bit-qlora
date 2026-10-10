# DQ11 Amendment 2: repair the locked-wheel origin check

Prospective registration for replacement draw `dq11-5090-2`, under
[DQ11-PREREG.md](DQ11-PREREG.md) and
[DQ11-AMENDMENT-1.md](DQ11-AMENDMENT-1.md). Those records and the first draw's
source and evidence remain unchanged. The repair and this amendment must merge,
and the maintainer must ACK the replacement before any offer query or rental.

## First draw and repair

`dq11-5090-1` ran from the reviewed merge `d90473e4ee66328d8169cccb6174eea4283e626a`.
Its code/input hash checks and standard VRAM/egress probes passed, then bootstrap
refused the first locked NVIDIA wheel with `ValueError: unregistered wheel
origin`. The committed lock contains sixteen `pypi.nvidia.com` URLs, but the
bootstrap's allowed origins omitted that host. The unchanged bootstrap reproduces
the refusal on CPU before a download or installation. The main reducer gives
`VOID`, with no recommendation: there are no binding proofs or timed readings.
The private operational receipt retains the raw logs, invoice and successful
guard destruction plus verified instance absence; its actual cost is $0.022.
This is a bootstrap/lock consistency defect and provides no scientific result.

The repair admits `pypi.nvidia.com`, which is already in the committed lock, and
validates every locked origin before downloading anything. Mandatory CPU checks
parse all 101 URLs in `requirements.lock`, match them to `wheels.json`, and
validate their origins. Mutants removing each used origin must refuse before
network or installation. Package versions, URLs and SHA-256 values do not move;
downloaded bytes must still match the lock before any installation or authority
seal. `science.sha256` includes this amendment and the repaired bootstrap.

## Replacement and aggregate ceiling

The registered work, model/data/canonical initializer, three arms, six fresh
readings and their reversed order, optimizer/update count, K8 quality budget,
headroom cutoff, proof/removal checks, controlled-comparison scope and teardown
requirement are unchanged. The clean reviewed merge SHA is bound at launch.
No default-Unsloth, capacity, shipping or timing claim follows from a refusal.

The **$2.80 ceiling covers both draws together**. Subtracting the first draw's
$0.022 leaves **$2.778 maximum** for `dq11-5090-2`: runtime/storage reservation
at most **$1.678** (at most **$0.839/hour for two hours**) plus the unchanged
100 GB transport reservation at most **$1.10**. The normal 32 GB RTX 5090,
320 GB ordered storage, secure provider policy and two-hour teardown guard
remain required. Actual quote, role and global admission bind; refuse a larger
reservation. Inspect the quote's driver before purchase: the pinned cu130 stack
requires a driver at least 580; hold an older or unknown driver. This is a
runtime prerequisite, with no exact DQ10 driver pin. Record the admitted driver.

No third draw, hotpatch, fallback, widened budget or retuned gate is authorized.
Any further refusal or incomplete evidence remains `VOID` and needs review.
