# DQ11 Amendment 3: shared fetch and a mandatory pre-rental gate

Prospective registration for `dq11-5090-3`, under
[DQ11-PREREG.md](DQ11-PREREG.md),
[DQ11-AMENDMENT-1.md](DQ11-AMENDMENT-1.md) and
[DQ11-AMENDMENT-2.md](DQ11-AMENDMENT-2.md). Those records and the evidence of both
earlier draws remain unchanged. No quote or rental for this draw is authorized
until the fetch repair/gate and this amendment merge, the separately reviewed
rental-driver ceiling fix merges, and the maintainer ACKs the third draw.

## Refusal and CPU diagnosis

The second draw ran from reviewed merge `20c6c30d4e771cb1308a4d388f87830d27c12c4f`.
It passed the source/input and normal-VRAM/egress checks, then bootstrap stopped
at an HTTP 403 before the pinned-stack install, registered model preparation,
binding proofs or timed readings. The main reducer gives `VOID` with no
recommendation. Its raw operational receipt records HTTP 200 guard destruction
and verified absence of instance 55173446, with actual cost $0.030. The first
draw cost $0.022; total actual spend is $0.052. Neither draw gives a scientific
result.

On the CPU launch host, the unchanged Python 3.11 bootstrap opener returned 403
for the locked `download-r2.pytorch.org` Torch, Torchvision and Triton URLs. The
first such URL in lock order is
`https://download-r2.pytorch.org/whl/cu130/torch-2.12.1%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl`.
The old box log did not include the URL, so its exact attribution is inferred
from this reproducible refusal, rather than directly observed in that log.
Changing only the Range header still returned 403. Both `pip/26.0` and the
honest `dq11-bootstrap/1` user agent returned 200 for full GET and 206 for a
one-byte range GET on each affected URL. The repair uses `dq11-bootstrap/1`.
Every fetch refusal identifies its URL and HTTP status (or UNKNOWN for a
transport failure). Full downloads retain the same SHA-256 verification before
installation and source-authority construction.

## Mandatory launch-host gate

`dq11_launch.py` is the required pre-rental controller. It verifies clean merged
source, the manifest's exact source binding, the science closure and canonical
assets. It then uses the bootstrap's own shared fetch/read functions to issue a
one-byte range GET to **all 101 locked URLs on the CPU launch host**. The gate
records each URL and status, the user agent, wheel/science/manifest hashes and
the actual source SHA. Any non-2xx, transport failure or empty response refuses
before the quoting/rental driver can run. It never installs wheels or creates
compute during this gate. A gate failure remains a refusal, with no automatic
retry, alternative URL or user-agent fallback.

The controller's explicit `--gate-only` mode performs the same checks without
invoking the rental driver, and is required before read-only quote inspection.
The actual launch performs the gate again against its final bound manifest and
then calls the unchanged reviewed `pod-launch.sh`. Gate output stays outside
the clean source checkout and is preserved in the private operational evidence.
A saved earlier pass cannot replace a gate on the actual launch source.

Mandatory CPU fixtures exercise all 101 requests with the shared honest user
agent, prove a single 403 prevents driver invocation while retaining all URL
statuses, and check successful gate/launch ordering and dirty/unmerged/wrong
source or changed canonical-asset refusals. The committed
`fetch-preflight-cpu.json` records the real pre-review CPU gate: all 101 returned
206 and one byte, with no installation, quote, rental or scientific reading.
The gate must pass again on the actual reviewed merge before launch.

## Remaining ceiling and work

The original **$2.80 aggregate ceiling** minus **$0.052 actual spend** leaves
**$2.748 maximum** for the third draw: at most **$1.648** runtime/storage
(at most **$0.824/hour for two hours**) plus the unchanged 100 GB transport
reservation at most **$1.10**. The normal 32 GB RTX 5090, 320 GB ordered storage,
secure provider policy, quote driver at least 580 and two-hour teardown guard
remain required. Exclude machine 45511 using the committed `sc5-5090-1`
SSH-readiness refusal receipt. Role/global admission and actual quotes bind.

The previous driver replaced a declared lower hourly cap with the fixed policy
rate $0.85. Although the selected offer and actual spend fit Amendment 2, the
driver did not enforce its narrower reservation. This draw therefore waits for
the separately reviewed driver fix that honors a declared rate below or equal
to policy and refuses one above policy, without changing policy rates. There is
no valid earlier DQ11 preparation/proof/read timing with which to certify a
1.9-hour guard with margin; the two-hour guard is retained.

Model, data, canonical initializer, package versions/URLs/hashes, arms,
optimizer/update count, six fresh readings and their reversed order, K8 quality
budget, headroom cutoff, binding/removal checks and controlled-comparison scope
are unchanged. No fourth draw, hotpatch, fallback, widened budget, shortened
registered work or retuned gate is authorized. Any further incomplete evidence
remains `VOID` and needs review.
