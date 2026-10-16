# Census direct staging and absolute phase deadlines

Operational follow-up to the second paid census harness refusal. No science,
threshold, driver, tariff, $0.90 ceiling, 1800-second paid wall or 300-second
teardown reserve changes. The next template uses unused run id
`dq11-l-census-3`; a template is not paid authority.

The observed initial staging payload was 170,378,038 bytes across 57 files.
Its proxy SSH endpoint was `ssh2.vast.ai:17364`; the outer staging interval
was 1265 seconds, about 134,686 bytes/s (1.08 Mbps). This interval includes
remote directory setup. A direct/proxy paired measurement was unavailable
once the instance was destroyed; the proxy bottleneck attribution is an
operational diagnosis, not a comparative benchmark.

The census offer gate now requires a public IPv4 address and a positive
integer `direct_port_count`. After creation the operational wrapper performs
only an authenticated instance GET, checks the exact instance id and resolves
`public_ipaddr` plus the single `ports["22/tcp"]` host-port mapping. Missing or
ambiguous direct endpoints refuse; there is no proxy fallback. This changes
only the census workload's SSH environment, not the shared rental controller.
The direct host key is recorded on first connection in a separate per-run
known-hosts file, matching the controller's existing trust policy; subsequent
key changes refuse. Endpoint provenance, policy and actual pinned fingerprint
are retained in `direct-transport.jsonl`. No HF upload or new hosting exists.

Staging first authenticates one foreground SSH control connection, records its
setup duration, then reuses that connection for SCP. A missing control connection
cannot silently open another one (`ProxyCommand=false`). The largest file is
copied first and its measured transfer rate excludes handshake time; projection
adds setup once, then scales only payload time to the whole closure. Setup still
counts against the fixed 180-second staging budget. The first file's share of
the remaining budget is an
external process deadline, so a slow initializer refuses before consuming the
paid wall. The remaining files share the original remaining staging budget.
The gate also respects the absolute work deadline minus the teardown reserve.
A timeout kills the transfer process group, then returns a refusal to the
existing controller, which owns authenticated teardown. There is no retry.

The wheel helper no longer touches SIGALRM or ITIMER_REAL. It keeps monotonic
per-wheel checks, bounded socket operations and small available-data reads.
An independent Python supervisor now times each box phase against both its
existing phase cap and the absolute work deadline minus 300 seconds, killing
its process group on expiry. Downloader signal changes cannot cancel this
parent timer. The supervisor observes child exit with `waitid(WEXITED|WNOWAIT)`,
signals the group while the unreaped leader still reserves its PID, then reaps.
This prevents a post-exit group-id reuse signal. Missing deadlines/host keys and
empty commands produce coded refusals. Phase start/end/cap/elapsed records are
retained separately.

## Phase accounting before another envelope

The following are evidence and refusal caps, not a claim that this design has
already met the paid wall:

| Phase | Evidence | Existing cap / new gate |
|---|---|---|
| Provisioning and controller preflight | Census2 create to drive:184s | controller wall remains1800s |
| Initial staging | Census2 proxy1265s; direct duration unmeasured | direct-only180s total, first-bulk projected gate |
| Box setup and probes | Census2 lane start to bootstrap:8s | existing individual probe caps |
| Exact wheel download and bootstrap | Census2 wheels130s, full bootstrap270s | existing900s cap, now externally enforced |
| Model preparation/fetch | no census2 measurement | existing600s cap |
| Proof | historical same-5090 L process20.346s | existing240s cap |
| Fresh unprofiled reference | historical L process50.506s | existing300s cap |
| Separate capture | historical second L50.501s; profiler overhead unknown | existing300s cap |
| Reduction/closure | historical conservative full-tail132s | existing120s reducer cap; transfer separately reserved |
| Packet transfer and destruction | census2 refusal to destroy39s | fixed final300s reserve |

The prior launch note records a combined496s provisioning/staging/bootstrap/
prepare measurement and the historical process evidence above. That combined
number cannot supply a separately measured model-fetch duration. Do not call
unmeasured phases zero or present the sum of timeout caps as a feasible plan.
The RED run3 request stays withheld until conservative phase accounting fits
1500s work plus300s teardown, within unchanged tariff bounds. The new structured
timing records make that accounting auditable. A direct staging gate passing
alone is not proof that census capture fits.

## Zero-cost verification

CPU tests exercise missing/direct/foreign endpoint refusals, a real throttled
staging subprocess, shared staging budget, a child that cancels its own alarm
but cannot cancel the external watchdog, and an outer ITIMER_REAL armed before
exec that still fires after a successful wheel fetch. The offer filter is also
exercised against a GET-only live listing; no create or rental is permitted.
This is an operational rehearsal, not a GPU census result or paid authority.
