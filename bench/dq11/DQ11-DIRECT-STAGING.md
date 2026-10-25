# Census direct staging and absolute phase deadlines

Operational follow-up to the second paid census harness refusal. The prospective
census-3 budget amendment extends the paid wall from 1800 to 2700 seconds
(45 minutes), with the same 300-second teardown reserve inside it: 2400 seconds
for work. Science, thresholds, driver, tariffs and the single-draw limit stay fixed. The next template uses unused run id
`dq11-l-census-4`; a template is not paid authority.

The observed initial staging payload was 170,378,038 bytes across 57 files.
Its proxy SSH endpoint was `ssh-redacted-a18487f6`; the outer staging interval
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

The maintainer's RED budget amendment is prospective. It does not authorize a
rental or reuse either spent envelope. The following accounting distinguishes
historical measurements, planning proxies and enforced refusal gates:

| Phase | Basis | Conservative allowance (seconds) |
|---|---|---:|
| Provisioning and controller preflight | Census2 create to drive: 184s, multiplied by 1.5 | 276 |
| Authenticated direct endpoint resolution | Unmeasured API allowance | 45 |
| Initial direct staging | Unmeasured direct rate; fixed total staging gate | 180 |
| Box setup and probes | Census2 lane start to bootstrap: 8s, multiplied by 1.5 | 12 |
| Exact wheel download and bootstrap | Census2 full bootstrap: 270s, multiplied by 1.5 | 405 |
| CUDA/import | Historical 5s attention canary multiplied by 1.5; proxy for the new phase | 7.5 |
| Model preparation/fetch | Historical same-science prepare interval: 118s, multiplied by 1.5 | 177 |
| L proof | Historical interval: 22s, multiplied by 1.5 | 33 |
| Fresh unprofiled reference | Historical interval: 52s, multiplied by 1.5 | 78 |
| Separate capture | Historical unprofiled L interval multiplied by 1.5; profiler overhead unknown | 78 |
| Reduction and closure | Historical conservative full tail: 132s, multiplied by 1.5 | 198 |
| **Work subtotal** | Includes the planning proxies above | **1489.5** |
| Packet transfer and destruction | Fixed teardown reserve, inside the paid wall | 300 |
| **Total** | 2700-second paid wall minus 1789.5 seconds | **1789.5, leaving 910.5** |

Replacing the 7.5-second CUDA proxy with the full existing 90-second phase cap
gives 1572 seconds of work and 1872 seconds including teardown, leaving
828 seconds. Even reserving 1872 seconds for work before adding teardown leaves
528 seconds. These margins allow for bookkeeping and uncertainty; they do not
turn the direct rate or capture overhead into measurements. Each existing phase
cap and the absolute work deadline remain enforced by the external watchdog.
A direct staging refusal still ends the one draw without retry.

The historical preparation interval comes from the canonical `dq11-5090-4f`
summary, whose science manifest matches the census science closure. Census2
supplies provisioning, setup and bootstrap evidence. The private accounting
archive records source hashes and the distinction between measurements and
proxies. The prior combined 496-second launch note is not used to invent a
separate preparation measurement.

The hourly cap remains $0.85, with 27 GB download at no more than $0.011/GB and
3 GB upload at no more than $0.05/GB. At 45 minutes the full reservation is
`0.85 * 0.75 + 27 * 0.011 + 3 * 0.05 = 1.0845` dollars. The approved all-in ceiling is $1.10, leaving $0.0155 above this full
reservation. The upload allowance remains included. A fresh RED envelope
request follows review and merge of the budget amendment; the template alone
is never paid authority.

## Zero-cost verification

CPU tests exercise missing/direct/foreign endpoint refusals, a real throttled
staging subprocess, shared staging budget, a child that cancels its own alarm
but cannot cancel the external watchdog, and an outer ITIMER_REAL armed before
exec that still fires after a successful wheel fetch. The offer filter is also
exercised against a GET-only live listing; no create or rental is permitted.
This is an operational rehearsal, not a GPU census result or paid authority.

## Instance creation and endpoint refusal evidence

Census3 refused before staging because the controller created proxy-only SSH.
The prospective census4 template explicitly opts in with `vast_direct_ssh: true`,
forwarded by the reviewed controller as `--vast-direct-ssh`. The provider sends
`ssh_direc ssh_proxy`; other runs keep the existing `ssh` default.

Endpoint refusal records actual_status, whether public_ipaddr is set, and only
HostIp/HostPort from the SSH bindings. The public address is omitted; non-wildcard
HostIp values are redacted. IPv4 and IPv6 wildcard bindings sharing one HostPort
are one mapping; distinct ports still refuse. There is no workload proxy fallback.
A fresh census4 envelope at both merged heads is required after controller deployment.
