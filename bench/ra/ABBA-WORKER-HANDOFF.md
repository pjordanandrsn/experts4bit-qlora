# ABBA worker handoff (proposal)

Schema 2 adds `startup.tools` (the complete fixed worker pin set) and
`startup.manifests.old/new` (path/SHA pairs). The existing schema 1 path remains
explicitly unverified. No launch CLI or rental authority is added.

Before each phase, the controller checks the stage, common input lock, copied
interpreter, fixed native specifications and startup bindings. It inspects only
bounded top-level wheel metadata: both retained manifests must contain exactly
the frozen common dependency names, versions, archive names, sizes and pinned
hashes plus two distinct release distributions. This controller inspection does
not establish archive payload integrity; each selected child independently
verifies retained archives and its entire installed payload before startup.

Every old/new/new/old position runs the same six phases via selected Python
`-I -S -B`, the fixed handoff and a freshly derived worker specification. No
caller command, retries or substituted production pin set is accepted. After
return, the process/guard/cleanup, spec, manifest, handoff and worker receipts
must join on PID and hashes. Installed common dependency payload hashes must
match the lock and every completed phase across both environments. Native
TC1, decode, quality and three capacity input projections must match the common
input lock. TC1 uses its retained nested frozen output; capacity additionally
requires the verified server/driver return evidence. Failure retains the active
phase, raw receipts and completed prefix without advancing.

Completion is `ORDERED_COMPONENTS_RECORDED_PENDING_GATES`. Its verified startup
flag concerns these wrapper receipt joins only. Runtime input consumption,
source/publication integration, native context absence, collected on-card
premises, default/fallback telemetry, full-image installation and reviewed
provider/ledger/launch/retrieval/teardown gates remain. No GPU engagement or
release clearance follows from CPU controls, synthetic protocol rows or caller
observations. The advisory lock excludes cooperating RA controllers only.
