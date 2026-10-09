# Same-process worker continuation (proposal)

`ra_handoff.py --worker-spec <absolute JSON> --worker-sha256 <reviewed digest>`
continues only after its no-site payload, guarded native startup and requested
release-import checks pass. It loads the fixed adjacent `ra_verified_worker.py`
against the worker spec's pin, then checks the complete fixed RA Python/support
inventory before importing wrappers. The spec chooses one of training,
training_profile, decode, quality or capacity; it cannot supply a command.
The handoff seals the installed-file inventory; the worker compares that seal
before taking its own snapshot, closing the gap between handoff and dispatch.
It derives sites from the verified venv explicitly, including Python 3.11 where
`sysconfig` can retain base paths after native site activation.

The continuation binds native-spec bytes, the frozen stage and adjacent source
pins, fresh separate native output/receipt paths and fixture environment before
the target. It retains its PID, passed handoff hash/evidence, worker phase and failure.
After return it rechecks tools, native inputs, stage, interpreter/config,
installation and archives, bootstrap evidence, loaded origins, environment and
path/finder order. A successful return remains pending the other gates.

The supplied expected spec digest is a proposal for later controller review,
not proof or launch authority. CPU controls substitute their own tiny release
wheels, narrowed stage pins and fixed target files; real wrappers are not run.
The real setuptools hook and full proof installation remain unproven. Nested
training/capacity subprocesses still enter their existing startup paths; this
component explicitly records `nested_workers_verified=false`. ABBA wiring,
source/publication/common-input joins, on-card engagement/context absence and
fleet/launch/retrieval/teardown gates remain. No registered instrument changes.

Validation: 43 worker and 36 handoff CPU controls pass without skips. Selected
extracted Linux CPython 3.11.13 accepts a copied-venv three-wheel synthetic shim
and fixed target, preserves one PID across the handoff/target and refuses a
post-target search-path mutation. Earlier fixture layout failures are retained.
Native host libc differs from the pinned image; this is not full-image proof.
