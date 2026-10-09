# Proposed startup/import handoff

`ra_handoff.py` runs with the selected venv's `python -I -S -B`. It performs
`ra_prestartup` in the same process before startup. Its manifest contains
`schema: 1`, the pre-startup manifest as `payload`, the two release module
lists as `imports`, and the SHA-256 of `startup-execution-proposal.json` as
`execution_policy_sha256`.

Native `site.main()` establishes the isolated venv prefix and search path.
A guarded `site.addpackage` replaces `.pth` execution with one literal
adapter: the retained setuptools hook's default-local `add_shim()` call.
The distribution, version, whole archive, installed hook and hook body must
match. Unknown hooks, overrides, customisation modules and preloaded site
code refuse. The native venv and site passes must each visit the hook; both calls must
leave exactly one finder installed. No `.pth` line is evaluated as Python.

Requested modules are selected against wheel owners before import. Afterwards
all loaded module origins, path/finder order, environment, retained archives,
installed files and bootstrap inputs are checked again. Failed receipts retain
the phase and any completed payload evidence. Output paths must be fresh.

The execution policy is a **proposal**, not approval or launch authority.
The original audit registry stays `PROPOSED_AUDIT_ONLY`; the original full
import probe still refuses every `.pth`. CPU tests use tiny synthetic wheels
and a synthetic shim, never the real setuptools startup hook. Selected-image
installation and real dependency/import behaviour require reviewed proof
execution. A successful requested import proves neither GPU engagement nor
that every module in a wheel was imported.
