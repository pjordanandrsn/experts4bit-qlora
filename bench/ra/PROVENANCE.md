# Wheel and import provenance

Run `ra_provenance.py` with the selected venv's `python -I -B`, before any
measurement. Its reviewed manifest names that venv, every retained wheel's
absolute path and SHA256, and the e4b/gnf4 modules to import. The installed
distribution inventory must match the manifest exactly, including pinned pip
when generated entry-point wrappers need verification.

The stdlib probe hashes each archive, verifies its wheel RECORD, compares
installed payload bytes to the archive, checks installed RECORD coverage and
hashes, and rejects editable origins, symlinks, startup `.pth` files, cached
bytecode and unlisted files. Resealing installed metadata cannot hide modified
wheel payload. Only purelib/platlib relocations are supported; other wheel data
schemes fail until a reviewed adapter exists.

Pip-generated scripts are derived from the already verified pip wheel's own
installer template, including its versioned pip entry points. No script is
executed or rewritten. Installer metadata and direct URLs must agree with the
retained wheel; an unknown installer convention fails. This follows the
[PyPA wheel format](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)
and [installed RECORD specification](https://packaging.python.org/en/latest/specifications/recording-installed-packages/).

After imports, each release module must resolve to that wheel's files. Loaded
modules and search paths must remain in verified wheels or the interpreter's
stdlib, and all verified installation bytes are rechecked. The receipt records
module paths, independently derived versions, wheel and tree hashes, interpreter
ABI, manifest hash and a common dependency digest excluding e4b/gnf4.

This verifies a supplied installation against retained archives. It does not
select or authorize releases, resolve the Linux lock, check feature engagement,
hash model/calibration inputs, prove GPU premises or grant rental clearance.
The future supervisor must enforce the reviewed manifest's release and shared
dependency bindings and retain failed process receipts.

CPU tests install tiny wheels into fresh isolated venvs without network access.
They exercise the real pip script template using re-archived test-host installer
bytes. These are controls, not validation of the actual Linux proof wheels.
