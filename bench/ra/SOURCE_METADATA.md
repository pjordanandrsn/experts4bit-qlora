# Source dependency metadata

Run `ra_source_metadata.py --manifest <absolute-json> --out <new-absolute-json>`
under `python -I -S -B`. The schema-1 manifest extends the runtime source pair
with `parser` (`path`, `sha256`) and `parser_lock_sha256`. The latter pins the
toolset's sibling `proof-wheel-lock.json`; the parser must match its packaging pin.

Runtime Git-to-wheel binding runs first. The archive verifier checks the parser's
complete RECORD before import. The gate derives dependencies and optional extra
gates from each committed pyproject, then compares every `Requires-Dist` and
`Provides-Extra` declaration with wheel metadata. Names, requested extras, bounds
and complete marker expressions participate. Missing, surplus or duplicate
declarations, changed gates, ambiguous extras and direct URLs refuse.

The marker adapter canonicalizes the pinned parser's Boolean structure. Ordering,
spacing and redundant grouping can differ; comparison operands and precedence
stay intact. Unsupported parser structures refuse. Module owners, parser/lock
bytes and both release inputs are checked again after comparison.

This checks source equality of dependency declarations, not dependency closure,
reproducible builds, release authorization, installation, release imports or GPU
engagement. No startup hook or build backend is executed. Test-host parser wheels
and synthetic Git fixtures are CPU controls, not the Linux proof installation.
