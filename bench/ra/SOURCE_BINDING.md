# Release runtime source binding

`ra_source_binding.py --manifest <absolute-json> --out <new-absolute-json>`
runs under `python -I -S -B`. The schema-1 manifest has two `releases`, each
with an absolute `repo`, full reviewed `commit`, `version`, and retained
`wheel` (`path`, `sha256`). One release of each project is required.

The gate reads Git objects at the pinned commits, with replacement objects
disabled, and recomputes the commit, all tree and selected blob identifiers from
their bytes. It derives the complete runtime file mapping from the committed
static setuptools configuration and compares every mapped byte with the
archive. Missing, extra or substituted runtime files refuse, including package
data and gnf4's native C sources. Source symlinks refuse. Dirty worktrees do not
affect the comparison. Unknown packaging conventions require a reviewed adapter.

It also compares project name/version, Python floor, license expression/files,
README bytes and top-level module census. The archive verifier checks the
complete outer RECORD; both archives and source trees are checked again.

This proves runtime payload equality to the selected Git commits. It does not
prove release authorization, reproducible builds, source equality of dependency
metadata, installation, imports or GPU engagement. Those remain separate gates;
the manifest is an input for review and grants no rental clearance.
