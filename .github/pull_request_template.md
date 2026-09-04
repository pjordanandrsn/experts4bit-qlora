<!-- Keep what applies; delete what does not. The claims register, not this text, decides whether a number is current. -->

## What changed and why

## Checks

- [ ] **Public API changed?** If yes: docstrings updated (when to use, layout expected, return/success assertion, refusal conditions), `docs/capabilities.json` entry points updated, `AGENTS.md` API list updated.
- [ ] **Solution pages** (`docs/solutions/*.md`, `docs/SOLUTIONS.md`) still answer their H1 for this change; new capability → new or updated page.
- [ ] **Capabilities** (`docs/capabilities.json`) updated; `python scripts/check_capabilities.py` passes.
- [ ] **Claims**: every number in new prose links an ACTIVE claim ID in `docs/claims.json`; no retired or superseded claim is repeated as current; `docs/STATUS.md` updated if the position moved.
- [ ] **llms bundle** regenerated (`python scripts/build_llms_bundle.py`; `--check` passes) when README opening, SOLUTIONS, STATUS, capabilities or the listed docs changed.
- [ ] **Discovery contract**: `python scripts/check_discovery_contract.py` passes (queries still route to pages that carry their concepts, canonical install route and limitations).
- [ ] **PyPI metadata**: `pyproject.toml` description/keywords/urls/extras still accurate; if an extra or dependency floor changed, README install section, `docs/capabilities.json` install commands and `AGENTS.md` say so.
- [ ] **README routing** (Use this when / Do not use this when / Start here) still accurate.
- [ ] **Examples** in new docs are executed in CI, executed in a hardware lane, or explicitly marked as needing GPU / network / model download / large storage. No example silently falls back.
- [ ] **Related repository** updated (or an issue filed there) if the kernel/consumer contract changed.
- [ ] **Anchored docs untouched**: nothing under `docs/` with an `ots-attestation-footer` was edited (`grep -l ots-attestation-footer docs/*.md`).

## Evidence

<!-- Lane / box / receipt paths for any measured number; "capability only, no performance claim" otherwise. -->

## Release note draft (if this ships in a release)

<!-- First paragraph, in ordinary language: which problem changed, which users / model families / environments are affected, and whether they should upgrade. Mechanism, measurements, receipts, corrections and caveats follow. -->
