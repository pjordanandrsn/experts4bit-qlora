### Changelog entries are one file each in `changelog.d/`; `## Unreleased` is no longer edited by hand (tooling only)

- **Why.** About ten pull requests an hour each inserted a `###` section at the top of `## Unreleased`. Every merge left every
  other open pull request DIRTY on that one hunk. #1136 (commit a5f88f2a) replaced the whole 8,187-line file with 31 lines,
  and CI caught it. GitHub's mergeability check ignores `.gitattributes` merge drivers, so `merge=union` is no fix. A new file
  never conflicts.
- **The rule.** A change adds `changelog.d/<pr-or-slug>.md`: its `### Title` and body, exactly what it would have put under
  `## Unreleased`. That section's body is now one pointer paragraph. The release
  (`scripts/changelog_fragments.py --release`) writes the fragments into the new version's section, newest first by the
  commit that added each, and deletes them. `--render` previews the section.
- **CI** (discoverability job): `scripts/changelog_fragments.py --check --base <PR base>` refuses entries under `## Unreleased`
  and malformed fragments. It also checks that released history is append-only: every line of the merge base's released sections
  survives, in order, and an unreleased fragment leaves only by being released. It flags a5f88f2a (all 7,959 released lines
  lost). Replayed over the last 400 first-parent commits to `CHANGELOG.md`, it flags four, each a deliberate edit: two
  misplaced entries moved, a conflict-marker repair, and a redaction of owner quotes. A deliberate edit passes with the
  `changelog-history-edit` label.
- **Migration.** Every section under `## Unreleased` moved, unchanged, into `changelog.d/<original PR>-<slug>.md`. Joined in
  their old order they reproduce the old section byte for byte. The released sections are byte-identical (678,465 bytes).
- `scripts/check_change_impact.py` (shared; grouped-nf4-gemm first) accepts a fragment as the `CHANGELOG.md` companion where a
  repository keeps `changelog.d/`. A version bump still needs `CHANGELOG.md`, because the release writes it.
