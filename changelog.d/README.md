# changelog.d — one file per unreleased change

Every change that would have added a section under `## Unreleased` in
[`CHANGELOG.md`](../CHANGELOG.md) adds one file here instead. Never edit
`CHANGELOG.md`'s `## Unreleased` by hand: its body is one pointer paragraph,
and CI fails a pull request that adds entries there.

**Name:** `<pr-or-slug>.md`, for example `1150.md` or
`1150-tc1-amendment-40-read.md` (letters, digits, `.`, `_`, `-`). New files
never conflict, so pick any name nobody else has.

**Content:** exactly the section you would have written under `## Unreleased`:
the first line is its `### Title`, the body follows, and there is no other
`###`, `##` or `#` heading (use `####` inside it). One change, one file. To
correct an unreleased entry, edit its file.

**Release:** the maintainer runs
`python scripts/changelog_fragments.py --release "## X.Y.Z — YYYY-MM-DD — title"`,
which writes these files, newest first by the commit that added each one,
into the new section of `CHANGELOG.md` and deletes them. Fragments added by
the same commit go in reverse name order. Released sections are append-only:
CI fails a pull request that removes or edits a released line, unless it
carries the `changelog-history-edit` label.

Preview the assembled section with
`python scripts/changelog_fragments.py --render`; check your fragment with
`python scripts/changelog_fragments.py --check --base origin/main`.
