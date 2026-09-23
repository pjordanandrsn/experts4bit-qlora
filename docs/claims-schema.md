# Claims register — schema

**One schema for both registers.** This file is byte-identical in
experts4bit-qlora and grouped-nf4-gemm, and so is its checker,
`scripts/check_claims_register.py` (both listed in `SHARED` in
`scripts/check_shared_tooling.py`). Until 2026-09-23 each repository kept its own
copy of both, and they had drifted into two schemas that refused each other's
data. They were converged that day: every rule below is enforced in both
repositories, and both registers were migrated to it (the last section says what
moved). The prose that quotes claim ids is checked separately, by
`scripts/check_readme_claims.py`.

A claim is one sentence a reader could act on, with a number where there is one.
Every README / docs number maps to a claim. `docs/claims.json` holds them:

```json
{
  "schema": "docs/claims-schema.md",
  "package": "experts4bit-qlora",
  "generated": "2026-09-05",
  "status_vocabulary": {"measured": "one or more runs, receipt public in this repo", "...": "..."},
  "claims": [ { "id": "e4b.serve.b1.qwen3-30b.nf4.5090", "...": "..." } ]
}
```

Those five top-level keys are the only ones. `status_vocabulary` names the
statuses this register uses, each a status of this schema, with a one-line
gloss. Ids are unique, never reused, and share the register's namespace
(`e4b.` / `gnf4.`).

## Fields

A row carries only these fields. The checker refuses any other key, so a typo is
a finding rather than an ignored field.

| field | when | what |
|---|---|---|
| `id` | always | stable slug in the register's namespace, never reused |
| `status` | always | a status below |
| `claim` | always | one sentence, present tense, the thing a user gets |
| `area` | optional | `train` · `offload` · `serve` · `kernel` · `parity` · `quality` · `provenance` · `portability` · `roadmap` |
| `package` | optional | this repository's package name, when written |
| `value`, `unit` | quantitative claims | the headline number and its unit |
| `model`, `hardware`, `conditions` | optional | what the number was measured on and how |
| `measured_on` | measured / measured-private / confirmed / verified | ISO date of the run (below) |
| `tier` | optional, legacy | `confirmed` · `measured` · `projected` (a repository's older tier word) |
| `evidence` | public-run statuses (non-empty) | public receipts (below) |
| `evidence_private` | measured-private (non-empty) | where a private receipt lives: strings, listed and never resolved |
| `supersedes`, `superseded_by` | successors | below |
| `retired_reason`, `retired_on` | retired rows | the sentence, with the measurement that retired the claim; its date |
| `licensed_by`, `pack_fingerprint` | licence labels | below |
| `quoted_in` | optional | locations where the claim is quoted (below) |
| `validity`, `row_status`, `parity_verdict`, `row_reason` | lane arms | below |
| `notes` | optional | caveats, refused arms, corrections — never the headline |

## Statuses

The statuses are these eight. A register's own `status_vocabulary` glosses the
ones it uses, and the system manifest's `evidence_vocabulary` defines them for
readers of both.

- **verified** — reproduced under stated conditions, with a public receipt in this
  repository.
- **confirmed** — pre-registered, OpenTimestamps-stamped blind confirmatory run.
  The protocol and pass/fail criteria were fixed before the data.
- **measured** — one or more runs, with a committed receipt in this repository.
- **measured-private** — the run happened and its receipt exists, but only in a
  private audit tree, so a reader of this repository cannot check it. Prose must
  flag it as such, or the receipt must be published.
- **projected** — arithmetic, not a run.
- **retired** — was published and is now known to be wrong. The row is kept so the
  retraction can be found; it is never deleted.
- **superseded** — still true as measured, but a later row is the number to quote.
- **open** — a question the docs raise that has no evidence either way yet.

*Active* means verified, confirmed, measured or measured-private: a row presented
as current.

## Locations

An `evidence` path and every `quoted_in` entry are **locations**, written `path`
or `path#anchor`.

- The **path** is a file in the git tree at HEAD. The checker reads `git ls-files`,
  not the working tree, because `*.log` is gitignored and a receipt log becomes
  evidence only once it is force-added. The path is relative and never goes
  through `..`. It is never a directory: cite the directory's index or manifest
  file instead. It carries no annotation and no glob characters. Outside a git
  checkout the working tree stands in, and the output says so.
- The **anchor** is either `L<n>` / `L<n>-L<m>`, lines the file has, or the anchor
  **github.com itself** gives a Markdown heading in that file. That anchor is the
  heading's rendered text, lower-cased, with every character other than letters,
  digits, `_`, `-` and spaces removed, and spaces turned into `-`. A repeated
  heading gets `-1`, `-2`, …. So `## 0.24.0 — 2026-08-31` is `#0240--2026-08-31`,
  and `## 10. Energy — measured ([bench/_upstream/…](…))` is
  `#10-energy--measured-bench_upstream…`: link text is kept, the target is
  dropped, and the underscore stays. A non-Markdown file takes only line anchors.

Locations are real links, and that is the point of this rule. The consumer site
links a row's `evidence[0]` at the pinned commit, and GitHub scrolls to exactly
this anchor. The checker's anchors were verified against the anchors github.com
rendered for 371 headings across both repositories' READMEs, CHANGELOGs and docs,
with every one identical.

## Evidence

Each `evidence[]` entry takes one of three forms:

| form | meaning | resolved how |
|---|---|---|
| `"bench/p58/RESULTS-p58.md"`, `"CHANGELOG.md#0301--2026-09-05"` | a receipt in this repository | a location (above) |
| `{"url": "https://github.com/<owner>/<repo>/issues/N"}` | an issue or pull request, for `open` items and refusals tracked there | an issue / pull URL of one of the system's repositories (`docs/system-manifest.json` `packages[].repository`); never fetched |
| `{"repository": "grouped-nf4-gemm", "path": "kernel/RESULTS.md"}` | a receipt in the other package's repository | `repository` is a package of the system other than this one; `path` is a location resolved in a `--sibling` checkout of it, and listed as SKIP without one |

- **Rows with a public run.** `measured`, `confirmed` and `verified` rows need a
  non-empty `evidence`, and its **first** entry is a location string: the receipt a
  reader lands on.
- **Private runs.** `measured-private` rows need a non-empty `evidence_private`.
- **Everything else is refused.** Free text, globs, URL strings, notes inside an
  entry, or any other object shape is a finding. What a path was run with goes in
  `notes`, and a script that was never committed is not evidence: say so in
  `notes` and drop it.

## Dates

`measured_on` is required on `measured`, `measured-private`, `confirmed` and
`verified` rows. Wherever it is present, it is an ISO calendar date `YYYY-MM-DD`:
a month is not a date, and `null` is not a date (omit the field). It is the date
of the run the numbers come from: the receipt's own stated date or, where the
receipt states none, the receipt's first commit, which `notes` then records.

## Successors and retirements

- **Superseded rows.** A `superseded` row carries `superseded_by`. It names the
  ACTIVE row to quote instead, **directly**. When that row is itself superseded
  later, every row that pointed at it is re-pointed to the new active row; there
  are no chains to follow.
- **Retired rows.** A `retired` row carries a non-empty `retired_reason`, and no
  other row does. It may carry `superseded_by` when a later row restates the claim
  correctly; that pointer follows the same rule. A restatement with no receipt of
  its own is not a successor, so the old row stays `retired`, not `superseded`.
- **Active rows** never carry `superseded_by`.
- **Links go both ways.** The row a `superseded_by` names lists the old id in its
  `supersedes`. Every `supersedes` entry exists, is superseded or retired, and
  names this row as its `superseded_by`.

## Quotes (`quoted_in`)

`quoted_in` lists locations where the claim is quoted:
`README.md#what-is-measured`, `CHANGELOG.md#0280--2026-09-03`,
`docs/METHODOLOGY.md#L13`. Each entry is a location (above). It is a location,
not a containment check: the file need not spell the id on that line. When a
quote moves, the entry moves. When the quote is gone, the entry goes, and `notes`
may keep the history.

## No placeholder on an active row

The `claim` and `notes` of an ACTIVE row never say `pending`, `TBD` or `TODO`.
An active row is presented as current, so it states the fact, with ids, or the
row becomes `open`. No check reads numbers from `notes`: the headline numbers are
`value`, `unit` and the `claim` sentence.

## Licence labels (`licensed_by`)

An ACTIVE row whose `claim` asserts a licence carries `licensed_by`. A licence is
asserted by any occurrence of the word "licensed" that is not the citation form
below and that no negation immediately precedes ("not licensed", "not a licensed",
"never licensed", "no licensed"; "unlicensed" is its own word). `licensed_by` is
the id of the ACTIVE claim whose receipt holds the verdict that licenses the
configuration: the two-text pass for a calibrated pack, the one-text pass for an
uncalibrated one, in the gate's own units. A row whose own receipt carries the
verdict names itself.

- **The rule is per occurrence.** A sentence that says "unlicensed" of one arm and
  "the licensed stack" of another still asserts the second. A VOID or FAIL row
  says "not licensed" or "unlicensed" of itself.
- **No instrument, no licence.** A family with no instrument (Gemma-4, gpt-oss on
  raw text) has no verdict row. So no active sentence about it may say "licensed";
  it says "measured, not licensed".
- **A later FAIL rewords the earlier row.** When a later row records FAIL for the
  same configuration class, the earlier sentence is reworded, or superseded by the
  licensed configuration's row. It is never left saying "best licensed".
- **Only a licensed configuration carries the label.** `licensed_by` never goes on
  a row whose own configuration is not licensed.

**Citing another row's licence.** ``licensed by `<id>` `` is a citation, not an
assertion. It refers to that row's licence, and the citing row needs no
`licensed_by` for it. `<id>` must be a row of this register that itself carries
`licensed_by` (a licensed row, or a verdict row). Every citation is resolved, in
`claim` and in `notes`.

**Pack identity (`pack_fingerprint`).** A calibrated serving licence is a
property of pack bytes, not of the calibration recipe. When present,
`pack_fingerprint` is `sha256:<64 lowercase hex>`. It is the root hash of a
canonical pack manifest (`experts4bit_qlora.engines.pack_manifest`), which records
the ordered `(path, size, sha256)` of the packed tensors, the scales, the identity
payload and, for a calibrated artifact, the recorded gptq/rtn assignment.

- **An artifact-backed licence names the bytes on both rows.** An ACTIVE row with
  `licensed_by` becomes artifact-backed once either it or its verdict carries the
  field. From then on both carry it, and the two are equal.
- **Observations may record without licensing.** Unlicensed and VOID observations
  may record the fingerprint they observed without `licensed_by`.
- **Never invent a hash.** Licence rows without a published artifact omit the
  field.

## Lane fields (`validity`, `row_status`, `parity_verdict`, `row_reason`)

Rows that are one arm of a pre-registered lane carry the lane reducer's verdicts
beside `status`. `status` says what kind of evidence the row is; these fields say
what the lane made of it. They are copied from the receipt and never edited by
hand, and the `claim` sentence opens with them in brackets (`[VALID]`,
`[OK · VOID]`, `[REFUSED]`). Only `status` decides whether a row may back a
capability or a README number.

- **`validity`** is `VALID` or `VOID`: whether the arm counts under the lane's
  rules. A VOID row is measured but is never a position, and no ratio is derived
  from it.
- **`row_status`** says whether the arm ran:
  - `OK`: it ran and wrote its receipt.
  - `HARNESS_ERROR`: it died before a receipt, and `row_reason` says where.
  - `REFUSED`: the path refused the configuration by design.
  - `EXPERIMENTAL`: it ran on an experimental path and licenses nothing.
- **`parity_verdict`** is the arm's loss-parity verdict against its family's
  reference arm:
  - `REF`: this arm is the reference.
  - `PASS`: inside the registered band.
  - `VOID`: the pair cannot be read as parity.
  - `no pair`: no reference arm exists.
  - `null`: no verdict.
- **`row_reason`** is free text beside a non-OK `row_status` or a re-run attempt.

## What moved in the 2026-09-23 convergence

Each repository's data was migrated to the rules above. Every migrated entry
still resolves, and the free text that had to leave a field was kept verbatim in
the row's `notes`.

- **grouped-nf4-gemm.**
  - `{"path", "section"}` objects became `path#anchor` locations.
  - URL strings became `{"url"}` objects. The bare issue-tracker URL on
    `gnf4.open.issues` became one URL per open issue.
  - The directory `kernel/receipts-m3/` became its index,
    `kernel/receipts-m3/m3_manifest.txt`.
  - Heading-prefix quotes became GitHub anchors. Two K17/K18 entries named
    `Unreleased`; they had been resolving, under the old prefix rule, to an
    unrelated historical `## Unreleased` heading. They now name the 0.33.0 entry,
    where that text shipped.
- **experts4bit-qlora.**
  - Free-text `quoted_in` entries became locations. Entries whose file no longer
    quotes the id were dropped, and their text kept in `notes`.
  - One evidence fragment was fixed. `docs/METHODOLOGY.md#10-energy--…` had dropped
    an underscore GitHub keeps, so the link had never scrolled to its section.
  - The cross-repository entry names the package, not the GitHub slug.
  - The successor graph was made direct and bidirectional. The gemma4 parity chain
    now points straight at its active row. Three retired or superseded rows gained
    the back-links their successors lacked.

Human-readable companion: each repository's `docs/STATUS.md` renders the register
as "what you get today" (active), "what changed" (superseded and retired, with the
one-line reason) and "what is open".
