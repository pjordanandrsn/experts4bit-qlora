# Claims register — schema (draft for docs/claims.json in both repos)

One JSON object per claim. A claim is one sentence a reader could act on,
with a number where there is one. Every README/docs number must map to an
entry; CI can enforce that later.

```json
{
  "id": "e4b.serve.b1.qwen3-30b.nf4.5090",         // stable slug, never reused
  "package": "experts4bit-qlora" | "grouped-nf4-gemm",
  "area": "train" | "offload" | "serve" | "kernel" | "parity" | "provenance" | "portability",
  "claim": "one sentence, present tense, the thing a user gets",
  "value": 98.3, "unit": "tok/s",                    // omit for qualitative claims
  "model": "Qwen/Qwen3-30B-A3B", "hardware": "RTX 5090 (sm_120), rented Vast host",
  "conditions": "B=1, NF4 experts, fp8 paged KV, 512-token prompt, --no-fuse-qkv",
  "measured_on": "2026-09-03",
  "status": "verified" | "measured" | "projected" | "retired" | "superseded" | "open",
  "tier": "confirmed" | "measured" | "projected",   // repo's existing evidence tiers
  "evidence": ["bench/hybrid-g9/b1/RESULTS-b1-decomposition.md"],  // PUBLIC files in the git tree; see "Evidence entries"
  "evidence_private": ["INT4B16/P25-PARITY.md"],    // exists but not in this repo -- reader cannot check it
  "supersedes": ["<id>"], "superseded_by": "<id>",
  "retired_reason": "why, in one sentence, with the measurement that retired it",
  "licensed_by": "<id>",                             // the K8 verdict row behind a licence label; see below
  "quoted_in": ["README.md#L45", "docs/METHODOLOGY.md#13"],
  "validity": "VALID" | "VOID",                      // lane arms only; see "Lane fields"
  "row_status": "OK" | "HARNESS_ERROR" | "REFUSED" | "EXPERIMENTAL",
  "parity_verdict": "REF" | "PASS" | "VOID" | "no pair" | null,
  "row_reason": "free text beside a non-OK row_status or a re-run attempt"
}
```

## What the register check enforces (`scripts/check_claims_register.py`, CI)

Added 2026-09-05 after an audit found every check keying on `status` alone.
Each rule below is mechanical and fails the discoverability job.

**Evidence entries.** Every element of `evidence[]` is one of:

- a bare repository-relative path of a FILE in the git tree, optionally with
  a `#fragment` (`docs/METHODOLOGY.md#10`). The check reads `git ls-files`,
  not the working tree: `*.log` is gitignored, so a receipt log is evidence
  once it is force-added and never before; a directory, an absolute path or
  a path through `..` is a finding (outside a git checkout the working tree
  stands in and the output says so). No free text, globs or annotations --
  what a path was run with goes in `notes`, and a script that was never
  committed is not evidence (say so in `notes` and drop it);
- `{"repository": "owner/name", "path": "kernel/RESULTS.md"}` -- a file in
  another repository (the kernel package's receipts); verified against a
  checkout when the check is given `--sibling PATH` and that checkout's
  slug matches, otherwise accepted by shape (a sibling whose slug cannot be
  resolved fails the check rather than skipping these silently);
- `{"url": "https://github.com/owner/name/issues/N"}` -- an issue or pull
  request on github.com, for `open` items and refusals tracked there.

**Dates.** `measured`, `measured-private`, `verified` and `confirmed` rows
carry `measured_on` as `YYYY-MM-DD`. When the run's own date is not on
record, the receipt's date is used and `notes` says so.

**Successors.** A `superseded` row carries `superseded_by`; following it
(through other superseded rows, never a cycle) reaches an ACTIVE row. A
`retired` row that names what replaced it carries `superseded_by` too, and
it resolves the same way; an ACTIVE row never carries `superseded_by`. A
`retired` row carries `retired_reason` and no other row does -- one status
per row. Every `supersedes` id exists. A restatement with no receipt of its
own is not a successor: the old row is `retired`, not `superseded`.

**Quotes.** Each `quoted_in` entry is `<path>[#fragment][ free text]` and the
path exists (`CHANGELOG.md 0.28.0`, `README.md (results table)`).

**No pending on an active row.** `claim` and `notes` of an ACTIVE row never
say "pending" or "TBD": state what is measured, with ids, or open a row.

**Licence labels (`licensed_by`).** An ACTIVE row whose `claim` sentence
asserts a licence -- an occurrence of the word "licensed" that no negation
immediately precedes ("not licensed", "not a licensed", "never licensed",
"no licensed"; "unlicensed" is its own word) and that is not the citation
form below -- carries `licensed_by`: the
id of the ACTIVE claim whose receipt holds the K8
verdict that licenses the configuration (the two-text pass for a calibrated
pack, the one-text pass for an uncalibrated one, in the gate's own units).
A row whose own receipt carries the verdict names itself (the bo3 Granite
row's notes hold its +0.019 ppl pass; the bo6c Qwen3 row IS the verdict).
A family with no instrument (Gemma-4, gpt-oss on raw text) has no verdict
row, so no active sentence about it may say "licensed": it says "position
with the no-instrument caveat" or "measured, not licensed". When a later
row records FAIL for the same configuration class, the earlier sentence is
reworded (numbers unchanged) or the row is superseded by the row of the
configuration that is licensed -- never left saying "best licensed". The
rule is per occurrence: a sentence that says "unlicensed" of one arm and
"the licensed stack" of another still asserts the second (label it, cite
it, or say "the licence label" where no assertion is meant), and a VOID or
FAIL row says "not licensed" / "unlicensed" of itself. `licensed_by` says
the row's OWN configuration is licensed by that verdict; it never goes on a
row whose configuration is not (the p37 head-to-head quotes no licensed
ratio and carries none).

**Citing another row's licence.** The form ``licensed by `<id>` `` is a
citation, not an assertion: "no ratio against the stack licensed by
`e4b.serve.buildout.bo6c…`" refers to the bo6c verdict and needs no
`licensed_by` on the citing row. `<id>` must be a claim in the register
that itself carries `licensed_by` -- a licensed row, or a verdict row, which
names itself -- and the check resolves every citation in `claim` and in
`notes`: a missing id, or a row that carries no `licensed_by`, is a finding.

**Lane fields (`validity`, `row_status`, `parity_verdict`, `row_reason`).**
Rows that are one arm of a pre-registered lane (the p37 / p38 head-to-heads,
the tp1 training-parity matrix) carry the lane reducer's verdicts beside
`status`: `status` says what kind of evidence the row is (`measured`), these
say what the lane made of it. They are copied from the receipt, never edited
by hand, and the `claim` sentence opens with them in brackets (`[VALID]`,
`[OK · VOID]`, `[REFUSED]`). No check keys on them beyond the register's own
structure; `status` alone decides whether a row may back a capability or a
README number.

- `validity` -- `VALID` | `VOID`: whether the arm counts under the lane's
  pre-registered rules. `VOID` is measured, never a position (the p37 arms of
  the recipe whose pack fingerprint did not match and whose gate then
  failed); the sentence says why, and no ratio is derived from a VOID arm.
- `row_status` -- `OK` | `HARNESS_ERROR` | `REFUSED` | `EXPERIMENTAL`:
  whether the arm ran. `OK`: it ran and wrote its receipt. `HARNESS_ERROR`:
  the process died before a receipt; the attempt is kept as a row and
  `row_reason` says what died and where. `REFUSED`: the path refused the
  configuration by design (gpt-oss fused / batched training). `EXPERIMENTAL`:
  ran on an experimental path and licenses nothing.
- `parity_verdict` (tp1 rows) -- `REF` | `PASS` | `VOID` | `no pair` |
  `null`: the arm's loss-parity verdict against its family's reference arm
  on the same box. `REF`: the reference arm itself. `PASS`: inside the
  registered band. `VOID`: the pair cannot be read as parity (the batched
  arm reached its kernel fewer times per step than the rule requires).
  `no pair`: no reference arm to compare against (attention-only gpt-oss).
  `null`: no verdict -- the arm did not run, or was refused.
- `row_reason` -- free text beside a non-`OK` `row_status` or a re-run
  attempt: what happened, with the log line that records it.

Status meanings:
- **verified** — reproduced under stated conditions, receipt public in this repo.
- **measured** — one run, receipt public. (The repos' own tier language.)
- **measured-private** — the receipt exists only in the private audit tree; the
  number is real but a reader of this repo cannot check it. MUST be flagged in
  prose, or the receipt published.
- **projected** — arithmetic, not a run.
- **retired** — was published, now known wrong; keep the entry so the retraction
  is findable, never delete.
- **superseded** — still true as measured, but a later entry replaces it as the
  number to quote (e.g. v0 offload tok/s superseded by the paged engine).
- **open** — a claim the docs make that has no evidence either way yet.

Human-readable companion: docs/STATUS.md renders this as three lists —
"what you get today" (verified/measured), "what changed" (superseded/retired
with the one-line reason), "what is open".
