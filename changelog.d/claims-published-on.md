### Claims register: retired rows can say when the claim was published

A `retired` row may now carry `published_on: {date, evidence}`, the first public appearance of the claim it retracts,
so `published_on` to `retired_on` is how long a wrong claim stood. The evidence is `<commit>:<path>`, which the check
resolves: the commit is in this repository's history, the path is in its tree, and the date is its committer date in UTC. A
commit date is only a lower bound on first publication (git does not record the push), so a
lifetime read from it is an upper bound. A URL is also accepted; it is listed and never resolved. The field is refused on any other status and on a date after
`retired_on`. One row is backfilled: the 13.47x training speedup, published 2026-08-26 UTC (7718440d1) and retired
2026-08-31. The rest stay unset rather than guessed. `docs/claims-schema.md`, `scripts/check_claims_register.py` and
its test change in lockstep with grouped-nf4-gemm.
