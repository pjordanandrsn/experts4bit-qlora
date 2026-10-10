### SD2 registered (#1313): SD1 Phase 1, an EAGLE-3 speculative decode path in `serve_paged`, measured end to end

- **The registration.** `bench/sd2/PREREG-sd2.md` registers the build and its read on one RTX 5090. It replaces SD1's
  modelled speedup with measured terms: the verify step at k + 1 rows, the in-engine EAGLE-3 draft, the loop's own cost,
  and decode tokens per second with speculation on against off.
- **The prior work it answers.** It cites S2-lite and S3 (2026-08-25), which refuted prompt-lookup speculation at 17–65
  verify rows on that stack. SD2 measures 2–4 rows first, and stage V can refute the lane on its own.
- **The quality gate.** P115 Phase B's instrument at the verify shape, against P110's bar.
- **The build.** Everything is behind `E4B_PAGED_SPEC` (default off) and lands in reviewed PRs. They are smoked together
  on a short 5090 proof before they merge.
