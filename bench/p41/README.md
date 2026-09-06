# p41 — training seq × rank sweep, downstream-task instrument, 32 GB capability-boundary fixture (design only)

Pre-registration for lane **p41**: [`P41-PREREG.md`](P41-PREREG.md), written 2026-09-06 05:08Z under issue [#433](https://github.com/pjordanandrsn/experts4bit-qlora/issues/433) by the CXO seat (`[kimi · CXO · mbp · kimi-k3]`), in the form of `bench/h2h-20260906/tp2/P40-PREREG.md` and `bench/h2h-20260905/p37/PREREG.md`.

**Status: DESIGN ONLY — nothing has been rented, run or measured.** This directory holds the pre-registration and nothing else. The lane launches only after Warden read-only review + a Claude Code review on the PR and Jordan's `APPROVE P41` on #433; runs go through the launcher of #430 once it exists, else the CEO, with receipts to `docs/run-receipt-schema.json` (landing via PR #432) under `bench/runs/` and a ledger line.

What P41 registers when it runs (three sub-lanes, one fixture lineage — P38's fixture as P40 generalised it):

- **p41a** — the seq ∈ {512, 1024, 2048} × rank ∈ {8, 16, 32} training sweep per family (qwen3, mixtral, granite, olmoe; gemma4 only if #426 has merged; gpt-oss REFUSED), e4b reference/fused arms everywhere, Unsloth where its tp2 anchor was VALID. The (512, 8) cell of each family is the tp2 cross-lane anchor (±10 % or STOP).
- **p41b** — one downstream-task instrument beside perplexity: greedy ROUGE-L on the 48 held-out clinical rows, with a permuted-weights positive control that must fire before any reading is quoted.
- **p41c** — the 32 GB capability-boundary fixture: the sweep's peak-VRAM columns plus a seq-4096 probe per family, registered as `.footprint`-style rows, never speed ratios.

Cost plan: five runs, ≈ 16 h wall, ≈ $17 estimated (≤ $24 predicted bound), every run ≤ $20 in the approval bands of `docs/compute-policy.json` (PR #432; confirmed by Jordan 2026-09-06T04:55Z), $35/run hard cap, inside the $100/day global budget. STOP rules, validity vocabulary, and the PP1–PP8 prediction set are in the pre-registration; a FAIL stays FAIL and predictions are never edited after the fact.

When the lane has run, this README is replaced by the receipt-bundle README (the tp2/p37 form) in the receipts PR — receipts and code/claims land separately.
