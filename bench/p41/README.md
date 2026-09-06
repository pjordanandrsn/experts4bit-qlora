# p41 — training seq × rank sweep, downstream-task instrument, 32 GB capability-boundary fixture (design only)

Pre-registration for lane **p41**: [`P41-PREREG.md`](P41-PREREG.md), written 2026-09-06 05:08Z under issue [#433](https://github.com/pjordanandrsn/experts4bit-qlora/issues/433) by the CXO seat (`[kimi · CXO · mbp · kimi-k3]`), in the form of `bench/h2h-20260906/tp2/P40-PREREG.md` and `bench/h2h-20260905/p37/PREREG.md`.

**Status: DESIGN ONLY — nothing has been rented, run or measured.** This directory holds the pre-registration and nothing else. The lane launches only after Warden read-only review + a Claude Code review on the PR and Jordan's `APPROVE P41` on #433; runs go through the launcher of #430 once it exists, else the CEO, with receipts to `docs/run-receipt-schema.json` (landing via PR #432) under `bench/runs/` and a ledger line.

What P41 registers when it runs (three sub-lanes, one fixture lineage — P38's fixture as P40 generalised it):

- **p41a** — the seq ∈ {512, 1024, 2048} × rank ∈ {8, 16, 32} training sweep per family (qwen3, mixtral, granite, olmoe; gemma4 only if #426 has merged; gpt-oss REFUSED), e4b reference/fused arms everywhere, Unsloth where its tp2 anchor was VALID. The (512, 8) cell of each family is the tp2 cross-lane anchor (±10 % or STOP).
- **p41b** — one downstream-task instrument beside perplexity: greedy ROUGE-L on the 48 held-out clinical rows, with a permuted-weights positive control that must fire before any reading is quoted.
- **p41c** — the 32 GB capability-boundary fixture: the sweep's peak-VRAM columns plus a seq-4096 probe per family, registered as `.footprint`-style rows, never speed ratios.

Cost plan: five runs, ≈ 16 h wall, ≈ $17 estimated (≤ $24 predicted bound), every run ≤ $20 in the approval bands of `docs/compute-policy.json` (PR #432; confirmed by Jordan 2026-09-06T04:55Z), $35/run hard cap, inside the $100/day global budget. STOP rules, validity vocabulary, and the PP1–PP8 prediction set are in the pre-registration; a FAIL stays FAIL and predictions are never edited after the fact.

When the lane has run, this README is replaced by the receipt-bundle README (the tp2/p37 form) in the receipts PR — receipts and code/claims land separately.


## Drivers (R1 onward; e4b#433, executed under the CTO role)

- `p41_run.sh` — the box-side lane script in tp2's proven form: e4b from PyPI at the shipped cut (versions recorded), helpers from the archive tarball at that cut, the train-anchor gate, the sha-verified clinical set, tokenisation per seq via `bench/tp3/tp3_arm.py --prepare`, then the family's grid seq-ascending / rank-ascending with the (512, 8) anchor first, fused_attn4 (PRIMARY) then reference_attn4 per cell, `alpha = 2r`, `--expect-trainable = (r/8) × the tp2 anchor count`, `--attn-4bit 1`, one process / one JSON / one `perl alarm` per arm from the planning curve, the seq-4096 p41c probe, stubs for every non-run, `TP_DONE`. STOP-1 (anchor ±10 %), STOP-4 (projected spend > 1.5 × estimate) and STOP-5 (80 % of the guard) are enforced on the box. `p41_run.sh --plan` prints the arm plan without a box (the tests read it).
- `p41_drive.sh` — the controller-side `rent.py --command`: reads the box from the launcher's `E4B_RENT_*` environment (e4b#464), stages the harness and the lane script, starts the lane detached, polls `TP_DONE` while the launcher keeps the heartbeat fresh, rsyncs receipts/stubs/logs/forensics/versions into the run directory. Exit 0 = the lane finished; anything else is HARNESS_ERROR in the receipt. It never creates, approves or tears down compute.
- Receipts land in `bench/runs/<date>/<run-id>/p41/` beside the launcher's `receipt.json`; the reducer and the claims come in the receipts PR.
