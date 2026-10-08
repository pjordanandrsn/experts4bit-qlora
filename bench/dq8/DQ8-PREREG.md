# DQ8 — dense execution at the 24 GB boundary

Registration proposal; no compute or measurement in this change. References #1083, DQ7 (PR1359) and
Loggetta D7 (PR22). DQ8 supplies the independent 24 GB-card reading required before the dense executor opt-in is removed.

Use the exact synthetic public-config subjects and per-tensor seed policy from DQ7. These are architecture-only
checkpoints, not pretrained models; no loss or quality claim. Pin the merged DQ7 e4b harness, Loggetta merge
`1dd862f73433b5d9fdd42d835dac7504726574ae`, GNf4 `6ee2e10408161a9d3c874975c9191a7f2957e6f4`,
torch2.8.0+cu128, bnb0.50.2, transformers5.18.0 and PEFT0.21.2. No D7 calibration observations enter these plans.

## Registered arms

One ordinary verified RTX4090 with nominal 24 GB, gen4/x16 or better, normal host RAM >=98 GB.
Run DQ7's exact deterministic tiny CUDA proof first, including cuBLAS workspace before initialization. A failure
stops the lane with FUNCTION_FAIL. No capacity or timing evidence is taken from the A2000 rehearsal.

Five completed training arms, each in a fresh process at seq4096:

- Qwen3-14B: resident and streamed.
- Llama-3.1-8B: resident and streamed.
- Qwen3-32B: streamed.

Each uses the DQ7 setup: NF4/doublequant64/bf16compute, fp32 LoRA r16/alpha32 on all four roles (seven projections),
SDPA, non-reentrant checkpointing, micro-batch1/grad-accum1, two AdamW steps at constant2e-4/clip1, default allocator.
Qwen loss chunk512; Llama stock logits/loss chunk0. Check exact config hashes and full4096 real-token rows with no
padding, pair row hashes, planned/executed setup, all pinned homes and every streamed layer/late-bound projection.

Sixth arm: Qwen3-32B resident at seq4096, with the same setup and no observations, must be REFUSED by the planner
before tokenizer, weights or optimizer allocation. Save the complete refused plan and actual hardware probe.
If unexpectedly feasible, record BOUNDARY_CHANGED and return to review; this registration does not license forcing
an infeasible execution or adding an unplanned measurement. For the five training arms, an infeasible plan stops
with PLAN_REFUSED and no partial capacity pass.

## Gates fixed before the data

The decisive allocator gate for each completed arm is unchanged estimate E >= measured training allocator peak A.
One byte below is ESTIMATE_UNDER. Separately record measured reserved peak, driver peak, load peak and itemized
estimate lines. The unchanged total plan must cover reserved bytes plus its stated context charge; report that
headroom test separately from the allocator gate. No fitted reserve, coefficient or 5090-card transfer is permitted.

Proof PASS + every completed arm passing allocator and reserved-headroom gates + expected refused control licenses BOUNDARY_PASS for this
registered card/setup/subject/rung only. Missing/duplicate arms, hardware outside the card band, changed software,
setup/row/config or unengaged streaming are VOID. Failed proof is FUNCTION_FAIL; insufficient reserved-headroom is RESERVE_UNDER.
Report every failed gate and receipt;
no redraw, threshold adjustment or tuning is authorized. This is memory/execution evidence, not a performance claim.

## Lifecycle and budget

One draw using existing pod-launch/TC1 nonce, guard, heartbeat, fetch and teardown machinery. RTX4090, two-hour guard,
total-hourly ceiling $0.60 including the guarded provider's 320 GB disk order; runtime ceiling $1.20.
The existing launcher additionally reserves its default 100 GB download allowance at $0.011/GB ($1.10),
so the ledger reservation is $2.30. Synthetic checkpoints are generated locally; preserve actual provider invoices
for image/dependency transfer and storage rather than treating the reservation as a measured cost. The actual provider
quote must fit the standing no-ask policy and ledger. No additional draw follows a failure without reviewed proposal.
Prelaunch tests must exercise the real planner controls, reducer mutation failures, wrong-card refusal before install,
runtime/config checks and shell syntax. Stage no credentials; deterministic synthetic shards stay under excluded hf-cache.
Fetch original proof/plans/receipts/logs/reducer output, prove teardown, commit receipt and ledger together, and publish
immutable checksummed evidence even on a failure. Maintainer review and merge precede launch.

The harness and tests must be present before this proposal is marked ready. The final dense gate-removal/release
remains a separate evidence-linked Loggetta change with DENSE/TRAINING/README and estimate register rows updated.
