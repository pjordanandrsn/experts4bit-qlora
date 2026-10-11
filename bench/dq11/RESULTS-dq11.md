# DQ11: a valid matched-configuration read and a fusion build candidate

**Measured-private, 2026-10-11.** The run and its receipts exist in the private
audit tree; a reader of this repository cannot independently check them.
Claim: `e4b.train.dense-adapter.dq11.matched-sdpa.5090.2026-10-11`.

**U is faster than L in this matched SDPA, eager, full-logits configuration,
not Unsloth's default.** The reducer's scope is exactly: "against Unsloth in
this matched SDPA, eager, full-logits configuration, not Unsloth's default".
No default-Unsloth, capacity or shipping claim follows.

## What ran

Run `dq11-5090-4f` used one RTX 5090 and the reviewed instrument at
`13b55990433f357d63232f75926ce19f5bda5b42`, under the
[original registration](DQ11-PREREG.md) and its seven amendments, ending with
[matched SDPA attention and backward admission](DQ11-AMENDMENT-7.md).
The model was the registered Mistral-7B-v0.1 checkpoint, with the sealed tokens,
canonical FP32 adapter initializer and unchanged frozen NF4 weights.

| Arm | Executed path |
| --- | --- |
| L | The actually admitted Loggetta dense streamed plan, PEFT on bitsandbytes, and the shipped e4b training loop |
| U | Pinned Unsloth, resident, with adapter fusion enabled |
| U0 | The same Unsloth backbone with only adapter fusion disabled |

U/U0 restored checkpoint embedding row 0 after Unsloth's padding-token change.
They also used native SDPA with xformers and Flash Attention absent: attention
kernel differences were excluded by design. These are named comparator
deviations, not a reading of Unsloth's defaults.

Three configuration-C bitwise observer-removal proofs preceded the initial
quality gate. Six fresh processes then ran 40 AdamW updates each in the order
L/U/U0/U0/U/L, with five trained warmups excluded from steady-update summaries.
These are two reversed-order repetitions of one seed on one physical host.
Initial and trained quality passed the unchanged K8 absolute perplexity budget
on both Alpaca heldout and Wikitext test, for both registered comparisons in
both repetitions. Native attention canaries admitted forward and backward;
proofs used math SDPA, while reads used the registered shipped policy.
The profiler was removed before model loading and the timed loop.

## Decision and evidence

The actual merged reducer, supplied with verified teardown and instance
absence, returns `VALID_CONTROLLED_READ`, `quality_pass=true`,
`BUILD_CANDIDATE` and `U_FASTER_CONTROLLED`. Every deciding field was independently
re-derived from the raw receipts. `BUILD_CANDIDATE`
licenses a separately reviewed fusion build investigation; it does not mean
e4b already implements or ships the measured lever. The U0/U ablation measures
headroom on Unsloth's own backbone, separately from the L/U user-path comparison.

Optional zero-update shipped-policy gradient spreads ran last and remain
reporting-only. Their raw files are retained separately in the audit tree;
they neither retune the bitwise proof nor change the science verdict.

The untouched receipt SHA-256 is
`67dc2ad03f924bee68ffb82179412e74f35abbdbc4549a9640edd1cee8cf9029`.
Earlier refused and failed draws remain preserved in their original records.
No earlier registration is rewritten by this result.
