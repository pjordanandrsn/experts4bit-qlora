# DQ11 amendment 7: matched SDPA attention and backward admission

Prospective registration for `dq11-5090-4c`, under the original preregistration
and Amendments 1–6. Earlier registrations and receipts remain immutable.
This amendment authorizes no quote or rental. Reviewed code on main, the
verification below, and a new exact-machine/offer maintainer ACK are required.

## Two strikes and the root cause

The last two paid attempts failed before a complete proof set: draw 3 refused
while hashing its scalar loss; draw 4b passed proof-L, then refused proof-U.
Draw 4b used `e8629240fed3c815b6bb7fd8f74902f92a9aad01`, instance 55296677,
machine 145701 and offer 54798130. Its xformers backward had no admissible
operator at SM12.0 with registered determinism enabled: FA2 rejected
determinism, while FA3 and CUTLASS rejected the architecture. Unsloth's SM12
admission probe tested forward only, so it admitted xformers without proving
backward support. No initial gate or timed read completed. Destroy returned
HTTP 200, followed by an authenticated empty instance list. Actual cost was
$0.224. Receipt and ledger were committed together in the receipts repository
at `09116ac941e0015eda49dabd01fdab93595377ba`.

## Executed subject

Remove only xformers from both exact install sets: 101 packages become 100,
with every remaining package version, URL and hash identical. Flash Attention
was already absent. The hash-verified Unsloth 2026.9.14 source selects SDPA
when both packages are absent; no library source or module is patched. Before
U/U0 import, refuse unless neither `xformers` nor `flash_attn` is importable.
Verify the actual Mistral forward, selector and dispatcher bindings against
the sealed wheel sources, and require the selector to return SDPA.

The comparison is now against Unsloth with SDPA attention matched to e4b's;
attention-kernel differences are excluded from the measured gap by design.
Name this deviation beside the existing checkpoint row-0 restoration in every
U/U0 receipt and eventual RESULTS. U0 still disables adapter fusion only.

Configuration C remains unchanged: cuBLAS workspace `:4096:8`, deterministic
algorithms with `warn_only=True`, forced math SDPA and a bitwise observer
removal proof. Reads and optional spreads retain the registered shipped SDPA
policy with all backends enabled and determinism disabled. Record actual
dispatch per phase; do not force math in timed reads.

## Required admission and verification

After bootstrap and before model fetch, run one untimed forward/backward
through the exact Unsloth Mistral `run_attention` binding at registered shape:
B1, T2048, 32 query heads, 8 KV heads, D128, BF16, causal, no packing or padding
mask, with gradients enabled. Preserve the native forward and backward event
names and their parent scopes, finite Q/K/V gradients, selected backend,
settings, source bindings and package absence. C requires actual SDPA math
forward plus its decomposed softmax/BMM backward. Refuse missing, ambiguous,
unmatched or xformers dispatch. The early canary has a 120-second cap inside
the existing deadline minus reserve.

Repeat this untimed canary before full model loading in every U/U0 process:
C for proofs, shipped policy for reads/spreads. Save it immediately and embed
it in the phase receipt. Remove the profiler before model loading; never
profile the timed loop. The reducer independently validates native events and
receipt identity. The full proof's sealed callable census covers the model's
attention binding.

Before another rental, require an isolated legacy-101 CPU regression calling
actual xformers 0.0.35 backward `.not_supported_reasons` and its dispatcher
with mocked SM12.0 and C, proving no admissible op. Paired controls must admit
FA2 without determinism on SM12 and CUTLASS under C on SM86. Keep this test
environment outside the paid install set. Require a fresh full A2000 fixed-arm
correctness rehearsal and real-model initial quality read, since changing
attention can change scores. These are $0 correctness evidence; they cannot
certify 5090/cu130 kernels, speed, capacity or shipping.

## Remaining budget and unchanged decisions

Aggregate actual spend is $0.459 of the original $2.800 ceiling, leaving
**$2.341**. Replace the previous 100 GB/$1.100 transport reservation with the
exact sealed input bytes, a 1.5× margin and the registered $0.011/GB cap:

| Input | Exact bytes | Evidence |
| --- | ---: | --- |
| Full registered model, all nine files | 14,485,813,893 | `model_files.json` |
| All 100 cu130 wheels | 2,954,832,429 | Actual cached file sizes; every SHA-256 matches `wheels.json` |
| Canonical tokens | 554,866 | `locked_inputs.json`, verified canonical asset |
| Canonical initializer | 167,816,528 | `locked_inputs.json`, verified canonical asset |
| Total | **17,609,017,716** | Sum of the four exact inputs |

The per-wheel byte/hash receipt is a private review packet,
SHA-256 `9b911fb101a0a91ec52205d116dbbbbd0f1d955cc337710647691595578d75e3`;
an independent readback matches. It records the exact 100 science-lock
filenames, actual file sizes and verified hashes.

Use decimal GB (1,000,000,000 bytes):
17,609,017,716 × 1.5 / 1,000,000,000 = 26.413526574 GB.
At $0.011/GB this is $0.290548792314; round the reservation **up to $0.291**.
The two-hour runtime plus 320 GB storage must therefore cost at most
($2.341 − $0.291) / 2 = **$1.025/hour** combined. The draw-4b provider receipt
recorded 16.8 GB downloaded and $0.131 at $0.0078/GB; that observation is
context, not the reservation basis. The new margin is prospective and does
not change either historical receipts or the aggregate ceiling.

Retain the two-hour guard,
300-second receipt/teardown reserve, normal 32 GB RTX 5090, driver at least
580, excluded machine 45511, secure provider and exact machine/offer pins.
Require a fresh all-100-URL and actual CPU policy gate at merged source before
quotes, and repeat it at launch. Refusal does not authorize substitution.

Keep the original model, tokens, initializer, full frozen equality, all three
C proofs before six fresh reads, 40 AdamW updates, five trained warmups,
initial/trained K8 absolute |delta PPL| <= 0.05 on both texts and both
comparisons/repetitions, both 5% decision bars, and report-only optional
spreads last. QUALITY_FAIL licenses no recommendation. No default-Unsloth,
capacity or shipping claim follows.
