# DQ10 — frozen training reserve and new-family dense holdouts

Work: [#1397](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1397).
This prospective registration follows DQ9's known-subject diagnosis
([receipt PR1396](https://github.com/pjordanandrsn/experts4bit-qlora/pull/1396)). DQ7 stays VOID and original D7
cannot import its receipts. DQ9 licenses neither capacity nor calibration. No box has been acquired for DQ10.
Review and merged registration precede compute. This changes no shipped default, gate or activation coefficient.

## Frozen design and fitting closure

Loggetta [PR36](https://github.com/pjordanandrsn/loggetta/pull/36), merged as
`05ee04d6a4005cbf50bbcbda4d189a6869fe7f8b`, supplies the explicit `dq10` policy. Its canonical packaged artifact
is copied byte-identically to [dq10_policy.json](dq10_policy.json), SHA256
`766020d5d9b848e533d87bb64f231976c74b7c55b0a85946b1e002db04fb5d59`.
The runner verifies both the installed artifact and this staged copy. Fractions and context values live in that
artifact; no holdout derives, modifies or imports them.

[dq10_derive.py](dq10_derive.py) reads all sixteen original DQ9 receipts, requiring their fixed
[source checksums](dq9-source-sha256.json). It derives separately for each placement:
`f = max(1/5, max((R_peak - A_peak) / A_peak))`, using exact integer fractions, and
`C = max(512 MiB, max(D_sampled_peak - R_peak))` in bytes. It prints every fitting row with E, A, R, D,
`ceil(E*f)`, C, `E + ceil(E*f) + C` and each residual. These are noncontemporaneous peak comparisons, not
a census of simultaneous cache or CUDA-context tensors. No additional residual/intercept charge is fitted.

| placement | frozen f | frozen C, bytes |
|---|---:|---:|
| resident | 1/5 | 660602880 |
| streamed | 2197697/5494591 | 731906048 |

All sixteen fitting rows close at allocator, reserve component, reserved total, context component and sampled
driver total. The worst driver residual is -563596500 B. That is in-sample closure only. The loss
workspace correction remains the source-derived 12*T*V full-logit term, and DQ4's activation coefficient/brackets
remain unchanged. This registration tests a new hypothesis; it neither repairs the historical DQ7 verdict nor
quotes fitting closure as a holdout pass.

## Untouched new-family subjects

Both families are absent from DQ4/DQ7/DQ9. Only config metadata has been screened; no capacity observations or
pretrained weights have been taken for these subjects. Preserve each config byte for byte:

| subject | config source at immutable Hub revision | config SHA256 |
|---|---|---|
| mistral7b_v03 | [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/caa1feb0e54d415e2df31207e5f4e273e33509b1/config.json) | affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99 |
| smollm3_3b | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/config.json) | c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1 |

Mistral has 32 layers and vocab32768. SmolLM3 has 36 layers, vocab128256 and a tied embedding/head. Both use their
stock full-logit loss (`loss_chunk=0`); neither is in e4b's verified chunked-loss table. This includes a new family
with a large vocabulary on the full-loss path.

Synthetic checkpoint generation follows DQ7's one-tensor-at-a-time bf16 policy, but explicitly preserves aliases.
Build the meta tree, leave Accelerate's registration context, and call `tie_weights()`. Parameter object identity
with `named_parameters(remove_duplicate=False)` determines canonical names; never use meta data pointers.
For an alias, seed from its canonical name so independently written shards contain identical bytes. Seed is the
low eight little-endian bytes of SHA256(subject + ':' + canonical name), modulo 2^63-1. Matrices are normal bf16
draws scaled by initializer_range; one-dimensional weights are ones and biases zeros. Manifest each alias, seed,
shape, shard SHA and config SHA. Do not untie or change the public config to make the loader accept it.
No pretrained loss or quality result is reported.

## Correctness, then actual admission, then reading

Use one ordinary RTX5090, 30-33 GiB driver-reported capacity, driver595.91.07, gen5/x16, host RAM >=98 GB. The
wrong-card/driver check occurs before installation. [runtime.json](runtime.json) pins torch2.8.0+cu128,
bnb0.50.2, transformers5.18.0, PEFT0.21.2, Loggetta's reviewed merge and GNF4's v0.43.0 commit. The launch checkout
pins e4b. Verify installed direct_url source commits, package versions, bnb source mirror and policy SHA before proof.
Keep both allocator configuration environment variables unset before CUDA initialization; plans explicitly declare
`allocator_profile='default'`. No silent setup switch, policy fallback or post-hoc plan overlay is permitted.

Proofs: independently generated tiny Mistral and tiny tied SmolLM3, each 2 layers, hidden1024/intermediate4096,
heads8/KV4, vocab512, max positions4096. SmolLM3's two layer types are full_attention and no_rope_layers=[1,1].
For each family separately, fresh resident and streamed processes must have bitwise-equal losses, all 28 pre-clip
LoRA gradient tensors, two AdamW updates, and exported/reloaded logits and adapter tensors. Seed731, TF32 off,
deterministic algorithms, cuBLAS :4096:8 before CUDA, math SDPA, 64 real token IDs, dropout0, clip1.
Require sampled frozen decoder bytes, full embedding/head fingerprints and tied parameter identity unchanged.
Fourteen NF4 PEFT wrappers engage; each tiny streamed arm pins 2 layers and late-binds its 6 MLP projections.
Tiny proofs use the unchanged default policy because they are correctness subjects outside the capacity-policy
scope; no memory, timing or card-transfer claim follows. Failure prevents every holdout.

Before generating either full checkpoint, actual Loggetta admission must accept all twelve shapes using the
explicit `dq10` policy and development opt-in. An infeasible plan stops with PLAN_REFUSED and preserves its plan;
never force execution or omit an arm. Each capacity arm plans again on the box, executes that selected plan and
records the policy SHA. All use NF4/doublequant64/bf16 compute, fp32 all-projection LoRA r16/alpha32/dropout0,
SDPA, non-reentrant checkpointing, micro-batch1/accum1, two AdamW steps at constant2e-4, clip1, stock full loss.

Order is Mistral then SmolLM3, sequence512/2048/4096 ascending, resident then streamed: twelve fresh processes.
Exactly seq real tokens per row, two rows, no padding or masked shrinkage; preserve token IDs and paired SHA.
All seven classified projections receive adapters/quantization. Mistral streams seven projections per layer
(224 late-bound total); SmolLM3 streams five (180 total), because K/V packed codes remain below e4b's unchanged
1 MiB threshold. Require all homes pinned and 32/36 streaming layer handles respectively.
Loading peak, training allocated/reserved peaks and sampled driver peak remain distinct. Reuse DQ9's baseline
phase collector; no cache clear, new synchronization, peak reset or tensor-data reads are added to the capacity
instrument. Proof fingerprint reads are outside this restriction and never counted as capacity evidence.

## Fixed gates and resulting scope

For every arm retain E (unchanged allocator estimate), A (training allocator peak), R (training reserved peak),
D (sampled driver peak), reserve charge S=ceil(E*f), frozen C, unchanged 20% charge and each measured-minus-priced
residual. Headroom remains an admission margin outside every estimate and gate. All five one-byte gates must pass:

- Allocator: E >= A.
- Reserve component: S >= R-A; conservative E cannot hide an underpriced slack component.
- Reserved total: E+S >= R.
- Context component: C >= D-R; conservative E/S cannot hide an underpriced driver component.
- Full driver total: E+S+C >= D.

Four proof PASS + twelve complete arms passing every gate gives HOLDOUT_PASS for these registered configs,
5090/driver/runtime/default allocator/recipe/rungs only. It makes a separate default-policy PR eligible for review;
it grants no automatic observation import and does not remove the execution opt-in. Any valid byte miss gives
HOLDOUT_UNDER with every row and miss retained; no fraction/context adjustment, narrowed success quote, redraw
or holdout contribution is licensed. Missing/duplicate/failed/OOM/changed arms or unengaged mechanisms are VOID.
Failed proof is FUNCTION_FAIL and stops before reading. Existing DQ4 allocator bracket tests must stay green;
new-family E/A ratios are descriptive and cannot retune the activation coefficient.

DQ8 remains the independent 24 GB gate before opt-in removal. Its original DQ7 precondition remains closed.
Only a separate reviewed successor proposal/amendment after DQ10 disposition can authorize its draw, updated pins
or a 24 GB candidate policy. A measured 5090 reserve does not automatically transfer to a 4090. Gate removal and
release remain separate, evidence-linked Loggetta changes reviewed by the maintainer.

## Lifecycle and prelaunch checks

One draw through the existing TC1 nonce/heartbeat/fetch and guarded provider. RTX5090 total-hourly ceiling $0.85
including the existing 320 GB disk order, two-hour guard ($1.70 runtime/storage), plus the launcher's unchanged
100 GB download reservation ($1.10): total reservation $2.80. The actual quote must fit the standing no-ask policy
and ledger; no policy bypass. No credentials/HF token are staged. Synthetic shards stay under excluded hf-cache;
all adapter tensors stay under explicitly excluded adapters/. Fetch original proof/receipts/plans/logs/sources,
verify teardown even on failure, and commit this receipt with its own ledger row before another launch.

Local mutation tests must cover every one-byte component miss, missing/duplicate arms, config/runtime/card/policy/
row/tied/mechanism drift, changed source receipts, and nonce preflight refusals. Test the real controller's local
stage closure against every checksum subject. Rehearse all four tiny CUDA proofs on A2000 under a resource claim;
this is correctness only and changes no shared environment. Review the complete instrument and merged pins before
launch. rc9 = instrument/install/config/disk; rc11 = proof/arm/admission/deadline; rc12 = reducer; rc14 = egress;
rc18 = VRAM floor; rc19 = wrong card/driver. No partial or failed lane becomes a pass.
