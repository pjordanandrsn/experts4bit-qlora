# P97 — Hybrid paged serving on the card: the paged runner against transformers' own forward on Qwen3.6-35B-A3B, held to an all-attention control (OLMoE-1B-7B), on one RTX 5090 (registered 2026-10-02, before any run)

Issue: experts4bit-qlora#564. Code under test: #889 (per-slot Gated DeltaNet state) and #897 (compact fp8 pool,
`build_engine` wiring).

**Why.**
- **#889** taught `PagedModelRunner` to serve hybrid linear-attention models (Qwen3.5 / Qwen3.6 MoE, Qwen3-Next). Each
  Gated DeltaNet layer reads and writes a per-slot conv window and recurrent state through transformers' own
  `LinearAttentionLayer`.
- **#897** sized the fp8 KV pool to the attention layers only (Qwen3.6: 10 of 40, through
  `PagedAttentionContext.layer_map`) and wired `build_engine`.
- Both are tested on CPU only. There, a stand-in replaces the fp8 decode kernel (gnf4's Triton, which needs sm_89+).
  `docs/SERVING.md` says so: "no GPU run yet".
- **This lane** is that GPU run. It asks one question: on the card, through the real kernel, does the paged runner
  track transformers' forward on a hybrid model as closely as it does on an all-attention model?
- Speed is not asked. Decode graphs are refused for hybrids, so any decode time read here is eager and reported only.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

- **Box and software.** One RTX 5090.
  - e4b at the launch commit.
  - grouped-nf4-gemm at `34da93d` (v0.34.1, e4b CI's pin: the `fp8_paged_attn` kernel).
  - transformers 5.17.0, the release #889 and #897 were built and tested on; bitsandbytes 0.50.2.
  - No `fla` or `causal_conv1d` is installed, so the Gated DeltaNet layers take transformers' torch path, the one the
    CPU tests ran. The record names which kernel modules were importable.
  - Every serving lever unset (e4b's defaults).
- **Models.** Each loads through `load_moe_4bit_streaming(..., quant_type="nf4")`, NF4 experts resident.
  - **Subject:** `Qwen/Qwen3.6-35B-A3B` at `995ad96` (qwen3_5_moe: 40 layers, 30 linear-attention, 10 attention; kv
    heads 2, head_dim 256). Its linear-attention projections and shared experts stay bf16 under e4b (#899); both paths
    here read the same weights.
  - **Control:** `allenai/OLMoE-1B-7B-0924-Instruct` at `7f1c97f` (P96's revision; 16 attention layers). Its only
    paged error is the fp8 KV, read through the same kernel.
- **The measurement** (`p97_box.py`), per model, on the same in-process weights.
  - **Reference:** paged attention is registered but no paged context is bound, so the model runs transformers' forward
    with a `DynamicCache`, one window at a time.
  - **Paged:** `PagedModelRunner` with the fp8 pool sized by `kv_layers()`. Each window is bound to its own slot,
    prefilled in 128-token chunks, then the four decode together, one row each.
  - **Text:** wikitext-2-raw test, the K8 corpus. Window k starts at token k × 4096 (k = 0–3), each a 512-token prompt
    and a 256-token continuation, teacher-forced in both paths.
  - **Per step:** KL(reference ‖ paged) in nats over the full vocabulary, the true token's nll, and whether the argmax
    agrees. Log-probs are kept in fp32: bf16 rounding (~2e-3 at log-prob −0.5) is the size of the effect.
  - **The mutant pass (subject only).** A second paged pass with the linear-state write-back dropped (stores counted,
    not applied), so each window reads whatever state its slot last held. It shows the rule can see broken state.
- **Engagement, counted.** The box counts every decode-kernel call with the pool layer it names, and every per-slot
  linear-state store with its layer.
- **The premise, on the card, before anything is fetched** (rc 25): `tests/test_linear_state_gpu.py` must PASS, not
  skip. On tiny models, through the real kernel, a hybrid with a compact pool must stay within 2× its all-attention
  control's worst relative logit error, with greedy tokens equal.
- **The order:** both checkpoints fetched (control, then subject), then the control's measurement, then the subject's.

**The reducer** (`p97_reduce.py`, 20-case self-test).
- **VOID** if any of these holds:
  - a record is missing, or ran a rehearsal knob (`--offload`, `--stand-in-attention`);
  - a record is not the registered shape (4 windows, 512 / 256 / 128), or loaded another model or commit;
  - a layer plan is not its model's. The subject must read 40 layers, 10 attention, 30 linear, on a 10-layer pool. The
    control must read 16 attention layers and no linear state;
  - an engagement count is off:
    - decode-kernel calls ≠ 255 × attention layers (2,550 subject, 4,080 control), or not over pool layers 0..L−1;
    - linear-state stores ≠ 30 × (4 windows × 4 chunks + 255) = 8,130, or not over every linear layer;
  - the control's mean KL is zero (the paged path read the reference's own numbers);
  - the subject's mutant pass is missing, or would itself pass the rule.
- **SUPPORTED** if the subject's mean KL ≤ 2 × max(the control's mean KL, 1e-3) **and** its argmax agreement ≥ the
  control's − 0.02.
- **NOT_SUPPORTED** otherwise.
- **Reported, not gated:**
  - each model's max KL, mean Δnll, max |Δnll| and the prefill step's max |Δ log-prob|;
  - the eager decode time per step at 4 rows;
  - the linear-state pool's size and peak GPU memory;
  - the mutant's numbers.

**The registered consequence.**
- **SUPPORTED:** `docs/SERVING.md`'s hybrid paragraph cites P97 as the GPU reading, and the next hybrid item, decode-graph
  capture of the state gather and scatter, starts from it.
- **NOT_SUPPORTED:** the hybrid path stays documented as CPU-tested only, and an issue records the reading before any
  change to it.
- **VOID:** nothing moves.

## Predictions (written before the data)

- **SUPPORTED.** The subject reads at or below the control's mean KL. Only 10 of its 40 layers read fp8 K/V, against
  all 16 of OLMoE's.
- **The control's mean KL** lands between 3e-4 and 1e-2 nats, with argmax agreement ≥ 0.97 for both models.
- **The mutant's mean KL** is above 0.1 nats. On CPU, the same mutant read 480× the hybrid's KL.
- **The engagement counts** match their expected values exactly. The subject's linear-state pool holds 0.24–0.25 GiB
  for 4 slots: 30 layers × (a 32,768-value conv window + a 32 × 128 × 128 recurrent state) per slot.

## Box and cost

- **`p97-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P97_PROVE=1` runs:
  - the refusals;
  - the install with its tripwire;
  - the reducer's self-test;
  - the premise on the real kernel;
  - an HF CDN egress probe.

  It loads no model.
- **`p97-5090-<n>`:** one RTX 5090, after a passing proof, with ≥ 150 GB of disk (the runner refuses below 130 GB: a
  72 GB and a 14 GB checkpoint). **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).** About 45 minutes:
  - install and premise ~6;
  - the fetch ~10–15 (86 GB);
  - the control ~5;
  - the subject's load ~10 and its three passes ~10.

  The runner stops starting steps 10 minutes before the deadline.
- **Lane ceiling $2.50; hard stop $3.50.**

## Rehearsal

To be run on the NAS RTX A2000 (sm_86, 12 GB) before the launch, and recorded here. It needs four knobs:
- the local checkpoints (`P97_MODEL_DIR`);
- `--offload --stand-in-attention` with a shortened shape (`P97_BOX_EXTRA`), since the card has no native e4m3 and 12 GB
  does not hold Qwen3.6 resident;
- `P97_PREMISE_ALLOW_SKIP=1`, since the premise skips below sm_89;
- `P97_GPU_CLASS=A2000 P97_MIN_DISK_GB=20`.

Any of these marks the run REHEARSAL, and the reducer voids its records. The rehearsal checks the path: the install,
the tripwire, the loader on the composite Qwen3.6 config, the reference and both paged passes, the counts and the
reducer. It quotes no time and no quality number.

Amendments, dated, go below this line before any data is read.
