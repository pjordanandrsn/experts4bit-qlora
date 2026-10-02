# TC3 — the memory frontier: Qwen3-30B-A3B at the field recipe on a 24 GB RTX 4090 (rented) and on the owned 12 GB RTX A2000, every framework with its own memory levers

Registered 2026-10-02T03:49Z, before any box. Issue: experts4bit-qlora#835. Drafted in the campaign's `TC3-PREREG-draft`; built as the `qwen3frontier`
and `qwen3frontier12` tokens of the TC1 harness (`bench/tc1/tc1_run.sh`, `tc1_arm.py`, `tc1_reduce.py` R11); upstream facts read from source in
`bench/tc1/UPSTREAM-NOTES.md` ("TC3 addendum"). Lineage: on the 32 GB card every resident arm of this family peaked at 24.3–27.9 GB (TC1,
`tc1-5090-16`: Unsloth 24.27, e4b matched 27.85, e4b as shipped 24.58), so a 24 GB card is exactly the boundary; the 12 GB card is e4b's
offload territory. TC1's registered facts this lane leans on: the matched trajectory (tokens sha `bfc742f67e37`, name-free
`matched_init_sha f7832488926eda91`, fp32 adapters, N = 20) and its resident e4b arm, read with `--tc1-dir`.

## Claim under test

Which frameworks can train this problem at all under a 24 GB and a 12 GB VRAM budget, each with its own documented memory levers
engaged, and at what cost (s/step, peak VRAM, host-RAM high-water, J/step) — and does e4b's expert-offload path keep the matched
trajectory (equivalence) while fitting?

## Fixture

TC1's field recipe unchanged (alpaca template, seq 2048, micro-batch 2 x accum 4, r 16 / alpha 16, lr 2e-4, adamw_8bit wd 0.001, linear
with 5 warm-up steps, seed 3407, N = 20, 8 held-out rows at 0 and N, matched init `matched:3407`, fp32 adapters on every matched arm);
the same model pin (`Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`), the same tokens file (sha asserted). "mb1" = micro-batch 1
x accum 8, the same tokens per step (TC1's secondary recipe). Every arm records `memory_lever`, `peak_vram_gb`, `host_ram_high_water_gb`
(the larger of the process's peak RSS and the cgroup's peak when it rose during the arm) and `host_ram_total_gb`; an OOM is a row with the
peak it reached; a lever arm whose loop raises a non-OOM exception is a `refused` row carrying the exception.

## Arms, in this order

**24 GB box** (`qwen3frontier`; one RTX 4090, Vast verified-secure, host RAM >= 98 GB, driver >= 580 — a lower driver is TC1 amendment 1's code-18
lane refusal and a redraw). Alarms in seconds after each arm.
1. `e4b/fused_attn4_m` resident (1200; expected OOM).
2. `e4b/fused_attn4_m_offload` — `--offload 1`: the frozen 4-bit expert stacks in pinned host RAM, streamed per layer (3600). **The box's anchor.**
3. `e4b/fused_attn4_m_mb1` resident at mb1 (3600).
4. `unsloth/ckpt_unsloth_m` — Unsloth 2026.9.14 at TC1's comparator configuration (venv-unsloth, torch 2.12.1+cu130, grouped_mm backend,
   `use_gradient_checkpointing="unsloth"`: its own lever, already on) (3600; expected OOM).
5. `unsloth/ckpt_unsloth_m_mb1` (3600). Unsloth documents no further single-GPU memory lever for this path (UPSTREAM-NOTES): no sixth row.
6. `hf/hf_peft_m` — transformers 5.18.0 / PEFT 0.21.2 / bitsandbytes 0.50.2, bf16 experts (1800; expected OOM at load).
7. `hf/hf_peft_m_offload` — `--hf-offload 1`: `device_map="auto"` under `max_memory = {0: <GPU GiB - 2>, "cpu": <host RAM GiB>}`, which the
   bnb-4-bit quantizer accepts only with `llm_int8_enable_fp32_cpu_offload=True` (passed by default; the CPU-resident modules are then NOT
   quantised — recorded under `hf_offload`); the device map is summarised in the receipt (3600; expected REFUSED or far slower).
8. `axolotl/ckpt_axolotl_m` — axolotl 0.20.0 (`load_in_4bit`, `adapter: qlora`, `quantize_moe_experts: true`, double-quant off; torch 2.14.0+cu130,
   axolotl's own `torch<=2.14.0` pin, transformers 5.17.0, peft 0.21.0) (2700).
9. `axolotl/ckpt_axolotl_m_layeroffload` — `layer_offloading: true`: axolotl's lever is a trainer mixin (`LayerOffloadingMixin`), so the harness
   builds its trainer-free `LayerOffloadManager` after `ModelLoader.load()` and enters `_LayerOffloadContext` around every micro-batch; an import
   failure or a manager that does not enable is a `refused` row (3600).
10. `axolotl/ckpt_axolotl_m_zero3` — DeepSpeed ZeRO-3 parameter + optimizer offload on one GPU: read from source, the engine is created by
    transformers' `Trainer` inside `axolotl.train.train`, not by `ModelLoader`, so this harness cannot drive it without faking it; the arm is a
    `refused` row that records the config it would have used and the installed deepspeed version (its extra installed as a separate, non-fatal
    step) (3600). If a later cut drives it, it is re-registered first.
11. `e4b/reference_attn4_m_offload` — the per-expert reference under the same offload: parity and equivalence control (5400; `can_run 900`).

**12 GB box** (`qwen3frontier12`; the owned RTX A2000 12 GB in the `gpu-dev` container, a hand run under `TC1_LOCAL_BOX=1` from the registered
snapshot with the config.json pin proof; no rental, no receipt in the private store — the receipts ship in-repo, labelled as the owned card; the
host driver is 575 by design, so the cu130 venvs are not built: Unsloth runs in `venv-unsloth-t28` (torch 2.8.0+cu128, the loader-default
`native_torch` backend), HF in venv-e4b (torch 2.8), and the axolotl arm is the driver-gate `refused` row). The card is shared with on-demand
sidecars: the operator starts the run with the card idle and records `nvidia-smi`'s used memory at the start in the run notes.
1. `e4b/fused_attn4_m_offload` (7200). 2. `e4b/fused_attn4_m_offload_d2` (the second draw; 7200). 3. `e4b/reference_attn4_m_offload` (14400).
4. `e4b/fused_attn4_m` resident (1200; expected OOM).
5. **The mb1 secondary, only if arm 1 OOMed** (fp32 adapters, their grads and the 8-bit Adam states on 642 M parameters alone are ~6.5 GB):
   `e4b/fused_attn4_m_offload_mb1` (7200) and `e4b/reference_attn4_m_offload_mb1` (14400), one draw each; the box's anchor then falls back to the
   fused mb1 arm.
6. `e4b/fused_attn4_shipped_offload` — e4b as the loader builds it (bf16 expert adapters, N(0, 1/r) init) under offload: a FIT row, never a position
   (7200).
7. `unsloth/ckpt_unsloth_m_mb1` (3600), 8. `hf/hf_peft_m_mb1` (1800), 9. `axolotl/ckpt_axolotl_m` (2700: the refused row).
Hand-run deadline 12 h (`TC1_LOCAL_HOURS`); the alarms are literals in the plan line.

## Validity and verdicts

TC1's (tp4's predicates; matched-init completeness, fp32 adapters, one name-free `matched_init_sha` across the matched arms; C1 bit-exact with the
real-storage flip control; the tokens sha; `lora_path_loop == 0` and `n_patched == 48` on every e4b fused arm, offloaded or not — an offload arm
whose kernel took the loop on any step is VOID), read against the box's own e4b offload anchor (the reference-under-offload arm, then the mb1
secondary, stand in when it did not complete). Verdict column as TC1 (VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM /
NOT_RUN); "completed" = status OK whatever the validity; OOM and UNSUPPORTED are the two ways not to.

## Readings

- **(a) FIT TABLE**, per framework: every arm with its lever, verdict, peak VRAM, host-RAM high-water, s/step, J/step; a framework FITS on a box
  iff at least one of its arms completed; an OOM-only column reads so, with the peaks reached.
- **(b) Equivalence under offload**: every matched arm against the box's anchor under TC1's fixed R4 bands (EQUIVALENT: median per-step |delta
  train| and |delta held-out at N| both <= 0.02; COMPARABLE <= 0.05; else DIVERGENT), tp1's parity on the fused/reference offload pair, and with
  `--tc1-dir` the anchor against TC1's RESIDENT `e4b/fused_attn4_m` (same tokens sha, same `matched_init_sha`, fp32-only adapters, same N asserted
  first): median per-step |delta| <= 0.02 reads EQUIVALENT-TO-RESIDENT, else DIVERGENT-FROM-RESIDENT.
- **(c) Speed is reported, never as a cross-box ratio**: the only ratios are within one box.
- **(d) P1–P4** scored mechanically (`score_frontier_predictions`).

## Predictions

- **P1 (24 GB)**, four clauses, all within the box: (i) e4b completes under offload; (ii) Unsloth OOMs at both recipes; (iii-a) HF OOMs resident
  and (iii-b) its offload arm is REFUSED or > 20x slower per step than e4b offload — an HF-offload OOM reads FALSIFIED (neither a refusal nor a
  measured slowdown); (iv) axolotl's ZeRO-3 arm, if it runs, is > 5x slower than e4b offload at a >= 60 GB host-RAM high-water — a ZeRO-3 arm
  that did not run is a conditional clause, vacuously held, and said so.
- **P2 (12 GB)**: the e4b FUSED offload arm completes (at the field recipe, or at its mb1 secondary after a field-recipe OOM) and every other
  framework's arm is OOM or UNSUPPORTED (an UNSUPPORTED is said as "not an OOM reading"). The second draw and the reference arm are reported, never
  scored; an e4b resident arm that also completes is noted, never a refutation (P2 is about frameworks); the as-shipped offload row is outside P2.
  UNTESTED when a non-e4b arm, or the deciding e4b arm, is NOT_RUN / HARNESS_ERROR / ALARM.
- **P3** (per token): the anchor is EQUIVALENT to its in-box reference and, on the 24 GB token with `--tc1-dir`, EQUIVALENT-TO-RESIDENT.
- **P4** (24 GB, with `--tc1-dir`): e4b offload s/step on the 4090 within 2x of TC1's resident e4b on the 5090 — two measurements with the factor
  applied, never a ratio; UNTESTED when the resident reading could not be made.

## Decision rules

- P1 holds -> the 24 GB fit table is registered as claims rows per framework (fit / no fit, the lever, the peaks), with the e4b offload row's
  cost beside it; any clause refuted -> the refuting row is the finding and the fit table is still registered as measured.
- P2 holds -> the 12 GB fit claim is registered, labelled as the owned card and the hand run; refuted -> no 12 GB claim, the row that refuted it
  quoted.
- P3 refuted -> no "offload keeps the trajectory" claim; TC4-style bisection before any such claim.
- Nothing from this lane moves a TC1 position.

## Budget and stop rules

24 GB: one RTX 4090, Vast verified-secure, ceiling $0.49/h, guard 5 h, estimate $2.45 (< $15; the standing no-ask tier for a single run, set
2026-09-26). 12 GB: the owned card, $0. TC1's pre-flight and STOP rules; a draw that cannot hold the plan before its deadline marks the
remaining arms `not_run` host-limited.

## Environments

24 GB: e4b main at this registration's merge (0.38.1; `experts4bit/` byte-identical to TC1's 079a422) + grouped-nf4-gemm 846b512 (TC1's pin);
Unsloth 2026.9.14 + unsloth_zoo 2026.9.9 on torch 2.12.1+cu130 (TC1's comparator venv); transformers 5.18.0 / PEFT 0.21.2 / bnb 0.50.2 on
torch 2.8.0+cu128 for HF; axolotl 0.20.0 in its own uv venv (CPython 3.12, torch 2.14.0+cu130 by axolotl's pin, transformers 5.17.0, peft 0.21.0,
bnb 0.50.2; TC1 amendment 3's `--index-strategy unsafe-best-match`). 12 GB: the venvs built from the `gpu-dev` k3rel interpreter under the hand
run (`TC1_LOCAL_PYTHON`); versions recorded in `versions.txt` as on every box.


## Amendments

### Amendment 1 (2026-10-02T04:20Z, after the first 12 GB hand run died in its venv build, before any arm on either box): every venv the box script makes upgrades pip first

**What the hand run showed.** The first `qwen3frontier12` run (gpu-dev, run id `local-20261002T041406Z`, 04:14Z) built `venv-e4b` from the
k3rel interpreter with `python -m venv`, whose ensurepip bundles pip 22.0.2 on that Ubuntu-22.04 image; that pip cannot read the
`setuptools>=77` (PEP 621) metadata of experts4bit-qlora and grouped-nf4-gemm and reported both as `unknown 0.0.0 ... ResolutionImpossible`
(`logs/pip_e4b.log`), so the script stopped at rc 9 (`PIP FAIL (e4b)`) with no arm run and nothing to read. The rented images carry a current
pip, which is why no TC1 / TC2 box met it.

**The amendment.** `tc1_run.sh` upgrades pip in each venv it makes with `python -m venv` (`venv-e4b`, `venv-unsloth-t28`, `venv-unsloth`)
before the install that needs it — logged to `logs/pip_upgrade_<venv>.log`, never fatal (a pip that cannot move leaves the install that
follows to fail or succeed on its own). The axolotl venv is uv's and is not touched. Nothing in the arms, the alarms, the predicates or the
readings moves; the 12 GB run restarts from this amendment's merge with the same registered snapshot and knobs.

### Amendment 2 (2026-10-02T04:59Z, after the first 12 GB arm refused itself in the harness; before any e4b offload row on either box): C1 hashes the offload home, not the GPU placeholder

**What the hand run showed.** The restarted 12 GB run (`local-20261002T042928Z`) loaded the model under offload in 991 s and then
refused its first arm in the harness: `AssertionError: C1 saw 97 empty frozen tensors`. Under e4b's expert offload the base's packed
`gate_up_proj` / `down_proj` and the `*_absmax` buffers are 0-element GPU placeholders while evicted; the bytes live in the handle's
pinned-CPU `home` (`experts_lora._offload`, `engines/offload.py`). TC1's phase-3 hasher hashes every frozen tensor it can see and
asserts no empties, so every offloaded arm of this lane (both TC3 boxes) and TC2's Mixtral offload arms (box B, not yet launched)
would refuse themselves the same way. The rented 24 GB box `tc3-4090-1`, already running on the pre-amendment harness, carries that
refusal on its e4b offload rows; its other rows stand.

**The amendment.** `tc1_arm.py` maps each offload handle's `home` tensors to the base module's qualified names
(`offload_homes`); `frozen_tensors` yields the home copy for those names (the placeholder is never hashed), `hashes_frozen` counts no
empties, the positive control flips a byte of the home copy, and the receipt records `C1_offloaded_homes` (0 when resident). The
resident regime is byte-for-byte as before (same hasher, same names). The 12 GB run restarts from this merge; the 24 GB token is
redrawn once (`tc3-4090-2`) so its e4b offload rows are measured, with `tc3-4090-1`'s non-e4b rows reported beside as a second draw.
TC2's box B launches from this merge (TC2-PREREG amendment 1). Nothing in the arms, the alarms, the predicates or the readings moves.

### Amendment 3 (2026-10-02T05:09Z, before the 24 GB redraw; while `tc3-4090-1` was still running): e4b as shipped, resident, as a fit row on the 24 GB token

**What the first 24 GB box showed.** `tc3-4090-1` (RTX 4090, 24,564 MiB, EPYC 7B13 host): Unsloth's matched arm trained RESIDENT at
the field recipe -- 8.43 s/step, peak 24.22 GB, held-out 1.9515 -> 0.8527 -- while e4b's matched arm (fp32 adapters) OOMed at both
recipes (24.32 / 24.30 GB at the first step). P1's clause (ii) ("Unsloth OOMs at both recipes") is therefore FALSIFIED on that box, as
registered, and the question the box raises is whether e4b at its own defaults (bf16 expert adapters, N(0, 1/r) init; 24.58 GB peak on
the 32 GB card in TC1) fits a 24 GB card resident.

**The amendment.** The 24 GB token gains `e4b/fused_attn4_shipped` -- resident, native precision and init, the field recipe -- as a
FIT row after the mb1 resident arm (alarm 1200 as the resident arm's): never a position, outside P1, reported in the FIT TABLE and
beside the matched arms (the 12 GB token already carries its offloaded counterpart). The reducer registers the row (`EXPECTED`,
`FRONTIER_LEVER`); nothing in the predictions, bands or decision rules moves.
