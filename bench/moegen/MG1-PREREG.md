# MG1: the new families through tp1's licence instrument, and the Qwen stack's portability ladder, on one RTX 5090

*Registered 2026-10-04 on branch `session/moe-generalize` before any box. This file is `bench/moegen/MG1-PREREG.md`. The
box side is [`mg1_run.sh`](mg1_run.sh), started by `bench/tc1/tc1_drive.sh` as its `TC1_RUNNER`. The reducer is
[`mg1_reduce.py`](mg1_reduce.py).*

## Question

Do the families this branch generalises the training stack to pass tp1's fused-vs-reference parity on real weights, so
that their `fast_train` path can be licensed? Those families are LFM2-MoE, Granite-4.0-H, ERNIE-4.5, Nemotron-H and
Qwen3.5/3.6. And how much of the Qwen performance stack does each family's shipped path carry?

## The instrument (unchanged, so a PASS here means what a tp1 PASS means)

* Arms: tp1's arm driver `tp1_train_smoke.py`, byte-for-byte as committed at
  `bench/train-parity-20260905/tp1/logs/tp1_train_smoke.py`: `reference` (no accelerator) and `fused`
  (`enable_fast_train(dgrad=True)`).
* Fixture: the registered clinical text (`n9_datasets.py`, sha `76fb9036de80…`, refused on mismatch). N = 60 steps,
  seq 512, r = 8, alpha 16, lr 1e-4, seed 0, bf16 base, fp32 attention adapters.
* Verdict: tp1_reduce v2.1's `verdict()`, copied verbatim into `mg1_reduce.py`. The fused arm PASSES when its final-step
  loss and its median per-step loss are both within 0.05 of the reference arm's, after the identity checks: identical
  `init_sha`, C1 bit-exact frozen experts on both arms with the byte-flip control firing, `n_patched > 0`, and kernel
  calls per step ≥ 2·n_patched on every step. Otherwise FAIL, or VOID with the reason.
* Box class: the train anchor (`bench/train-anchor/`), strict as in tp1. A refused box ends the lane (exit 12).
* Revisions pinned in `mg1_run.sh` (the snapshots downloaded on 2026-10-04; OLMoE and Gemma-4 by amendment 1): OLMoE-Instruct
  `7f1c97f4`, Gemma-4-26B-A4B-it `4d7ae498`, LFM2-8B-A1B `c1c44ff9`,
  granite-4.0-h-tiny `791e0d3d`, ERNIE-4.5-21B-A3B-PT `87db9548`, Nemotron-3.5-Lightning-30B-A3B `a9904d24`, Qwen3.6-35B-A3B
  `995ad96e`.
* OLMoE runs first, as the regression anchor: a licensed family re-read on this branch's code (amendment 1: on the licensed
  `-Instruct` checkpoint, and Gemma-4 beside it).
* Qwen3.6 runs resident first. On an OOM stub its reference arm is renamed `.resident_oom` and **both** arms rerun under
  expert offload (TC2's working configuration). The two arms are never split across modes.

## The ladder (informational, never a licence)

Per family after its pair: `bench/moegen/ladder.py --rungs fused,fused_pre,keep` on the same snapshot, 8 timed steps after
3 warmup, seq 512, bf16 adapters (the shipped configuration). Rungs are read interleaved A..Z Z..A in one process. Recorded:

* wall s/step and the device-busy seconds of one profiled step;
* peak VRAM, and host syncs per step;
* the engagement census: modules patched and skipped, the dgrad route with loop reasons, the LoRA route, RMSNorm variants
  and fallbacks, RoPE patched and refused, MoE-keep layers and offload skips, ring overflow, the train-GEMM route;
* device time by kernel family.

## Predictions (scored by the reducer's table; each is a row whatever it reads)

* **P1** All six load and pass `verify_moe_4bit(strict=True)`. Fused patches every MoE layer: OLMoE 16, LFM2 22,
  Granite-H 40, ERNIE 27, Nemotron-H 23, Qwen3.6 40.
* **P2** `DGRAD_STATS["loop"] == 0` on every resident fused arm: the dgrad kernel serves every frozen-GEMM backward,
  including Nemotron-H's non-gated relu² stack.
* **P3** PASS on all five new families and on the OLMoE anchor. Expected margins are of tp1's order (d_final ≤ 0.02).
  A FAIL is a finding about that family's composition and stays a row.
* **P4** Ladder, on device time: `keep` < `fused` < `fused_pre` on every family. The keep gain is largest where the routed
  experts are the biggest share of the layer (OLMoE, LFM2, ERNIE) and smallest on Nemotron-H and Granite-H, whose
  Mamba blocks are outside the expert runtime.
* **P5** The RMSNorm census reads the variant per family: plain Llama-rounded on LFM2, ERNIE, Nemotron-H and Granite-H;
  centered fp32 on Qwen3.6, every norm fused where the shipped code fused none. RoPE is refused on semantics on ERNIE.

## Decision rule

A family whose fused arm PASSES, with every identity check clean and P1/P2 engagement, enters `docs/capabilities.json`
`training_support.by_model_type.<type>.fast_train = supported` with its claim ids, and joins `model_families`. STATUS and
claims are updated in the same diff. Any other reading enters with its status (`fail` / `void` / `refused` / `oom`) and the
reason. Under expert offload the box-side mode is named in the row, as Mixtral's tp1 row names it.

## Budget

One RTX 5090 at the policy's $0.85/h ceiling. Estimated wall about 4 h (fetches about 210 GB; per family two N=60 arms and
one ladder). Guard 5.5 h, so about $4.7, inside the CTO band. The box disk holds one checkpoint at a time; each family's cache
is removed after its ladder.

## Amendment 1 (2026-10-04, before any box): the regression anchors re-read what was licensed

* **OLMoE is `allenai/OLMoE-1B-7B-0924-Instruct` @ `7f1c97f4`.** That is the checkpoint tp1 licensed. The registration named the
  base model, which no licence reads, so as registered the anchor could not show a regression.
* **Gemma-4-26B-A4B-it @ `4d7ae498` is a second regression anchor.** This branch changes its fused RMSNorm numerics: `main`
  fused its norms with the Llama rounding, and they now run their own fp32 multiply, which is closer to the reference composite.
  tp1's licensed reading was d_final 0.02385 and median 0.04742, against a 0.05 band, so it is the licensed family with the
  least margin.
  - **P6.** Gemma-4 PASSES, with a median no larger than 0.0574 (tp1's 0.04742 + 0.01). This is a regression signal beside the
    verdict, not a second verdict.
  - A FAIL, or a median above 0.0574, is recorded, and Gemma-4's licence row is flagged for a re-read. Nothing is removed
    automatically.
* **Order:** olmoe, gemma4, lfm2, graniteh, ernie, nemotron, qwen3_5 (`MG1_FAMILIES`' registered default).
* **Budget:** Gemma-4's fetch (51.6 GB) and two arms add about 30 min. Guard 6 h, so about $5.10 at $0.85/h, still under the
  $15 no-ask tier.

## Amendment 2 (2026-10-04, after the reading and before its own box): P2 for Qwen3.6

The reading ([`mg1/mg1-5090-2/RESULTS-mg1.md`](mg1/mg1-5090-2/RESULTS-mg1.md)) PASSED all seven families. P2 went unread for
Qwen3.6, because tp1's arm driver predates `DGRAD_STATS` and Qwen3.6's ladder OOMed at its default r 16. The decision rule needs
P2, so `qwen3_5_moe` entered as `experimental`. This amendment reads that one counter.

* **Run.** `MG1_FAMILIES=qwen3_5 MG1_LADDER_ONLY=1 MG1_LADDER_ARGS='--rungs fused --r 8 --alpha 16 --adapter-dtype fp32 --attn4 0 --attn-lora 1 --profile 0 --warmup 1 --steps 2'`: the pinned Qwen3.6 revision, the `fused` rung
  only, at the licensed arms' configuration (r 8, alpha 16, fp32 expert and attention adapters, bf16 attention), with no
  profiler, 1 warmup and 2 timed steps. The runner prints `AMENDMENT 2 SHAPE (registered)`. It skips the train anchor and the
  arms: P2 is an engagement count, not a timing, and the arms are already read.
* **P7.** The `fused` rung's census reads `DGRAD_STATS["loop"] == 0`, with `kernel` > 0, on Qwen3.6 resident.
* **Decision.** If loop is 0, `qwen3_5_moe.fast_train` becomes `supported`, citing the reading's PASS and this receipt, and the
  family joins `model_families`. If loop is above 0, it stays `experimental`, with the loop's reasons named. An OOM or harness
  error is a row, and the status is unchanged.
* **Budget.** One RTX 5090: a 67 GB fetch and one load. Guard 0.75 h, about $0.64. The guard is under one hour, so no proving
  run is required.

## Staging, the proving run and the rehearsal

* **Controller.** `bench/tc1/tc1_drive.sh` with `TC1_BOX=A`, `TC1_RUNNER=mg1_run.sh`, and `TC1_EXTRA_STAGE` = `bench/moegen/mg1_run.sh
  bench/moegen/ladder.py bench/moegen/mg1_reduce.py bench/train-anchor/train_anchor.py bench/train-anchor/train_anchor_gate.py
  bench/train-parity-20260905/tp1/logs/tp1_train_smoke.py`. tc1's own stage brings `n9_datasets.py` and the registered
  `ds_manifest.json`. `GNF4_SHA` is the grouped-nf4-gemm commit that carries `DGRAD_STATS`: the box tripwire refuses any other.
* **Proving run first** (the standing rule for a guard over one hour): the same image, provider class and controller with
  `MG1_PROVE=1`. That runs the install and the tripwire, then finishes clean without anchor, fetch or arm. Budget: ≤ $0.15,
  ≤ 10 min. A failed proof is a row and a defect, never a retry of the reading.
* **Shape knobs.** `MG1_FAMILIES` / `MG1_STEPS` are forwarded by `tc1_drive.sh`. Any value other than the registered six families
  and N = 60 prints `NON-REGISTERED SHAPE` into `summary.txt`, and such a run is not this reading.
* **`MG1_REHEARSAL=1`** exists for the $0 A2000 container rehearsal only. It records the train anchor's refusal and continues, so
  the arms and the ladder run on a card outside the anchor band. `tc1_drive.sh` does not forward it, so a rented box cannot set
  it.
* The runner installs `git` when the image lacks it (pip's `git+https` needs it; P55's lesson).

## A2000 rehearsal

The same arms and ladder ran first on the owned RTX A2000 12 GB (sm_86), outside tp1's anchor band and so informational
only: [`tp1-a2000/`](tp1-a2000/), [`receipts/`](receipts/). They validate the harness path per family. Big families ran
as layer slices (`slice_snapshot.py`) for the ladder.
