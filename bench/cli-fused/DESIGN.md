# The CLI trainer on the fused path — design packet (before any code)

**Goal.** `python -m experts4bit_qlora.train` should reach the configuration P137 licensed:
- `enable_fast_train(model, dgrad=True)`;
- expert offload;
- `layer_major_step`.

P137 measured layer-major at 2.365× the stock loop per step there (`e4b.train.p137.layer-major-offload.qwen3.4090.2026-10-18`).
Today the trainer cannot reach that configuration, so the measured gain is not a capability a user of the trainer has.

## What the trainer does today (e4b main)

- **No fused path at all.** Nothing in `experts4bit_qlora/` calls `enable_fast_train` except `engines/fast.py` itself.
  - Every `python -m experts4bit_qlora.train` run trains through the `ExpertsLoRA` reference loop: per-expert
    bitsandbytes.
  - That holds resident or offloaded, on every family.
  - The fused path, the headline of `qlora-fused-moe-experts` in `docs/capabilities.json`, is reachable only from
    Python and from the bench harnesses (TC1, TC3, P137).
- **`LAYER_MAJOR=1`** (needs `OFFLOAD_EXPERTS=1`) calls `layer_major_step` over that reference loop. TC3 read the
  reference loop under offload at 77.4 s/step on an RTX 4090, against 10.5 s/step for the fused path. P137's licence
  explicitly excludes it.
- **Recipe differences from P137:**
  - torch `AdamW` (fp32 moments), not bitsandbytes `AdamW8bit`;
  - token-budget micro-batches (`TOKEN_BUDGET`), not fixed rows of 2 × 2048;
  - a cosine schedule with warm-up.
- **Load cost.** Every start pays the full load. Under offload on CUDA, the loader fuses the expert rows on the host CPU,
  which took 2,078 s on `p137-4090-7`'s EPYC 7742 and about 38 min on a contended host.

## The change

1. **`FAST_TRAIN=1`** (default `0`, so no default flips) calls `enable_fast_train(model, dgrad=True,
   absmax_dq=ABSMAX_DQ)`. That is `dgrad=True` always, the only configuration with receipts.
   - **When it runs:** once the adapters and trainables are set, before the chunked-loss hook and before the BEFORE
     evaluation. So both evaluations and every training step run the same expert path.
   - **It refuses:**
     - a patch count of 0 (`SystemExit`, as `TRAIN_ATTN_4BIT` does for a vacuous arm);
     - `TRAIN_ARENA`, because `enable_hybrid_train` patches the same wrappers and the composition is untested;
     - an `EpilogueContractError`, which propagates.
   - **The log line names** the patch count and `dgrad=True`.
2. **`LAYER_MAJOR=1` means the fused path.** Unset, `FAST_TRAIN` defaults to `1` under `LAYER_MAJOR=1`.
   - An explicit `FAST_TRAIN=0` keeps today's reference loop, with a log line saying it is the reference path and not
     the licensed configuration, so earlier census reads still reproduce.
   - `LAYER_MAJOR=1` still needs `OFFLOAD_EXPERTS=1`.
3. **Help text and docs.** The `--help` table gains `FAST_TRAIN`. The `LAYER_MAJOR` line says it implies it.
   `docs/STATUS.md` changes in one place only: once the A2000 leg passes, the P137 paragraph names the trainer
   invocation, and no number moves.
4. **Nothing else changes** in the loop. `layer_major_step` is already called with `[1 / GRAD_ACCUM] × GRAD_ACCUM`
   weights; the OOM back-off, the census phase `layer_major` and `offload_stats_report` stay as they are.

## The $0 A2000 leg (correctness only; no timing is read)

This uses the tiny Qwen3-MoE fixture (`bench/p137/p137_tiny.py`, 2 MoE layers), `OFFLOAD_EXPERTS=1` and `STEPS=2`.
Each check is STRICT:
- **The trainer is the library call.** With `FAST_TRAIN=1 LAYER_MAJOR=1`, step 0's per-micro-batch losses are
  bit-identical to `layer_major_step` called directly on the same micro-batches after `enable_fast_train(dgrad=True)`
  (a probe script that imports the trainer's batcher).
- **The fused path is engaged.** The patch count equals the number of MoE layers, and the step-0 losses differ from
  `FAST_TRAIN=0`'s by the expected forward-fusion error, nonzero and under 1e-2.
- **Stagings.** `offload_stats_report` reads 2n − 1 per steady step with `LAYER_MAJOR=1` and GRAD_ACCUM × (2n − 1)
  without, as in P137.
- **Same adapters, same start.** The adapter sha256 after loading is equal across the arms.
- **Refusals.** `FAST_TRAIN=1` together with `TRAIN_ARENA` refuses before any expert storage is released, and a model
  with nothing to patch refuses.
- **CPU unit tests** cover the knob's parsing, the `LAYER_MAJOR` ⇒ `FAST_TRAIN` default and the help table.

## The paid read: options (the maintainer's call)

The licence covers the library call. What remains open is whether the trainer, at its own recipe, gets the gain.

| option | what it reads | cost and risk |
|---|---|---|
| **A. No paid read.** | Nothing new is measured. The A2000 leg proves the trainer makes the licensed calls, and the docs say "at P137's recipe". | $0. It does not cover the trainer's AdamW and token-budget recipe. |
| **B. Two CLI processes** (`LAYER_MAJOR=0`, then `=1`), with one load each. | S at the trainer's recipe. Self-consistency comes from first-half against second-half medians over 64 timed steps per run. | About $1.5–2.5: two loads of 35–40 min plus steps. This is weaker than ABBA, because there is no between-process self-pair. |
| **C. Four CLI processes, ABBA.** | The P137 design, through the CLI. | About 2.3 h of loads alone, against a 2.5 h guard on a contended host. It needs a guard raise or a quiet-host screen. |

**Proposal: A now, with B registered only if a user-facing number at the trainer's recipe is wanted.** The CLI change is
plumbing whose correctness the A2000 leg settles. The speed claim stays the library's, quoted with its recipe.

**The load is the real lever behind B and C.** A 35-minute host-side fuse on every start is a cost every trainer user
pays. A quantized-expert cache keyed by sha256 (no quantized save path exists today) would be its own packet.

## If deferred

The trainer keeps training through the reference loop only. `LAYER_MAJOR=1` stays a 77 s/step path on a 4090, and the
measured 2.4× stays a Python-only capability.
