# tp4 — same-box training head-to-head: e4b (GitHub `main`) vs Unsloth vs plain HF+PEFT+bnb, every supported MoE family

Pre-registration: [`TP4-PREREG.md`](TP4-PREREG.md) (read it first; the harness refuses a run that does not name it).

| file | role |
|---|---|
| `tp4_arm.py` | per-arm driver, three frameworks through ONE `run_arm` (copy of `bench/tp3/tp3_arm.py` + T11–T16; `--selftest` on CPU) |
| `tp4_alpaca.py` | builds the fixed Alpaca text from the pinned `unsloth/alpaca-cleaned` revision (seed 3407, 1,200 / 48 rows; registered sha) |
| `tp4_run.sh` | BOX side: installs the three environments, fetches each family, tokenises once, runs the arms in the registered order, reduces |
| `tp4_drive.sh` | CONTROLLER side (the launcher's `--command`): stages, starts under a nonce, heartbeats, fetches receipts |
| `tp4_reduce.py` | the per-family support / validity / position / quality table, the anchor and the P1–P9 scoring (stdlib only) |
| `../../tests/test_tp4_arm.py` | CI: the selftest end to end, the `--prereg` refusal, the dataset builder's refusal path |

Boxes (one RTX 5090 each, launched in parallel through `adertha-agents/tools/pod-launch.sh`):
`TP4_BOX=A` granite · olmoe · gpt-oss + the Qwen3 anchor pair (tp2's fixture) + the two NOT_RUN rows;
`TP4_BOX=B` qwen3 · qwen3_5 (Qwen3.6-35B-A3B); `TP4_BOX=C` gemma4 · mixtral.

Receipts land privately first (the adertha receipt store), then the curated bundle is committed here as
`bench/h2h-<date>/tp4/` exactly as tp2's was.

## Where an arm's time goes before step 1 (e4b#548)

Every receipt — `ok`, `refused`, `oom`, `phase_alarm`, all of them — carries:

| field | meaning |
|---|---|
| `phase_seconds` | named, flat intervals: `preamble`, `snapshot`, `load_weights`, `verify`, `attn4`, `lora`, `enable`, `tokenizer`, `census`, `trainable_sha`, `counters`, `c1_before`, `eval0`, `optimizer`, then `eval_final`, `c1_after`, `adapter_save` |
| `prologue_s` | process start → the instant the timed training window opens, measured independently of the parts |
| `prologue_unattributed_s` | `prologue_s` minus the named prologue phases — **the residual, always present** |
| `phase_budget_s` | what the prologue was judged against on this arm (null = recording only) |

The residual is the point. #548 was raised because ~33 min sat in a window the receipt could not describe, so a
breakdown that can silently omit a phase would reproduce the same failure with more JSON. If a future prologue
surprise lands somewhere nobody has named, it shows up as a number in `prologue_unattributed_s` first.

`load_s` is unchanged and still means the whole `load_fn` call. It always *did* include the attention-4-bit
conversion, `add_attention_lora` and `enable_fast_train` — those run inside `load_e4b` — so at `load_s = 29.4 s`
those three together could never have been the missing half hour. `phase_seconds` is what says which of them it was.

The arm also prints `PROLOGUE {...}` as it opens the timed window, and `tp4_run.sh` copies that line into
`summary.txt` beside the `CELL` line, so the box's own summary answers the question without fetching receipts.
`tp4_reduce.py` prints one bullet per arm under the support table.

### The watchdog: loud, not after the fact

`tp4_run.sh` exports `TP4_ARM_ALARM_S` — the same `A` it gives `perl -e "alarm $A"` — and the arm gives the whole
prologue `PROLOGUE_BUDGET_SHARE` (0.35) of it. Pass the budget through the prologue and the arm **refuses itself
from inside the offending phase**: `PHASE ALARM` on stdout, `faulthandler.dump_traceback()` for every thread, a
receipt with `status: phase_alarm` naming `phase_in_flight` and every phase already closed, and **exit 16**.

It replaces a real tp4 row — `status: alarm`, *"the process could not write its own stub"* — where SIGALRM killed an
arm 3570 s into its prologue and no phase could be named. A share of the alarm rather than a literal is deliberate:
the budget has to mean *"this arm can no longer finish"*, which is a fact about the rental window the run script
chose, not about seconds. `tp4_reduce.py` maps `phase_alarm` onto the already-registered word **ALARM**, so the
status vocabulary and every verdict are unchanged — what changed is that the row can name the phase.

Turn it off with `--phase-budget-s 0` or `TP4_PHASE_BUDGET_S=0`; `phase_seconds` is recorded either way.

### What is already ruled out, and what is not

Measured, not assumed — and none of it names the missing half hour, which is the point of shipping the
instrumentation rather than a theory.

From the committed Qwen3-30B-A3B receipt `bench/h2h-20260906/tp2/qwen3_e4b_fused_attn4.json` (same family and card
class as #548, **tp2's fixture**: clinical, seq 512, batch 1 × accum 1, N 60):

| phase | what the receipt or a measurement says |
|---|---|
| `load_weights`+`verify`+`attn4`+`lora`+`enable`+`tokenizer` | `load_s = 24.3 s` — the whole loader, already including the three steps #548 lists as invisible |
| `c1_before` + `c1_after` | `C1_bytes_hashed = 16,760,438,784` (15.61 GiB, 384 tensors) × 2 passes ≈ **33 s** of host work |
| `trainable_sha` | `trainable_params = 321,257,472` → 1.20 GiB of fp32 ≈ **1.3 s** of host work |
| a held-out eval of 48 rows | `window_wall_s − train_wall_s = 88.77 s` over 3 in-window evals → **29.6 s each** |

The C1 and `trainable_sha` figures are `numpy.tobytes()` at 1.94 GiB/s and `hashlib.sha256` at 1.82 GiB/s, measured
on an Apple-silicon Mac; they are a **lower bound** on those phases as they run on a rented 5090 because they
exclude the device→host copy, and the rented host's sha256 rate is its own.

Two things that bound is NOT:

1. **`eval0` is the one prologue phase nothing measured bounds.** It is the first forward the process ever runs, so
   any Triton/JIT compilation or autotuning on the forward path is billed to it and to nothing else. 29.6 s is the
   *steady-state* cost of the same work later in the arm; `eval0` is that plus a first-call cost no existing receipt
   separates. `phase_seconds` isolates it for the first time.
2. **The fixtures differ.** #548's arm ran the Alpaca fixture (seq 2048, micro-batch 2 × accum 4) at ~24.67 s/step
   against tp2's 4.11 s/step — 6× — so every row above scales, and eval rows at 4× the sequence length do not scale
   linearly. The table says which phases are *small at a fixture we have a receipt for*; it does not transfer.

**So: the ~33 min is not attributed here, and attributing it needs a GPU.** The arms that produced it ran under
another lane on a rented box, and reproducing it means renting — which is pre-registration-and-approval work, not
something this change does. What this change guarantees is that the *next* large-family arm answers the question
from its own receipt.
