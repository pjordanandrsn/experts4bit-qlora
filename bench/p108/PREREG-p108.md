# P108 — is e4b's paged path on Gemma-4 worse than transformers' own forward, judged against transformers' own chaos? Gemma-4-26B-A4B on one RTX 5090, 32 wikitext windows with the sliding window binding, the paged path set against a floor of arithmetically neutral perturbations of the same model (registered 2026-10-03, before any run)

Issue: experts4bit-qlora#359. Uses the two fixes that preparing this lane turned up: #964 (`serve_paged` sizes a Gemma-4
KV pool) and #966 (the unbound fallback keeps sliding windows), both shipped in 0.41.1.

**Why.**
- **Gemma-4 has no parity reading.** `docs/SERVING-PARITY.md` carries it as "no reference at this resolution" (P27,
  2026-09-03).
- **One window is chaos.** On three 512-token windows the paged path read +0.247 / +0.093 / +0.114 nats against a
  one-shot forward. transformers' own cached forward read +0.081 / −0.107 / +0.271 against the same one-shot forward.
- **The cause is the model, not e4b.** Gemma-4 amplifies bf16 batch-shape variance in the per-expert GEMMs: the NLL of
  identical tokens moves by tenths of a nat with only what follows them.
- **What remains unknown** is whether the paged path carries a BIAS beyond that chaos. The fp8 KV's own cost was
  estimated at 0.017–0.046 nats on one window (fake-quant, #359).
- **P27's windows never bound the 1,024-token sliding window**, so it never touched the paged path's sliding windows.
  Prompts here are 1,280 tokens.

**The method follows P67's training floor** (`bench/p67/`): the reference is re-run under arithmetically neutral
perturbations, the spread of those draws is the floor, and the subject is judged against it.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

**Setup.**
- **Checkpoint:** `google/gemma-4-26B-A4B-it` at `4d7ae49` (P67's revision), loaded once through e4b's streaming
  loader (NF4 experts on the GPU).
- **Paged attention is registered.** With no paged context bound, the forward is transformers' own with e4b's unbound
  fallback, which carries the window and the last-key causal alignment (#966).
- **Text:** wikitext-2 test (the K8 corpus). Window k starts at token k × 4,096.
- **Windows:** **32**, each a **1,280-token prompt** and a **256-token teacher-forced continuation**, processed in
  groups of **8**.
- **Scoring:** per window and arm, the continuation's 256 positions on fp32 log-probs: the true token's NLL, the
  argmax, and KL against the reference.

**Arms** (`p108_box.py`):
- **R (the reference):** a DynamicCache, batch 1. The prompt is prefilled in one call, then 255 cached decode steps.
- **rep:** R again, first group only. Bit-identity is recorded; if it is not identical, rep joins the floor.
- **Floor draws** (the same model and weights, arithmetically neutral changes):
  - **oneshot:** one forward over prompt and continuation, no cache;
  - **chunk:** R with the prompt prefilled in 256-token pieces, the paged path's chunking;
  - **batch:** R with the group's 8 windows batched (equal lengths, no padding).
- **paged (the subject):** `PagedModelRunner`, the group's windows on their own slots, prefilled in 256-token chunks
  and decoded together through the fp8 KV pool and gnf4's real fp8 kernel. This is P97's `_paged_pass` at its
  registered bytes.
- **mutant_scale:** the subject with the decode attention's softmax scale halved. The bar must catch it.
- **mutant_window:** the subject with decode attention ignoring the sliding window. Reported only: whether a dropped
  window is visible at this length.
- **Engagement:** every decode attention call is counted with its layer and window:
  - 255 × 30 per group;
  - 255 × 25 of them at window 1,024.

**The premise** (on this card, before anything is fetched): `tests/test_gemma4_paged_window_gpu.py`, **2 passed**, none
skipped.
- **Setup:** a tiny dense Gemma-4 whose window binds, through the paged path. One variant stands SDPA in for the
  decode attention; the other runs gnf4's REAL fp8 kernel.
- **Bar:** within 2× a non-binding-window control on every one of 8 seeds.
- **The proof it can fail:** on the A2000 (stand-in), a mutant that ignores the window fails at 13.5–27.2× on every
  seed, while the test itself reads 0.55–1.33×.

**The rule** (`p108_reduce.py`, self-test 17 cases). Per window w and arm X, d_X(w) = mean NLL of the continuation under
X minus under R.
- **The floor:** `B_floor` = max over floor draws of |mean_w d_f|, and `S_floor` = max of mean_w |d_f|.
- **The subject:** bias `b` = mean_w d (positive means paged is worse), and spread `s` = mean_w |d|.
- **passes(X):** `b_X ≤ B_floor + 0.05` and `s_X ≤ 2 × max(S_floor, 0.01)`.
  - The spread term also catches a paged path that is much *better* than R, which is what a future-token leak reads
    like.
- **NO_READING:** no premise line saying it held, or no box record.
- **VOID**, any of:
  - the loaded commit is not `4d7ae49`;
  - fewer than 32 windows, or an arm short of its windows;
  - engagement counts or windows not exactly as above;
  - `mutant_scale` passes (the gate cannot fail).
- **AT_PARITY** if the subject passes; **COST** otherwise.
- **Reported:** every arm's bias, spread, standard error, mean KL against R and argmax agreement; rep's bit-identity;
  mutant_window.

**The registered consequence:**
- **AT_PARITY:** `docs/SERVING-PARITY.md`'s Gemma-4 row moves from "no reference at this resolution" to "at parity
  with transformers' own perturbations", with the bias, spread and floor. A register row. #359 closes.
- **COST:** SERVING-PARITY.md records the bias above the floor. `docs/SERVING.md` notes it for Gemma-4. A register row.
  #359 stays open with the number.
- **VOID / NO_READING:** nothing moves; the reason goes on #359.

**Not measured:**
- context beyond 1,536 tokens;
- decode graphs (the box decodes eager);
- `serve_paged`'s `build_engine` and scheduler (the box drives `PagedModelRunner` directly, as P97 did);
- any corpus but wikitext-2;
- any host but this one.

## Predictions (written 2026-10-03, before the A2000 rehearsal's output and before any 5090 data)

- **AT_PARITY.**
- **The floors:**
  - each floor draw's |bias| ≤ 0.05 over 32 windows;
  - the floor spread between 0.05 and 0.25 nats (Gemma-4's chaos).
- **The subject:**
  - bias between 0 and +0.05 (the fp8 KV);
  - spread ≤ 1.5 × the floor spread.
- **The reference** repeats bit for bit.
- **mutant_scale:** bias above +0.3.
- **mutant_window:** visible, bias above +0.05 (weak).
- **Peak GPU memory** ≤ 28 GB; **the box** ≤ 60 min.

## Box and cost

- **`p108-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P108_PROVE=1`, no model.
- **`p108-5090-<n>`:** one RTX 5090 (adertha's Vast filter: ≥ 98 GB host RAM, ≥ 320 GB disk). **Guard 2 h at ≤ $0.75/h
  (≤ $1.50).**
- **Lane ceiling $3.50; hard stop $4.50.**

## Rehearsal

**Rehearsed 2026-10-03 on the NAS RTX A2000 (sm_86, driver 575.64.05), at e4b `e9b235f` (0.41.1), staged exactly as
`p108_drive.sh` stages.** The knobs were class A2000, disk 10 GB, RAM 8 GiB and premise skips allowed. Times are the
logs' own stamps. Gemma-4-26B-A4B does not fit the 12 GB card, so the box ran on a tiny model.

1. **The proving path** (`P108_PROVE=1`, `a2000/rehearse.sh`, 13:53:24Z–13:54:41Z): rc 0, with `PROVED` and the
   `REHEARSAL` marker.
   - **Tripwire:** e4b 0.41.1, gnf4 `34da93d`, torch 2.8.0+cu128, transformers 5.17.0. Both fixes are present: #966's
     `_fallback_mask`, and #964's per-layer KV geometry on a tiny Gemma-4 config.
   - **Reducer self-test:** 17 cases.
   - **Premise:** 1 passed, 1 skipped (the real-kernel variant on sm_86). The stand-in variant read 0.55–1.33× its
     control on all 8 seeds.
   - **HF CDN probe:** 56.6 MB/s.
   - **A first attempt** (13:51Z) stopped at the tripwire, on the tripwire's own expectation. Its tiny config lacked
     `attention_k_eq_v=True`, without which Gemma-4's full layer keeps `num_key_value_heads`. Fixed, and the run above
     is the second attempt.
2. **The box on the GPU** (`a2000/box_rehearsal.py`; `p108_box.measure()` on a tiny Gemma-4 MoE, window 16, 40-token
   prompts, 24-token continuations, 8 windows in groups of 4, stand-in attention honouring the window; 13:54:41Z–13:55:11Z).
   rc 0.
   - **Engagement exact:** 276 of 276 decode attention calls, 230 of them at window 16 and 46 at 0.
   - **The reference** repeated bit for bit.
   - **The floors:** |bias| ≤ 0.0012, spread ≤ 0.0013. A tiny random model has no chaos.
   - **The subject:** bias +0.0001, spread 0.0151, KL 7.6e-3 (the fp8 KV).
   - **mutant_scale:** spread 0.0205 against a bar of 0.020. It fails, but barely: on near-uniform logits, halving the
     scale moves little.
   - **mutant_window:** spread 0.032, KL 4.4e-2.
   - **The reducer** read VOID on the window count (8 of 32), by design.
   - These are a random model's numbers. They show the path works, not what the reading will find.

Amendments, dated, go below this line before any data is read.

### Amendment 1 (2026-10-03, written at 15:02Z by `date -u`, before any data is read): the box's time budget

**What happened.** `p108-5090-1` (launched 2026-10-03, lane started 14:14:19Z) loaded the model at 14:29Z. By 15:01Z it
had finished only about one group of four.
- **The evidence:** the box prints one line per paged pass, and two had appeared.
- **The cause:** transformers' batch-1 eager decode of Gemma-4-26B-A4B runs at about 0.25 s a step. The reference, the
  repeat and the chunked floor each take 8 × 255 of those steps per group.
- **The consequence:** the box needs about 95–100 minutes. Its 75-minute alarm (`step_alarm 4500`) ends it at about
  15:44Z, before `box.json` is written. So that run cannot produce a reading, and it is left to end on its alarm.
- **Nothing was read.** No per-window value was seen: the box writes its record only at the end.

**What changes, and only this:**
- **The box's alarm:** 75 → **150 minutes** (`step_alarm 9000` in `p108_run.sh`, re-pinned).
- **The reading's guard:** 2 h → **3 h at ≤ $0.75/h (≤ $2.25)**.
- **The lane ceiling** stays at **$3.50**, hard stop $4.50.
- **A progress line:** the box prints `P108_GROUP <k> of 4 done at <s>` after each group (logging only).

**What does not change:** the windows, the arms, the floor, the rule, the premise, the predictions and the consequence.

**One prediction is already wrong.** "The box ≤ 60 min" is refuted by the timing above, and the read will say so.
