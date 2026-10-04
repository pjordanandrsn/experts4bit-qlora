# SC2b: does a CUDA-graphed prefill chunk lift e4b `serve_paged`'s request-level capacity? The prefill graph OFF against ON, paired, on today's serving stack, one RTX 5090 (lane SC2b of #846; registered 2026-10-04)

Registered 2026-10-04, before any SC2b run. Reviewed before registration by the maintainer session ("approve with
4 changes and 3 small ones"; all applied: the licence decoupled from P1, P1's basis, the default `auto` and its scope, early
engagement, the OFF-arm wording, the error record, the VRAM record).

**The code under test** is e4b `373c89ac` (#1070, the knob's merge). The box runs this registration's merge commit, whose
`experts4bit_qlora/` tree must equal `373c89ac`'s except for the version string; the launch chain checks this.

**The `/health` contract**, read from the knob's code at `ec5beacc` and unchanged at `373c89ac`:
- the engaged `status` is `"on"`;
- the startup verification's replays are not counted, so `replays` equals every request admitted.

**The proof model verifies.** The maintainer's GPU tests ran a tiny GraniteMoE NF4 store under SC1's `GR_ENV`: it
engaged, passed the startup check, and its prefills were bitwise against eager. The knob's lifetime fix (the graph keeps
everything it reads; the check runs after the capture returns and the allocator churns; a mutation test is refused) is
in `373c89ac`.

## Why this lane

SC2 (`bench/h2h-2026-10-02/sc2/`, corrected in #1061) read e4b `serve_paged`'s capacity ceiling at 1 req/s, against
vLLM's and SGLang's 8. Every SLO miss was a TTFT miss. e4b's own trace fit each request's decode time as 6.15 ms/token
plus 0.356 s for every other request's 512-token prefill that landed while it decoded (R² 0.985). So each prefill
forward stalls every running decode, and capacity is set by the cost of that forward.

That forward is launch-bound:
- P102 counted about 15k device kernels per 512-token chunk, and its TTFT-512 moved 4× with the host CPU alone on the
  same GPU class.
- The maintainer's census (A2000, e4b `6178a15b`) found zero host syncs inside the prefill forward, and a 512-token
  first chunk that captures as-is and replays bitwise-equal to eager.

The lever is `E4B_PAGED_PREFILL_GRAPH=1`:
- it captures the first-chunk prefill once and replays it per prompt;
- it verifies itself at startup (capture, replay, bitwise against eager in logits and every layer's K/V), and refuses
  otherwise;
- it covers every prompt of at most `chunk_tokens` (512) tokens, which is all of SC2's workload.

SC2's e4b rows ran SC1's inherited prefill route pins (M-tile + `math`) on gnf4 v0.34.1. This lane's OFF arm is main's
default routes (k19 + flash) at `373c89ac` with gnf4 v0.38.0. That is today's server, **not** a re-read of SC2's registered
configuration (main's defaults at `887940e` with v0.34.1). Its numbers stand beside SC2's as a description, never as
SC2's missing row.

## The box (F) and the stack it pins

- **One box, e4b only:** `SC1_BOX=F` (`bench/sc2/sc2b_box_f.sh`, sourced after `sc2_box_e.sh`).
- **e4b**: the code at `373c89ac` (the knob's merge), with grouped-nf4-gemm **v0.38.0** (`5a887c48`, tag object `b7b5ef74`):
  KV-step-select on, `GNF4_PDL` capped at 8 rows, `GNF4_TRAIN_GEMM=auto`. SC1's v0.34.1 is kept for boxes A–E.
- **Arm `e4b_int4`.** SC2's e4b_int4 levers (`SPEEDENV`, the three folds, `E4B_PAGED_FUSE_QKV=1`, `ROUTEENV`), with
  `max_seqs` 16 and `E4B_PAGED_MAX_TOKENS_PER_SEQ=2048`. The only difference between OFF and ON is
  `E4B_PAGED_PREFILL_GRAPH=0|1`.
- **Prefill routes are main's defaults, checked in the server's environment, not the launch text.**
  - Box F exports neither `E4B_INT4_PREFILL` nor `E4B_PAGED_PREFILL_ATTN`, and every e4b server is started with
    `env -u` on both.
  - `tests/test_sc2b_box.py` runs `sc1_run.sh`'s export block and reads the child's environment.
  - Every server's own `/health` `prefill_routes` must read `int4_prefill` k19, `int4_prefill_above_256_rows` k19,
    `prefill_attn` flash, `device_grouping` true, and both raw env values null.
  - Its `engine` block must show `chunk_tokens` 512 and `max_prefill_tokens_per_step` 512 (the default budget). A larger
    budget could split a first chunk, and every 512-token prompt must be one first chunk.
  - Otherwise the arm **STOPs** (rc 47).
- **Engagement is checked early, before the paid workload.** Right after each server's 4 warm requests, its `/health`
  must already read as registered: OFF `status` "off"; ON `status` "on" with `replays` 4 and `eager_chunks` 0. Every warm
  response must report `usage.prompt_tokens` 512. Otherwise the arm **STOPs** (rc 48). A token drift between the pool
  and the server would otherwise only VOID at the end. The proof checks this on Granite's pool; this check covers
  Qwen3's.
- **Records, no gate:** the full `/health` JSON whenever a server reports `error` (a log line truncates
  `prefill_graph.why`), and `nvidia-smi` memory used at ready for every arm. The graph's private pool costs VRAM;
  full-sequence logits alone are about 155 MB on Qwen3.
- **Image and checkpoint:** SC1's `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`; Qwen3-30B-A3B @ `ad44e77`, baked to the
  NF4 arena on the box.

## The instrument

- **Driver and prompts:** SC2's driver and prompt pool, unchanged. Each request is one 512-token wikitext-2 row with
  `max_tokens` drawn from U[64, 256]: greedy, `ignore_eos`, streaming, 16 in flight. TTFT, TPOT and SLO attainment are
  as SC2 defines them (TTFT ≤ 1.0 s and TPOT ≤ 100 ms).
- **Paired arms.** The knob is read at server start, so each (draw, arm) is its own server: draw 1 runs OFF then ON,
  draw 2 ON then OFF. **Both arms of a draw use the same plan seeds**, so they see identical arrivals. That fixes, for
  this A/B, the seed effect that made SC2's knee rows UNSTABLE; between draws it remains.
- **Per server:**
  1. 4 warm serial requests;
  2. Q1 serial, 24 requests;
  3. Q2 Poisson at 1, 2, 4 and 8 req/s, 120 requests each;
  4. `/health` at start (routes) and at the end (engagement).
  Draw 1's OFF server also repeats the serial plan, as the determinism control.

## The rule (`bench/sc2/sc2b_reduce.py`, self-tested on 10 cases)

The gates run in order:
1. **ROUTES.** Every arm's start `/health` reads the registered routes, or the run is VOID.
2. **ENGAGED.**
   - OFF servers carry `prefill_graph` with `status` "off" and `requested` false.
   - ON servers carry `status` "on", `requested` true, `T` 512, `replays` equal to every request the server admitted
     (warm-up included; the startup check's replays are not counted), and `eager_chunks` 0.
   - A refused or errored ON server is not engaged.
   - Otherwise VOID.
3. **DETERMINISM.** OFF draw-1 serial against its repeat must be IDENTICAL (`sc2_identity.py`: every request's
   streamed text byte-equal), or the identity gate is UNREAD.
4. **IDENTITY.** OFF against ON on each draw's serial plan must be IDENTICAL; otherwise DIFFERS, and the lever cannot be
   a default.
5. **PROMPTS.** Every request in every run must report `usage.prompt_tokens` 512, one first chunk each. Otherwise VOID.

**Rows and ceilings** use SC2's rule (`sc2_reduce.row` / `ceiling`) per arm.

**Predictions**, paired per draw:

| # | prediction | basis |
|---|---|---|
| P1 | serial p50 TTFT, OFF / ON, ≥ 1.5 in both draws (a prediction only; it does not gate the licence) | the census counted ~107 launches per layer, so ~5k per 512-token chunk on 48 layers. At ~10–20 µs of host time each that is ~50–100 ms against ~40 ms of device time, putting OFF / ON at ~1.5–3×. SC2's 0.27 s is **not** OFF's baseline: OFF runs k19 + flash on v0.38.0 and may already be faster. The ~47 ms device figure is P107's 4096-token flash time ÷ 8, unmeasured for a first chunk. 2.0 would be near a coin flip |
| P2 | serial p50 TPOT, ON / OFF, in [0.95, 1.05] in both draws | the graph touches prefill only |
| P3 | ON's capacity ceiling ≥ 2 req/s | SC2's fit: capacity is set by the per-prefill stall, which P1 cuts |
| P4 | no regression: at every rate and draw, ON's attainment ≥ OFF's − 0.05, and ON's ceiling ≥ OFF's | |

**Reported, no bar:** OFF's own rows and ceiling under today's defaults, beside SC2's e4b-as-run (1 req/s; SC1's pins,
v0.34.1, another board). Descriptive only.

**Licence, decoupled from P1.** DEFAULT_LICENSED iff ROUTES, ENGAGED, PROMPTS, DETERMINISM and IDENTITY pass, P4 holds,
**and** serial p50 TTFT OFF / ON is ≥ 1.10 in both draws. A bitwise-identical, non-regressing lever that cuts TTFT
should be the default whether or not it reaches P1's guess. Otherwise NOT_LICENSED, with the reason.

**What the default would be.** On DEFAULT_LICENSED, the maintainer's release makes `E4B_PAGED_PREFILL_GRAPH` default to
**`auto`**:
- it engages wherever the per-server startup check passes;
- where the check refuses (`max_seqs` 1 / no device grouping, a hybrid linear-state model, a capture that raises, a
  mismatch), the server runs eager prefill and `/health` reports `status` "refused" with `requested` "auto";
- an explicit `=1` keeps refusing at startup.

A default that refused at startup would break every `max_seqs` 1 server and every hybrid.

**The licence's scope.** Speed is read on Qwen3-30B-A3B int4, `serve_paged`, one RTX 5090, 512-token prompts, and only
there. Correctness elsewhere is the per-server startup check's job (bitwise against eager on every engaged server).
Speed elsewhere is unread.

**No position against vLLM, SGLang or llama.cpp comes from SC2b.** It is e4b against itself on one box. SC2's comparator
rows stand.

## Proof and budget

**Proof** (`sc2b-prove-*`, guard 1.0 h):
- self-tests (driver, SC2 reducer, identity, SC2b reducer);
- the proof model's prompt pool;
- an OFF server and an ON server on the proof model (Granite-3.1-3b-a800m, NF4 store, SC1's `GR_ENV` unchanged), each on
  the registered routes,
  each answering a 6-request serial smoke and a 16-request Poisson smoke with every request VALID;
- the serial smokes IDENTICAL OFF against ON;
- the knob ENGAGED on the ON server and absent on the OFF server.

**Reading** (`sc2b-5090-*`, guard 2.5 h):
- installs about 10 min, fetch and bake 25–40 min, four servers at about 9 min each.
- **Guards counted honestly, after SC2.** Hours × $0.75 plus download at $0.0078/GB (about 70 GB for the reading, about
  10 GB for the proof).
  - proof: 1.0 h ≤ $0.75 + about $0.10;
  - reading: 2.5 h ≤ $1.88 + about $0.55;
  - lane, both guards: about $3.3, each run under the $15 no-ask tier.
- The deadline drops draw 2 first; draw 1 alone reads every gate except P1/P2/P4's "both draws".

## Out of scope

- Prompts over 512 tokens (later chunks stay eager in phase 1). Per-chunk-index graphs are phase 2.
- The `lm_head` last-row restriction.
- Comparators.
- The NF4 expert store in the reading.
