# SC2d: bulk KV bookkeeping engages and changes nothing on a hybrid model (Qwen3.6-35B-A3B) and on gpt-oss-20b, FLIP_LICENSED (lane SC2d of #846; 2026-10-05; correctness only)

Pre-registration: [`../../sc2/SC2d-PREREG.md`](../../sc2/SC2d-PREREG.md) (#1176, reviewed by the maintainer before it
was registered). These are the two engagement reads SC2c registered for its flip ([`../sc2c/README.md`](../sc2c/README.md),
#1166: DEFAULT_LICENSED on Qwen3-30B-A3B), from #1131's review:
- a hybrid model, whose KV pool holds only its attention layers;
- gpt-oss, which carries attention sinks.

Engagement is read with #1174's `flush_bulk_fallback`. It counts what *ran*, not what was called.

| run (adertha-receipts commit) | host (driver; board power limit) | outcome | $ |
|---|---|---|---|
| `sc2d-prove-1` (`1c4fa968`) | AMD EPYC 7702P, 126 vCPU (595.71.05; 575 W), machine 45379 | **PROVED box=K**: the #1174 tripwire, self-tests, Granite smokes, and box K's whole flow on Granite read ENGAGED_IDENTICAL | 0.121 |
| `sc2d-5090-1` (`6894127e`) | machine 46782 | NOT_RUN: pre-flight ssh never authenticated (a dud host); nothing ran | 0.002 |
| `sc2d-5090-2` (`b1d83845`) | machine 40093 (570.133.07) | HARNESS_ERROR: the box refused at rc 18, driver < R580, before any workload | 0.025 |
| `sc2d-5090-3` (`ade38b20`) | — | REFUSED by the launcher (an exclusion flag that takes NOT_RUN receipts only); no box | 0.000 |
| `sc2d-5090-4` (`f650e0e5`) | AMD Threadripper PRO 3955WX, 32 vCPU (590.48.01; 575 W, 3090 MHz), machine 26157 | **OK**: four servers, every gate passed on both models | 0.739 |

SC2d total: **$0.887 across 5 receipts.**
- Every run came in under its guard (proof ≤ $1.95, reading ≤ $3.44), and each sat under #846's standing no-ask tier.
- The three failed launches were host or launcher refusals before any workload, so none is a reading.
- The launcher rents offers whose driver is older than the CUDA 13 image's floor. This was flagged on the bus and not
  changed here.

**Stack.**
- e4b `8245ad32`, which is #1176's merge.
  - The serving files this read exercises are byte-identical to #1174's merge `16074418`. The launch script checked
    this before renting: `serve_paged.py`, and in `engines/` `paged_runner`, `fp8_paged_kv`, `linear_state`,
    `paged_attention`, `scheduler` and `step_trace`.
- grouped-nf4-gemm v0.41.0 (`dc8f94ab`).
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1.
- Main's defaults: no route pins, decode graphs `auto`, the prefill graph `auto`.
- **gpt-oss-20b** at `6cee5e81`, on SC2g's e4b path.
- **Qwen3.6-35B-A3B** at `995ad96e`, on the server's defaults over P98's NF4 arena (baked on the box in 42 s). This
  is its first run through `serve_paged`'s HTTP server.

## The outcome by the registered rule

`sc2d_reduce.py` re-derives `verdict_sc2d.json` byte-identically from the committed run files (see Reproduce).

| gate | gpt-oss-20b | Qwen3.6-35B-A3B |
|---|---|---|
| SERVED | OFF 36 / 36 VALID, ON 20 / 20 | OFF 36 / 36 VALID, ON 20 / 20 |
| ARCH (the checkpoint's own facts) | 24 layers, all attention (full and sliding alternating), **`self_attn.sinks` on 24 of 24** | 40 layers, **10 attention + 30 linear** (Gated DeltaNet): the pool holds 10 |
| PROMPTS | one pool (gpt-oss tokenizer) in both arms | one pool (Qwen3.6 tokenizer) in both arms |
| ENGAGED, OFF | `flush_layers` 36, `ready_layers` 36, all bulk counters 0 | `flush_layers` 36, `ready_layers` 36, all bulk counters 0 |
| ENGAGED, ON | **`flush_bulk` 20, `flush_bulk_fallback` 0**, `ready_at_flush` 20, `flush_layers` = `ready_layers` = 0 | **`flush_bulk` 20, `flush_bulk_fallback` 0**, `ready_at_flush` 20, `flush_layers` = `ready_layers` = 0 |
| DETERMINISM (OFF serial vs its repeat) | IDENTICAL (16 requests, 9,906 characters) | IDENTICAL (16 requests, 11,516 characters) |
| **IDENTITY** (OFF serial vs ON serial) | **IDENTICAL** (16 requests, 9,906 characters) | **IDENTICAL** (16 requests, 11,516 characters) |
| **outcome** | **ENGAGED_IDENTICAL** | **ENGAGED_IDENTICAL** |

**The flip: FLIP_LICENSED.** Both models are ENGAGED_IDENTICAL. As registered, the flip PR makes `E4B_PAGED_BULK_KV`
default to `1`, keeping `0` as the escape, and cites this read with SC2c's. The prediction (both ENGAGED_IDENTICAL)
held.

**What it shows.**
- **The bulk path ran on every prompt.** On every ON server the bulk path *wrote* every prompt: `flush_bulk` equals the
  requests admitted, with zero per-layer fallbacks. Every slot's block claims were made at the flush, so its first
  graphed decode claimed nothing.
- **The pool-layer subset.** On Qwen3.6 the flush and the claims ran over the 10 attention layers the pool holds.
- **Sinks and sliding layers.** On gpt-oss they ran over 24 layers that carry sinks and alternate sliding and full
  windows.
- **Output identical.** In both models every request's streamed text was byte-equal between the arms.

**Recorded, no gate** (from each server's `/health`):

| | gpt-oss-20b | Qwen3.6-35B-A3B |
|---|---|---|
| prefill graph | `on` in both arms: replays = prompts, 0 eager chunks, pool 246 MiB, 5,796 MiB free after. ON also counted the bulk flush's transient (168 MiB) in its headroom | `refused` in both arms, by design: a hybrid's linear-attention state is per slot and prefill mutates it |
| decode graphs | every bucket (1–16) captured | every bucket (1–16) captured |
| VRAM at ready | 26,316 MiB, both arms | 25,470 MiB, both arms |

The driver also recorded serial p50 TTFT:
- gpt-oss: 163 ms OFF, 111 ms ON;
- Qwen3.6: 336 ms OFF, 317 ms ON.

TPOT was within 3% between arms (gpt-oss 6.16 / 6.03 ms, Qwen3.6 13.06 / 13.00 ms). **As registered, these are not read as speed:** one box, 16 requests, no draws,
nothing preregistered on them.

## Scope

**What the read covers.** Correctness and engagement on two real checkpoints through the HTTP server:
- 512-token, single-chunk prompts;
- serial requests;
- one 5090.

**What licensed speed.** SC2c's Qwen3-30B-A3B reading did, and nothing here adds to it.

**What it does not cover.**
- Other families: the bitwise tests cover equivalence there, and every server's `kv_bookkeeping` reports engagement.
- Longer prompts.
- Concurrency beyond the serial plan.

## Reproduce

- **Rerun:** `SC1_BOX=K bash bench/sc1/sc1_drive.sh` at `8245ad32`.
- **Re-reduce:**
  `python bench/sc2/sc2d_reduce.py --dir bench/h2h-2026-10-02/sc2d/receipts/sc2d-5090-4/sc2 --out verdict_sc2d.json`.
  The result is identical to the committed `verdict_sc2d.json`.
- **What the receipts leave out:** per-token chunk gaps are dropped from the driver records (`trimmed`), and the prompt
  pools and logs are left out. All are complete in the receipt store at `f650e0e5`.
