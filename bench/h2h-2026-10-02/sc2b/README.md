# SC2b: e4b's prefill graph is value-identical, cuts serial TTFT 1.30–1.65×, and is licensed as a default (`auto`); capacity stays at 1 req/s, because most of the stall a prefill imposes under load is not the graphed forward (lane SC2b of #846; 2026-10-04)

Pre-registration: [`../../sc2/SC2b-PREREG.md`](../../sc2/SC2b-PREREG.md) (#1072, reviewed by the maintainer before it was
registered). The lever: `E4B_PAGED_PREFILL_GRAPH=1` (#1070, `373c89ac`). It captures the first 512-token prefill chunk
once, verifies the replay bitwise against eager at startup (logits and every layer's K/V), and replays it per prompt.

| run (receipt) | e4b | host (driver; board power limit) | outcome | $ |
|---|---|---|---|---|
| `sc2b-prove-1` (adertha-receipts `b4f13d8b`) | `f8c9dde` | AMD EPYC 7C13, 256 vCPU (595.71.05; **400 W**, 3090 MHz), machine 45511 | **PROVED box=F**: routes, engagement and identity on Granite NF4, every smoke request VALID | 0.240 |
| `sc2b-5090-1` (`93d86fb5`) | `f8c9dde` | same machine, 45511 | OK: four servers (draw 1 OFF→ON, draw 2 ON→OFF), every gate passed | 1.326 |

SC2b total: **$1.566 across 2 receipts**. Both runs came in under their registered guards, which now include download
charges (proof ≤ $0.85, reading ≤ $2.43).

**Stack.**
- e4b `f8c9dde`, whose package code is `373c89ac` plus the version string; the launch chain checks this.
- grouped-nf4-gemm v0.38.0 (`5a887c48`).
- **Prefill routes:** k19 / k19 / flash with device grouping on, read from every server's own `/health`. Neither SC1
  pin was in any server's environment.
- **The arm:** e4b_int4 with SC2's levers on Qwen3-30B-A3B. The knob was the only difference between the arms.

**The board.** Same board as SC2's reading (machine 45511, capped at 400 W), so absolute times are a 400 W 5090's.

**What the OFF arm is.** It is today's default server, **not** a re-read of SC2's registered configuration (main's
defaults at `887940e` with v0.34.1). It is reported beside SC2's e4b-as-run as a description only.

## The outcome by the registered rule

Full tables: [`receipts/sc2b-5090-1/RESULTS-sc2b.md`](receipts/sc2b-5090-1/RESULTS-sc2b.md). `sc2b_reduce.py` re-derives
`verdict_sc2b.json` identically from the committed run files.

| gate | result |
|---|---|
| ROUTES | pass: every server k19 / k19 / flash, device grouping on, both pins null, chunking 512 / 512 |
| ENGAGED | pass: OFF `status` "off"; each ON server `status` "on", **508 replays = 508 requests admitted, 0 eager chunks**. The early check after warm-up passed on all four servers |
| PROMPTS | pass: every request reported 512 prompt tokens |
| DETERMINISM | IDENTICAL: OFF draw-1 serial against its repeat, 24 requests, 17,701 characters |
| **IDENTITY** | **IDENTICAL on both draws:** every request's streamed text byte-equal, OFF against ON (24 requests each; 17,701 and 15,204 characters) |

| arm | serial p50 TTFT, draw 1 / draw 2 | serial p50 TPOT | attainment at 1 / 2 / 4 / 8 req/s (draw 1, draw 2) | ceiling |
|---|---|---|---|---|
| OFF | 0.263 / 0.222 s (UNSTABLE) | 4.51 / 4.48 ms | 1.00, 0.98 / 0.57, 0.83 (UNSTABLE) / 0.10, 0.19 (UNSTABLE) / 0.04, 0.03 | 1 |
| ON | 0.160 / 0.171 s | 4.46 / 4.52 ms | 1.00, 1.00 / **0.90, 1.00** / 0.12, 0.29 (UNSTABLE) / 0.06, 0.06 | 1 |

| prediction | verdict | detail |
|---|---|---|
| P1 serial TTFT OFF / ON ≥ 1.5 in both draws | **REFUTED** | 1.645 (draw 1), 1.299 (draw 2) |
| P2 serial TPOT ON / OFF in [0.95, 1.05] | **HOLDS** | 0.988, 1.009 |
| P3 ON's capacity ceiling ≥ 2 req/s | **REFUTED** | 1. At 2 req/s ON reached 0.90 and 1.00 attainment; the ceiling needs ≥ 0.95 in both draws |
| P4 no regression | **HOLDS** | ON ≥ OFF − 0.05 at every rate and draw (it improved at 2 and 4 req/s); ceiling 1 ≥ 1 |
| **Licence** | **DEFAULT_LICENSED** | Every gate passed, P4 held, and TTFT OFF / ON was 1.645 and 1.299, both ≥ 1.10 |

**What the licence means**, as registered. The maintainer's release can make `E4B_PAGED_PREFILL_GRAPH` default to
`auto`:
- it engages wherever the per-server startup check passes;
- elsewhere the server runs eager prefill and reports "refused";
- an explicit `=1` keeps refusing.

**Scope:** speed was read on Qwen3-30B-A3B int4, a 400 W 5090, and 512-token prompts. Correctness elsewhere is the
per-server startup check's job.

## What it means

**The graph does what it was built to do on the serial path.** First-token time fell from 0.22–0.27 s to 0.16–0.17 s
with identical output, and decode pace was unchanged. The two draws' ratios differ (1.65 against 1.30) because the
OFF arm is **host-sensitive** and the ON arm is not:
- OFF's serial TTFT was 0.263 s (repeat 0.269 s) in draw 1, which ran right after the bake at load average 66, and
  0.222 s in draw 2, at load 19.
- ON's was 0.160 s and 0.171 s.

Taking the host out of the prefill is the point, and OFF's own spread shows it.

**But capacity stayed at 1 req/s.** This analysis is post hoc and descriptive, from each server's own trace; the tool
is `sc2_trace.py --plan` and the fits are pinned by `tests/test_sc2_trace.py`. Each request's decode time fits:

| server | ms per token | stall per other request's prefill landing during decode | R² |
|---|---|---|---|
| OFF draw 1 | 5.68 | 0.326 s | 0.987 |
| OFF draw 2 | 6.10 | 0.313 s | 0.988 |
| ON draw 1 | 6.06 | 0.262 s | 0.985 |
| ON draw 2 | 6.13 | 0.269 s | 0.990 |

- **The stall shrank much less than TTFT.** The graph cut serial TTFT by 23–39% (draw 2, draw 1) but the per-prefill
  stall under load by only about 17% (0.32 s to 0.27 s, means of the two draws).
- **Most of the stall is outside the forward.** Even with the graph, the stall each prefill imposes on its neighbours
  (0.27 s) is about 1.6× the whole serial TTFT (0.16 s), so most of it under load is per-prefill work outside the
  graphed forward. Plausible candidates are named, not measured: the prompt's K/V staging and flush into the FP8 pool,
  and host bookkeeping that grows with the 16 live streams. The maintainer's phase 2 (a static staging buffer) is
  aimed at the first.
- **Where it helps.** TTFT at moderate load improved sharply: p50 at 2 req/s went from 0.77 s to 0.28 s in draw 1, and
  from 0.36 s to 0.19 s in draw 2. That is enough to lift draw 2's attainment at 2 req/s to 1.00, but not draw 1's
  (0.90).

**The cost is memory.** `nvidia-smi` at ready read 20,308 MiB with the knob OFF and **23,686 MiB ON (+3,378 MiB)**, out
of 32,607. That is the graph's private pool for a captured 512-token MoE forward, far more than the full-sequence logits
(~155 MB) the design estimate counted. An `auto` default should know it costs about 3.3 GiB on this model. Restricting
`lm_head` to the last row would trim only the logits part. This was recorded only, with no gate.

**Beside SC2 (descriptive; different gnf4 and e4b, same board).** SC2's e4b-as-run (SC1's pins: M-tile + `math`)
read 0.25–0.29 s serial TTFT and a ceiling of 1. Today's defaults (k19 + flash) read 0.22–0.27 s OFF, also ceiling 1.
On this host the route change barely moves prefill, consistent with prefill being host-bound.

## Next

- **The lever after this one** is the per-prefill work outside the forward, which now dominates the stall under load.
  Measure it first: a census of what a prefill step does besides the graphed forward (K/V flush, staging, scheduler,
  SSE), then the maintainer's phase 2. A chunked prefill that interleaves with decode is the other route.
- **The default decision** follows the licence. Its memory cost (+3.3 GiB at `max_seqs` 16 on Qwen3-30B-A3B) is stated
  here for the release note.

## Reproduce

- **Rerun:** `SC1_BOX=F bash bench/sc1/sc1_drive.sh` at `f8c9dde`.
- **Re-reduce:** `python bench/sc2/sc2b_reduce.py --dir receipts/sc2b-5090-1/sc2 --out verdict_sc2b.json`.
- **Re-fit the stall:** `python bench/sc2/sc2_trace.py receipts/sc2b-5090-1/sc2/trace_e4b_on_d1.jsonl --plan warm:4,serial:24,r1:120,r2:120,r4:120,r8:120`
  (OFF draw 1 adds `serial_repeat:24` after `serial`).
- **What the receipts leave out:** the driver records drop per-token chunk gaps and the prompt pool is left out. Both
  are complete in the receipt store.
