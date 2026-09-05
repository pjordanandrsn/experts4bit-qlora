# Head-to-head lane p37 — 2026-09-05, one RTX 5090: vLLM 0.28.0 vs e4b, same box, same session, identical prompt token ids

**Status: the speed lane is complete — `TP_DONE` 2026-09-05T20:13:55Z (16 registered arms, every receipt written, no alarm,
no init failure, no DRIFT) and `TP2_DONE` 20:15Z (`p37b.sh`, the amendment-2 follow-up: `NOT NEEDED`, the comparator
had run in-lane).** The reducer's registered verdict is **NO RATIO QUOTED against e4b's licensed stack**: every e4b
licensed arm is **VOID under the pre-registered pack-fingerprint rule** — this box's streamed calibration packed
**11522 gptq / 766 rtn** expert matrices where the licensed 64k pack (bo6b, bo6c, bo7) reads **11512 / 776**: the same
recipe, ten of 12,288 matrices classified the other way at the `min_rows` threshold, therefore *not the licensed bytes*,
and speed cannot inherit a licence. What IS quoted: vLLM 0.28.0's numbers (all six arms VALID), e4b's NF4 control (VALID,
licence-free — nothing beyond NF4 is quantised) and the ratio vLLM / e4b-NF4 on this box; the e4b recipe arms' speed is
reported as an **unlicensed observation**. **Amendment 3** (pre-registered 20:30Z, `p37c.sh`, `TP3_DONE` 22:18:07Z) ran
the registered K8 gate on this box's own pack: **NOT LICENSED — wikitext Δ −0.0230 ppl PASS, c4val1 Δ +0.1093 ppl FAIL
against the +0.05 budget** ([`gate_verdict.json`](gate_verdict.json)). **VOID stands**: no ratio against e4b's licensed
stack is quoted on this lane, the recipe's speed rows stay measured-and-unlicensed, and the recipe is shown not to
reproduce its licence across boxes. Box torn down 22:22Z (proven).

**What was compared.** Qwen3-30B-A3B (48 layers, 128 experts top-8) through two engines on one box in one session,
decode-vs-decode on **identical prompt token ids** (the e4b harness's own `step_decomp._k8_window` rows dumped once to
[`prompts_b1.json`](prompts_b1.json) / [`prompts_b16.json`](prompts_b16.json) — wikitext-2 test, 512-token rows ≥ 18k tokens
apart, sha256 `a8e6ea1d…` / `f67e7e4d…` — and fed to vLLM verbatim as `prompt_token_ids`; every receipt carries the file
sha and the reducer refuses to divide two receipts that disagree on it). **e4b** in the image python — experts4bit-qlora
0.35.0 + grouped-nf4-gemm 0.30.0 (main @`f4b639fd2640` / @`ddcb850e05c3`, bo7's cut), transformers 5.16.1, bitsandbytes
0.50.1, torch 2.8.0+cu129 (the image's; bo7 ran cu128 — the one environmental difference from bo7, recorded), triton
3.4.0, hook v6 + `step_decomp.py` + `k8_bake.py` + `calib.json` staged **byte-identical** to the bo7 receipt
([`staged.sha256`](staged.sha256) = [`logs/staged.actual.sha256`](logs/staged.actual.sha256)); the bf16 checkpoint
`Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` baked to an NF4 arena, the int4 packs built at arm start
by the hook exactly as bo7b (`E4B_CALIB_NSEQ=128` = 64k C4-validation tokens, streamed, 24 GiB Hessian budget, GPU solve;
C4-calibrated int4 attention, 192 projections). **vLLM 0.28.0** in its own venv (torch 2.13.0+cu130, triton 3.7.1) serving
Qwen's official `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441` (GPTQ 4-bit g128 sym;
`MarlinLinearKernel` for the dense projections, `MARLIN` WNA16 MoE backend, `MarlinExperts` — the engine's own
selection, asserted in every log), `dtype auto` → float16, `max_model_len 2048`, `gpu_memory_utilization 0.90`,
`enable_prefix_caching=False`, `seed 0`, `cudagraph_mode FULL_AND_PIECEWISE` (the graph arms) / `enforce_eager=True`
(the eager arm), `kv_cache_dtype auto` / `fp8` (the fp8-KV arm), every knob read back from the engine's config into the
receipt. **Quality is quoted, never equated** (pre-registered rule 5): e4b's arm carries — was meant to carry — the
register's two-text K8 licence; vLLM's checkpoint carries its author's evaluation; this lane scores neither.

**Box:** Vast.ai instance **49975016**, RTX 5090 (sm_120, driver 595.84, 32607 MiB) on an AMD EPYC 7Q83 64-core host
(251 GiB, container cgroup `memory.max` 183,318,347,776 B = 170.7 GiB; [`forensics.txt`](forensics.txt)) — the same host
class as bo7 and tp1; image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` (vLLM on sm_120 needs a ≥ 12.9 toolkit in the
container); pre-flight 42 MB/s against the 40 MB/s floor. B=1 is host-bound and the 5090 class carries ~8.5 %
inter-box dispersion, so **every ratio on this page is same-box, same-session; no absolute is compared to bo7's or
P20's** (they are cited beside, with their boxes and dates).

**When (box time = UTC, from [`outer.log`](outer.log)):** launched 15:40Z; the first run refused at the staging check
15:43Z (`STAGE MISSING: hook/usercustomize.py`, [`outer.attempt1.log`](outer.attempt1.log) — amendment 1) and relaunched
15:44Z; e4b installed 15:44:10Z (`p37 tripwire OK (e4b): e4b 0.35.0 gnf4 0.30.0 torch 2.8.0+cu129 triton 3.4.0 hook v6
loadable, ENABLE_USER_SITE True`); vLLM 0.28.0 installed 15:44:52Z–15:50Z (`p37 tripwire OK (vllm): 0.28.0 torch
2.13.0+cu130 triton 3.7.1`); bf16 fetched 15:50:54Z (staged in 13.6 min); NF4 arena baked 16:04:29Z (`BAKE OK`, 6144 rows,
16.3 GB in 29 s); prompts dumped 16:05:36Z; GPTQ fetched 16:05:50Z — **the resumed read hung for ≈ 10 min** (amendment 2)
and recovered on its own, staged in 19.7 min at 16:25:30Z, inside the 1800-s alarm; arms 16:25:30Z e4b `nf4_r1` B=1 →
16:26:25Z B=16 → 16:27:33Z vLLM `graph_r1` B=1 → `eager` → `fp8kv` → 16:36:17Z the same three at B=16 → 16:39:50Z e4b
`lic_r1` B=1 (≈ 50 min, the calibration inside it) → 17:31:59Z `lic_r1` B=16 → 18:25:37Z `nf4_r2` B=1 → B=16 → 18:27:14Z
vLLM `graph_r2` B=1 → B=16 → 18:29:23Z e4b `lic_r2` B=1 → 19:22:06Z `lic_eager` B=1 → reduced and **`TP_DONE`
20:13:55Z**; `p37b.sh` at 20:15Z: `NOT NEEDED` → `TP2_DONE`. **Lane wall 15:44Z → 20:14Z ≈ 4 h 30 min** against the
pre-registration's ≈ 5.5 h. Snapshot of the speed lane taken 20:20Z; amendment 3 (`p37c.sh`) on the same box: `nf4` on wikitext 20:21:13Z →
`nf4` on c4val1 20:24:33Z → `all` on wikitext 20:27:58Z (the pack rebuilt, ≈ 55 min) → `all` on c4val1 21:23:26Z → verdict
and **`TP3_DONE` 22:18:07Z**; box torn down 22:22Z.

## The identical fixture (pre-registered; [`PREREG.md`](PREREG.md) is the full text, verbatim)

- **Decode isolation, both sides (the P20 protocol, unchanged):** 512-token prompt, 128 generated tokens, greedy, context
  640 within `max_model_len 2048`. e4b: bo7's `step_decomp.py` protocol — B=1 prefill through the scheduled path, one step
  captured and replayed, **127 timed steps** (`B1D_TIMED_GRAPH steps=127 step=X ms`; tok/s = 1000 / step ms); B=16 the
  `BV3_GRAPH batch=16 steps=70 … agg=Y tok/s` aggregate; fp8 paged KV; `--no-fuse-qkv`; `E4B_RECOMPILE_LIMIT=64`;
  recompiles in the window asserted 0. vLLM: offline `LLM.generate`, the **slope method** — 32 and 128 tokens from the same
  prompts (`temperature 0`, `ignore_eos`, `min_tokens = max_tokens`), one untimed warm at each length, 3 timed
  repetitions, `decode_ms_per_step = (min wall_128 − min wall_32) / 96` (P20's estimator; the median-of-3 printed beside
  it), `decode_tok_s = 96·B / Δwall`, and `end_to_end_tok_s_long` (prefill included) beside. Per-step scheduling and
  detokenisation do not cancel in the slope, so vLLM's number includes its serving loop while e4b's replay window has no
  scheduler — the asymmetry understates vLLM's engine, it does not inflate it.
- **Arms (16; each one process, one JSON, one `perl -e 'alarm N'`):** e4b `nf4_r1` / `nf4_r2` (NF4 experts, bf16
  attention, no folds — the same-box control, bo7's `nf4`) at B=1 and B=16; vLLM `graph_r1` / `graph_r2` (default graphs,
  kv auto — **primary**), `eager` (`enforce_eager=True`), `fp8kv` (default graphs, `kv_cache_dtype="fp8"`, e4b's cache
  class) at B=1 and B=16; e4b `lic_r1` (**the licensed configuration** = bo7b's `calibexp_all_n128`: streamed 64k
  calibrated int4 experts + calibrated int4 attention + round-1/2 folds + router epilogue + glue, graph loop — **primary**)
  at B=1 and B=16, `lic_r2` at B=1 (the self-pair; B=16 not repeated, pre-registered), `lic_eager` at B=1 (the eager
  pairing for vLLM's eager arm). e4b first (a vLLM install failure cannot void the e4b arms), then interleaved;
  the calibrated pack rebuilt per calibrated arm (no pack cache; ≈ 50 min each).
- **Peak VRAM, two instruments:** a 1-s `nvidia-smi memory.used` sampler brackets every arm (`vram_<engine>_b<B>_<arm>.txt`);
  vLLM's own accounting line is in its log and receipt (`Actual usage is 16.7 GiB for consumed memory (weights +
  non-torch), 1.19 GiB for peak activation, and 0.58 GiB for CUDAGraph memory`; `Available KV cache memory: 10.34 GiB`)
  because its `memory.used` is a policy number (it reserves 0.90 of the card up front).
- **TTFT:** defined, judged not fair, recorded where present, **no ratio** (pre-registered).

## Validity and reading rules (the reducer, [`p37_reduce.py`](p37_reduce.py), applies these and no others)

1. **VOID arm** if — e4b: the run log lacks `INT4EXP … 48 layers (qwen3_moe)` **and** `ATTNINT4 192 projections` **and**
   the streamed-calibration lines whose pack counts read **11512 gptq / 776 rtn** (bo6b's, bo6c's and bo7's counts for
   the 64k pack; "a different count is a different pack"), or `recompiles_in_window ≠ 0`, or `fuse_qkv ≠ false`, or
   `n_steps ≠ 127` (B=1) / `70` (B=16); vLLM: the log lacks the Marlin lines, the graph arm lacks `Capturing CUDA graphs`
   (or the eager arm has it), a row generated fewer tokens than asked, or the receipt's `prompts_sha` differs from the
   e4b side's.
2. **Self-pair rule:** a ratio is quoted only when both sides' self-pairs are inside **1.03×** (`e4b nf4 r1/r2`,
   `e4b lic r1/r2` at B=1, `vllm graph r1/r2`); outside it the row reads DRIFT and no ratio is quoted.
3. **The number:** per batch size, `vllm_graph_r1 decode tok/s / e4b_lic_r1 tok/s`, both absolutes, the median-of-3
   beside the min, r2 beside r1; `> 1` = vLLM ahead. Secondary rows print the same way, labelled secondary.
4. **The instrument check, reported not gated:** `e4b_lic / e4b_nf4` on this box beside bo7's ×2.067 / ×2.602.
5. **Three axes** for e4b's arm; **6. supersession** of `e4b.serve.h2h.vllm.same-box`'s current-position use by this
   lane's claim; **7. quality quoted, never equated; 8. every receipt carries `prompts_sha`, the box forensics and the
   engine versions.**

## The three amendments (each dated in the private record before the data it touches)

1. **Amendment 1 (15:44Z, before any data) — the staging refusal.** The first run refused at the lane's own staging check
   (`STAGE MISSING: hook/usercustomize.py`; the launcher had copied the hook from the lane directory root while it lived
   under `hook/` on the mini) and touched `TP_DONE`; the hook was staged (its sha256 `574546fd…` matches the bo7 receipt
   copy, as do `step_decomp.py`, `k8_bake.py` and `calib.json` — [`staged.sha256`](staged.sha256)), the marker removed,
   the lane relaunched at 15:44Z ([`outer.attempt1.log`](outer.attempt1.log) kept). The refusal worked as designed.
   **Nothing measured; nothing changed but the staging.**
2. **Amendment 2 (16:22Z, before any comparator data) — the GPTQ fetch hang, left to its alarm; a guarded follow-up.**
   The comparator fetch (started 16:05:50Z) hung on a resumed read — 9,740 MB of the ~17 GB shard in a `.incomplete` blob,
   cache size flat across two minutes, every thread in `futex_wait`, no established socket, the log ending in
   `The read operation timed out / Trying to resume download…` (XET already disabled; the same class as tp1's Gemma-4
   hang). Per the STALL rule (look, never kill) it was left to the lane's own 1800-s fetch alarm, after which the lane
   would record `DL FAIL (gptq) -- NO COMPARATOR`, run the e4b arms only and write `SKIPPED` rows for the vLLM arms;
   [`p37b.sh`](p37b.sh) (waiter [`p37b_wait.sh`](p37b_wait.sh)) was pre-registered to resume the fetch (≤ 3 × 2400 s) and
   run the eight vLLM arms in the registered order against the SAME prompt files after `TP_DONE`, then re-reduce
   (`TP2_DONE`). **Outcome (16:30Z):** the lane's own resume recovered — `staged in 19.7 min` at 16:25:30Z, inside the
   alarm — so `VLLM_OK=1` and the vLLM arms ran in-lane; `p37b.sh` was given a guard before the waiter fired (it acts
   only on `SKIPPED (no comparator)` rows in `summary.txt`, else writes `NOT NEEDED`) and did exactly that at 20:15Z.
   Nothing re-run; the vLLM arms ran ≈ 1 h after the first e4b arms on the same box — an ordering difference only (the
   pre-registration ran e4b first anyway). **Fixture, arms, criteria, knobs unchanged.**
3. **Amendment 3 (pre-registered 20:30Z, after `TP_DONE`, before any gate data) — the registered K8 gate on THIS box's
   pack.** The fingerprint rule's intent is "the e4b arm is the licensed stack"; the count is a proxy for the bytes. The
   direct test is the registered gate itself on the pack this box derives: `p37c.sh` (same box, e4b unchanged, bo6c's
   `k8()` verbatim — `step_decomp.py --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager
   --no-fuse-qkv --ppl-source <text>`, `E4B_CALIB_SOURCE=c4`, `E4B_INT4_HESSIAN_BUDGET_GB=24`) runs four arms in this
   order: `nf4` on wikitext, `nf4` on c4val1, `all` (`exp=1 calib=1 fuse=all`, `E4B_SERVE_EXP_INT4_CALIB=1
   E4B_CALIB_NSEQ=128`) on wikitext, `all` on c4val1. **Verdict rule (the registered calibrated-pack gate, unchanged):
   Δppl = ppl(all) − ppl(nf4) ≤ +0.05 on BOTH texts ⇒ this box's pack is LICENSED**, the reducer's fingerprint check is
   replaced for this lane by `gate_verdict.json` (a stronger licence than a count fingerprint), the four e4b licensed
   arms become VALID and the ratio is quotable from the receipts already in this directory (B=1 `vllm graph_r1 / e4b
   lic_r1` = 286.0 / 236.4 = 1.21; B=16 2030.0 / 1305.3 = 1.56; the fp8-KV secondary 1.27 / 1.69). **If either text fails,
   VOID stands, P37 quotes no e4b ratio, and the recipe's speed stays an unlicensed observation.** No threshold, arm,
   prompt set or knob changes; the speed receipts are not re-run. Marker `TP3_DONE`; cost ≈ 3 h of the same box (each
   `all` arm recalibrates ≈ 50 min). **Outcome (`TP3_DONE` 22:18:07Z; [`gate_verdict.json`](gate_verdict.json), the four
   receipts `qwen3_ppl_{nf4,all}_{wikitext,c4val1}.json`, their logs, [`p37c.sh`](p37c.sh)): NOT LICENSED — VOID stands.**
   wikitext: nf4 6.41984 → all 6.39679 ppl, **Δ −0.0230, PASS** (sha `9ef10d760ad9`); c4val1: nf4 16.49703 → all 16.60631 ppl,
   **Δ +0.1093 against the +0.05 budget, FAIL** (sha `4bcb55179b96`); 2048 steps, eager, fp8 compute, this box's own NF4 as the
   reference on both texts. The `all` arms rebuilt the same pack the speed arms had (2418/142, 2398/162, 2412/148, 2338/222,
   1956/92 = **11522 / 766**; `INT4EXP enabled: 48 layers`, `ATTNINT4 calibrated: 192 projections`). In nats the c4val1 delta is
   +0.0066, inside Qwen3's 0.0095-nat arithmetic-order floor — and the floor is not the budget: the registered gate is in
   perplexity and reads FAIL. Beside it, never divided into: bo6c's licensed pack (11512 / 776, box 49861751) read c4val1
   **−0.0662** against its own NF4, and this box's NF4 reference reproduces bo6c's to 0.0002 / 0.0000 ppl (6.41984 / 16.49703
   there too) — the reference travels, the calibrated pack does not. **Reading:** ten of 12,288 expert matrices classified
   differently at the `min_rows` threshold go with a real quality difference; the streamed calibration recipe does NOT
   reproduce its licence across boxes. The fingerprint rule's proxy is confirmed by the gate, not replaced. Register:
   `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05.gate` (measured, FAIL); the four e4b licensed arms stay VOID; the
   1.21 / 1.56 these receipts would give is not a position and is not quoted. Open: the calibration's host-dependence —
   [e4b#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405) ("serve: the streamed int4 expert calibration
   does not reproduce its K8 licence across boxes (11522/766 vs 11512/776; c4val1 +0.109 vs −0.066)"; docs/STATUS.md "What
   is open").

Full text: the private receipt `INT4B16/P25-PARITY.md` ("P37 / P38 launch log": amendments 1–3, the in-lane progress
notes, the `TP_DONE` reading) and [`PREREG.md`](PREREG.md) here.

## The table, exactly as reduced ([`RESULTS-p37.md`](RESULTS-p37.md) = [`RESULTS.txt`](RESULTS.txt))

**B=1** (prompts_sha `a8e6ea1d7d140dbe`, rows 1, row step 298426 tokens)

| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | lic_eager | VOID | 12.2 | 81.99 | 19.99 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | lic_r1 | VOID | 236.4 | 4.23 | 19.9 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | lic_r2 | VOID | 235.8 | 4.24 | 19.9 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | nf4_r1 | VALID | 113.4 | 8.82 | 18.91 | |
| e4b | nf4_r2 | VALID | 113.5 | 8.81 | 18.86 | |
| vllm | eager | VALID | 20.8 | 48.114 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 20.4 tok/s; e2e incl. prefill 20.7 |
| vllm | fp8kv | VALID | 300.9 | 3.323 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 301.1 tok/s; e2e incl. prefill 282.5 |
| vllm | graph_r1 | VALID | 286.0 | 3.497 | 30.39 (policy: reserves 0.90) | min-of-3; median-of-3 286.1 tok/s; e2e incl. prefill 270.5 |
| vllm | graph_r2 | VALID | 286.0 | 3.496 | 30.39 (policy: reserves 0.90) | min-of-3; median-of-3 286.1 tok/s; e2e incl. prefill 270.5 |

- self-pairs: e4b nf4 1.0011; e4b lic 1.0024; vllm graph 1.0 (rule: inside 1.03× or DRIFT)
- **NO RATIO QUOTED at B=1 — missing/VOID primary arm: e4b lic_r1** (both readings above; a re-run is a new lane)
- secondary — eager pairing: vllm eager 20.8 tok/s (0.073 of vllm graph); e4b lic_eager 12.2 tok/s (0.052 of e4b graph);
  eager-vs-eager ratio 1.705 (secondary, never the headline)
- secondary — fp8 KV vs kv auto (vLLM only): vllm fp8kv 300.9 tok/s (1.052 of vllm graph)

**B=16** (prompts_sha `f67e7e4d592b002b`, rows 16, row step 18651 tokens)

| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |
|---|---|---|---|---|---|---|
| e4b | lic_r1 | VOID | 1305.3 | 12.26 | 20.11 | 64k pack counts 11512/776 not in log (different pack or not calibrated) |
| e4b | nf4_r1 | VALID | 500.1 | 31.99 | 19.41 | |
| e4b | nf4_r2 | VALID | 499.9 | 32.0 | 19.41 | |
| vllm | eager | VALID | 322.5 | 49.616 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 322.4 tok/s; e2e incl. prefill 311.5 |
| vllm | fp8kv | VALID | 2206.5 | 7.251 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 2217.4 tok/s; e2e incl. prefill 1705.1 |
| vllm | graph_r1 | VALID | 2030.0 | 7.882 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 2023.7 tok/s; e2e incl. prefill 1595.9 |
| vllm | graph_r2 | VALID | 2022.2 | 7.912 | 31.0 (policy: reserves 0.90) | min-of-3; median-of-3 2020.5 tok/s; e2e incl. prefill 1593.2 |

- self-pairs: e4b nf4 1.0003; e4b lic n/a at B=16 (not repeated, pre-registered); vllm graph 1.0039 (rule: inside 1.03× or DRIFT)
- **NO RATIO QUOTED at B=16 — missing/VOID primary arm: e4b lic_r1**
- secondary — eager pairing: vllm eager 322.5 tok/s (0.159 of vllm graph)
- secondary — fp8 KV vs kv auto (vLLM only): vllm fp8kv 2206.5 tok/s (1.087 of vllm graph)

(The vLLM rows' notes column in `RESULTS-p37.md` also carries the engine's memory-accounting line verbatim; it is
abbreviated here. vLLM's `memory.used` is a policy number: the comparable figure is its own accounting — 16.7 GiB weights
+ non-torch, 1.19 GiB peak activation, 0.58 GiB CUDA graphs, 10.34 GiB reserved for KV on the graph arms.)

## Reading (registered units; the register's ids in `docs/claims.json`)

- **Quoted — vLLM 0.28.0 against e4b's NF4 control on this box** (`e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05`):
  B=1 vLLM graph 286.0 tok/s (3.497 ms/step; r2 286.0) vs e4b NF4 113.4 / 113.5 tok/s (8.82 / 8.81 ms) → **vLLM / e4b-NF4
  2.52**; B=16 vLLM graph 2030.0 (7.882 ms; r2 2022.2) vs e4b NF4 500.1 / 499.9 → **4.06**. Both sides VALID, both
  self-pairs inside 1.03×, the same `prompts_sha` on both sides. The NF4 stack quantises nothing beyond NF4 and needs no
  gate (it is 0.00173 nats from the model's own attention, `e4b.parity.qwen3.paged-vs-own-attention`), so this ratio is
  licence-free — and it is the ratio against e4b's *slowest* configuration, not its position.
- **Not quoted — vLLM against e4b's licensed stack.** Every licensed e4b arm is VOID (fingerprint 11522/766 vs 11512/776:
  per-chunk lines 2418/142, 2398/162, 2412/148, 2338/222, 1956/92 against bo7's 2418/142, 2394/166, 2412/148, 2334/226,
  1954/94; total 12,288 either way). The recipe's speed on this box is **measured and unlicensed**: B=1 236.4 tok/s
  (4.23 ms; r2 235.8 / 4.24), B=16 1305.3 (12.26 ms), eager 12.2 (81.99 ms) — reported as an observation, never divided
  into a position. Amendment 3 ran the registered gate on that pack and it **failed on c4val1 (+0.1093 ppl)**: that does
  not change (`e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05.gate`).
- **Secondary:** eager-vs-eager 1.705 (vLLM 20.8 vs e4b 12.2 at B=1); vLLM fp8-KV / kv-auto 1.052 at B=1, 1.087 at B=16.
- **The old position** (`e4b.serve.h2h.vllm.same-box`, 2026-09-03, box 49702459, vLLM 0.28.0 vs the 0.27.0/0.21.0 RTN
  stack: ×1.47 / ×1.55) is **superseded** for current-position use by this lane's claim and stays true as measured; the
  2026-08 "6.31×" figure is retired by id. No number from that box, bo7 or P20 is divided into a p37 number.

## Predictions P1–P6 (pre-registered; scored at this snapshot)

| | predicted | outcome |
|---|---|---|
| **P1** | B=1: vLLM ahead of the licensed stack, ratio 1.15–1.30 (central 1.24); vLLM ≈ 283 tok/s, e4b 220–240 | **Not scorable as registered** (the licensed arm is VOID, and amendment 3's gate confirmed it). The absolutes fell inside the predicted ranges: vLLM 286.0, e4b recipe 236.4 — an unlicensed observation, no ratio. |
| **P2** | B=16: vLLM ahead, 1.15–1.45 (central 1.30); vLLM ≈ 1,900–2,000, e4b 1,300–1,700 | **Not scorable as registered.** vLLM 2030.0 (above the range), e4b recipe 1305.3 (inside) — an unlicensed observation, no ratio. |
| **P3** | vLLM eager costs 1.5–3× at B=1 and 1.2–2× at B=16; e4b eager 1.5–2.5× at B=1 | **Wrong by an order of magnitude on both engines:** vLLM eager is 13.7× slower at B=1 (286.0 → 20.8) and 6.3× at B=16 (2030.0 → 322.5); e4b's eager loop 19.4× at B=1 (236.4 → 12.2). Neither changes a headline (secondary rows). |
| **P4** | vLLM fp8 KV within ±5 % of kv-auto | **Held at B=1 (1.052), missed at B=16 (1.087).** |
| **P5** | e4b's same-box ratio reproduces bo7's ±10 %; self-pairs inside 1.03×; the 64k pack counts read 11512 / 776 | **Two of three held, the third is the finding:** self-pairs 1.0011 / 1.0024 / 1.0000 (B=1) and 1.0003 / 1.0039 (B=16); the recipe's ratio over NF4 reads 2.08 / 2.61 (236.4 / 113.5; 1305.3 / 500.1) against bo7's 2.067 / 2.602 — inside 1 % — **but the pack counts read 11522 / 766**, so the arm is VOID as registered. |
| **P6** | the two engines' continuations diverge within 128 tokens on most rows (not a gate) | **Held:** the per-row `tokens` in both receipts differ from the first tokens on (e.g. row 0: `…659, 758, 220…` vs `…659, 1260, 1030…`). |

## Layout

- `e4b_b<B>_<arm>.json` (8) and `vllm_b<B>_<arm>.json` (8) — one receipt per arm; [`RESULTS-p37.md`](RESULTS-p37.md) =
  [`RESULTS.txt`](RESULTS.txt) (`python p37_reduce.py .`); [`summary.txt`](summary.txt) (one result line per arm, the
  prompt shas, the `p37b: NOT NEEDED` line); [`versions.txt`](versions.txt); [`prompts_b1.json`](prompts_b1.json) /
  [`prompts_b16.json`](prompts_b16.json) (the identical token ids, per-row and file shas); [`forensics.txt`](forensics.txt);
  [`staged.sha256`](staged.sha256) (the expected shas of the four bo7 pieces); the 16 `vram_<engine>_b<B>_<arm>.txt`
  1-s `nvidia-smi` traces; the markers `TP_DONE` / `TP2_DONE` / `TP3_DONE`.
- Amendment 3 (the gate on this box's pack): [`gate_verdict.json`](gate_verdict.json), the four K8 receipts
  `qwen3_ppl_nf4_wikitext.json`, `qwen3_ppl_all_wikitext.json`, `qwen3_ppl_nf4_c4val1.json`, `qwen3_ppl_all_c4val1.json`
  (2048 steps, `mean_nll`, `ppl`, the text sha, the mechanism tally), [`p37c.sh`](p37c.sh) (bo6c's `k8()` verbatim apart
  from the log directory) and `logs/run_qwen3_ppl_*.log` (the pack lines, the `K8_PPL` result line); `summary.txt` and
  `outer.log` are the snapshot's final copies (the `k8 …` and `GATE …` lines appended by `p37c.sh`).
- Lane: [`p37_run.sh`](p37_run.sh) (the pre-registered lane script, the one that ran), [`p37_vllm.py`](p37_vllm.py) (the
  vLLM arm driver: P20's `h2h_vllm_wt.py` with the prompt ids fed verbatim and every knob read back),
  [`p37_reduce.py`](p37_reduce.py), [`p37b.sh`](p37b.sh) + [`p37b_wait.sh`](p37b_wait.sh) (amendment 2's guarded follow-up,
  which ran as `NOT NEEDED`), the lane console [`outer.log`](outer.log) and the refused first start
  [`outer.attempt1.log`](outer.attempt1.log).
- [`logs/`](logs/) — `pip_e4b.log`, `pip_vllm.log`, `tripwire_vllm.log`, `fetch_bf16.log`, `fetch_gptq.log`, `bake_qwen3.log`,
  `prompts.log`, `staged.actual.sha256` (the four staged pieces' shas as measured at install, = `staged.sha256`), and
  **every per-arm `run_*.log`** (the e4b logs carry the engagement banners — `INT4EXP calibrated experts: … gptq / … rtn`
  per chunk, `INT4EXP enabled: 48 layers`, `ATTNINT4 calibrated: 192 projections`, the `B1D_TIMED_GRAPH` / `BV3_GRAPH`
  result line; the vLLM logs the Marlin selection, the graph capture, the memory accounting). Force-added past the
  repository's `*.log` ignore rule, as tp1's and p38's were.
- [`PREREG.md`](PREREG.md) — the pre-registration, verbatim from the private tree, added by this bundle.
- Nothing in this directory is edited: every file is the byte-for-byte copy of the box's snapshot (checked with `cmp`
  when the bundle was written) except the two written here, `README.md` and `PREREG.md` (the amendment-3 files were added from the box's final snapshot, `cmp`-checked the same way). Not shipped: the four pieces
  staged verbatim from the bo7 receipt — `hook/usercustomize.py`, `step_decomp.py`, `k8_bake.py`, `calib.json` — whose
  bytes are already in this repository under
  [`../../hybrid-g9/throughput-20260904/bo7/`](../../hybrid-g9/throughput-20260904/bo7/README.md) (`logs/hook/`,
  `logs/step_decomp.py`, `logs/k8_bake.py`, `calib.json`; the shas in `staged.sha256` are theirs), `INSTANCE_ID`
  (49975016, also in every vLLM receipt's `vast_instance_id`), the empty `p37b_wait.log`, the NF4 arena and the two
  checkpoints.

## Reproduce

Rent the class in `forensics.txt` on the `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` image (≥ 320 GB disk, ≥ 98 GB host
RAM, a link that sustains ≥ 40 MB/s), stage `p37_run.sh`, `p37_vllm.py`, `p37_reduce.py`, `hook/usercustomize.py`,
`step_decomp.py`, `k8_bake.py` and `calib.json` (the last four from the bo7 receipt directory; the lane refuses any other
bytes) under `/root/p37/`, put the Hugging Face token at `~/.cache/huggingface/token`, and run `p37_run.sh` (its pip lines
pin both cuts and vLLM's version; its tripwires refuse anything else; it dumps the prompt ids once and feeds both engines
from the files). Then `python p37_reduce.py .`. Expect the pack-count fingerprint to be host-dependent: this box read
11522 / 766 against the licensed 11512 / 776.

## What is NOT claimed

- **No ratio against e4b's licensed stack.** The reducer's verdict stands and amendment 3's gate confirmed it (c4val1
  +0.1093 ppl, FAIL); the recipe's speed (236.4 / 1305.3 tok/s on this box) is an unlicensed observation and is never
  quoted as a position or divided into one. The 1.21 / 1.56 these receipts would give against it is not quoted anywhere.
- **No licence is withdrawn from bo6c.** The licensed pack's two-text pass on its box stands as measured; what this lane
  shows is that the recipe re-derived on another host is a different pack with a different verdict.
- **The NF4 ratio is not e4b's position.** 2.52 / 4.06 is vLLM against the one e4b configuration that needs no gate — its
  slowest; it is quoted because it is the only licence-free ratio this lane can produce.
- **Quality is not equated.** The GPTQ checkpoint's quality is its author's; e4b's is the register's; neither is scored
  here, and no "at equal quality" is said or implied.
- **No TTFT ratio, no server path, no other batch sizes, prompt lengths, checkpoints or engines** (pre-registered out of
  scope). No anchor-class projection is made for the recipe arms (a VOID arm has no axes).
- **No cross-box number.** bo7's ×2.067 / ×2.602, bo6c's −0.0528 / −0.0662 ppl and P20's ×1.47 / ×1.55 are cited beside,
  with their boxes and dates; the recipe's same-box ratio over NF4 (2.08 / 2.61) is reported under rule 4 as an instrument
  reading, not a position.
- **The prediction table is scored, not argued.** P3 was wrong by an order of magnitude on both engines; P1/P2 are not
  scorable as registered; P5's pack-count clause failed and the gate confirmed why it matters; the lane ships them.
