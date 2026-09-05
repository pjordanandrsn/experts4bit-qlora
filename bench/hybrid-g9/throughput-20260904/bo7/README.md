# Throughput census lane bo7 — 2026-09-05, one RTX 5090: all six families, speed only, under the shipped code

**Box:** Vast.ai instance 49916675, RTX 5090 (sm_120, driver 595.84, 32607 MiB) on an AMD EPYC 7Q83 64-core host
(128 CPUs, 251 GB RAM, container cgroup `memory.max` 183,318,347,776 B ≈ 170.7 GiB — [`forensics.txt`](forensics.txt) is
the lane script's own probe at install, [`forensics2.txt`](forensics2.txt) the orchestrator's at the 05:31Z snapshot);
torch 2.8.0+cu128, transformers 5.16.1, bitsandbytes 0.50.1, triton 3.4.0; installed `grouped-nf4-gemm 0.30.0`,
`experts4bit-qlora 0.35.0` (`pip show`, in `forensics2.txt`). No memory-bandwidth probe ran on this box.
**When:** 2026-09-05. Per [`logs/outer.log`](logs/outer.log): install 02:00:46Z (`pip rc=0`, tripwire
`bo7 tripwire OK: e4b 0.35.0 main (streamed calibrated experts + gpu solve, swiglu/combine) + gnf4 0.30.0 main`,
`cuda ok`), Granite bake 02:01:32Z, its eight arms 02:02–02:18Z (`GRANITE DONE`); OLMoE bake 02:18Z, arms 02:20–02:26Z;
gpt-oss bake 02:26Z (5-min download), arms 02:32–02:34Z; Qwen3 bake 02:34–02:41Z, arms 02:41Z → (the calibrated arms
~40 min each: 02:51 → 03:34Z, 03:34 → 04:13Z, 04:24 → 05:05Z); **this bundle is the 05:31Z snapshot — 31 of 48 arms**;
Qwen3's last arm, then Gemma-4's eight and Mixtral's eight follow (`TP_DONE` expected ≈ 08:00Z), then **amendment 2's
two arms** (below; `TP2_DONE` ≈ 10:00Z) — all land in this directory before merge. Rented 01:26Z on bo6's `TP2_DONE` (the image took ~35 min to load; ssh auth at try 79); a 10-h
hard-kill guard on the mini was armed before the first ssh (fires ≈ 12:05Z); bandwidth pre-flight 96 MB/s.
**Cut — the shipped code, both packages at their `main` after the 0.35.0 / 0.30.0 releases** (the pip line in
[`logs/bo7_run.sh`](logs/bo7_run.sh)):

- **experts4bit-qlora main @`f4b639fd2640bb603b6b7b63ea010312b0bb351d`** = v0.35.0 (faaaef5) + #396 (docs / tooling);
  carries #384 (streamed calibrated int4 experts), #385 (decode glue), #388. The tripwire asserts `__version__ == "0.35.0"`,
  the streamed driver `enable_serve_experts_int4_calibrated`, the GPU-solve knob and the `swiglu`/`combine` glue.
- **grouped-nf4-gemm main @`ddcb850e05c3595e2bb87813df70427f7e4bafce`** = v0.30.0 + #341 (docs); the tripwire imports
  `rope_heads`, `reduce_partials`, `combine_rows`, `swiglu_rows` and the GPTQ packer.
- Hook **v6** ([`logs/usercustomize.py`](logs/usercustomize.py); the installed copy [`logs/hook/usercustomize.py`](logs/hook/usercustomize.py)
  is byte-identical, and both are byte-identical to bo6's) at its defaults: streamed calibration, **16k calibration tokens**
  (8 batches × 4 × 512 of C4 validation shard 00000; `E4B_CALIB_NSEQ` not set), damping 0.01, `min_rows=32`, 24 GiB
  Hessian budget, GPU solve (`E4B_INT4_GPTQ_DEVICE=cuda`). [`logs/step_decomp.py`](logs/step_decomp.py) is byte-identical
  to `bench/hybrid-g9/step_decomp.py` on this cut; [`logs/k8_bake.py`](logs/k8_bake.py) and [`calib.json`](calib.json)
  are byte-identical to bo6's (and bo5's, bo3's).

## The amendment, and why (P35 amendment 1, 2026-09-04 23:05Z — before any bo7 box was rented)

bo7 as pre-registered (13:58Z) carried the cut of the moment — e4b integration-8 @db2a070 + grouped-nf4-gemm @0b25d13 —
hook v4, a 3600-s alarm on every arm, and Mixtral's `calibexp_lic` arms. Two things happened on bo6 before bo7 could
start: (a) its queue's first 8-h window (13:58Z → 21:57Z) expired with nothing launched because bo6b's streamed
calibrations ran past it, and the queue was re-armed at 22:09Z (deadline 06:09Z; it fired at 01:26Z); (b) bo6b's Mixtral
speed arms died by their own 3600-s alarm at streamed-calibration pass 23 of 32 — a 32-layer Mixtral calibration at an
8 GiB budget takes ~85 min — and bo7's pre-registered form would have alarmed out the same way (or, under hook v4's
all-at-once path, OOM-killed on ~230 GB of Hessians). Amended in the lane script on the mini before the rent, the
pre-amendment copy kept for the record as [`logs/bo7_run.sh.pre-amend`](logs/bo7_run.sh.pre-amend) (`diff` it against
[`logs/bo7_run.sh`](logs/bo7_run.sh), the script that ran):

1. **cut = the shipped code** (above) instead of the integration branch; the tripwire additionally asserts the streamed
   driver and `__version__ == "0.35.0"`;
2. **hook v6** (streamed calibration) instead of v4;
3. **arm alarm 5400 s for calibrated arms** (`E4B_SERVE_EXP_INT4_CALIB=1` on the line), 3600 s otherwise — the `arm()`
   helper reads it off the arm's own environment and prints `alarm=N` on the console line;
4. **Mixtral `calibexp_lic` dropped** (two arms): FAIL as registered on bo6b (wikitext +0.077 ppl,
   `e4b.serve.buildout.bo6.mixtral.lic-calibexp-streamed.k8.2026-09-04`) and ~85 min of calibration per arm inside a
   10-h guard; Mixtral `lic` and `all` stay for the record;
5. the arm count corrected: the script carries 25 configurations per batch size (Granite 4, OLMoE 4, gpt-oss 2, Qwen3 6,
   Gemma-4 4, Mixtral 5) = 50 arms as written, not the 46 the pre-registration said — **48 after the drop**.

Reading rules (1)–(5) unchanged; speed only; the guard stays 10 h. The full text is in the private receipt
(`INT4B16/P25-PARITY.md`, "P35 AMENDMENT 1" and "P35 log — bo7 launched").

## Amendment 2 (P35 amendment 2, pre-registered 2026-09-05 06:05Z — before its arms ran)

The lane as amended measured Qwen3's calibrated stack at the hook's **16k** default, while the configuration the
register licenses is bo6c's streamed **64k** pack — so the licensed stack's speed would have stayed unmeasured on this
box. Amendment 2 adds two arms **on this box after `TP_DONE`, same session and install**: `qwen3/calibexp_all_n128` at
B=1 and B=16 — streamed GPTQ-calibrated int4 experts at 64k C4 tokens (`E4B_CALIB_NSEQ=128`) + C4-calibrated int4
attention + round-1/2 folds + router epilogue + glue, the licensed Qwen3 configuration — run by `/root/bo7/bo7b.sh`
(shipped as `logs/bo7b.sh` with the final snapshot) under 5400-s alarms, receipts `qwen3_b1_calibexp_all_n128.json` /
`qwen3_b16_calibexp_all_n128.json`, marker `TP2_DONE` (≈ 10:00Z). Their ratios are to bo7's own `nf4` arms (rule 1
holds: same box, same session); the licensed-best row then carries the three axes for real, the anchor projection
(159.2 × the B=1 ratio, a projection) included. If they alarm out or refuse, they are `alarm` / `refused` rows and the
Qwen3 census claims stay `open` with the cause. **Pending in this snapshot.**

## Protocol

bo5's ([`../bo5/README.md`](../bo5/README.md)) speed arms, without the K8 arms: per family an NF4 bake of the released
checkpoint (`logs/k8_bake.py`; [`models.txt`](models.txt)), then arms under `logs/step_decomp.py` — **B=1** decode
(512-token prompt, 128 generated, graph loop, 127 timed steps, `--no-fuse-qkv`, fp8 paged KV, placement all-vram,
`--amort off`) and **B=16** aggregate (graph loop, 70 steps) — the families in size order with the arena and the
checkpoint freed between them (Granite → OLMoE → gpt-oss → Qwen3 → Gemma-4 → Mixtral). Every arm is one
`step_decomp.py` invocation; the `arm()` helper carries the flags; `fenv` maps `0 / r12 / r1epi / all` to
`E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`, `E4B_FUSE_ROUTER_EPI`; `E4B_SERVE_EXP_INT4`, `E4B_SERVE_ATTN_INT4_CALIB`,
`E4B_SERVE_EXP_INT4_CALIB`, `E4B_INT4_KEEP_NF4` select the packs, and each arm's environment is on its line in
`outer.log`. tok/s at B=1 is `1000 / step_ms_clean` with the step rounded to the 0.01 ms the logs print (bo5's
reducer's convention); at B=16 it is the receipt's `aggregate_tok_s`.

Arms per family (each at B=1 and B=16): **Granite** `nf4` / `r12epi` / `int4_r12epi` / `calibexp_r12epi`; **OLMoE**
`nf4` / `folds` / `calattn` / `int4all`; **gpt-oss** `nf4_r12` / `store_r12` (MXFP4 GEMV for single rows, NF4 kept for
batched rows); **Qwen3** `nf4` / `folds` / `calattn` / `int4all` / `calibexp_all` / `calibexp_folds`; **Gemma-4** `nf4` /
`r1epi` / `int4_r1epi` / `calattn_r1epi` (round 2 refuses by design on this family); **Mixtral** `nf4` / `folds` / `lic` /
`all`. 48 arms. **An alarmed or refused arm is a row**, never the end of a family.

## The queue expiry and the alarm sizing

The queue ([`logs/bo7_queue.sh`](logs/bo7_queue.sh), on the mini) waits on bo6's `TP2_DONE` marker or bo6's proven
teardown, with an 8-h deadline; its first window expired unfired (bo6b overran it), the second (22:09Z → 06:09Z) fired at
01:26Z. The renter ([`logs/bo7_all.sh`](logs/bo7_all.sh)) draws an RTX 5090 offer (≥ 320 GB disk, ≥ 96 GB RAM, ≤ $1.20/h,
sm_80+), arms the 10-h hard-kill guard **before** the first ssh and verifies the guard's pid against this instance id,
runs a 15-s bandwidth pre-flight (re-roll under 15 MB/s), stages the scripts and launches `bo7_run.sh` under `setsid`.
Alarms: **5400 s on calibrated arms, 3600 s otherwise**, sized from bo6b — a Qwen3 streamed 16k calibration takes
~40 min on this host (well inside 5400 s), a Mixtral one ~85 min (which is why those arms were dropped rather than given
a longer alarm inside a 10-h guard). No alarm fired in this snapshot.

## Reading rules (pre-registered, P35; applied verbatim)

> (1) Every ratio is the family's own NF4 arm on this box, same session. (2) The licence label of each arm comes from
> docs/claims.json and the K8 verdicts already on record (bo3/bo5, and bo6 for the calibrated-expert arms) at the time
> the bundle is written — bo7 measures speed, it does not license anything. (3) Three axes are quoted for every
> licensed best: ratio ×N over NF4 (this box), rental-measured tok/s (this box, one 5090), and the anchor-class
> projection marked as a projection. (4) Unlicensed arms appear in the table labelled "measured, not licensed" and
> never as a position. (5) No cross-lane ratio: bo3/bo5 numbers are cited beside, never divided into, bo7's.

Applied on this page as: the verdict sources are bo3, bo5, bo6 **and bo6c** (the register as merged through #399, main
2a632fb); where the register is silent on an arm the label is **`no quality verdict on record`**, never "licensed"; the
anchor-class projection is `159.2 tok/s × the B=1 ratio` and exists **only for Qwen3-30B-A3B at B=1** (the anchor class
was never certified — 12 refusals; the memory of that campaign is in the private tree), so every other family and every
B=16 cell says *no anchor projection*; gpt-oss's NF4 arm on this lane is `nf4_r12` (NF4 experts + the exact folds), so its
ratios are to that arm and its quoted best is the reference itself.

## What is licensed, and what is not (the register at bundle time — nothing here is decided by this lane)

- **Granite-3.1-3B-A800M:** `r12epi` is the licensed stack (K8 +0.019 ppl wikitext, pass, bo3
  `e4b.serve.buildout.granite.b1.5090.2026-09-04` / `.b16`; re-measured bo5 `…bo5.granite.*`). `int4_r12epi` (RTN
  experts, +0.063 ppl FAIL, retracted #381) and `calibexp_r12epi` (c4val1 +0.387 FAIL, bo5) are measured, not licensed.
- **OLMoE-1B-7B:** nothing above NF4 is licensed. The tp claim (`e4b.serve.tp.olmoe.b1.5090.2026-09-04`) called the full
  stack "best licensed" on one wikitext text before the two-text clause was applied to calibrated packs (#386); the
  calibrated attention in it is refused on quality on this family (+0.60 ppl C4-val, `e4b.serve.b1.qwen3-30b.int4attn-calib.5090`
  notes: Qwen3-specific), and int4-b32 experts at ≤1B active are the class STATUS records at ~1.2–1.8% ppl. The folds
  alone have no K8 on record. `folds` → no verdict; `calattn`, `int4all` → measured, not licensed.
- **gpt-oss-20b:** no raw-text instrument (OOD-flattery regime). `nf4_r12` is the register's quoted best
  (`e4b.serve.buildout.gptoss.b1.5090.2026-09-04`: "licensed configuration" in the claim text, "no instrument" in its
  notes) and this lane's reference arm; `store_r12` is measured with the quality gate open (`…bo5.gptoss.*`) — not licensed.
- **Qwen3-30B-A3B:** the licensed configuration is the **streamed 64k** full stack (bo6c,
  `e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`). bo7's `calibexp_all` and `calibexp_folds`
  ran the 16k default → "measured; the licensed configuration is the 64k pack — the 16k streamed full stack has no
  two-text verdict"; **the licensed stack's speed is amendment 2's `calibexp_all_n128` arm, pending in this snapshot**
  (its label: "the licensed configuration (bo6c)"). `int4all` = bo5's
  RTN `all`, FAIL as registered (c4val1 +0.063); the exact `folds` carry bo5's verdict FAIL-by-improving (−0.073 ppl,
  sub-floor) — not a licensed position; `calattn` alone has no verdict on record.
- **Gemma-4-26B-A4B:** no K8 instrument exists for this family (`e4b.parity.gemma4.no-reference`, #359), so the register
  carries no verdict for any arm. `r1epi` is the exact-arithmetic position (round-1 norm fold + epilogue on NF4 experts);
  `int4_r1epi` is the register's quoted best (bo3 `stack`, `e4b.serve.buildout.gemma4.b1.5090.2026-09-04` — "licensed
  configuration" in the claim text, "NO quality instrument" in its notes) and `calattn_r1epi` a calibrated pack with an
  unreadable K8: both measured, no quality verdict.
- **Mixtral-8x7B-Instruct:** nothing above NF4 is licensed. `lic` (RTN int4 experts + folds + epilogue) FAILS as
  registered on c4val1 (+0.0575; the P30 "licensed stack" label withdrawn, `…bo5.mixtral.*`), `all` FAILS (+0.116),
  the calibrated stack FAILS on wikitext (bo6b, +0.077; dropped here). The combined `folds` arm has no receipt (bo3 scored
  r12 +0.0009 and the epilogue −0.0087 separately, one text each) → no verdict on record.

## Instrument notes

- **Receipts reproduce their console lines on all 31 arms** (step ms to 0.01, aggregate tok/s to 0.1, step counts —
  checked mechanically); `fuse_qkv: false` and `recompiles_in_window: 0` on every receipt.
- **The decode-attention compute path is chosen per family, not per arm:** the receipts' `mech.compute` tally reads
  `fp8` on OLMoE and Qwen3 and `f32` on Granite and gpt-oss, as bo5's and the tp lane's receipts do (an unset
  `GNF4_ATTN_COMPUTE` selects capability-conditionally). Identical across every arm of a family, so ratios do not see it.
- **The fusions print no banner.** `E4B_FUSE_*` refuse aloud when nothing matches (#333) and are silent when they engage;
  their engagement is read from the step time. The int4 / calibrated arms print `INT4EXP` / `ATTNINT4` banners, which
  `RESULTS.md` reproduces in the `engaged` column.
- **Calibration counts reproduce across hosts:** Granite 2524 gptq / 36 rtn (bo5's counts), Qwen3 10820 / 1468 (bo6b's
  and bo6c's counts) — the calibration set and the `min_rows` fallbacks are host-independent at the count level; the
  calibrated packs are speed-identical to RTN packs (same kernel, same bytes).
- **No K8 arm ran on this lane.** Every verdict is the register's; nothing here licenses a ratio.

## The table

[`RESULTS.md`](RESULTS.md) is the verbatim output of [`census_reduce.py`](census_reduce.py) (`python census_reduce.py .`)
followed by the reading of it. The reducer reads the JSON receipts (one per arm; its result line is in `summary.txt`),
each arm's status and alarm from `logs/outer.log`, what each arm engaged from its `logs/run_*.log`, and prints per family
the arm table (configuration · licence label from the register with the claim id · engaged · B=1 ms · B=1 tok/s · ×NF4 ·
B=16 tok/s · ×NF4 · status), then the licensed best per family on the three axes, then a cross-family summary of ratios
(no absolute is compared across families — they differ in size).

## Layout

- `<family>_b1_<arm>.json` — timed graph window (`step_ms_clean`, `n_steps`, the kernel's `mech` tallies);
  `<family>_b16_<arm>.json` — B=16 (`aggregate_tok_s`, `step_ms_clean`, the slots and prompts). [`calib.json`](calib.json)
  is the attention-calibration manifest (byte-identical to bo6's / bo5's / bo3's). [`summary.txt`](summary.txt) (each run
  log's final result line, in lane order), [`models.txt`](models.txt), [`forensics.txt`](forensics.txt),
  [`forensics2.txt`](forensics2.txt).
- [`logs/`](logs/) — [`bo7_run.sh`](logs/bo7_run.sh) (the amended lane script, the one that ran),
  [`bo7_run.sh.pre-amend`](logs/bo7_run.sh.pre-amend) (the pre-amendment copy, kept for the record),
  [`bo7_all.sh`](logs/bo7_all.sh) (the renter / guard / pre-flight / launcher on the mini),
  [`bo7_queue.sh`](logs/bo7_queue.sh) (the queue behind bo6), `bo7b.sh` (amendment 2's two arms; ships with the final
  snapshot), the hook (`usercustomize.py` = v6, `hook/usercustomize.py`
  its installed copy), `k8_bake.py`, `step_decomp.py`, the four bake logs, **every per-arm `run_*.log`** (the hook banners,
  the calibration pass lines, the result line) and the lane console `outer.log`. The logs are force-added past the
  repository's `*.log` ignore rule, as bo6's were. Nothing in this directory is edited: every file is the byte-for-byte
  copy of the box's, except the four written here (`README.md`, `RESULTS.md`, `census_reduce.py`; `forensics2.txt` is the
  orchestrator's probe). The mini-side watcher scripts (`bo7_loop.sh`, `bo7watch.sh`), `pip.log` (a root-user warning
  only), the `TP_DONE` marker and the hook's `__pycache__` are not shipped.

## Reproduce

Rent the class in `forensics.txt`, stage `logs/step_decomp.py`, `logs/k8_bake.py` and `logs/hook/usercustomize.py` under
`/root/bo7/` with `calib.json` and `models.txt` beside them, put the Hugging Face token at `~/.cache/huggingface/token`,
and run `logs/bo7_run.sh` (its `pip install` line pins both cuts; its tripwire refuses anything else). Then
`python census_reduce.py .`.

## What is NOT in the table

No K8 arm: no perplexity was scored on this lane, and no label here was decided by it. **The speed of Qwen3's licensed
stack (the streamed 64k pack) is amendment 2's arm, pending in this snapshot** — the lane's own calibrated arms ran the
hook's 16k default. Mixtral's `calibexp_lic` was dropped (amendment 1) and is a row that says so. No anchor projection is
computed in this snapshot (the only arm one applies to has no receipt yet; the reducer prints it, marked as a projection,
when it does). **In this snapshot Gemma-4 and Mixtral are pending** (16 arms), Qwen3's `calibexp_folds` B=16 was
running, and amendment 2's two arms had not started; all arrive before merge. No kernel census ran.
