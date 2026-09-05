# Build-out validation lane bo6 (+ bo6b, bo6c) — 2026-09-04/05, one RTX 5090: close the gap, not the gate

**Box:** Vast.ai instance 49861751, RTX 5090 (sm_120, driver 580.173.02, 32607 MiB) on an AMD Ryzen Threadripper
PRO 7975WX host (64 CPUs, 251 GB, container cgroup `memory.max` 183,093,952,512 B = 170 GiB — see
[`forensics.txt`](forensics.txt)); torch 2.8.0+cu128, transformers 5.16.1, bitsandbytes 0.50.1, triton 3.4.0. No
memory-bandwidth probe ran on this box.
**When:** 2026-09-04 13:25Z onward. Per [`logs/outer.log`](logs/outer.log) and the two earlier attempts: attempt 1
13:25:09Z (died at the bake), attempt 2 13:27:23Z (baked Qwen3, scored the NF4 references, the first calibrated arm
refused), attempt 3 13:50:05Z (`QWEN3 DONE` 15:39Z, Mixtral bake 15:39–15:54Z, Mixtral NF4 references 15:54–15:57Z,
its four calibrated arms killed one after another, `TP_DONE` 16:43:25Z), bo6b 16:43:43Z (Qwen3 re-bake, the repeat
controls and the sweep, `QWEN3 SWEEP DONE` 18:56Z; Mixtral re-bake 18:56–19:10Z; `lic_calibexp_c4val` 19:10–20:33Z;
`lic_calibexp` on wikitext 20:33–21:54Z; then three arms that never produced a receipt — `lic_calibexp` B=1 21:54Z,
killed by the `arm()` helper's own 3600-s alarm at 22:54Z at streamed-calibration pass 23 of 32; B=16 22:54Z, alarm
23:54Z, pass 23 of 32; `lic_calibexp_n128_c4val` 23:54Z, killed by the `k8()` helper's 5400-s alarm at 01:24Z at pass
20 of 32 — `MIXTRAL SWEEP DONE`, `TP2_DONE` 2026-09-05T01:24:40Z). Those three arms print as **`alarm`** rows in
[`RESULTS.md`](RESULTS.md): a harness limit, not a model result, and no number is quoted for them. **bo6c** followed on
the same box ([`logs/bo6c.sh`](logs/bo6c.sh)): reinstall 01:25:01Z — the console line says `reinstall e4b @d286dd5
(damp knob)`, which is the stale plan message; the script's `pip install` line names **@ae9dc122** and the tripwire
that passed asserts the streamed driver (`enable_serve_experts_int4_calibrated`) and hook v6 — so the cut and hook
are verified live, with `E4B_INT4_HESSIAN_BUDGET_GB=24`; Qwen3 re-bake 01:25–01:34Z; `calibexp_c4val_rep2` 01:34Z;
`calibexp_n128` on wikitext 01:54Z; `all_calibexp_n128` on wikitext 02:21Z and on c4val1 02:50Z; `QWEN3 STREAMED
STACK DONE`, `TP3_DONE` 2026-09-05T03:18:54Z. The box was torn down after `TP3_DONE`.
**Lane:** the user's decision at 13:10Z — *close the gap rather than move the gate*. The registered K8 gate stays in
perplexity: a calibrated pack passes when Δppl ≤ +0.05 against NF4 on **every** text scored, an improvement is
claimable only with the same sign on ≥ 2 texts, and wikitext is the text outside the calibration domain. The
int4-expert arms that failed their second text on bo5 with round-to-nearest experts (Qwen3 `all` +0.063 ppl,
`int4exp` alone +0.071; Mixtral's `lic` stack +0.058) are re-run with per-expert GPTQ calibration (e4b#384's
mechanism) in place of RTN. Granite is not re-run: its calibrated arm read +0.387 ppl on bo5, eight times the budget,
which this lever does not close. Pre-registration, every reading with its timestamp, and the instrument notes are in
the private receipt (`INT4B16/P25-PARITY.md`, section P34).

**Cuts** (all with grouped-nf4-gemm main @`0b25d1389701bb793d60075e5b870212c848e33a`, the pin CI adopted in #391;
its metadata reports 0.29.0):

- **@`f42924d509ecb3f534f57a87976219986fe5b69f`** — e4b integration-8 = main 7dcc16f (0.34.0 + #385 decode glue +
  #388 discoverability) + #384 (calibrated int4 experts) + the GPU-solve knob (`E4B_INT4_GPTQ_DEVICE=cuda`).
  Attempts 1 and 2 ([`logs/outer.attempt1.log`](logs/outer.attempt1.log), [`logs/outer.attempt2.log`](logs/outer.attempt2.log)):
  the Qwen3 NF4 references (`qwen3_ppl_nf4*.json`) were scored on this cut, and NF4 does not touch #384.
- **@`db2a0703b803e50de1e3ac4d18b4705cb90c17fe`** — the GPTQ pack returns to the stack's device. Attempt 2's first
  calibrated arm had refused: the GPU solve returned GPTQ-packed experts on `cuda` while the 94 experts that saw no
  rows were RTN-packed on the CPU, and the per-layer stack's `torch.cat` refused to mix them (the calibration itself
  had worked: `INT4EXP hessians: 48 layers, 6050 (layer, expert) pairs`). Attempt 3 ([`logs/bo6_run.sh`](logs/bo6_run.sh),
  [`logs/outer.log`](logs/outer.log)) relaunched on this cut keeping the bake and the two NF4 references; every
  attempt-3 calibrated arm is **all-at-once** calibration (below). Mixtral's NF4 references were scored here.
- **@`d286dd52a8b23222479fb9de880bbdd00f6bf671`** — adds `E4B_INT4_GPTQ_DAMP` (default = the kernel's 0.01;
  numerically identical for arms that do not set it). Planned as bo6b's cut at 14:12Z; superseded by the next cut
  before bo6b launched, so **no arm ran on it** — bo6b's `reinstall e4b @d286dd5 (damp knob)` console line is the
  stale message of that plan; the `pip install` on the line below it in [`logs/bo6b.sh`](logs/bo6b.sh) names
  @ae9dc122, and the tripwire that follows asserts the streamed entry point that only @ae9dc122 has.
- **@`ae9dc122e25216e8f8713c1631f5904e059efeb6`** — `enable_serve_experts_int4_calibrated`: **streamed** calibration
  and packing per layer chunk under `E4B_INT4_HESSIAN_BUDGET_GB`, with layer filters on both steps. bo6b
  ([`logs/bo6b.sh`](logs/bo6b.sh)): the repeat controls, the Qwen3 sweep and every Mixtral `lic_calibexp` arm; and
  **bo6c** ([`logs/bo6c.sh`](logs/bo6c.sh), 24 GiB budget): the calibration-determinism repeat, the 64k-token pack on
  wikitext, and the shipping stack on both texts. #384 with this mechanism is merged on main as 6bca732 and shipped
  in 0.35.0.

## The false starts

1. **Attempt 1 (13:25Z) died at the bake:** the run script had dropped the `transformers/bitsandbytes/datasets/...`
   line when copied from a truncated read of bo5's, so `safetensors` was missing and `k8_bake.py` failed at import
   for both families (`outer.attempt1.log`).
2. **Attempt 2 (13:27Z) refused at the first calibrated arm** — the device mismatch above. Fixed on @db2a070; the bake
   and the NF4 references were kept.
3. **Attempt 3's Mixtral calibrated arms were OOM-killed by the container** (`Killed` on all four in `outer.log`;
   the cgroup's `memory.events` recorded `oom_kill 1` at 170 GiB on a 251 GB host). Read from the code:
   `calibrate_expert_hessians` bounded the per-*pass* accumulators (24 GiB) but returned every layer's fp32 Hessians
   before packing began. Mixtral's 8 experts × (4096² + 14336²) × 4 B ≈ 7.1 GB per layer × 32 layers ≈ 230 GB;
   Qwen3's 128 × (2048² + 768²) × 4 B ≈ 2.4 GB × 48 ≈ 115 GB had survived only because it fits under the limit. The
   fix is @ae9dc122's streaming; bo6b re-ran the Mixtral arms at an 8 GiB budget (32 passes of one layer).
4. **bo6b's Mixtral speed arms and its 64k-token arm hit the lane's own per-arm alarms** (`Alarm clock` in
   `outer.log`; the run logs stop mid-calibration). The `arm()` helper's 3600-s and the `k8()` helper's 5400-s alarms
   were sized for bo5's RTN arms, which need no calibration; the streamed 32-pass Mixtral calibration at an 8 GiB
   budget takes ~85 min before a timed window can start (B=1 and B=16 both died at pass 23 of 32), and the 64k-token
   set makes each layer ~4.5 min (`lic_calibexp_n128_c4val` died at pass 20 of 32). The Mixtral speed of the
   calibrated stack was therefore **not measured** on this lane, and neither was its 64k-token reading; bo7 dropped
   those arms too (the stack is not licensed, below). A harness limit, not a model result.

## Protocol

bo5's ([`../bo5/README.md`](../bo5/README.md)), two families: per family an NF4 bake of the released checkpoint
(`logs/k8_bake.py`), then arms under `logs/step_decomp.py` (byte-identical to `bench/hybrid-g9/step_decomp.py`):
K8 teacher-forced NLL (2048 steps, sha-matched windows, `--b1d-loop eager`, `--no-fuse-qkv`, fp8 paged KV,
placement all-vram) on **wikitext** and on **`c4val1`**, B=1 decode (512-token prompt, 128 generated, graph loop,
timed window) and B=16 aggregate (graph loop, 70 steps). The NF4 references are **re-scored on this lane**
(`nf4` / `nf4_c4val` per family) and every delta is against them — never against bo5's or bo3's rows. The hook
(`logs/hook/usercustomize.py`, v6; `logs/usercustomize_v4.py` is the version attempt 3's first arm ran under — the
v4 → v5 → v6 lineage differs only by `E4B_CALIB_NSEQ` and the streamed call) attaches the int4 lanes after the
hybrid tier is enabled; `E4B_SERVE_EXP_INT4_CALIB=1` selects calibrated experts, `E4B_CALIB_NSEQ` the calibration
set size, `E4B_INT4_GPTQ_DAMP` the damping, `E4B_INT4_HESSIAN_BUDGET_GB` the streamed budget. Calibration: 8
batches × 4 sequences × 512 tokens (16k tokens; ×4 / ×16 in the sweep) through the fused forward's Hessian tap,
`min_rows=32` (an expert that saw fewer rows is packed RTN), damping 0.01, GPU solve.

## The two methods, and why they differ

Every calibrated arm's method is read from its own run log and printed in the table's `method` column:

- **All-at-once** (attempt 3; hook v4/v5 path on @db2a070): `calibrate_expert_hessians` runs the calibration batches
  once through the NF4 model, accumulating **every** layer's per-expert Hessians against the *unquantised* prefix,
  and then every layer is packed. Banner: `INT4EXP hessians: 48 layers, 6050 (layer, expert) pairs from 8 batches`.
- **Streamed** (bo6b; hook v6 on @ae9dc122): `enable_serve_experts_int4_calibrated` packs the layers in chunks —
  the first chunk's Hessians are accumulated, that chunk is packed to int4, and only then are the next chunk's
  Hessians accumulated, so layer *L* is calibrated against the **int4 outputs of layers < L** — the sequential
  convention GPTQ is defined with. Banner: `INT4EXP calibrating (streamed): 8 batches of c4 budget GB 24`, then one
  `calibrated experts` line per pass and `calibrated streaming: 48 layers in 5 passes`.

On the same box, the same 8 batches, the same damping and `min_rows`, that reordering moves Qwen3's c4val1 reading
by 0.200 ppl (16.64678 → 16.44653) and flips the registered verdict from FAIL to pass; the RTN-fallback count is
the same under both (1462 vs 1468 packs), so it is the order, not the row count. **Attempt 3's "calibrated experts
alone FAIL +0.150" is therefore a verdict on the all-at-once method. The streamed method is what ships** (0.35.0,
#384). The all-at-once path remains the two-step API (`calibrate_expert_hessians` + `enable_serve_experts_int4(...,
expert_hessians=)`); `all_calibexp` — the full stack — was scored under it on both texts and passes as registered,
and its speed arms carry that pack.

## The three texts

- **Calibration text:** C4 *validation* shard `00000-of-00008` (the hook's `_calib_batches`: the first 4000
  documents, 512-token windows spread evenly through them). The C4-calibrated int4 *attention* pack in `all_calibexp`
  reads the same batches.
- **`c4val1`, the second text:** C4 validation shard `00001-of-00008` (`step_decomp.py --ppl-source c4val1`; window
  sha `4bcb55179b96` for Qwen3, `061534dadd4d` for Mixtral). Same split and domain as the calibration text, **disjoint
  documents** — every c4val1 number here is in-domain generalisation, not the calibration set scored on itself.
- **wikitext:** outside the calibration domain (sha `9ef10d760ad9` / `31fd7d408809`). Under the registered rule an
  improvement is claimable only when it holds with the same sign here.

bo5's README described its packs as calibrated on C4 *train*. bo5 did not ship its hook, so that is not checkable
from its bundle; the copy of bo5's hook kept in the private audit tree is byte-identical to this lane's
[`logs/usercustomize_v4.py`](logs/usercustomize_v4.py) and reads the same validation shard 00000 — the "train"
wording there is an erratum. It softens nothing: bo5's second-text verdicts were FAILs scored *inside* the
calibration domain, the stronger reading. This lane ships its hook.

## What is licensed by this lane, and what is not

- **LICENSED as registered — the shipping stack (bo6c):** Qwen3 `all_calibexp_n128` = sequentially (streamed)
  calibrated int4 experts at 64k C4-validation tokens + C4-calibrated int4 attention (192 projections from the same
  32 batches) + round-1/2 folds + router epilogue + #385 glue: wikitext 1.85114 (6.36709) = −0.0528 ppl
  (−0.0083 nats), c4val1 2.79916 (16.43081) = −0.0662 ppl (−0.0040 nats) — pass on both texts under the unchanged
  gate, the same sign on both. Both deltas are inside the family's 0.0095-nat floor, so the reading is **at parity or
  better on both texts**; no improvement is claimed by a number. Its speed is not on this lane: bo7 measures the
  calibrated stack at B=1 / B=16 with the hook's 16k-token default, not the 64k pack scored here.
- **Licensed as registered (all-at-once pack, the two-step API):** Qwen3 `all_calibexp` — Δppl −0.0597 on wikitext,
  +0.0352 on c4val1, both within +0.05; *no improvement is claimed* (the signs differ). Its 158.0 / 993.6 tok/s are
  rental-measured on this box, with no ratio (no NF4 speed arm here; bo7 measures it) — the speed of a different
  pack from the shipping one.
- **Pass on both texts, parity out of domain:** the streamed 64k experts-only pack (`calibexp_n128`) — c4val1
  −0.2109 (bo6b), wikitext −0.0002 ppl / −0.00003 nats (bo6c). The improvement clause is met on its letter (same
  sign on both texts), but the wikitext delta is inside the floor: *parity on the out-of-domain text, −0.211
  in-domain*; no improvement is claimed by a number.
- **Pass on one text, improvement not claimed:** the streamed 16k (−0.0505; repeated bit-identically by bo6c) and
  256k (−0.1410) expert arms on c4val1 — wikitext was not scored at those sizes.
- **FAIL as registered, and recorded as such:** Qwen3 calibrated experts alone under the all-at-once method
  (+0.1498) and under streamed calibration with damping 0.1 (+0.0544); and **Mixtral `lic_calibexp`** — c4val1
  +0.0391 (pass) but wikitext +0.0771 ppl (+0.0234 nats; floor unmeasured; receipt `mixtral_ppl_lic_calibexp.json`)
  — **measured, not licensed**. Sequential calibration closed Qwen3's gap and did not close Mixtral's: the
  calibrated stack passes the in-domain text and fails the out-of-domain one, the mirror image of bo5's RTN `lic`.
  Its B=1 / B=16 speed was not measured (the arms alarmed, false start 4), and bo7 dropped those arms too. Next
  levers are not gate changes: a per-expert NF4 fallback for the largest-residual experts, or the 64k calibration
  set scored on wikitext (the `lic_calibexp_n128` arm that alarmed was c4val1-only).
- **Not touched:** Granite (NF4 experts stay licensed; bo5's +0.387 is not closable by this lever), gpt-oss, Gemma-4,
  OLMoE. Nothing on this page licenses a *ratio*; nothing is compared across lanes.

**bo6c ran on this box behind bo6b** (01:25–03:18Z, 2026-09-05): the streamed 16k arm repeated for calibration
determinism, the 64k arm on wikitext, and the full stack at 64k under the streamed method on both texts. Its four
receipts are in this directory and its rows are in the table.

## Instrument notes

- **Deterministic on one box and cut.** bo6b re-baked Qwen3 and re-scored NF4 three times (c4val1 ×2, wikitext ×1):
  bit-identical `mean_nll` at full float precision (the repeat-controls table in `RESULTS.md`). bo6c re-baked again
  and repeated the streamed 16k calibrated-expert arm end to end (Hessian tap, GPU solve, `min_rows` fallbacks,
  packing, K8): `calibexp_c4val_rep2` = `calibexp_c4val_rep1` bit-identically (2.800114648339439). Any sub-0.01-nat
  difference *within* this lane is real arithmetic, not run noise.
- **The cross-lane 0.006-nat shift stays OPEN.** Qwen3's NF4 c4val1 reads 2.80318 here and 2.80923 on bo5 (box
  49841214, integration-6 + grouped-nf4-gemm @587eb7a) on the identical window sha, while Mixtral's NF4 references
  agree across the same two lanes to 0.001 nats. Determinism rules out run noise; what remains is the install or the
  box. bo5's kernel install was exposed to the stale `build/lib` race (grouped-nf4-gemm tracked stale modules until
  #336; bo5's tripwire proved two of them fresh, and the race is per file), which cannot by itself explain a
  Qwen3-only shift. Consequence, applied on this page: **no sub-0.01-nat comparison is made against bo5 or against any
  lane installed from a commit before grouped-nf4-gemm#336**; within-lane and bo6/bo7 comparisons are.
- **Speed is quoted with its box only.** Qwen3 `all_calibexp` reads 6.33 ms / 993.6 tok/s here; bo5's RTN-expert
  `all` read 4.90 ms / 1251.6 on its EPYC box. Different host, different cut, no same-box NF4 arm and no bandwidth probe
  on this box (the 5090 class carries measured bandwidth heterogeneity): the difference is not read. bo7 carries the
  NF4 arm.
- The Mixtral NF4 references come from attempt 3's bake; bo6b's Mixtral arms run on a re-bake. Qwen3's repeat controls
  show bake + score determinism; Mixtral's NF4 was not repeated on the re-bake.

## The table

[`RESULTS.md`](RESULTS.md) is the verbatim output of [`buildout_reduce.py`](buildout_reduce.py)
(`python buildout_reduce.py .`) followed by the reading of it. The reducer reads the JSON receipts (one per arm;
its result line is in `summary.txt`), reads each arm's method from its run log and its damping / set size from
`outer.log`, and applies the registered rule (`experts4bit_qlora.k8_gate`) **in its own units, perplexity, per
text**; nats are printed beside every verdict, read against the family's arithmetic-order floor (Qwen3 0.0095,
Mixtral unmeasured), and never change it. It also prints the repeat-controls table, the method / size / damping
sweep with the method effect at equal settings, and the speed rows with `ratio: not measured on this lane`.

## Layout

- `<family>_ppl_<arm>.json` — K8 receipt (`mean_nll`, `ppl`, `ppl_source`, `text_sha`); `_c4val` on the arm name =
  the second text; `_rep1` / `_rep2` = repeat controls; `_d01` / `_n128` / `_n512` = the sweep; bo6c's four are
  `qwen3_ppl_calibexp_c4val_rep2`, `qwen3_ppl_calibexp_n128` (wikitext) and `qwen3_ppl_all_calibexp_n128[_c4val]`.
  `<family>_b1_<arm>.json`
  — timed graph window (`step_ms_clean`); `<family>_b16_<arm>.json` — B=16 (`aggregate_tok_s`).
  [`calib.json`](calib.json) is the attention-calibration manifest, byte-identical to bo5's and bo3's.
- [`logs/`](logs/) — the three lane scripts with their pip pins (`bo6_run.sh`, `bo6b.sh`, `bo6c.sh`), the hook
  (`usercustomize.py` = v6, `hook/usercustomize.py` its installed copy, `usercustomize_v4.py` the version before the
  set-size knob), `k8_bake.py`, `step_decomp.py`, the two bake logs, **every per-arm `run_*.log`** (the method
  banners, calibration statistics and result lines), and the three lane consoles `outer.attempt1.log`,
  `outer.attempt2.log`, `outer.log`. The logs are force-added past the repository's `*.log` ignore rule, as bo5's
  `outer.log` was. `run_mixtral_b1_lic_calibexp.log`, `run_mixtral_b16_lic_calibexp.log` and
  `run_mixtral_ppl_lic_calibexp_n128_c4val.log` are the alarmed bo6b runs (the lane overwrote attempt 3's OOM-killed
  b1/b16 logs in place); they end mid-calibration with no result line. `bake_qwen3.log` is bo6b's re-bake (the one
  behind the repeat controls and the sweep); bo6c's re-bake overwrote the box's copy and is shipped as
  `bake_qwen3.bo6c.log` (a renamed verbatim copy — the bake behind bo6c's four receipts). `outer.log` runs through
  `TP3_DONE`; `summary.txt` holds all 20 result lines.
- [`summary.txt`](summary.txt) (each run log's final result line, in lane order), [`models.txt`](models.txt),
  [`forensics.txt`](forensics.txt) (`nvidia-smi`, `lscpu`, `free`, the cgroup limit, the package versions).
  The lane's `TP_DONE` / `TP2_DONE` markers are empty files and are not shipped; nor are the pip logs (warnings
  only, including bo6c's), the watcher / restart scripts or the `.sha` pin files (their contents are the cut list
  above).

## Reproduce

Rent the class in `forensics.txt` (a container whose cgroup allows ≥ 170 GiB, or set
`E4B_INT4_HESSIAN_BUDGET_GB` lower), stage `logs/step_decomp.py`, `logs/k8_bake.py` and `logs/hook/usercustomize.py`
under `/root/bo6/`, put `calib.json` beside them, and run `logs/bo6_run.sh` (its `pip install` line pins the cuts),
then `logs/bo6b.sh` (waits on `TP_DONE`), then `logs/bo6c.sh` (waits on `TP2_DONE`). Every arm is one
`step_decomp.py` invocation; the `k8()` / `arm()` helpers
carry the flags, `fenv` maps `0 / all` to `E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`, `E4B_FUSE_ROUTER_EPI`, and the
per-arm environment (`E4B_SERVE_EXP_INT4`, `E4B_SERVE_EXP_INT4_CALIB`, `E4B_SERVE_ATTN_INT4_CALIB`,
`E4B_INT4_GPTQ_DAMP`, `E4B_CALIB_NSEQ`, `E4B_INT4_HESSIAN_BUDGET_GB`) is on each arm's line in `outer.log`. Then
`python buildout_reduce.py .`.

## What is NOT in the table

Every K8 row is the 2048-step window; nothing was probed short. **Mixtral's calibrated-stack speed (B=1, B=16) and
its 64k-token c4val1 reading are not measured** — the three arms were killed by their own alarms during the streamed
calibration (false start 4) and print as `alarm` rows with no number. **The licensed Qwen3 stack's speed is not on
this lane** (bo7 measures the calibrated stack at the hook's 16k default, not 64k). The streamed 16k and 256k expert
packs were not scored on wikitext. Granite, gpt-oss, Gemma-4 and OLMoE were not run. No kernel census ran on this
lane.
