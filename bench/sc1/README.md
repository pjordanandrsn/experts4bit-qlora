# SC1 — Qwen3-30B-A3B serving head-to-head on RTX 5090s: e4b vs vLLM / SGLang / llama.cpp / ExLlamaV3 / LMDeploy (orchestration)

Pre-registration: `SC1-PREREG.md` (the PI's; registered as draft v3 plus the round-3 patches — the box script follows its three boxes and
their phases IN THAT ORDER). Upstream facts: `UPSTREAM-NOTES.md`. Issue: experts4bit-qlora#846.
Lineage: `bench/p88/p88_run.sh` (the NF4 bake, the licensed pack build with `LICENV`/`SPEEDENV`/`K8ARGS`, the first-chunk watchdog, the
K8 rows, the premise test), `bench/p58/p58_run.sh` + `p58_drive.sh` (nonce handshake, `can_run`/`arm_alarm`, the prompt dump, the
staging map, heartbeat / liveness, fetch), `bench/tc1/tc1_drive.sh` (the forwarded-knob list, `%q`), `bench/p39/step_decomp.py` (the
window arm, K8, the upstream oracle), `experts4bit_qlora/serve_paged.py` (`build_engine`) and `engines/scheduler.py` (the sched arms).

| file | role |
|---|---|
| `sc1_run.sh` | BOX side, `SC1_BOX=A\|B\|C`: nonce + `finish` markers (`SC1_EXIT_CODE.<nonce>`, `SC1_SUCCESS.<nonce>`, `TP_DONE.<nonce>`); the staged-pin check; refusals BEFORE any install (class rc 15, disk < 320 GB rc 13, driver < 580 rc 18, box A not AuthenticAMD rc 18; each leaves `REFUSAL`); forensics incl. `nproc`, `cpu.max`, `/proc/loadavg`, PCIe gen/width; the e4b venv (`--system-site-packages`, P58 pins, gnf4 v0.34.1) + tripwire (K19/K23 defaults `auto`, K25 `0`, `Int4Linear.fuse`, `serve_paged.build_engine`, `add_request(stop_ids=)`); the box's comparator installs under alarms with UNSUPPORTED stubs; the quiescence gate; `SC1_PROVE=1` (installs + tripwires + the Granite paged-engine smoke at B=1/16, no Qwen3 fetch); the real lane: fetches, bake, prompts, box A's `build_engine` smoke with BOTH fusion sets before the pack build, P88's `k8 build` → `PACK <fp>`, the premise (rc 25), then the per-box phase list (`box_a` / `box_b` / `box_c` near the end) |
| `sc1_e4b_sched.py` | e4b timed THROUGH `ContinuousScheduler.step()` via `serve_paged.PagedServeConfig.from_env()` + `build_engine`: the p37 slope (32→128, warm + 3 reps, exactly N tokens per `done` request asserted, 512 prompt tokens as the engine counted them), `--ttft`, `--sameprompt`, `--energy` (1024 tokens, prints its epoch window), `--smoke` (census; exit 0 only if every SET lever counts nonzero and every bucket captured), `--selftest`. Receipt in p37's shape with `engine: "e4b-sched"` + census / graph_status / graph_stats / scheduler stats / banners / the four route knobs / `same_stack` fields. Written for the FIXED `_apply_fusions` (fix/paged-fusions `2719171`): `E4B_PAGED_FUSE_QKV=1` + the fold flags = the registered fused-with-folds set |
| `sc1_prompts.py` | `prompts_b1.json`, `prompts_b16.json`, `prompts_b1_4096.json` (P58's record, `step_decomp._k8_window` byte for byte), `prompts_b16_same.json` (16 copies of row 0; says so), `k8_window_{wikitext,c4val1}.json` (`ids[:2561]` + `text_sha` re-derived with the stdlib and asserted equal to the harness's); `--selftest` |
| `sc1_sampler.sh` | per-arm `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw.instant,pcie.link.gen.current -lms 50` → `samples/<tag>.csv` + a 1 s host sampler (loadavg, top-5 RSS/%CPU) → `<tag>.host.txt`; `<tag>.meta.json` on stop |
| `sc1_drive.sh` | CONTROLLER side (the launcher's `--command`): checks every pin against its source, stages the flat pieces + the comparator dirs (tar, `COPYFILE_DISABLE`) + P39's harness + P42's hook + the K19 premise test, starts under a nonce, heartbeats (stall reported, never acted on; a dead lane ends the wait with rc 25), fetches receipts / logs / samples / `bake.json` / the pack `manifest.json` (never venvs, caches, arenas, checkpoints, payloads); forwards every `SC1_*` knob the box reads (`%q`); `SC1_DRIVE_DRYRUN=1` |
| `staged.sha256` + `make_pin.sh` | the pin, named as the box sees the files; `make_pin.sh` regenerates it (`sc1_reduce.py`, the reducer, is pinned since it landed on `campaign/sc`) |
| `vllm/`, `sglang/`, `llamacpp/`, `exl3/`, `lmdeploy/` | the comparator drivers (their own reports and tests); the runner calls them through the contracts those reports define |
| `../../tests/test_sc1_staged_pin.py` | CI: every pinned file matches its source; pinned == staged; the harness is P86/P88's pinned bytes; `make_pin.sh` reproduces the pin; the driver reaches its dry run |
| `../../tests/test_sc1_run_shape.py` | CI: `bash -n`; refusals precede the install; the proof fetches no Qwen3; the three boxes' phase order as v3 lists it; every `SC1_*` the box reads is forwarded; every v3 arm name appears; P88's env strings byte-identical; the quiescence gate placement; the two selftests |

## The three boxes (each its own draw; every ratio within its box)

- **A** (AMD host, `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`): Phase 0 (fetch bf16 + GPTQ, bake, prompts, `build_engine` smoke
  fused/unfused at B=1/16, quiesce, the licensed pack build, premise) → A (`e4b/lic_b16_r1`, `vllm/gptq_graph_b16_r1`, `e4b/lic_b1_r1`,
  `vllm/gptq_graph_b1_r1`) → A2 (`e4b/lic_sched_b{16,1}_r1`) → B (K8 lic `auto` + `=1`, wikitext + c4val1; NF4 control both texts) →
  D-vLLM (vLLM prefill + served quality both windows; e4b `--ppl-oracle eager` rows) → G1 (second draws of A, A2) → C (`fp8kv`, `rtn`,
  `nf4_ctrl`; the AWQ arm is CUT, v4 Phase C) → F (`vllm/gptq_graph_b16_SAMEPROMPT`, `e4b/lic_sched_b16_SAMEPROMPT`, `e4b/lic_b16_degraded`) → E (TTFT 512/4096
  e4b-sched + vLLM) → EN (energy `lic_sched_b{1,16}`, `vllm_b{1,16}`) → G2 (second draws of C + controls, third draws on > 3 % disagreement,
  the nodetok pair at both B as the first droppable, `nf4_ctrl` r2 as the second, the fp8-KV served quality rows last).
- **B** (same image): Phase 0 (fetches incl. two GGUFs + EXL3, bake, prompts; llama.cpp build + ExLlamaV3 cu128 + LMDeploy venvs) → anchors
  `e4b/int4_b{16,1}` + `e4b/int4_sched_b{16,1}` + `vllm/gptq_graph_b{16,1}` → the bf16 upstream oracle (both windows) → `llamacpp/q4km`,
  `exl3/4bpw`, `lmdeploy/w4a16` at B=16/1, `llamacpp/iq4xs_b1` → their quality (prefill + served shape; LMDeploy's served shape is
  `decode_prefix`, partial-block) → anchors r2 → TTFT on every engine → energy B=1 (three comparators) → comparators r2 + third draws.
- **C** (`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` + python 3.12; e4b on torch 2.8 cu128 in its own venv): Phase 0 (fetch bf16 for the bake,
  GPTQ, EXL3; bake; prompts) → anchors → SGLang install + first JIT (matched server) + quiescence → `sglang/gptq_matched_b{16,1}`,
  `sglang/gptq_native_b{16,1}` → SGLang quality → `exl3/4bpw_cu13_b{16,1}` (labelled native) → TTFT (e4b-sched, vLLM, SGLang, ExLlamaV3) →
  SGLang r2 + third draws → the anchors' r2 as droppables.

Alarms: bake 5400, build 5400 (first-chunk watchdog 1500), timed arm 1800, quality arm 2400, oracle 3600, install 2700. `can_run <need> <name>`
never starts an arm that cannot finish 10 min before the deadline; a skipped arm is a `SKIPPED ... host-limited deadline` line.

## Receipts (the reducer's vocabulary, flat in `$W`; one JSON + `logs/run_<stem>.log` + a `summary.txt` line + `samples/<stem>.{csv,host.txt,meta.json}` per arm)

Timed arms `<engine>_<arm>_b<B>_r<n>.json`, engine `e4b` (window) | `e4bsched` (scheduler) | `vllm` | `sglang` | `llamacpp` | `exl3` | `lmdeploy`:
`e4b_lic_b16_r1`, `e4b_rtn_b1_r2`, `e4b_nf4_ctrl_b16_r1`, `e4b_int4_b1_r1`, `e4b_lic_degraded_b16_r1`, `e4bsched_lic_sched_b16_r1`,
`e4bsched_int4_sched_b1_r2`, `e4bsched_lic_sched_sameprompt_b16_r1`, `vllm_gptq_graph_b16_r1`, `vllm_gptq_fp8kv_b1_r1`,
`vllm_gptq_graph_sameprompt_b16_r1`, `vllm_gptq_graph_nodetok_b{1,16}_r1`, `sglang_gptq_matched_b16_r1`, `sglang_gptq_native_b1_r1`,
`llamacpp_q4km_b16_r1`, `llamacpp_iq4xs_b1_r1`, `exl3_4bpw_b16_r1`, `exl3_4bpw_cu13_b16_r1`, `lmdeploy_w4a16_b1_r1`; a third draw is `_r3`.
Quality: `k8_build_wikitext`, `k8_lic_{auto,1}_{wikitext,c4val1}` (+ `logs/census_k8_lic_1_<src>.txt` from the `e4b_k19census_b1` arm),
`k8_nf4_<src>`, `nll_e4b_prefill_<src>` (the `--ppl-oracle eager` rows), `nll_vllm_{prefill,served}_<src>`, `nll_vllm_fp8kv_served_<src>`,
`nll_sglang_{prefill,served}_<src>`, `nll_llamacpp_{prefill,decode}_<src>`, `nll_exl3_{prefill,decode}_<src>`, `nll_lmdeploy_{prefill,decode_prefix}_<src>`,
`oracle_<src>`. TTFT `ttft_<engine>_{512,4096}` (engine `e4b` = the sched engine). Energy `energy_<tag>.json` + `energy_<tag>.csv`
(`start_epoch`, `stop_epoch`, `tokens`, `engine`, `batch`, `basis`; tags `e4b_b{1,16}`, `vllm_b{1,16}`, `llamacpp_b1`, `exl3_b1`, `lmdeploy_b1`), beside
the arm receipts `e4bsched_energy_b<B>`, `vllm_energy_b<B>`, `<engine>_energy_<arm>_b1` (no `_r<n>`: not timed rows). Smokes `smoke_<tag>.json`.
`unsupported_<engine>.json` for an engine whose install failed; a stub (`status: unsupported|harness_error|alarm|not_run|load_fault`, `written_by: sc1_run.sh`,
`log_tail`) for every arm that produced no receipt of its own. Also `box.json` (`SC1_BOX`, host facts), `quiesce_<tag>.json`, `forensics.txt`, `versions.txt`,
`manifests/experts.json` (the pack manifest), and `PACK <fp>` / `PHASE` / `QUIESCE` / `FETCH` / `ENERGY` / `K8CENSUS` lines in `summary.txt`.
The reducer is called as `sc1_reduce.py $W --box <A|B|C> --out-dir $W` when staged (its file is pinned at integration: `SC1_PIN_REDUCER=1 make_pin.sh`).

## Mappings and assumptions a reader must know (see the orchestration report for the full list)

- **The detokenisation pair.** The matched vLLM arms keep `detokenize=True` (vLLM's shipped default; a comparator's loop is never trimmed to
  e4b's omission) and `vllm_gptq_graph_b{1,16}_nodetok.json` (`SC1_ARM=nodetok`, driver commit `8cb77d1`) measures the detokeniser's share at
  both batch sizes -- v3's Phase G2 pair, the first thing the deadline may drop.
- **bf16 Qwen3 is fetched on every box**, box C included: `k8_bake.py` quantises the bf16 checkpoint into the NF4 arena the e4b anchors load.
  v3's "no bf16" on box C refers to the oracle (box B's), not the fetch.
- **The premise test runs after the pack build** (v3's Phase 0 order), where P88 ran it before any fetch.
- **The K8 `=1` rows' census.** step_decomp's eager K8 loop has no kernel census; the runner censuses the SAME stack (LICENV + pack + `=1`,
  `--no-fuse-qkv`) at T == 1 in a B=1 graph window (`e4b_k19census_b1`) and places the table under `logs/census_k8_lic_1_<src>.txt`, the names the
  reducer reads for LICENSED-B16; the `K8CENSUS` line says so.
- **The fp8-KV served quality rows** (`nll_vllm_fp8kv_served_<src>`, P8's quality clause; v3 Environments says the hazard is SCORED) are not in v3's
  phase list and run LAST on box A, after every registered droppable.
- **Energy windows.** e4b-sched decodes 1024 tokens with its own epoch window; vLLM runs `SC1_SHORT=32 SC1_LONG=1024 SC1_REPS=1` and the LONG
  rep's `timed_windows_epoch` is the window; the box-B comparator drivers expose neither a 1024-token mode nor an epoch window, so their rows are
  whole-arm windows under the sampler (server brought up outside the window for llama.cpp), labelled in `energy_<tag>.json`.
- **The sched buckets** are the registered list truncated at B (`1` at B=1, `1,2,4,8,16` at B=16), as vLLM derives its capture list from
  `max_num_seqs`; the receipt records them.
- **The e4b venv uses `--system-site-packages`**: a plain venv sets `site.ENABLE_USER_SITE=False` and the P42 hook (`usercustomize` on
  `PYTHONPATH`) silently never loads; the tripwire asserts both.
- **Box C's proof** installs SGLang and runs its tripwire; the Marlin MoE JIT + `/health` against `SC1_PROVE_SGLANG_MODEL=<repo@rev>`
  (a small GPTQ MoE checkpoint; the proof fetches no Qwen3) is REQUIRED. Every proof writes `PROVED` only when each install its box
  needs succeeded and each smoke (and box C's JIT) RAN and passed; a skipped smoke is rc 23, NOT PROVED (Amendment A2).
