# P235 — P58's same-box vLLM head-to-head, re-read at the release pins (e4b 0.58.0, grouped-nf4-gemm 0.51.0)

Registered 2026-10-24, before any box. One row of the head-to-head at the released pins: e4b serving against vLLM,
read with an existing reviewed harness, P58's ([`bench/p58/P58-PREREG.md`](../p58/P58-PREREG.md),
[`RESULTS-p58.md`](../p58/RESULTS-p58.md)). Registered and descriptive. The only bars are the harness's own, and the
reading is quoted **beside** P58's, never in place of it.

## The harness: P58's, by named edits

- `p235_run.sh` is `bench/p58/p58_run.sh` with the names changed from P58 to P235 (work directory `/root/p235`, the
  `P235_*` environment and marker names) and with the pin and provenance changes below. P58's 18 arms, their order,
  the per-arm alarms, STOP-2 (an arm that cannot finish 10 min before the deadline is skipped and recorded), the
  engagement greps and the reducer call are unchanged.
- `p235_reduce.py` is `bench/p58/p58_reduce.py` with its labels renamed (title, docstring, usage). The verdict code is
  byte-identical.
- `provenance.py` (new, stdlib only) holds the registered pins as constants and checks each installed distribution's
  version and provenance.
- P37's vLLM driver (`p37_vllm.py`), P39's `step_decomp.py`, `k8_bake.py` and `calib.json`, and P42's hook are staged
  at P58's digests (`staged.sha256`; `tests/test_p235.py` checks them against P58's pins).
- `p235_drive.sh` follows the pod-launch drive form (P233's): lane liveness by launch identity, the clean checkout HEAD
  as the launch commit with an `E4B_SHA` override refused, the `--no-links` fetch into an empty `$RUN_DIR/p235/`, and
  explicit run ids. The launch commit is the harness's (the staged pieces), recorded and never installed.
- At $0, before registration, every e4b call the staged harness makes was bound against e4b main's signatures
  (`47bf1ed2`), and the tripwire's API asserts hold at grouped-nf4-gemm main. Nothing in the harness needed an edit
  to run at these pins.

## Pins and provenance (what changes from P58)

A version string is not a release. Every distribution the lane reads is installed from PyPI as a wheel at a
registered version. Before any arm, `provenance.py` records each one's version AND provenance into the receipts: a
PyPI index install has no `direct_url.json`, or one whose url is an https PyPI archive with no `vcs_info` or
`dir_info`. Editable, `file://`, local-directory and other-host installs are refused. Every git install is refused,
whatever its URL or commit: a fork, a mirror or a local repo can carry the release's tag, so no tag is resolved
through the URL an install recorded. The pins are constants in `provenance.py` and in the runner, checked
unconditionally, never behind a flag.

**The release proof: the installed bytes.** Provenance says where an install came from; it does not prove the
bytes. `release_proof.py` is loggetta `bench/release_proof.py` at `f09e5676` (#126's merge),
sha256 `aff3fc80db8d35bbf94a9aaeed125c9d424cfe93bf16eb0e6868b6f9d03f13e0`, staged byte-identical and pinned in
`staged.sha256`. For each pin it takes the one published PyPI wheel whose filename carries the installed `WHEEL` Tag
lines (`wheel_pick` "filename"). When none does, it takes the one wheel with the install's python and abi tags whose
platform is `linux_<arch>` or `manylinux*`/`musllinux*_<arch>` (`wheel_pick` "arch"); zero or more than one
refuses. It downloads it and checks its digest against PyPI's, then requires every file in the wheel's own
`RECORD` to be on disk with an equal hash, and nothing installed beyond bytecode and entry-point launchers. It also
checks, through `find_spec` in the calling interpreter, that every top-level name the wheel installs imports from
the proven install, so nothing earlier on `sys.path` can shadow it. A wheel with `.data/` entries is refused.

Each proof is the last step before its build's first arm, after every install. It runs in that build's import
environment: cwd `$W`, `PYTHONPATH=$W/hook`, the arms' python, no `-I`. The receipts carry `release_proof.jsonl` and
`release_proof_vllm_<build>.jsonl`. Each line, one per distribution, holds the wheel filename, its sha256, "RECORD
equal", the import origins and `pycache_fresh`. A proof file whose lines are not all `"pycache_fresh": true`, or not
one per pin, refuses like the proof itself.

**Bytecode.** Every python the runner starts uses `PYTHONPYCACHEPREFIX=$W/pycache-<nonce>`, created empty per run and
recorded in `versions.txt`, so no in-tree `.pyc` is loaded. `RELEASE_PROOF_FRESH_PYCACHE` is exported on the same line
with the same value, which is what lets the proof report `pycache_fresh`. Every install uses `--no-compile`. With the prefix set,
pip writes its install-time bytecode outside site-packages and lists it in `RECORD` as `../` rows, which the proof
refuses. Checked at $0 on e4b 0.57.0 and grouped-nf4-gemm 0.50.0: refused with pip's default compile, "RECORD equal"
with `--no-compile`.

| | P58 (2026-09-22) | P235 |
|---|---|---|
| e4b | 0.36.4, git @ `4cab05a` | **`experts4bit-qlora==0.58.0`**, PyPI wheel; provenance checked, and the tripwire asserts `experts4bit_qlora.__version__ == "0.58.0"` |
| grouped-nf4-gemm | 0.32.1, git @ `65cb104` | **`grouped-nf4-gemm==0.51.0`**, PyPI wheel (P58's `--force-reinstall --no-deps` form); provenance checked, and the tripwire asserts `md.version("grouped-nf4-gemm") == "0.51.0"` |
| transformers | 5.16.1 | **5.17.0** |
| bitsandbytes | 0.50.1 | 0.50.1 (e4b's floor is `>=0.43`) |
| vLLM PRIMARY | latest PyPI at launch (0.30.0) | **0.32.0**, a constant, PyPI wheel, provenance checked. A newer vLLM is a new reading |
| vLLM SECONDARY | 0.29.0 | **0.30.0** (P58's primary), a constant, PyPI wheel, provenance checked |
| GPU UUID | collected | **not collected**: the forensics query drops `uuid`, and a test refuses `uuid`, `-L` or `-q` |

**What a provenance or release-proof failure does:**
- **e4b or grouped-nf4-gemm, provenance** (wrong version, editable, `file://`, another host, any git commit): the run
  ends with rc 9 before the tripwire and before any arm.
- **e4b or grouped-nf4-gemm, release proof:** the run ends with rc 9 before the first arm. This comes after the
  downloads and the bake, so a refusal here costs the setup time.
- **A vLLM build** (checked in that build's venv, on `vllm` only): that build's rows are refused (they read `SKIPPED`,
  no comparator), as P58's `PIP FAIL` rule does. The e4b arms and the other build continue.

**Also unchanged from P58:**
- the model, `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`;
- vLLM's checkpoint, `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441`;
- the prompts, from `step_decomp._k8_window`. P58's digests were `a8e6ea1d…` (B=1) and `f67e7e4d…` (B=16). A
  different digest here is reported beside the reading; it does not void it, since both engines decode the same file;
- e4b's two stacks (the NF4 control, and the int4 stack with fused q/k/v at B=1 and unfused at B=16);
- the image, `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`.

**Risks:**
- **vLLM's wheels are proven by arch.** The published vLLM 0.32.0 and 0.30.0 x86_64 wheels are named
  `manylinux_2_28_x86_64`, but their own `WHEEL` file says `Tag: cp38-abi3-linux_x86_64`, and pip copies that file
  into the install. No filename carries the installed tags, so the proof picks by arch (`wheel_pick` "arch"). It then
  proves the bytes by the same RECORD compare; `WHEEL` is itself a compared RECORD entry. Neither wheel has `.data/`
  entries. Checked at $0: vLLM 0.32.0 and 0.30.0 installed from PyPI `--no-deps --no-compile` and proved as the runner
  runs it, each rc 0, "RECORD equal", `wheel_pick` "arch" and `pycache_fresh` true (5,443 and 5,097 files). If the
  helper refuses a build anyway, that build's rows are refused, as for any vLLM failure.
- vLLM 0.32.0 pins the same torch (2.13.0) as 0.30.0, which P58 installed and ran on this image. Its Python floor
  is 3.11.
- `p37_vllm.py` has run against vLLM 0.28 to 0.30, not 0.32.0. An API break there reads as that build's failed rows
  (`vllm_init_failed`), never as a void of the e4b arms.
- The launch commit must carry the staged pieces at the pins in `staged.sha256`. The installed e4b is PyPI's 0.58.0,
  whatever the checkout's version.

## Bars (the harness's own; registered and descriptive)

1. **Self-pairs:** `nf4_r1/r2`, `int4_r1/r2` and each build's `graph_r1/r2` must be inside **1.03×**. Outside that,
   the row reads **DRIFT**, both readings are printed, and no ratio is quoted.
2. **The ratio** is `vLLM graph_r1 / e4b int4_r1` (decode tok/s), per build and batch size. It comes only from the
   PRIMARY pair, only with both self-pairs inside 1.03×, and only with an equal `prompts_sha` on both sides.
3. **VOID arms never enter a ratio** (P58's verdict rule 1, applied by the reducer).

**Quoted beside P58's reading, never instead of it:** vLLM 0.30.0 / e4b int4 = **1.087** at B=1 and **1.396** at
B=16 (vLLM ahead). This lane licenses nothing and changes no default. FAIL stays FAIL, and DRIFT stays DRIFT.

## What this lane cannot say

P58's not-measured list carries over:
- TTFT and prefill (informational only, with no ratio);
- other batch sizes or context lengths;
- memory footprint (vLLM reserves 0.90 of VRAM by policy);
- any other checkpoint;
- `GNF4_GEMV_FUSED_REDUCE=1`;
- an fp8-KV vLLM arm.

Two limits of the instrument:
- **The e4b window is not the shipped server's path.** `step_decomp`'s graph-replay window is not the server's
  scheduler.
- **The engine advantage is understated, not inflated.** vLLM's slope includes its scheduler and detokenisation,
  while e4b's window has neither.

## Sequencing

1. This registration is reviewed independently.
2. It merges after the tags exist AND both releases (e4b 0.58.0, grouped-nf4-gemm 0.51.0) are published on PyPI.
3. A $0 validation follows.
4. The maintainer gives an Approved.
5. The box runs.
6. The maintainer re-derives the reading from the receipts.
7. A results PR follows.

## Runs and budget

- **Box:** one RTX 5090 at ≤ $1.05/h, guard 3.0 h, download ≤ $0.02/GB × 80 GB.
- **Worst case:** $3.15 + $1.60 = **$4.75**, inside a **$6** lane ceiling. P58 ran the same arms in 42 minutes.
- **Run ids:** `p235-5090-1..-3` are the readings, and `p235-5090-r1..-r6` are for a box that did not start. The
  first run that completes is the reading. A run that ended before its arms (install, provenance, a dud box, a host
  fault) is followed by the next id. A completed run is never re-run, whatever it reads (P58 STOP-4).

**No shakedown is registered.** P58's runner ran all 18 arms on a 5090 in this exact shape. The changes here are
version pins, PyPI installs with a provenance check, and names, each checked at $0 (the API binding above, the
tripwire asserts, the provenance checker's tests, vLLM's torch pin). A failure at install, provenance or the tripwire
refuses (rc 9) before the first arm and costs a few minutes of one box, about what a shakedown would cost.

Amendments, dated, go below this line before any data is read.
