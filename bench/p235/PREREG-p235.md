# P235 — P58's same-box vLLM head-to-head, re-read at the release pins (e4b 0.58.0, grouped-nf4-gemm 0.51.0)

Registered 2026-10-24, before any box. One row of the head-to-head at the released pins: e4b serving against vLLM,
read with an existing reviewed harness, P58's ([`bench/p58/P58-PREREG.md`](../p58/P58-PREREG.md),
[`RESULTS-p58.md`](../p58/RESULTS-p58.md)). Registered and descriptive. The only bars are the harness's own, and the
reading is quoted **beside** P58's, never in place of it.

## The harness: P58's, by named edits

- `p235_run.sh` is `bench/p58/p58_run.sh` with the names changed from P58 to P235 (work directory `/root/p235`, the
  `P235_*` environment and marker names) and with the pin changes below. P58's 18 arms, their order, the per-arm
  alarms, STOP-2 (an arm that cannot finish 10 min before the deadline is skipped and recorded), the engagement greps
  and the reducer call are unchanged.
- `p235_reduce.py` is `bench/p58/p58_reduce.py` with its labels renamed (title, docstring, usage). The verdict code is
  byte-identical.
- P37's vLLM driver (`p37_vllm.py`), P39's `step_decomp.py`, `k8_bake.py` and `calib.json`, and P42's hook are staged
  at P58's digests (`staged.sha256`; `tests/test_p235.py` checks them against P58's pins).
- `p235_drive.sh` follows the pod-launch drive form (P233's): lane liveness by launch identity, the clean checkout HEAD
  as the launch commit with an `E4B_SHA` override refused, the `--no-links` fetch into an empty `$RUN_DIR/p235/`, and
  explicit run ids.
- At $0, before registration, every e4b call the staged harness makes was bound against e4b main's signatures
  (`47bf1ed2`), and the tripwire's API asserts hold at grouped-nf4-gemm main. Nothing in the harness needed an edit
  to run at these pins.

## Pins (what changes from P58)

| | P58 (2026-09-22) | P235 |
|---|---|---|
| e4b | 0.36.4 @ `4cab05a` | **0.58.0**: the launch commit. The tripwire asserts `experts4bit_qlora.__version__ == "0.58.0"` and refuses (rc 9) |
| grouped-nf4-gemm | 0.32.1 @ `65cb104` | **0.51.0** @ `GNF4_SHA`. The tripwire asserts `md.version("grouped-nf4-gemm") == "0.51.0"` (rc 9) |
| transformers | 5.16.1 | **5.17.0** |
| bitsandbytes | 0.50.1 | 0.50.1 (e4b's floor is `>=0.43`) |
| vLLM PRIMARY | latest PyPI at launch (0.30.0) | latest PyPI at launch (**0.32.0** today) |
| vLLM SECONDARY | 0.29.0 | **0.30.0** (P58's primary) |
| GPU UUID | collected | **not collected**: the forensics query drops `uuid`, and a test refuses `uuid`, `-L` or `-q` |

**The one placeholder: `GNF4_SHA`.** It is the commit of grouped-nf4-gemm's 0.51.0 release, which does not exist
yet. Amendment 1 names it after the tag and before any data. The drive passes it to the runner (P58's form) and refuses
any value that is not a 40-char hex sha.

**Also unchanged from P58:**
- the model, `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`;
- vLLM's checkpoint, `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441`;
- the prompts, from `step_decomp._k8_window`. P58's digests were `a8e6ea1d…` (B=1) and `f67e7e4d…` (B=16). A
  different digest here is reported beside the reading; it does not void it, since both engines decode the same file;
- e4b's two stacks (the NF4 control, and the int4 stack with fused q/k/v at B=1 and unfused at B=16);
- the image, `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`.

vLLM 0.32.0 pins the same torch (2.13.0) as 0.30.0, which P58 installed and ran on this image. Its Python floor is
3.11. A vLLM install or import failure leaves the e4b arms running and reads as that build's `PIP FAIL` or
`vllm_init_failed` row (P58's rule).

**The release pins bind.** The tripwire refuses any e4b whose version is not 0.58.0, so a launch from a commit past
the next version bump refuses before the first arm.

## Bars (the harness's own; registered and descriptive)

1. **Self-pairs:** `nf4_r1/r2`, `int4_r1/r2` and each build's `graph_r1/r2` must be inside **1.03×**. Outside that,
   the row reads **DRIFT**, both readings are printed, and no ratio is quoted.
2. **The ratio** is `vLLM graph_r1 / e4b int4_r1` (decode tok/s), per build and batch size. It comes only from the
   PRIMARY pair, only with both self-pairs inside 1.03×, and only with an equal `prompts_sha` on both sides.
3. **VOID arms never enter a ratio** (P58's verdict rule 1, applied by the reducer).

**Quoted beside P58's reading, never instead of it:** vLLM 0.30.0 / e4b int4 = **1.087** at B=1 and **1.396** at
B=16 (vLLM ahead). This lane licenses nothing and changes no default. FAIL stays FAIL, and DRIFT stays DRIFT.

## Runs and budget

- **Box:** one RTX 5090 at ≤ $1.05/h, guard 3.0 h, download ≤ $0.02/GB × 80 GB.
- **Worst case:** $3.15 + $1.60 = **$4.75**, inside a **$6** lane ceiling. P58 ran the same arms in 42 minutes.
- **Run ids:** `p235-5090-1..-3` are the readings, and `p235-5090-r1..-r6` are for a box that did not start. The
  first run that completes is the reading. A run that ended before its arms (install, a dud box, a host fault) is
  followed by the next id. A completed run is never re-run, whatever it reads (P58 STOP-4).

**No shakedown is registered.** P58's runner ran all 18 arms on a 5090 in this exact shape. The changes here are
version pins and names, each checked at $0 (the API binding above, the tripwire asserts, vLLM's torch pin). A failure
at install or the tripwire refuses (rc 9) before the first arm and costs a few minutes of one box, about what a
shakedown would cost.

Amendments, dated, go below this line before any data is read.
