#!/bin/bash
# bench/tc1/tc1_run.sh -- lane TC1, BOX side (bench/tc1/TC1-PREREG.md). Started detached by tc1_drive.sh with the run's
# nonce; writes TC1_RUN_NONCE first, then TC1_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and TC1_SUCCESS.<nonce>
# only when the plan completed (every arm reached a row and the reducer ran).
#
# PROVENANCE: copied from bench/tp4/tc1_run.sh @ e4b main 10ce711d (the tp4 tree is never edited). Kept: the nonce handshake,
# the box refusals (GPU class, overlay disk), the deadline-derived alarms, the venv installs + tripwires, the pinned Alpaca
# fixture, fetch (unpinned + the proof main == pin, e4b#404), tokenise, the `arm` wrapper (one process, one JSON, one alarm,
# the arm told its own alarm for the #548 watchdog), stubw, free_family. Removed: tp4's other boxes and plans (the anchor pair,
# the P43 diagnosis, the P45 profile, the P46 LoRA-path and the P56/P67 opt-ins), the clinical dataset -- lane TC1b brings the
# anchor pair and the clinical dataset back under its own `qwen3curve` token (tc1_curve_family). Added: `draw2`
# (the second draw of an arm: the same invocation, a fresh process, tag <tag>_d2), `todo_arm` (a not_run row for an arm this
# cut does not implement), and `tc1_family`, which runs TC1-PREREG's arm set IN ITS REGISTERED ORDER with the matched flags
# (`--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED`) on the matched set and the native flags on the native rows.
#
#   TC1_BOX=A  qwen3 (Qwen3-30B-A3B @ the pin) -- one RTX 5090, the only registered box of TC1
#   TC1_BOX=B  tc2big (lane TC2's box B: qwen3_5 + mixtral); lane TC2's box A token is TC1_FAMILIES=tc2small (granite, olmoe, gptoss) on TC1_BOX=A
#   TC1_FAMILIES=tc2small | tc2big  lane TC2 (TC2-PREREG-draft, registered by the PI as bench/tc1/TC2-PREREG.md; the box defaults PREREG to tc1/TC2-PREREG.md
#              for these tokens): the other MoE families at matched work on the current cuts -- tc2_small_family (N 60, eval 48 rows every 20 steps,
#              as tp4 registered for the small families; granite's second Unsloth target list; gptoss MODE with the REFUSED stubs, attn_only_m x2,
#              the 16-bit Unsloth load that keeps the MXFP4 experts packed) and tc2_big_family (N 20, 8 rows at 0 and N; qwen3_5's explicit expert
#              target_parameters; mixtral's e4b arms under expert offload, Unsloth resident; the e4b reference LAST; the _mb1 pair on an OOM)
#   TC1_FAMILIES=qwen3curve  lane TC1b (bench/tc1/TC1B-PREREG.md, the PI's; drafted in TC1b-PREREG-draft): the same pin on one RTX 5090 --
#              the matched pair at N 200 (eval every 40 on 16 held-out rows), the as-shipped e4b curve beside it, the tp2/P38 anchor pair
#              (tp4_run.sh's `anchor_pair` fixture and clinical text, byte-for-byte, plus the t28 variant that IS tp4's arm), the
#              tokens-per-step pair (`_t1`: micro-batch 1 x accum 1) and the rank pair (`_r64`); `tc1_curve_family` below
#   TC1_FAMILIES=qwen3axolotl  TC1-PREREG amendment 3 (2026-10-02): the axolotl rows re-asked on their own box after both TC1 boxes' axolotl venv
#              failed to install (uv's first-index strategy stopped at PyTorch's cu130 index): e4b fused_m x2, axolotl matched x2, axolotl native-best
#              (scattermoe) and the HF torch-2.14 grouped_mm mb1 row, each against this box's own e4b fused_m; no Unsloth venv is built; `tc1_axolotl_family` below
#
# Frameworks: e4b (GitHub main @ E4B_SHA + grouped-nf4-gemm @ GNF4_SHA, venv-e4b: transformers 5.18.0 / bitsandbytes 0.50.2 /
# peft 0.21.2), plain HF+PEFT+bnb (venv-e4b), Unsloth at the REGISTERED versions (unsloth 2026.9.14 + unsloth_zoo 2026.9.9) in
# TWO venvs -- venv-unsloth-t28 = tp4's install on the image's torch 2.8.0+cu128, venv-unsloth = unsloth[cu130-torch2121]
# (torch 2.12.1+cu130, the route its installer names for Blackwell; UPSTREAM-NOTES "Unsloth") -- and axolotl 0.20.0 in
# venv-axolotl (uv, CPython 3.12, torch 2.14.1+cu130: torch >= 2.13 Linux wheels exist only under cu130). Every cu130 venv
# is gated on the host driver >= 580 (checked BEFORE any install); below it the arms that need one are `refused` rows that
# name the driver -- never a silent fallback to the t28 venv.
set -uo pipefail
LANE=tc1
# TC3 (lane TC3, the owned 12 GB box; TC3-PREREG-draft "12 GB box", README "TC3 hand run"): TC1_LOCAL_BOX=1 is a HAND RUN without the launcher -- no nonce,
# no instance id and no deadline from rent.py (deadline = now + 6 h), the checkpoint from TC1_LOCAL_SNAPSHOT (a directory at the registered revision,
# never fetched; `local_snapshot` below links it into a private HF cache and proves the pin against the Hub's config.json when the network answers, else
# records `pin_proof: offline`), venvs / tokens / receipts under TC1_LOCAL_OUT, the venvs made from TC1_LOCAL_PYTHON. Everything else -- the tokens, the arm
# wrapper, the alarms, the reducer -- is the registered path. tc1_drive.sh never forwards TC1_LOCAL_* (tests/test_tc1_arm.py asserts it).
TC1_LOCAL_BOX=${TC1_LOCAL_BOX:-0}
if [ "$TC1_LOCAL_BOX" = 1 ]; then
  [ -n "${TC1_LOCAL_OUT:-}" ] || { echo "refusing: TC1_LOCAL_BOX=1 needs TC1_LOCAL_OUT (where the venvs, tokens and receipts go)"; exit 78; }
  W=$TC1_LOCAL_OUT
  TC1_BOX=${TC1_BOX:-A}; TC1_RUN_ID=${TC1_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}; TC1_INSTANCE_ID=${TC1_INSTANCE_ID:-local:$(hostname)}
  TC1_DEADLINE_EPOCH=${TC1_DEADLINE_EPOCH:-$(( $(date +%s) + ${TC1_LOCAL_HOURS:-12} * 3600 ))}
  TC1_RUN_NONCE=${TC1_RUN_NONCE:-local-$(date +%s)-$$}
  export HF_HUB_CACHE=$W/hf-cache          # the local snapshot is linked in here, so every loader resolves the pin offline exactly as after a fetch
else
  W=/root/$LANE
fi
export TC1_W=$W                            # the venv tripwires write versions.txt beside the receipts, wherever they are
PY_BASE=${TC1_LOCAL_PYTHON:-python}        # the interpreter the venvs are made from (the image's `python` on a rental)
mkdir -p $W/logs $W/adapters $W/data; cd $W || exit 9
say(){ echo "[$(date -u +%FT%TZ)] tc1/box${TC1_BOX:-?}: $*"; }
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/TC1_RUN_NONCE.tmp && mv $W/TC1_RUN_NONCE.tmp $W/TC1_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > TC1_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > TC1_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in TC1_BOX TC1_RUN_ID TC1_DEADLINE_EPOCH TC1_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$TC1_BOX" in A|B) ;; *) say "refusing: TC1_BOX must be A (TC1's one RTX 5090; TC1-PREREG 'Budget and stop rules') or B (lane TC2's box B, tc2big)"; finish 78;; esac
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# The pre-registration written into EVERY receipt and stub. Overridable because a draw that adds arms is governed by the
# document that REGISTERED those arms and authorised its spend -- and tc1_arm.py refuses a run without --prereg precisely so
# a receipt can never cite a pre-registration the run did not pass.
PREREG=${TC1_PREREG:-tc1/TC1-PREREG.md}
export TC1_INSTANCE_ID
# ---------------------------------------------------------------- the registered fixture (TC1-PREREG "Fixture" = tp4's field recipe): N 20, eval at 0 and N on 8 held-out rows
STEPS=${TC1_STEPS:-20}; SEQ=${TC1_SEQ:-2048}; MB=${TC1_MB:-2}; ACCUM=${TC1_ACCUM:-4}; R=${TC1_R:-16}; ALPHA=${TC1_ALPHA:-16}
LR=${TC1_LR:-2e-4}; WD=${TC1_WD:-0.001}; WARMUP=${TC1_WARMUP:-5}; SCHED=${TC1_SCHED:-linear}; OPTIM=${TC1_OPTIM:-adamw_8bit}; SEED=${TC1_SEED:-3407}
EVAL_EVERY=${TC1_EVAL_EVERY:-20}; EVAL_N=${TC1_EVAL_N:-8}; AUTOCAST=${TC1_AUTOCAST:-0}; TEMPLATE=alpaca
MATCHED_SEED=${TC1_MATCHED_SEED:-3407}            # --lora-init matched:<seed> on every matched arm (TC1-PREREG "Arms"); its own knob, distinct from the fixture seed
# TC1 amendment 39: TC1_PACK=1 packs the Alpaca text into rows of exactly SEQ tokens (tc1_arm.py --prepare --pack: the registered text extended
# in its own order, EOS between examples, no padding, full causal attention across example boundaries). A box-level fixture: only the packed-regime
# tokens run with it, and they run only with it (the refusal below the BOX line). 0 (default) = the tokens file byte-identical to before.
PACK=${TC1_PACK:-0}
DS_ALPACA_SHA=${TC1_DS_ALPACA_SHA:-5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb}   # tp4_alpaca.py output, registered
TF_VER=${TC1_TRANSFORMERS_VER:-5.18.0}; BNB_VER=${TC1_BNB_VER:-0.50.2}; PEFT_VER=${TC1_PEFT_VER:-0.21.2}   # TC1-PREREG "Environments"
# ---------------------------------------------------------------- TC1b (the qwen3curve token; TC1B-PREREG "Fixture" / "Arms"): the curve instrument over the SAME field recipe
# (N 200, eval every 40 on the first 16 held-out rows, the linear schedule's decay running to step 200 with the 5 warm-up steps --
# tc1_arm.py's _lam(step, N=a.steps) is transformers' formula), the `_t1` sub-fixture (micro-batch 1 x accum 1, N 20, 8 rows at 0 and N)
# and the `_r64` sub-fixture (r 64 / alpha 64, everything else the field recipe). Every knob here is forwarded by tc1_drive.sh.
CURVE_STEPS=${TC1_CURVE_STEPS:-200}; CURVE_EVAL_EVERY=${TC1_CURVE_EVAL_EVERY:-40}; CURVE_EVAL_N=${TC1_CURVE_EVAL_N:-16}
T1_MB=${TC1_T1_MB:-1}; T1_ACCUM=${TC1_T1_ACCUM:-1}; R64_R=${TC1_R64_R:-64}; R64_ALPHA=${TC1_R64_ALPHA:-64}
# ---------------------------------------------------------------- TC2 (the tc2small token; TC2-PREREG-draft "Fixture"): the small families' instrument -- N 60, eval 48 rows every
# 20 steps, as tp4 registered for them (TP4-PREREG; amendment 2 cut only the LARGE families to N 20 / 8 rows). The tc2big token runs the field recipe above
# (N 20, 8 rows at 0 and N: tp4 amendment 2's instrument). Every knob here is forwarded by tc1_drive.sh.
SMALL_STEPS=${TC1_SMALL_STEPS:-60}; SMALL_EVAL_N=${TC1_SMALL_EVAL_N:-48}; SMALL_EVAL_EVERY=${TC1_SMALL_EVAL_EVERY:-20}
# the anchor pair's fixture = tp2/P38 as tp4 RAN it (bench/tp4/tp4_run.sh A_*, TP4-PREREG "Anchor"): clinical text, seq 512, batch 1 x accum 1,
# r 8 / alpha 16, lr 1e-4, torch AdamW wd 0.01, constant, seed 0, N 60. Literals, as tp4's: the pair is byte-for-byte tp4's arms, not a knob.
A_STEPS=60; A_SEQ=512; A_MB=1; A_ACCUM=1; A_R=8; A_ALPHA=16; A_LR=1e-4; A_WD=0.01; A_WARMUP=0; A_SCHED=constant; A_OPTIM=adamw_torch; A_SEED=0; A_TEMPLATE=clinical
# 8 held-out rows at 0 and N: what tp4's anchor RAN (TP4-PREREG amendment 3's erratum: 8 rows, the first 8 of tp2's 48, a prefix) and what
# TC1b registers; bench/tp4/tp4_run.sh now carries A_EVAL_N=48 as the post-erratum fix (e4b#545), so this is NOT tp4_run.sh's current literal.
A_EVAL_N=8; A_EVAL_EVERY=20
SKIP=${TC1_SKIP:-}; PIN_FALLBACK=${TC1_PIN_FALLBACK:-0}; GPU_CLASS=${TC1_GPU_CLASS:-5090}
case "$TC1_BOX" in
  A) FAMILIES=${TC1_FAMILIES:-"qwen3"};;          # the judged family; TC1_FAMILIES=qwen3native for the labelled / native-best box (phase 3 I); TC1_FAMILIES=tc2small for lane TC2's box A
  B) FAMILIES=${TC1_FAMILIES:-"tc2big"};;         # lane TC2's box B (qwen3_5 + mixtral); TC1_FAMILIES overrides as on A
esac
# TC1-PREREG amendment 3 (2026-10-02): the qwen3axolotl token runs no Unsloth arm, so it builds neither Unsloth venv (~15 min of box time
# on the TC1 boxes); every other token builds both as before.
NEED_UNSLOTH=1; case " $FAMILIES " in " qwen3axolotl "|" qwen3nativebest200 "|" qwen3syncab "|" qwen3prof945 "|" qwen3leanab "|" qwen3tileab "|" qwen3rmsab "|" qwen3reuseab "|" qwen3keepab "|" routebench "|" fusedsweep ") NEED_UNSLOTH=0;; esac
# TC1 amendment 22: the dense-route A/B tokens run no Unsloth arm either -- alone, or both on their one box (in either order)
case " $FAMILIES " in " qwen3denseab "|" mixtraldenseab "|" qwen3denseab mixtraldenseab "|" mixtraldenseab qwen3denseab ") NEED_UNSLOTH=0;; esac
case " $FAMILIES " in " qwen3prebindab ") NEED_UNSLOTH=0;; esac   # TC1 amendment 26: an e4b-only A/B
case " $FAMILIES " in " qwen3dqab "|" mixtraldqab ") NEED_UNSLOTH=0;; esac   # TC1 amendment 28: e4b-only A/Bs
case " $FAMILIES " in " qwen3tritonab ") NEED_UNSLOTH=0;; esac   # TC1 amendment 32: an e4b-only A/B
case " $FAMILIES " in " olmoedecab "|" qwen3decab "|" olmoedecab qwen3decab "|" qwen3decab olmoedecab ") NEED_UNSLOTH=0;; esac   # TC1 amendment 46: e4b-only A/Bs in venv-e4b
: > summary.txt; echo "$TC1_INSTANCE_ID" > INSTANCE_ID
echo "FIXTURE field: template=$TEMPLATE steps=$STEPS seq=$SEQ micro_batch=$MB accum=$ACCUM r=$R alpha=$ALPHA lr=$LR wd=$WD warmup=$WARMUP sched=$SCHED optim=$OPTIM seed=$SEED eval_every=$EVAL_EVERY eval_n=$EVAL_N autocast=$AUTOCAST matched_seed=$MATCHED_SEED pack=$PACK" | tee -a summary.txt
case " $FAMILIES " in *" qwen3curve "*)
  echo "FIXTURE curve (TC1b): steps=$CURVE_STEPS eval_every=$CURVE_EVAL_EVERY eval_n=$CURVE_EVAL_N; t1: micro_batch=$T1_MB accum=$T1_ACCUM; r64: r=$R64_R alpha=$R64_ALPHA" | tee -a summary.txt
  echo "FIXTURE anchor (tp4's A_*): template=$A_TEMPLATE steps=$A_STEPS seq=$A_SEQ micro_batch=$A_MB accum=$A_ACCUM r=$A_R alpha=$A_ALPHA lr=$A_LR wd=$A_WD warmup=$A_WARMUP sched=$A_SCHED optim=$A_OPTIM seed=$A_SEED eval_every=$A_EVAL_EVERY eval_n=$A_EVAL_N" | tee -a summary.txt
  [ -n "${TC1_PREREG:-}" ] || PREREG=tc1/TC1B-PREREG.md     # TC1b: the curve token is governed by its own registration (the PI's); TC1_PREREG still overrides
  ;;
esac
case " $FAMILIES " in *" tc2small "*|*" tc2big "*|*" tc2mixtral "*|*" tc2qwen35off "*|*" tc2resident "*|*" tc2mixtralres "*|*" tc2qwen35mb1 "*)
  echo "FIXTURE tc2 (TC2-PREREG-draft): small (box A: granite olmoe gptoss): steps=$SMALL_STEPS eval_every=$SMALL_EVAL_EVERY eval_n=$SMALL_EVAL_N; big (box B: qwen3_5 mixtral): the field recipe (steps=$STEPS eval_every=$EVAL_EVERY eval_n=$EVAL_N)" | tee -a summary.txt
  [ -n "${TC1_PREREG:-}" ] || PREREG=tc1/TC2-PREREG.md      # TC2: governed by its own registration (the PI's, bench/tc1/TC2-PREREG.md); TC1_PREREG still overrides
  ;;
esac
case " $FAMILIES " in *" qwen3frontier "*|*" qwen3frontier12 "*)      # TC3: the memory-frontier tokens are governed by their own registration (the PI's); TC1_PREREG still overrides
  echo "FIXTURE frontier (TC3): the field recipe, matched init, fp32 adapters on a memory-frontier box (class $GPU_CLASS); levers: e4b --offload 1, hf --hf-offload 1 (device_map auto + max_memory), axolotl --axolotl-layer-offload 1 / --axolotl-zero3 1; local_box=$TC1_LOCAL_BOX snapshot=${TC1_LOCAL_SNAPSHOT:-<fetched>}" | tee -a summary.txt
  [ -n "${TC1_PREREG:-}" ] || PREREG=tc1/TC3-PREREG.md
  ;;
esac
echo "BOX $TC1_BOX families: $FAMILIES; e4b $E4B_SHA gnf4 $GNF4_SHA; run $TC1_RUN_ID instance $TC1_INSTANCE_ID deadline $TC1_DEADLINE_EPOCH; prereg $PREREG" | tee -a summary.txt
# TC1 amendment 39: packing is the box's fixture, so every token on the box agrees with it -- refused here, before anything is fetched or installed:
# a packed-regime token without TC1_PACK=1 / TC1_SEQ=4096 would run the field recipe under the packed token's name, and TC1_PACK=1 beside a
# field-recipe token would pack that token's rows under ITS name.
case "$PACK" in 0|1) ;; *) say "refusing: TC1_PACK must be 0 or 1 (got '$PACK')"; echo "BOX_REFUSED pack=$PACK" >> summary.txt; finish 78;; esac
for _f in $FAMILIES; do
  case "$_f" in
    qwen3samestack4k|qwen3samestack4kce|qwen3samestack4kce2|qwen3memc4k|qwen3padbk|qwen3samestack4kd|qwen3padbk28|qwen3prof28|qwen3ladder28|qwen3memc4kb|qwen3dqpack|qwen3memc4kt|qwen3ckptoff|qwen3evalce|qwen3combck) { [ "$PACK" = 1 ] && [ "$SEQ" = 4096 ]; } || { say "refusing: $_f is the packed 4,096-token regime -- it runs with TC1_PACK=1 TC1_SEQ=4096 (got pack=$PACK seq=$SEQ)"; echo "BOX_REFUSED $_f pack=$PACK seq=$SEQ" >> summary.txt; finish 78; };;
    *) [ "$PACK" = 0 ] || { say "refusing: TC1_PACK=1 packs every family on the box and $_f is a field-recipe token (packed-regime tokens: qwen3samestack4k, qwen3samestack4kce, qwen3samestack4kce2, qwen3memc4k, qwen3padbk, qwen3samestack4kd, qwen3padbk28, qwen3prof28, qwen3ladder28, qwen3memc4kb, qwen3dqpack, qwen3memc4kt, qwen3ckptoff, qwen3evalce, qwen3combck)"; echo "BOX_REFUSED $_f pack=$PACK" >> summary.txt; finish 78; };;
  esac
done
# ---------------------------------------------------------------- staged pieces, box class, forensics
for f in tc1_arm.py tc1_reduce.py tp4_alpaca.py n9_datasets.py ds_manifest.json; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done   # TC1b: + the clinical builder and its manifest (tc1_drive.sh's STAGE)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "BOX REFUSED: gpu '$GPU_NAME' is not the registered class ($GPU_CLASS)"; echo "BOX_REFUSED gpu=$GPU_NAME" >> summary.txt; finish 12;; esac
# K: the class label the arm records (TC1_BOX_CLASS): "RTX <n>" for a numeric class, the class string itself otherwise (H100 NVL / SXM / PCIE pass the substring check above)
case "$GPU_CLASS" in [0-9]*) BOX_CLASS="RTX $GPU_CLASS";; *) BOX_CLASS="$GPU_CLASS";; esac
# F [F19]: every arm runs with OMP_NUM_THREADS = the box's PHYSICAL core count (recorded in the receipt's arm_facts)
PHYS=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' '); case "$PHYS" in ''|0|*[!0-9]*) PHYS=$(nproc);; esac
echo "OMP_NUM_THREADS=$PHYS (physical cores) box_class=$BOX_CLASS" | tee -a summary.txt
# TC1 amendment 45 (2026-10-05): the container's CPU allotment -- the cgroup v2 cpu.max quota (else the v1 CFS quota) in CPUs, rounded up,
# capped by the affinity count; the affinity count when there is no quota. Every arm still runs OMP_NUM_THREADS=$PHYS unless its family
# hands it another count (qwen3ompab's om1 side runs $ALLOT).
ALLOT=$(python3 -c '
import math, os
n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
q = None
try:
    a, b = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
    q = None if a == "max" else int(a) / int(b)
except Exception:
    try:
        a = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); b = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        q = a / b if a > 0 else None
    except Exception:
        q = None
print(min(n, math.ceil(q)) if q else n)' 2>/dev/null); case "$ALLOT" in ''|0|*[!0-9]*) ALLOT=$(nproc);; esac
echo "CPU allotment $ALLOT (cgroup quota capped by affinity; physical cores $PHYS)" | tee -a summary.txt
case " $FAMILIES " in *" qwen3ompab "*)
  [ "$ALLOT" -lt "$PHYS" ] || { say "BOX REFUSED: the container's CPU allotment ($ALLOT) is not below the physical cores ($PHYS) -- amendment 45's threads A/B has no contrast on this host (host floor)"; echo "BOX_REFUSED cpu_allotment=$ALLOT phys=$PHYS" >> summary.txt; finish 18; };;
esac
# P48 run 1 (2026-09-19) drew a host whose container overlay was 32 GB and died ENOSPC mid-fetch; box B fetches ~120 GB of
# checkpoints. The launcher orders machine disk, the instance overlay is what the box gets: refuse here, before any fetch.
MIN_DISK_DEFAULT=200; [ "$TC1_LOCAL_BOX" = 1 ] && MIN_DISK_DEFAULT=40      # TC3 local box: nothing is fetched; the venvs need ~20 GB
MIN_DISK_GB=${TC1_MIN_DISK_GB:-$MIN_DISK_DEFAULT}; FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "${FREE_GB:-0}" -lt "$MIN_DISK_GB" ]; then say "BOX REFUSED: ${FREE_GB:-?} GB free on $W < ${MIN_DISK_GB} GB (instance overlay too small for the checkpoints -- host-limited)"; echo "BOX_REFUSED disk=${FREE_GB:-?}GB" >> summary.txt; finish 13; fi
# cu130 wheels (torch 2.12.1 / 2.14.1) need an NVIDIA driver >= 580: checked here, before any install; recorded in summary.txt
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' '); DRIVER_MAJOR=${DRIVER%%.*}
CU130_OK=1; case "$DRIVER_MAJOR" in ''|*[!0-9]*) CU130_OK=0;; *) [ "$DRIVER_MAJOR" -ge 580 ] || CU130_OK=0;; esac
CU130_REASON="cu130 wheels need driver >= 580; host has ${DRIVER:-unknown}"
# TC1-PREREG amendment 1 (2026-10-01): both TC1 family tokens need the cu130 venvs (the comparator and axolotl), so a host
# below the driver floor cannot produce the lane's readings. It is a REGISTERED HOST FLOOR -- rent.py's lane-refusal class
# 18 (P86's driver refusal, machine 37958 at 575.57 on tc1-5090-3) -- so the box refuses here, before any install or
# fetch, and the receipt names the machine for exclusion on the next draw instead of running four hours of refused rows.
# TC3's owned 12 GB box (TC1_LOCAL_BOX=1) sits at driver 575 by design: NOT a box refusal there -- its registered arms run on venv-e4b and venv-unsloth-t28,
# and every arm that needs a cu130 venv (axolotl) is a refused row naming the driver, as the gate below arranges.
if [ "$CU130_OK" != 1 ] && [ "$TC1_LOCAL_BOX" != 1 ]; then say "REFUSED: $CU130_REASON (registered host floor; TC1-PREREG amendment 1)"; echo "refused: driver ${DRIVER:-unknown} < 580" > REFUSAL; echo "BOX_REFUSED driver=${DRIVER:-unknown} floor=580" | tee -a summary.txt; finish 18; fi
# TC1 amendment 61 (2026-10-07): the image's torch must be able to use the GPU -- a REGISTERED HOST FLOOR (rent.py's lane-refusal
# class 18), checked here before any install or fetch. tc1-5090-119 (machine 34887, driver 595.58) passed the driver gate, then the
# image's torch raised CUDA error 803 ("system has unsupported display driver / cuda driver combination"): every venv inherits that
# torch, so the e4b tripwire failed as rc 9 -- a harness code that names no machine, so the next draw could land on it again. A host
# whose image has no importable torch is NOT refused here (that is the image, not the host): it reaches the tripwire as before.
CUDA_PROBE=$($PY_BASE - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
try:
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
except Exception as e:
    ok = False
print("ok" if ok else f"no-cuda torch {torch.__version__}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  no-cuda*) say "REFUSED: the image's torch cannot use the GPU on this host (${CUDA_PROBE}; driver ${DRIVER:-unknown}) -- registered host floor, TC1-PREREG amendment 61"
            echo "refused: cuda unusable (${CUDA_PROBE}, driver ${DRIVER:-unknown})" > REFUSAL
            echo "BOX_REFUSED cuda=unusable driver=${DRIVER:-unknown}" | tee -a summary.txt; finish 18;;
esac
[ "$CU130_OK" = 1 ] && say "driver $DRIVER: cu130 venvs (venv-unsloth, venv-axolotl) will be built" || say "driver ${DRIVER:-unknown}: $CU130_REASON -- venv-unsloth (cu130) and venv-axolotl are NOT built; their arms are refused rows"
echo "DRIVER $DRIVER cu130_ok=$CU130_OK" | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,power.limit,clocks.max.sm --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt; grep MemTotal /proc/meminfo | tee -a forensics.txt; cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
# The container's CPU allotment, beside the host's cores (2026-10-05): OMP_NUM_THREADS above is the host's PHYSICAL cores, and a rented
# container can be held to far fewer -- the RunPod H100 pod tc1c-h100-22 had 18 vCPUs under 72 threads, and Vast lists 5090 rentals at
# 24 of 96 cores (machine 152440). Recorded only; nothing here changes how an arm runs.
{ echo "cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo absent)"
  echo "cgroup cpu.cfs_quota_us/period_us $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo absent)/$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo absent)"
  echo "cgroup cpuset.cpus.effective $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo absent)"
  echo "affinity cpus $(nproc) of $(nproc --all)"; } | tee -a forensics.txt
python3 - "$TC1_BOX" "$TC1_RUN_ID" "$TC1_INSTANCE_ID" "$GPU_NAME" <<'PYB' > box.json
import json, os, subprocess, sys
box, run_id, iid, gpu = sys.argv[1:5]
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return f"ERR {e}"
print(json.dumps({"box": box, "run_id": run_id, "instance_id": iid, "gpu": gpu, "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
                  "cpu": sh("lscpu | grep 'Model name' | cut -d: -f2 | xargs"), "nproc": os.cpu_count(), "mem_total_kb": sh("grep MemTotal /proc/meminfo | awk '{print $2}'"),
                  "cgroup_memory_max": sh("cat /sys/fs/cgroup/memory.max 2>/dev/null"), "disk_root": sh("df -h /root | tail -1"), "hostname": sh("hostname"),
                  "cgroup_cpu_max": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"), "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  "cgroup_cpuset_effective": sh("cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null"),
                  "registered_gpu_class": os.environ.get("TC1_GPU_CLASS", "5090"), "prereg": "tc1/TC1-PREREG.md",
                  "local_box": os.environ.get("TC1_LOCAL_BOX"), "local_snapshot": os.environ.get("TC1_LOCAL_SNAPSHOT")}, indent=1))   # TC3: the hand-run facts
PYB
# ---------------------------------------------------------------- deadline-derived alarms (p39's rule): what is LEFT minus a fetch margin, never a literal that outlives the rental
left(){ echo $(( TC1_DEADLINE_EPOCH - $(date +%s) )); }
alarm_for(){ local want=$1 l; l=$(( $(left) - 900 )); [ "$l" -lt "$want" ] && want=$l; [ "$want" -lt 300 ] && want=300; echo "$want"; }
can_run(){ local need=$1 name=$2; [ $(( $(left) - 900 )) -ge "$need" ] && return 0; say "STOP-DEADLINE: $name needs ${need}s, $(left)s left -- skipped (host-limited)"; echo "SKIPPED $name host-limited deadline" >> summary.txt; return 1; }
# ---------------------------------------------------------------- installs
export DEBIAN_FRONTEND=noninteractive
PY_E4B=$W/venv-e4b/bin/python; PY_UNS=$W/venv-unsloth/bin/python; PY_UNS_T28=$W/venv-unsloth-t28/bin/python; PY_AX=$W/venv-axolotl/bin/python
say "venv-e4b (system torch): e4b @$E4B_SHA + gnf4 @$GNF4_SHA + transformers==$TF_VER bitsandbytes==$BNB_VER peft==$PEFT_VER"
$PY_BASE -m venv --system-site-packages $W/venv-e4b || { say "VENV FAIL (e4b)"; finish 9; }
# TC3-PREREG amendment 1 (2026-10-02): a venv made by `python -m venv` carries ensurepip's bundled pip -- 22.0.2 on the owned box's Ubuntu-22.04
# interpreter -- which cannot read the setuptools>=77 (PEP 621) metadata of e4b / grouped-nf4-gemm and reports both as "unknown 0.0.0 ...
# ResolutionImpossible" (the first 12 GB hand run, 04:17Z). The rented images carry a current pip. Every venv this script makes upgrades
# pip first (a no-op where it is current); the upgrade is logged, never fatal: the install that follows is what fails if pip cannot move.
pip_fresh(){ "$1" -m pip install -q --no-input -U pip > "logs/pip_upgrade_$2.log" 2>&1 || say "pip upgrade ($2) failed (logs/pip_upgrade_$2.log): continuing on $("$1" -m pip --version | cut -d" " -f1-2)"; }
pip_fresh $PY_E4B e4b
if [ "$TC1_LOCAL_BOX" = 1 ] && ! $PY_E4B -c "import torch; assert torch.cuda.is_available()" > logs/torch_probe_e4b.log 2>&1; then
  # TC3 local box: a venv made FROM a venv inherits the BASE interpreter's site-packages, not the venv's (the owned box keeps torch in a venv, so
  # --system-site-packages sees no torch there). venv-e4b then gets the base interpreter's own torch build explicitly: its version + CUDA tag,
  # from PyTorch's matching index -- recorded in versions.txt like every other torch.
  TORCH_SPEC=$($PY_BASE -c "import torch; print(torch.__version__)" 2>/dev/null); TORCH_CU=${TORCH_SPEC##*+}; TORCH_VER=${TORCH_SPEC%%+*}
  say "venv-e4b sees no CUDA torch; installing torch==${TORCH_VER:-?} from https://download.pytorch.org/whl/${TORCH_CU:-?} (TC1_LOCAL_PYTHON's build: ${TORCH_SPEC:-none})"
  [ -n "$TORCH_SPEC" ] && [ "$TORCH_CU" != "$TORCH_SPEC" ] || { say "VENV FAIL (e4b): TC1_LOCAL_PYTHON=$PY_BASE has no '+cuNNN' torch to mirror"; finish 9; }
  perl -e 'alarm 2400; exec @ARGV' $PY_E4B -m pip install -q --no-input "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$TORCH_CU" > logs/pip_torch_e4b.log 2>&1 \
    || { tail -4 logs/pip_torch_e4b.log; say "PIP FAIL (torch into venv-e4b)"; finish 9; }
fi
perl -e 'alarm 2400; exec @ARGV' $PY_E4B -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "transformers==$TF_VER" "bitsandbytes==$BNB_VER" "peft==$PEFT_VER" accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken > logs/pip_e4b.log 2>&1
rc=$?; echo "pip(e4b) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
E4B_SHA="$E4B_SHA" GNF4_SHA="$GNF4_SHA" TF_VER="$TF_VER" $PY_E4B - <<'PYT' > logs/tripwire_e4b.log 2>&1 || { tail -5 logs/tripwire_e4b.log; say "TRIPWIRE FAIL (e4b)"; finish 9; }
import importlib.metadata as md, inspect, json, os
import experts4bit_qlora as e, torch, triton, transformers, bitsandbytes, peft
from experts4bit_qlora import enable_fast_train, enable_batched_train, ExpertsLoRA, load_moe_4bit_streaming, verify_moe_4bit, disable_fast_train, disable_batched_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit, detect_attention_projections
from experts4bit_qlora.train import save_adapter
from nf4_qlora import fused_grouped_lora
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
def commit(dist):
    d = md.distribution(dist)
    du = d.read_text("direct_url.json")
    return (json.loads(du).get("vcs_info") or {}).get("commit_id") if du else None
ce, cg = commit("experts4bit-qlora"), commit("grouped-nf4-gemm")
assert ce == os.environ["E4B_SHA"], f"e4b installed from {ce}, wanted {os.environ['E4B_SHA']}"
assert cg == os.environ["GNF4_SHA"], f"gnf4 installed from {cg}, wanted {os.environ['GNF4_SHA']}"
assert transformers.__version__ == os.environ["TF_VER"], transformers.__version__
assert torch.cuda.is_available(), "no CUDA in venv-e4b"
print("tc1 tripwire OK (e4b):", e.__version__, "@", ce[:12], "gnf4", md.version("grouped-nf4-gemm"), "@", cg[:12], "torch", torch.__version__, "triton", triton.__version__,
      "transformers", transformers.__version__, "bnb", bitsandbytes.__version__, "peft", peft.__version__)
open(os.path.join(os.environ.get("TC1_W", "/root/tc1"), "versions.txt"), "a").write(f"e4b {e.__version__} @{ce} (GitHub main)\ngnf4 {md.version('grouped-nf4-gemm')} @{cg} (GitHub main)\ntorch(e4b/hf) {torch.__version__}\ntriton(e4b/hf) {triton.__version__}\n"
                                          f"transformers(e4b/hf) {transformers.__version__}\nbitsandbytes(e4b/hf) {bitsandbytes.__version__}\npeft(hf) {peft.__version__}\n")
PYT
tail -1 logs/tripwire_e4b.log
# Unsloth: the REGISTERED versions (TC1-PREREG "Environments": unsloth 2026.9.14 + unsloth_zoo 2026.9.9), NO transformers/bnb/peft
# pins from us (P38 amendment 1); torchao removed on the ScalingType tripwire (P38 amendment 2). TWO venvs (phase 2):
#   venv-unsloth-t28  unsloth[cu128-torch280]  -- tp4's install on the image's torch 2.8.0+cu128 (the field-image row, `ckpt_unsloth_t28`)
#   venv-unsloth      unsloth[cu130-torch2121] -- torch 2.12.1+cu130 (needs driver >= 580; UPSTREAM-NOTES "Unsloth": _auto_install.py:44),
#                     where torch._grouped_mm runs on sm_120 and the grouped_mm backend can engage (moe_utils.py:374-378)
UNS_VER=${TC1_UNSLOTH_VERSION:-2026.9.14}; ZOO_VER=${TC1_UNSLOTH_ZOO_VERSION:-2026.9.9}
UNS_T28_OK=0
if [ "$NEED_UNSLOTH" = 1 ]; then
UNS_T28_OK=1
say "venv-unsloth-t28: unsloth[cu128-torch280]==$UNS_VER unsloth_zoo==$ZOO_VER (the image's torch 2.8.0+cu128)"
$PY_BASE -m venv $W/venv-unsloth-t28 && pip_fresh $PY_UNS_T28 unsloth-t28 && perl -e 'alarm 2700; exec @ARGV' $PY_UNS_T28 -m pip install -q --no-input --no-cache-dir \
  "unsloth[cu128-torch280]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth_t28.log 2>&1
rc=$?; echo "pip(unsloth-t28) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_unsloth_t28.log; echo "PIP FAIL (unsloth-t28) -- its rows = install_failed"; UNS_T28_OK=0; }
else
  say "venv-unsloth-t28 SKIPPED: the $FAMILIES token runs no Unsloth arm (TC1-PREREG amendment 3)"
fi
UNS_OK=0
if [ "$CU130_OK" = 1 ] && [ "$NEED_UNSLOTH" = 1 ]; then
  UNS_OK=1; say "venv-unsloth: unsloth[cu130-torch2121]==$UNS_VER unsloth_zoo==$ZOO_VER (torch 2.12.1+cu130)"
  # TC1-PREREG amendment 2 (2026-10-01): the cu130 extra pins torch==2.12.1+cu130 / torchvision+cu130, which live on PyTorch's cu130
  # index, not PyPI -- without the index pip's resolver backtracks for the whole 2,700 s alarm (tc1-5090-7, 30 min at 99 % CPU, 0 sockets).
  $PY_BASE -m venv $W/venv-unsloth && pip_fresh $PY_UNS unsloth && perl -e 'alarm 2700; exec @ARGV' $PY_UNS -m pip install -q --no-input --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
    "unsloth[cu130-torch2121]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.log 2>&1
  rc=$?; echo "pip(unsloth) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_unsloth.log; echo "PIP FAIL (unsloth cu130) -- its rows = install_failed"; UNS_OK=0; }
elif [ "$NEED_UNSLOTH" != 1 ]; then
  say "venv-unsloth (cu130) SKIPPED: the $FAMILIES token runs no Unsloth arm (TC1-PREREG amendment 3)"
else
  say "venv-unsloth (cu130) SKIPPED: $CU130_REASON"
fi
cat > $W/tripwire_unsloth.py <<'PYU'
import importlib.metadata as md, os, torch, transformers, bitsandbytes, peft
import unsloth, unsloth_zoo
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches.common import is_transformers_v5_moe_quantization_available
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit, _is_bnb4bit_param, _moe_uses_bnb4bit_expert_weights
from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend, _should_use_separated_lora
assert is_transformers_v5_moe_quantization_available(), "the transformers-v5 4-bit MoE path is NOT available in this environment: Unsloth would not train 4-bit MoE here"
assert torch.cuda.is_available(), "no CUDA in venv-unsloth"
tri = None
try:
    import triton; tri = triton.__version__
except Exception: pass
tao = None
try:
    tao = md.version("torchao")
except Exception: pass
fm = hasattr(unsloth, "FastModel")
tag = os.environ.get("TC1_VENV_TAG", "unsloth")          # phase 2: one tripwire per Unsloth venv, lines suffixed with the venv
print(f"tc1 tripwire OK ({tag}):", unsloth.__version__, "zoo", unsloth_zoo.__version__, "torch", torch.__version__, "triton", tri, "transformers", transformers.__version__,
      "bnb", bitsandbytes.__version__, "peft", peft.__version__, "torchao", tao, "moe_backend", select_moe_backend(), "separated_lora", _should_use_separated_lora(), "FastModel", fm)
open(os.path.join(os.environ.get("TC1_W", "/root/tc1"), "versions.txt"), "a").write(f"unsloth({tag}) {unsloth.__version__}\nunsloth_zoo({tag}) {unsloth_zoo.__version__}\ntorch({tag}) {torch.__version__}\ntriton({tag}) {tri}\n"
                                          f"transformers({tag}) {transformers.__version__}\nbitsandbytes({tag}) {bitsandbytes.__version__}\npeft({tag}) {peft.__version__}\ntorchao({tag}) {tao}\nmoe_backend({tag}) {select_moe_backend()}\n")
PYU
# tripwire_unsloth VENV_TAG PY OK_VAR: the import tripwire for one Unsloth venv (torchao removed once on ScalingType, P38 amendment 2)
tripwire_unsloth(){ local TAG=$1 PY=$2 ok=1
  TC1_VENV_TAG=$TAG $PY $W/tripwire_unsloth.py > logs/tripwire_$TAG.log 2>&1; local trc=$?
  if [ $trc -ne 0 ] && grep -q "ScalingType" logs/tripwire_$TAG.log; then
    say "tripwire($TAG) failed on ScalingType -> removing torchao (P38 amendment 2)"; cp logs/tripwire_$TAG.log logs/tripwire_$TAG.attempt1.log
    $PY -m pip uninstall -y -q torchao > logs/pip_${TAG}_torchao_removed.log 2>&1
    echo "AMENDMENT-CLASS (P38 amendment 2): torchao removed from $TAG after the tripwire failed on ScalingType" | tee -a summary.txt
    TC1_VENV_TAG=$TAG $PY $W/tripwire_unsloth.py > logs/tripwire_$TAG.log 2>&1; trc=$?
  fi
  [ $trc -ne 0 ] && { echo "TRIPWIRE FAIL ($TAG) -- other arms still run; its Unsloth rows = install_failed"; tail -5 logs/tripwire_$TAG.log; ok=0; }
  tail -1 logs/tripwire_$TAG.log; return $(( 1 - ok )); }
[ "$UNS_T28_OK" = 1 ] && { tripwire_unsloth unsloth-t28 $PY_UNS_T28 || UNS_T28_OK=0; }
[ "$UNS_OK" = 1 ] && { tripwire_unsloth unsloth $PY_UNS || UNS_OK=0; }
# J [F7]: e4b + grouped-nf4-gemm at the same pins into venv-unsloth (torch 2.12.1+cu130) for the `fused_attn4_m_t212` row; a failed install
# or tripwire is a row (install_failed), the cu130 gate a refused row
T212_OK=0; T212_REASON=""
if [ "$UNS_OK" = 1 ]; then
  say "venv-unsloth + e4b @$E4B_SHA + gnf4 @$GNF4_SHA (the t212 row)"
  perl -e 'alarm 2400; exec @ARGV' $PY_UNS -m pip install -q --no-input --prefer-binary --extra-index-url https://download.pytorch.org/whl/cu130 \
    "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_e4b_t212.log 2>&1
  rc=$?; echo "pip(e4b-t212) rc=$rc"
  if [ $rc -ne 0 ]; then tail -6 logs/pip_e4b_t212.log; T212_REASON="e4b/gnf4 install into venv-unsloth failed rc=$rc (logs/pip_e4b_t212.log): $(tail -3 logs/pip_e4b_t212.log | tr '\n' ' ' | cut -c1-300)"
  else
    E4B_SHA="$E4B_SHA" GNF4_SHA="$GNF4_SHA" $PY_UNS - <<'PYT2' > logs/tripwire_e4b_t212.log 2>&1; trc=$?
import importlib.metadata as md, inspect, json, os
import experts4bit_qlora as e, torch
from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming
from nf4_qlora import fused_grouped_lora, LORA_PATH_STATS
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
def commit(dist):
    d = md.distribution(dist); du = d.read_text("direct_url.json")
    return (json.loads(du).get("vcs_info") or {}).get("commit_id") if du else None
assert commit("experts4bit-qlora") == os.environ["E4B_SHA"] and commit("grouped-nf4-gemm") == os.environ["GNF4_SHA"], (commit("experts4bit-qlora"), commit("grouped-nf4-gemm"))
assert torch.cuda.is_available()
print("tc1 tripwire OK (e4b-t212):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
open(os.path.join(os.environ.get("TC1_W", "/root/tc1"), "versions.txt"), "a").write(f"e4b(t212) {e.__version__} @{commit('experts4bit-qlora')}\ngnf4(t212) {md.version('grouped-nf4-gemm')} @{commit('grouped-nf4-gemm')}\ntorch(e4b-t212) {torch.__version__}\n")
PYT2
    if [ $trc -ne 0 ]; then tail -4 logs/tripwire_e4b_t212.log; T212_REASON="e4b-t212 tripwire failed (logs/tripwire_e4b_t212.log): $(tail -2 logs/tripwire_e4b_t212.log | tr '\n' ' ' | cut -c1-300)"; else T212_OK=1; tail -1 logs/tripwire_e4b_t212.log; fi
  fi
else
  T212_REASON="venv-unsloth (cu130) unavailable: ${CU130_REASON}"
fi
# ---------------------------------------------------------------- axolotl 0.20.0 (phase 2, axolotl-arm-spec.md): its own venv, uv, CPython 3.12, torch cu130
# Pins (wheel METADATA): python >= 3.12, torch >= 2.13.0 <= 2.14.0, transformers == 5.17.0, peft == 0.21.0, bitsandbytes == 0.50.2.
# torch >= 2.13 Linux cp312 wheels exist only under /whl/cu130 (verified 2026-10-01 by the coordinator), so the driver gate above applies.
AX_VER=${TC1_AXOLOTL_VERSION:-0.20.0}; AX_OK=0; AX_REASON=""
if [ "$CU130_OK" != 1 ]; then
  AX_REASON="$CU130_REASON"; say "venv-axolotl SKIPPED: $AX_REASON -- its arms are refused rows"
else
  AX_PKG="axolotl==$AX_VER"; AX_INDEX="https://download.pytorch.org/whl/cu130"      # TC3 reuses both for its deepspeed extra (a separate, non-fatal step)
  # TC1-PREREG amendment 3 (2026-10-02): uv's default first-index strategy takes `packaging` (and anything else PyTorch's index carries) from the
  # cu130 index alone, where axolotl's packaging==26.0 does not exist -> "No solution found" on both TC1 boxes (tc1-5090-14, -16: INSTALL_FAILED).
  # unsafe-best-match reads PyPI for the rest while the cu130 wheels still win on version (torch 2.14.0+cu130, torchao 0.18.0+cu130, triton 3.8.0;
  # dry-resolved for x86_64-manylinux_2_28 / cp312 with uv 0.12.22 before the amendment: axolotl 0.20.0, transformers 5.17.0, peft 0.21.0, bnb 0.50.2).
  say "venv-axolotl: uv venv --python 3.12 + axolotl==$AX_VER --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match (alarm 2700 s)"
  $PY_BASE -m pip install -q --no-input uv > logs/pip_uv.log 2>&1 && $PY_BASE -m uv venv --python 3.12 $W/venv-axolotl > logs/uv_venv_axolotl.log 2>&1 \
    && perl -e 'alarm 2700; exec @ARGV' $PY_BASE -m uv pip install --python $PY_AX "axolotl==$AX_VER" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > logs/pip_axolotl.log 2>&1
  rc=$?; echo "pip(axolotl) rc=$rc"
  # TC3 arm 10 (ckpt_axolotl_m_zero3): the `deepspeed` extra (METADATA lines 80-81: deepspeed>=0.18.6,<0.20.0 + deepspeed-kernels) ONLY under the
  # qwen3frontier token, as a SEPARATE, non-fatal step: deepspeed ships an sdist whose setup imports torch, so it is built without isolation against
  # the venv's torch (setuptools/wheel/ninja installed first -- a uv venv carries none). A failure here leaves the other axolotl arms their venv and
  # the ZeRO-3 refused row records `deepspeed_version: not installed` (the row is refused on the trainer either way). UNVERIFIED by execution here.
  AX_DS_OK=0
  if [ $rc -eq 0 ]; then case " $FAMILIES " in *" qwen3frontier "*)
    say "venv-axolotl + axolotl[deepspeed]==$AX_VER (TC3 arm 10; non-fatal, alarm 2700 s)"
    perl -e 'alarm 600; exec @ARGV' $PY_BASE -m uv pip install --python $PY_AX setuptools wheel ninja > logs/pip_axolotl_deepspeed_build.log 2>&1 \
      && perl -e 'alarm 2700; exec @ARGV' $PY_BASE -m uv pip install --python $PY_AX --no-build-isolation "axolotl[deepspeed]==$AX_VER" --extra-index-url $AX_INDEX --index-strategy unsafe-best-match > logs/pip_axolotl_deepspeed.log 2>&1
    dsrc=$?; echo "pip(axolotl[deepspeed]) rc=$dsrc"
    if [ $dsrc -ne 0 ]; then tail -4 logs/pip_axolotl_deepspeed.log; echo "AXOLOTL[deepspeed] INSTALL FAILED rc=$dsrc (logs/pip_axolotl_deepspeed.log): the ZeRO-3 row records deepspeed as not installed; the other axolotl arms keep their venv" | tee -a summary.txt
    else AX_DS_OK=1; fi;;
  esac; fi
  if [ $rc -ne 0 ]; then tail -6 logs/pip_axolotl.log; AX_REASON="axolotl venv install failed rc=$rc (logs/pip_axolotl.log): $(tail -3 logs/pip_axolotl.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; echo "PIP FAIL (axolotl) -- its rows = install_failed"
  else
    $PY_AX - <<'PYA' > logs/tripwire_axolotl.log 2>&1; trc=$?
import importlib.metadata as md, os, torch, transformers, peft, bitsandbytes
import axolotl
from axolotl.cli.config import load_cfg
from axolotl.loaders import ModelLoader, load_tokenizer
from axolotl.monkeypatch.moe_quant import get_moe_quantized_count, patch_moe_quantization_on_load
from axolotl.integrations.base import PluginManager
from bitsandbytes.nn.parametrize import replace_parameter_4bit
assert torch.cuda.is_available(), "no CUDA in venv-axolotl"
try:
    ds = md.version("deepspeed")          # TC3: present only under the qwen3frontier token's axolotl[deepspeed] install (read, never run by this harness)
except Exception:
    ds = None
print("tc1 tripwire OK (axolotl):", md.version("axolotl"), "torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "bnb", bitsandbytes.__version__, "python", __import__("sys").version.split()[0], "deepspeed", ds)
open(os.path.join(os.environ.get("TC1_W", "/root/tc1"), "versions.txt"), "a").write(f"axolotl {md.version('axolotl')}\ntorch(axolotl) {torch.__version__}\ntransformers(axolotl) {transformers.__version__}\npeft(axolotl) {peft.__version__}\nbitsandbytes(axolotl) {bitsandbytes.__version__}\npython(axolotl) {__import__('sys').version.split()[0]}\ndeepspeed(axolotl) {ds}\n")
PYA
    if [ $trc -ne 0 ]; then tail -5 logs/tripwire_axolotl.log; AX_REASON="axolotl tripwire failed (logs/tripwire_axolotl.log): $(tail -2 logs/tripwire_axolotl.log | tr '\n' ' ' | cut -c1-300)"; echo "TRIPWIRE FAIL (axolotl) -- its rows = install_failed"; else AX_OK=1; tail -1 logs/tripwire_axolotl.log; fi
  fi
fi
# ---------------------------------------------------------------- the fixed texts: Alpaca (field recipe) + clinical (the anchor pair only)
say "dataset alpaca (tp4_alpaca.py: unsloth/alpaca-cleaned @ pinned revision, seed 3407, 1200/48)"
(cd $W/data && perl -e 'alarm 900; exec @ARGV' $PY_E4B $W/tp4_alpaca.py --out $W/data/ds_alpaca.json > $W/logs/dataset_alpaca.log 2>&1) || { tail -3 logs/dataset_alpaca.log; say "DATASET FAIL (alpaca)"; finish 13; }
tail -1 logs/dataset_alpaca.log
GOT=$(sha256sum $W/data/ds_alpaca.json | awk '{print $1}'); [ "$GOT" = "$DS_ALPACA_SHA" ] || { say "DATASET MISMATCH alpaca: $GOT != $DS_ALPACA_SHA"; finish 13; }
echo "DATASET alpaca sha=$DS_ALPACA_SHA" | tee -a summary.txt
case " $FAMILIES " in *" qwen3curve "*)      # TC1b: the anchor pair's text, built and sha-verified exactly as tp4_run.sh does it (ds_manifest.json's clinical sha)
  say "dataset clinical (n9_datasets.py, sha-verified against ds_manifest.json) for the anchor pair"
  (cd $W/data && $PY_E4B $W/n9_datasets.py $W/data > $W/logs/dataset_clinical.log 2>&1); tail -1 logs/dataset_clinical.log
  CLIN_SHA=$($PY_E4B -c "import json; print(json.load(open('$W/ds_manifest.json'))['clinical']['sha256'])")
  GOT=$(sha256sum $W/data/ds_clinical.json | awk '{print $1}'); [ "$GOT" = "$CLIN_SHA" ] || { say "DATASET MISMATCH clinical: $GOT != $CLIN_SHA"; finish 13; }
  echo "DATASET clinical sha=$CLIN_SHA" | tee -a summary.txt;;
esac
# ---------------------------------------------------------------- helpers
# TC1 amendment 7: each second also writes the SM and memory clocks, the GPU temperature, the active clock-event (throttle) reasons, the
# host load average and the host's aggregate cpu counters to gpuclk_<arm>.txt, so a draw that slows (tc1-5090-33's e4b shipped pair, 6.1 %)
# can be put down to the card or to the host. The vram file's columns are unchanged; a field this driver does not know fails only its own
# line. The loop's stdout goes to /dev/null: it runs inside the caller's $(...), which would otherwise wait for it forever.
vram_start(){ : > $W/vram_$1.txt; : > $W/gpuclk_$1.txt
  ( while :; do
      echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)" >> $W/vram_$1.txt
      echo "$(date -u +%s) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits 2>&1 | head -1) | $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null) | $(head -1 /proc/stat 2>/dev/null)" >> $W/gpuclk_$1.txt
      sleep 1; done ) >/dev/null 2>&1 & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
status_of(){ $PY_E4B -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','missing'))" "$W/${1}_${2}_${3}.json" 2>/dev/null || echo missing; }
# stubw FAM FW TAG ARM STATUS REASON [extra-json]: a row for an attempt that never reached the harness
stubw(){ $PY_E4B - "$W" "$STEPS" "$SEQ" "$ACCUM" "$MB" "$PREREG" "$@" <<'PYS'
import json, os, sys
W, steps, seq, accum, mb, prereg, fam, fw, tag, arm, status, reason = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], *sys.argv[7:13]
extra = json.loads(sys.argv[13]) if len(sys.argv) > 13 else {}
rec = {"framework": fw, "fam": fam, "arm": arm, "tag": tag, "status": status, "reason": reason[:800], "steps": steps, "seq": seq, "accum": accum, "micro_batch": mb,
       "written_by": "tc1_run.sh", "prereg": prereg, **extra}
p = os.path.join(W, f"{fam}_{fw}_{tag}.json")
json.dump(rec, open(p, "w"), indent=1)
print(f"STUB {status.upper()} {fam}/{fw}/{tag}: {reason[:160]}")
PYS
  echo "$1/$2/$3 STUB $5: $6" | cut -c1-300 >> summary.txt; }
# fetch FAM MID REV ALARM: the snapshot UNPINNED (the loader resolves `main`, e4b#404), then the PROOF main == pin; refs/main written from the pin only then
fetch(){ local FAM=$1 MID=$2 REV=$3 AL=$4; say "fetch $FAM ($MID, unpinned; pin $REV)"
  perl -e "alarm $(alarm_for $AL); exec @ARGV" $PY_E4B - "$MID" <<'PYF' > logs/fetch_$FAM.log 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json", "*.tiktoken", "*.jinja"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
  local rc=$?; tail -2 logs/fetch_$FAM.log | head -1
  [ $rc -ne 0 ] && { echo "$FAM: FETCH FAILED rc=$rc" | tee -a summary.txt; FETCH_REASON="fetch failed rc=$rc (alarm $AL s; logs/fetch_$FAM.log)"; return 1; }
  local GOT; GOT=$(grep -a "^STAGED " logs/fetch_$FAM.log | tail -1 | awk '{print $2}')
  local RDIR=/root/.cache/huggingface/hub/models--${MID//\//--}
  if [ "$GOT" = "$REV" ]; then
    mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main
    echo "PIN OK $FAM staged=$GOT == pin; refs/main written from the pin (P38 amendment 2 / e4b#404)" | tee -a summary.txt
  elif [ "$PIN_FALLBACK" = "1" ]; then
    say "PIN MISMATCH $FAM: staged $GOT != pin $REV -> fetching the pinned revision (TC1_PIN_FALLBACK=1, an amendment)"
    perl -e "alarm $(alarm_for $AL); exec @ARGV" $PY_E4B -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MID', revision='$REV', allow_patterns=['*.safetensors', '*.json', 'tokenizer*', '*.model', '*.txt', 'merges.txt', 'vocab.json', '*.tiktoken', '*.jinja'], max_workers=4))" > logs/fetch_${FAM}_pinned.log 2>&1 \
      || { echo "$FAM: PINNED FETCH FAILED" | tee -a summary.txt; FETCH_REASON="main moved past the pin ($GOT != $REV) and the pinned fetch failed"; return 1; }
    mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main
    echo "AMENDMENT (TC1_PIN_FALLBACK): $FAM main=$GOT != pin=$REV; pinned snapshot fetched; refs/main := pin (cache pointer only)" | tee -a summary.txt
  else
    echo "PIN MISMATCH $FAM: staged=$GOT != pin=$REV -- main moved past the pin; the family is ABORTED with load_fault stubs (not coerced)" | tee -a summary.txt
    FETCH_REASON="staged snapshot $GOT != pinned revision $REV (main moved past the pin); family aborted, not coerced (set TC1_PIN_FALLBACK=1 as an amendment)"
    return 2
  fi
  df -h /root | tail -1; return 0; }
# local_snapshot FAM MID REV (TC3, TC1_LOCAL_BOX / TC1_LOCAL_SNAPSHOT): the checkpoint is a DIRECTORY at the registered revision, linked into the private
# HF cache as snapshots/<REV> with refs/main = REV, so every loader resolves the pin offline exactly as after a fetch (nothing in tc1_arm.py changes).
# The pin proof: the directory's config.json sha256 against the Hub's at REV (hf_hub_download, into a side cache) -> `pin_proof: config.json sha == hub@<rev>`,
# plus the safetensors count and bytes against the Hub listing when it answers; no network -> `pin_proof: offline` (recorded, the family runs);
# a MISMATCH aborts the family with load_fault stubs, as fetch's PIN MISMATCH does. Written to pin_proof_<FAM>.json beside the receipts and into summary.txt;
# every arm of the family carries it in --note.
local_snapshot(){ local FAM=$1 MID=$2 REV=$3 SNAP=$TC1_LOCAL_SNAPSHOT; say "local snapshot $FAM: $SNAP (pin $REV; TC1_LOCAL_BOX)"
  [ -s "$SNAP/config.json" ] || { echo "$FAM: LOCAL SNAPSHOT MISSING $SNAP/config.json" | tee -a summary.txt; FETCH_REASON="TC1_LOCAL_SNAPSHOT=$SNAP has no config.json"; return 1; }
  local RDIR=${HF_HUB_CACHE:-/root/.cache/huggingface/hub}/models--${MID//\//--}     # the private cache under TC1_LOCAL_BOX; the image's otherwise
  { mkdir -p $RDIR/snapshots $RDIR/refs && ln -sfn "$SNAP" $RDIR/snapshots/$REV && printf '%s' "$REV" > $RDIR/refs/main; } || { FETCH_REASON="could not link $SNAP into $RDIR"; return 1; }
  perl -e 'alarm 300; exec @ARGV' $PY_E4B - "$MID" "$REV" "$SNAP" "$FAM" <<'PYL' > logs/pin_proof_$FAM.log 2>&1
import glob, hashlib, json, os, sys
mid, rev, snap, fam = sys.argv[1:5]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
local = sha(os.path.join(snap, "config.json"))
shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
out = {"snapshot": snap, "model": mid, "revision": rev, "config_sha256_local": local, "n_safetensors_local": len(shards), "bytes_safetensors_local": sum(os.path.getsize(p) for p in shards)}
try:
    from huggingface_hub import HfApi, hf_hub_download
    p = hf_hub_download(mid, "config.json", revision=rev, cache_dir=os.path.join(os.environ["HF_HUB_CACHE"], "pin-proof"))
    hub = sha(p)
    out["config_sha256_hub"] = hub
    out["pin_proof"] = ("config.json sha == hub@" + rev[:12]) if hub == local else ("MISMATCH: config.json sha != hub@" + rev[:12])
    try:
        info = HfApi().model_info(mid, revision=rev, files_metadata=True)
        sib = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
        out["n_safetensors_hub"], out["bytes_safetensors_hub"] = len(sib), sum((s.size or 0) for s in sib)
        if out["n_safetensors_hub"] != out["n_safetensors_local"] or out["bytes_safetensors_hub"] != out["bytes_safetensors_local"]:
            out["pin_proof"] = "MISMATCH: safetensors count/bytes differ from the Hub listing at " + rev[:12]
        elif out["pin_proof"].startswith("config.json"):
            out["pin_proof"] += f" and {len(sib)} safetensors ({out['bytes_safetensors_hub']} bytes) match the Hub listing"
    except Exception as e:
        out["hub_listing"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
except Exception as e:
    out["pin_proof"], out["offline_reason"] = "offline", f"{type(e).__name__}: {str(e)[:160]}"
json.dump(out, open(f"pin_proof_{fam}.json", "w"), indent=1)
print("PIN_PROOF", out["pin_proof"])
PYL
  local rc=$? PROOF; PROOF=$(grep -a "^PIN_PROOF " logs/pin_proof_$FAM.log | tail -1 | cut -d' ' -f2-)
  if [ $rc -ne 0 ] || [ -z "$PROOF" ]; then tail -3 logs/pin_proof_$FAM.log; echo "$FAM: PIN PROOF FAILED rc=$rc" | tee -a summary.txt; FETCH_REASON="pin proof script failed rc=$rc (logs/pin_proof_$FAM.log)"; return 1; fi
  case "$PROOF" in MISMATCH*) echo "PIN MISMATCH $FAM (local snapshot $SNAP): $PROOF -- the family is ABORTED with load_fault stubs (not coerced)" | tee -a summary.txt; FETCH_REASON="local snapshot $SNAP: $PROOF"; return 2;; esac
  PIN_PROOF=$PROOF; echo "PIN LOCAL $FAM snapshot=$SNAP -> $RDIR/snapshots/$REV (refs/main := pin); pin_proof: $PROOF" | tee -a summary.txt; return 0; }
# expect_of FAM E4BTAG: the family's e4b trainable count (the primary receipt of the same recipe) for --expect-trainable
expect_of(){ $PY_E4B - "$W" "$1" "$2" <<'PYE' 2>/dev/null
import json, os, sys
W, fam, tag = sys.argv[1], sys.argv[2], sys.argv[3]
for t in (tag, "attn_only_m", "reference_attn4_m", "fused_attn4_m_offload", "reference_attn4_m_offload"):   # TC2: gpt-oss's e4b arm is attn_only_m; TC3: on a frontier box the resident e4b arm is expected to OOM, the offload arms carry the count
    p = os.path.join(W, f"{fam}_e4b_{t}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        if r.get("status") == "ok" and r.get("trainable_params"):
            print(r["trainable_params"]); break
PYE
}
# arm FAM FW TAG ARM ALARM MID REV OFFLOAD RECIPE(field|mb1|curve|anchor|t1|r64|small) TOK TOK_SHA [extra args...]: one process, one JSON, one alarm
arm_once(){ local FAM=$1 FW=$2 TAG=$3 ARM=$4 AL=$5 MID=$6 REV=$7 OFF=$8 RECIPE=$9 TOK=${10} TOK_SHA=${11}; shift 11
  { skip $FAM || skip $FAM/$FW/$TAG; } && { say "skip $FAM/$FW/$TAG"; stubw $FAM $FW $TAG $ARM not_run "skipped by TC1_SKIP"; return 0; }
  # phase 2: the interpreter per framework; UNS_VENV=t28 (a prefix assignment on the call) selects tp4's torch-2.8 venv for an
  # Unsloth arm, else the cu130 venv. A cu130 venv the driver gate refused -> `refused` rows naming the driver; a venv that
  # did not install/import -> `install_failed` rows. Never a silent fallback to the other venv.
  local PY=$PY_E4B
  if [ "$FW" = e4b ] && [ "${E4B_VENV:-}" = t212 ]; then PY=$PY_UNS          # J: e4b on torch 2.12.1+cu130
    [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-unsloth (cu130-torch2121) + e4b"}'; return 0; }
    [ "$T212_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "$T212_REASON"; return 0; }
  elif [ "$FW" = e4b ] && [ "${E4B_VENV:-}" = tf55 ]; then PY=$W/venv-e4b-tf55/bin/python   # TC1 amendment 34: venv-e4b with transformers 5.5.0
    [ "${TF55_OK:-0}" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "${TF55_REASON:-venv-e4b-tf55 was not built}"; return 0; }
  elif [ "$FW" = hf ] && [ "${HF_VENV:-}" = t214 ]; then PY=$PY_AX              # J: the HF arm on the axolotl venv's torch 2.14
    [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-axolotl (torch cu130)"}'; return 0; }
    [ "$AX_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "$AX_REASON"; return 0; }
  elif [ "$FW" = unsloth ]; then
    if [ "${UNS_VENV:-cu130}" = t28 ]; then PY=$PY_UNS_T28
      [ "$UNS_T28_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "venv-unsloth-t28 did not install/import (logs/pip_unsloth_t28.log, logs/tripwire_unsloth-t28.log)"; return 0; }
    else PY=$PY_UNS
      [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-unsloth (cu130-torch2121)"}'; return 0; }
      [ "$UNS_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "venv-unsloth (cu130) did not install/import (logs/pip_unsloth.log, logs/tripwire_unsloth.log)"; return 0; }
    fi
  elif [ "$FW" = axolotl ]; then PY=$PY_AX
    [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-axolotl (torch cu130)"}'; return 0; }
    [ "$AX_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "$AX_REASON"; return 0; }
  fi
  local s=$STEPS q=$SEQ m=$MB ac=$ACCUM r=$R al=$ALPHA lr=$LR wd=$WD wu=$WARMUP sc=$SCHED op=$OPTIM sd=$SEED en=$EVAL_N ee=$EVAL_EVERY ex_tag=fused_attn4_m
  case "$RECIPE" in
    mb1)    m=1; ac=$(( MB * ACCUM )); ex_tag=fused_attn4_m_mb1;;
    # TC1b sub-fixtures: each names its OWN e4b arm as the trainable reference (expect_of), as tp4's anchor did
    curve)  s=$CURVE_STEPS; en=$CURVE_EVAL_N; ee=$CURVE_EVAL_EVERY; ex_tag=fused_attn4_m_200;;
    anchor) s=$A_STEPS; q=$A_SEQ; m=$A_MB; ac=$A_ACCUM; r=$A_R; al=$A_ALPHA; lr=$A_LR; wd=$A_WD; wu=$A_WARMUP; sc=$A_SCHED; op=$A_OPTIM; sd=$A_SEED; en=$A_EVAL_N; ee=$A_EVAL_EVERY; ex_tag=fused_attn4_p38;;
    t1)     m=$T1_MB; ac=$T1_ACCUM; ex_tag=fused_attn4_m_t1;;
    r64)    r=$R64_R; al=$R64_ALPHA; ex_tag=fused_attn4_m_r64;;
    # TC2: the small families' instrument (N 60, 48 held-out rows every 20 steps); everything else the field recipe; the trainable reference is the family's fused_attn4_m
    small)  s=$SMALL_STEPS; en=$SMALL_EVAL_N; ee=$SMALL_EVAL_EVERY; ex_tag=fused_attn4_m;;
  esac
  local EXP EXPARG=""; [ "$FW" != e4b ] && { EXP=$(expect_of $FAM $ex_tag); [ -n "$EXP" ] && EXPARG="--expect-trainable $EXP"; }
  local A; A=$(alarm_for $AL)
  say "arm $FAM/$FW/$TAG (arm=$ARM recipe=$RECIPE steps=$s seq=$q mb=$m accum=$ac r=$r lr=$lr optim=$op sched=$sc offload=$OFF alarm=$A expect_trainable=${EXP:-none} $*)"
  local sp; sp=$(vram_start ${FAM}_${FW}_$TAG)
  # e4b#548: the arm is told the alarm it is running under, so it can refuse ITSELF while still inside an over-budget
  # prologue phase (status phase_alarm, exit 16) instead of leaving SIGALRM to kill a process that cannot write a stub.
  # P56: the batched arm is a PARITY arm, so it must run the batched arithmetic on every
  # call. `enable_batched_train` falls back to the reference forward above a pad-waste
  # ratio, and the tp1 bundle records exactly that producing VOID rows on OLMoE, Qwen3 and
  # Gemma-4 -- an arm that fell back is measuring the reference against itself. The guard
  # is a SPEED guard; raising it trades peak memory for engagement and never numerics, and
  # `batched_fallback_stats` puts the limit in force on the receipt.
  #
  # It goes through `env`, and that is NOT cosmetic. An unquoted expansion spliced into
  # the assignment prefix -- `A=1 $ARM_ENV B=2 cmd` -- makes the shell stop treating the
  # words after it as assignments, because it decides which words are assignments BEFORE
  # expanding: the next `VAR=value` becomes the COMMAND. Run p56-gemma4-ladder-1 lost all
  # four e4b arms to rc=127 `TC1_BOX_CLASS=RTX 5090: command not found` this way, and it
  # fails whether the variable is empty or set. As an argument to `env` the expansion is
  # an ordinary word and an empty one simply vanishes.
  local ARM_ENV=""; [ "$ARM" = batched ] && ARM_ENV="E4B_BATCHED_PAD_WASTE_LIMIT=${TC1_BATCHED_PAD_WASTE_LIMIT:-64}"
  # TC1 amendment 10 (#945): a family may hand one arm extra environment (TC1_ARM_EXTRA_ENV="K=V ..." as a prefix on the
  # arm/draw2 call); it rides the same `env` word list, so an empty value vanishes as ARM_ENV's does
  [ -n "${TC1_ARM_EXTRA_ENV:-}" ] && ARM_ENV="$ARM_ENV $TC1_ARM_EXTRA_ENV"
  # TC1c amendment 2: a box may hand EVERY e4b arm extra environment (TC1_E4B_ENV="K=V ...", forwarded by tc1_drive.sh) -- an opt-in
  # e4b setting measured inside an unchanged family; the comparator's arms never see it, and an empty value vanishes as above
  [ "$FW" = e4b ] && [ -n "${TC1_E4B_ENV:-}" ] && ARM_ENV="$ARM_ENV $TC1_E4B_ENV"
  # TC1 amendment 4: every arm runs with the Hub offline (the pinned snapshot is the only model bytes) except axolotl's scattermoe
  # native-best, whose KernelsPlugin fetches kernels-community kernels by version at load, as an axolotl user's run does; the
  # arm records the kernel commits it fetched (hub_kernels_cached), and the model revision stays pinned by sha.
  # TC1 amendment 6: both draws (ckpt_axolotl_best and its _d2) -- the exact match left the second draw offline (tc1-5090-33)
  # TC1 amendment 8: every scattermoe tag (ckpt_axolotl_best, _d2, _200, _200_d2) -- a prefix, so a new suffix cannot fall offline again
  local OFFL=1; case "$FW/$TAG" in axolotl/ckpt_axolotl_best*) OFFL=0;; esac
  # TC1 amendment 45: OMP_NUM_THREADS=$PHYS comes FIRST so a family's per-arm environment can override it (env applies left to right);
  # an arm whose environment does not name it runs exactly as before
  env OMP_NUM_THREADS=$PHYS $ARM_ENV HF_HUB_OFFLINE=$OFFL UNSLOTH_ENABLE_LOGGING=1 TC1_BOX_CLASS="$BOX_CLASS" TC1_ARM_ALARM_S=$A perl -e "alarm $A; exec @ARGV" $PY -u $W/tc1_arm.py --framework $FW --arm $ARM --tag $TAG --fam $FAM --model "$MID" --revision $REV \
      --steps $s --seq $q --micro-batch $m --accum $ac --autocast $AUTOCAST --lr $lr --r $r --alpha $al --seed $sd --offload $OFF \
      --optim $op --weight-decay $wd --lr-schedule $sc --warmup-steps $wu \
      --tokens $TOK --tokens-sha $TOK_SHA --eval-every $ee --eval-n $en --unsloth-loader FastLanguageModel $EXPARG \
      --prereg $PREREG --out $W --adapter-dir $W/adapters "$@" > logs/run_${FAM}_${FW}_$TAG.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/${FAM}_${FW}_$TAG.json ]; then stubw $FAM $FW $TAG $ARM alarm "arm alarm $A s (SIGALRM; the process could not write its own stub)"; fi
  grep -aE "^CELL |^LOAD OK|^PROLOGUE |^PHASE ALARM|^ENGAGE|^STUB|^LOADER FALLBACK|Enabling LoRA on MoE|MoE bnb4bit|Error|error:" logs/run_${FAM}_${FW}_$TAG.log | tail -3 | cut -c1-300 | sed "s/^/    /"
  # e4b#548: the phase table goes into summary.txt beside the CELL line, so the box's own summary answers "where did the prologue go"
  grep -aE "^PROLOGUE |^PHASE ALARM" logs/run_${FAM}_${FW}_$TAG.log | tail -1 | cut -c1-600 | sed "s|^|$FAM/$FW/$TAG |" >> summary.txt
  { echo -n "$FAM/$FW/$TAG rc=$rc "; grep -aE "^CELL " logs/run_${FAM}_${FW}_$TAG.log | tail -1 | cut -c1-400; echo; } >> summary.txt
  $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader
  rm -rf $W/adapters/* 2>/dev/null; }
# TC1 amendment 33 (2026-10-05): load-gated draws. Every arm runs through arm_once (the body above). With TC1_LOAD_GATE set (a host load
# average), an OK arm whose median host load1 over its own run (the second field group of gpuclk_<arm>.txt, sampled each second) exceeds the
# gate is set aside to $W/loadvoid/ -- receipt, gpuclk and vram samples, run log, each suffixed .a<k> -- and run again, at most
# TC1_LOAD_RETRIES (default 2) more times while the deadline allows; the last attempt stands whatever its load. Each attempt writes a
# LOADGATE line to summary.txt. Unset (every box before this amendment): arm is arm_once.
arm(){ local G=${TC1_LOAD_GATE:-} n=0 FAM=$1 FW=$2 TAG=$3
  while :; do
    arm_once "$@"
    [ -n "$G" ] || return 0
    local f=$W/gpuclk_${FAM}_${FW}_$TAG.txt j=$W/${FAM}_${FW}_$TAG.json
    { [ -s "$f" ] && [ -s "$j" ]; } || return 0
    local st med over
    st=$(status_of $FAM $FW $TAG)
    med=$($PY_E4B -c "import statistics, sys
v = []
for ln in open(sys.argv[1]):
    try: v.append(float(ln.split('|')[1].split()[0]))
    except Exception: pass
print(round(statistics.median(v), 2) if v else 'nan')" "$f" 2>/dev/null || echo nan)
    over=$($PY_E4B -c "import sys; m = sys.argv[1]; print(int(m != 'nan' and float(m) > float(sys.argv[2])))" "$med" "$G" 2>/dev/null || echo 0)
    echo "LOADGATE $FAM/$FW/$TAG attempt $n load1_median $med gate $G status $st over $over" | tee -a summary.txt
    [ "$st" = ok ] && [ "$over" = 1 ] && [ $n -lt ${TC1_LOAD_RETRIES:-2} ] || return 0
    can_run 600 $FAM/$FW/$TAG/loadgate_retry || return 0
    n=$((n + 1)); mkdir -p $W/loadvoid
    local x; for x in $j $f $W/vram_${FAM}_${FW}_$TAG.txt logs/run_${FAM}_${FW}_$TAG.log; do [ -e "$x" ] && mv "$x" "$W/loadvoid/$(basename "$x").a$n"; done
    echo "LOADGATE $FAM/$FW/$TAG attempt $((n - 1)) VOID (host load1 median $med > $G): re-run $n of ${TC1_LOAD_RETRIES:-2}" | tee -a summary.txt
  done; }
free_family(){ [ "$TC1_LOCAL_BOX" = 1 ] && { say "local snapshot kept ($1: TC1_LOCAL_SNAPSHOT is the owner's directory, never freed)"; return 0; }
  rm -rf /root/.cache/huggingface/hub/models--$2; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
tokenise(){ local FAM=$1 MID=$2 REV=$3 TEMPLATE_=$4 SEQ_=$5 DATA=$6 DATA_SHA=$7 TOK=$8 EVAL_N_=${9:-$EVAL_N}     # ${10}...: passed to --prepare as they are (TC1 amendment 39: the pack flags)
  say "tokenise $FAM ($TEMPLATE_, seq $SEQ_) -> $(basename $TOK)"
  HF_HUB_OFFLINE=1 $PY_E4B $W/tc1_arm.py --prepare --fam $FAM --model "$MID" --revision $REV --data $DATA --data-sha $DATA_SHA --seq $SEQ_ --eval-n $EVAL_N_ --template $TEMPLATE_ --tokens $TOK "${@:10}" > logs/prepare_${FAM}_$TEMPLATE_.log 2>&1 || return 1
  tail -1 logs/prepare_${FAM}_$TEMPLATE_.log; return 0; }
tok_sha(){ $PY_E4B -c "import json; print(json.load(open('$1'))['sha256'])"; }
# TC1b: the tokens file's sha256 covers {train, eval[:eval_n]} (tc1_arm.py prepare), so a file tokenised with 16 held-out rows cannot carry
# TC1's qwen3 sha (8 rows) even though its TRAIN rows are byte-identical. The train-only sha -- sha256 of json.dumps(train, separators=(',', ':'))
# -- is printed beside it so the registration can assert the training bytes against TC1's file (TP4-PREREG amendment 3's reading).
train_sha(){ $PY_E4B -c "import json, hashlib; print(hashlib.sha256(json.dumps(json.load(open('$1'))['train'], separators=(',', ':')).encode()).hexdigest())"; }
# ---------------------------------------------------------------- the TC1 arm set (TC1-PREREG "Arms, in this order")
dmon_start(){ ( nvidia-smi dmon -s ut -d 1 -o T > $W/logs/dmon_$1.txt 2>/dev/null ) & echo $!; }
dmon_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
# draw2 FAM FW TAG ARM ...: the SECOND draw of an arm -- the same invocation, the same everything, a fresh process, tag <TAG>_d2.
# The reducer reads the pair as one arm with two draws (median over both; STABILITY = |d1-d2|/mean <= 5 % e4b / 10 % others).
draw2(){ local FAM=$1 FW=$2 TAG=$3; shift 3; arm "$FAM" "$FW" "${TAG}_d2" "$@"; }
# todo_arm FAM FW TAG ARM: an arm this harness cut does not implement yet -- a not_run row with the reason, never a silent omission (unused since phase 2; kept for the next cut).
todo_arm(){ stubw "$1" "$2" "$3" "$4" not_run "arm not yet implemented (TC1 follow-up)"; }
UT7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"     # the notebooks' seven targets (TC1-PREREG arm 2)
# TC2 (TC2-PREREG-draft "Arms per family"; bench/tp4/tp4_run.sh's family table): UT4 = attention only, tp4's qwen3_5 target list (the family carries a SHARED
# dense expert); UT_GRANITE2 = tp4 amendment 4's second Unsloth arm naming Granite's ParallelExperts modules (never discovered by the notebooks' seven);
# UP_QWEN3_5 = the routed expert stacks named EXPLICITLY as PEFT target_parameters (tc1_arm.py --unsloth-target-parameters, T24) beside UT4
UT4="q_proj,k_proj,v_proj,o_proj"
UT_GRANITE2="q_proj,k_proj,v_proj,o_proj,input_linear,output_linear"
UP_QWEN3_5="mlp.experts.gate_up_proj,mlp.experts.down_proj"
PROFILE_STEPS=${TC1_PROFILE_STEPS:-3}; PROFILE_WARM=${TC1_PROFILE_WARM:-3}   # arm 10: P45's instrument, 3 warm + 3 profiled
# tc1_fetch_tokenise FAM MID REV FETCH_AL STUBLIST: fetch + tokenise, or stub every arm in STUBLIST (fw:tag:arm words); sets TOK / TS
tc1_prepare(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 ALL=$5 EVN=${6:-$EVAL_N}      # EVN: held-out rows written into the tokens file (TC1b: CURVE_EVAL_N; every arm takes its own --eval-n prefix)
  stub_all(){ local st=$1 why=$2 t fw tag arm; for t in $ALL; do IFS=: read -r fw tag arm <<< "$t"; stubw $FAM $fw $tag $arm $st "$why"; done; }
  if skip $FAM; then say "skip family $FAM (TC1_SKIP)"; stub_all not_run "family skipped by TC1_SKIP"; return 1; fi
  FETCH_REASON=""; local frc
  if [ -n "${TC1_LOCAL_SNAPSHOT:-}" ]; then local_snapshot $FAM $MID $REV; frc=$?; else fetch $FAM $MID $REV $FAL; frc=$?; fi   # TC3: the owned box's directory, never a fetch
  if [ $frc -ne 0 ]; then local st=not_run; [ $frc -eq 2 ] && st=load_fault; stub_all $st "$FETCH_REASON"; free_family $FAM ${MID//\//--}; return 1; fi
  TOK=$W/tokens_$FAM.json
  # TC1 amendment 39: on a packing box, rows of exactly SEQ tokens from the registered text extended in its own order (tp4_alpaca.py left its
  # source beside ds_alpaca.json), at least steps x micro-batch x accum of them -- the rows the arms read, none twice
  local PK=""; [ "$PACK" = 1 ] && PK="--pack 1 --pack-src $W/data/alpaca_data_cleaned.json --pack-min-rows $(( STEPS * MB * ACCUM ))"
  if ! tokenise $FAM "$MID" $REV alpaca $SEQ $W/data/ds_alpaca.json $DS_ALPACA_SHA $TOK $EVN $PK; then
    tail -3 logs/prepare_${FAM}_alpaca.log; echo "$FAM: TOKENS FAIL" | tee -a summary.txt
    local why; why="tokenise failed (logs/prepare_${FAM}_alpaca.log): $(tail -1 logs/prepare_${FAM}_alpaca.log | cut -c1-200)"
    stub_all harness_error "$why"; free_family $FAM ${MID//\//--}; return 1
  fi
  TS=$(tok_sha $TOK); echo "TOKENS $FAM alpaca sha=$TS eval_n=$EVN train_only_sha=$(train_sha $TOK)${PK:+ pack=1 seq=$SEQ}" | tee -a summary.txt; return 0; }
# tc1_frontier_family FAM MID REV FETCH ERES EOFF MB1 UNS HF HOFF AX ALO AZ3 ROFF -- lane TC3 (TC3-PREREG-draft "24 GB box"; registered as bench/tc1/TC3-PREREG.md by
# the PI): Qwen3-30B-A3B at the pin on ONE 24 GB card (TC1_GPU_CLASS=4090), the field recipe, matched init + fp32 adapters, N 20, 8 held-out rows at 0 and N,
# every framework with its own memory lever, one process per arm, IN THIS ORDER (an OOM is a row; every row records peak VRAM, the host-RAM high-water and the host total):
#   1 e4b/fused_attn4_m (resident; expected OOM)   2 e4b/fused_attn4_m_offload (--offload 1: frozen 4-bit experts in pinned host RAM, one layer GPU-resident)
#   3 e4b/fused_attn4_m_mb1 (resident, micro-batch 1 x accum 8)   4 unsloth/ckpt_unsloth_m (venv-unsloth, grouped_mm; expected OOM)   5 unsloth/ckpt_unsloth_m_mb1
#   6 hf/hf_peft_m (expected OOM at load)   7 hf/hf_peft_m_offload (--hf-offload 1: device_map auto + max_memory cap; what landed where and whether it trained ARE the row)
#   8 axolotl/ckpt_axolotl_m (its 4-bit path)   9 axolotl/ckpt_axolotl_m_layeroffload (--axolotl-layer-offload 1: layer_offloading: true, driven as the trainer mixin does)
#   10 axolotl/ckpt_axolotl_m_zero3 (--axolotl-zero3 1: DeepSpeed ZeRO-3 parameter offload without quantize_moe_experts -- a refused row naming why: the engine is the trainer's)
#   11 e4b/reference_attn4_m_offload (the parity / equivalence anchor under offload)
# Alarms (the draft's): e4b resident 1200 (an OOM is quick), offload 3600, mb1 3600, Unsloth 3600 each, HF 1800 / HF offload 3600, axolotl 2700 / 3600 / 3600, reference offload 5400.
tc1_frontier_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 ERAL=$5 EOAL=$6 MBAL=$7 UAL=$8 HAL=$9 HOAL=${10} AAL=${11} ALAL=${12} AZAL=${13} ROAL=${14}
  local ALL="e4b:fused_attn4_m:fused e4b:fused_attn4_m_offload:fused e4b:fused_attn4_m_mb1:fused e4b:fused_attn4_shipped:fused unsloth:ckpt_unsloth_m:unsloth unsloth:ckpt_unsloth_m_mb1:unsloth hf:hf_peft_m:hf hf:hf_peft_m_offload:hf axolotl:ckpt_axolotl_m:axolotl axolotl:ckpt_axolotl_m_layeroffload:axolotl axolotl:ckpt_axolotl_m_zero3:axolotl e4b:reference_attn4_m_offload:reference"
  say "===== FRONTIER family $FAM (TC3; $MID @ $REV; matched seed $MATCHED_SEED; box class $BOX_CLASS; alarms e4b resident $ERAL offload $EOAL mb1 $MBAL unsloth $UAL hf $HAL hf-offload $HOAL axolotl $AAL/$ALAL/$AZAL reference-offload $ROAL)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"      # every arm is a matched arm
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"                   # the notebooks' recipe (tp4's arm); double-quant OFF is the arm's default
  local NOTE=""; [ "$TC1_LOCAL_BOX" = 1 ] && NOTE="TC1_LOCAL_BOX hand run: snapshot ${TC1_LOCAL_SNAPSHOT:-?} (pin_proof: ${PIN_PROOF:-?})"
  can_run 600 $FAM/e4b/fused_m             && arm   $FAM e4b fused_attn4_m fused $ERAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/e4b/fused_m_offload     && arm   $FAM e4b fused_attn4_m_offload fused $EOAL "$MID" $REV 1 field $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/e4b/fused_m_mb1         && arm   $FAM e4b fused_attn4_m_mb1 fused $MBAL "$MID" $REV 0 mb1 $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  # TC3-PREREG amendment 3 (2026-10-02): e4b as shipped (bf16 expert adapters, N(0,1/r) init) RESIDENT -- a FIT row, never a position: tc3-4090-1 showed
  # Unsloth's matched arm fitting resident on 24 GB (24.22 GB peak) while e4b's matched arm (fp32 adapters) OOMs at both recipes; the question a 24 GB owner asks
  local NATIVE="--adapter-dtype native --lora-init native"
  can_run 600 $FAM/e4b/shipped             && arm   $FAM e4b fused_attn4_shipped fused $ERAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/unsloth/m               && arm   $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/unsloth/m_mb1           && arm   $FAM unsloth ckpt_unsloth_m_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/hf/m                    && arm   $FAM hf hf_peft_m hf $HAL "$MID" $REV 0 field $TOK $TS $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/hf/m_offload            && arm   $FAM hf hf_peft_m_offload hf $HOAL "$MID" $REV 0 field $TOK $TS --hf-offload 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/axolotl/m               && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/axolotl/m_layeroffload  && arm   $FAM axolotl ckpt_axolotl_m_layeroffload axolotl $ALAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json --axolotl-layer-offload 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/axolotl/m_zero3         && arm   $FAM axolotl ckpt_axolotl_m_zero3 axolotl $AZAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json --axolotl-zero3 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 900 $FAM/e4b/reference_m_offload && arm   $FAM e4b reference_attn4_m_offload reference $ROAL "$MID" $REV 1 field $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_frontier12_family FAM MID REV FETCH ERES EOFF UNS HF AX ROFF -- lane TC3's owned 12 GB box (TC3-PREREG-draft "12 GB box": the RTX A2000 in container gpu-dev;
# TC1_GPU_CLASS="RTX A2000", run by hand with TC1_LOCAL_BOX=1 -- README "TC3 hand run"). Its host's driver is 575, so every cu130 venv is refused by the driver
# gate: the Unsloth arm runs in venv-unsloth-t28 (UNS_VENV=t28; the venv and its torch land in the receipt's env and in versions.txt) with the loader-default
# backend, and the axolotl arm is a refused row naming the driver. In this order:
#   1 e4b/fused_attn4_m_offload   2 e4b/fused_attn4_m_offload_d2 (the second draw)   3 e4b/reference_attn4_m_offload   4 e4b/fused_attn4_m (resident; expected OOM)
#   5 unsloth/ckpt_unsloth_m_mb1 (t28)   6 hf/hf_peft_m_mb1   7 axolotl/ckpt_axolotl_m (refused: its venv needs cu130 torch)
# Alarms: the 24 GB token's classes (offload 3600, reference offload 5400, resident 1200, Unsloth 3600, HF 1800, axolotl 2700) -- a choice, not the draft's (it names none for this box).
tc1_frontier12_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 ERAL=$5 EOAL=$6 UAL=$7 HAL=$8 AAL=$9 ROAL=${10}
  local NATIVE="--adapter-dtype native --lora-init native"
  local ALL="e4b:fused_attn4_m_offload:fused e4b:fused_attn4_m_offload_d2:fused e4b:reference_attn4_m_offload:reference e4b:fused_attn4_m:fused unsloth:ckpt_unsloth_m_mb1:unsloth hf:hf_peft_m_mb1:hf axolotl:ckpt_axolotl_m:axolotl"
  say "===== FRONTIER-12 family $FAM (TC3, the owned 12 GB box; $MID @ $REV; matched seed $MATCHED_SEED; box class $BOX_CLASS; local_box=$TC1_LOCAL_BOX; alarms e4b resident $ERAL offload $EOAL unsloth $UAL hf $HAL axolotl $AAL reference-offload $ROAL)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local NOTE=""; [ "$TC1_LOCAL_BOX" = 1 ] && NOTE="TC1_LOCAL_BOX hand run: snapshot ${TC1_LOCAL_SNAPSHOT:-?} (pin_proof: ${PIN_PROOF:-?})"
  can_run 600 $FAM/e4b/fused_m_offload     && arm   $FAM e4b fused_attn4_m_offload fused $EOAL "$MID" $REV 1 field $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/e4b/fused_m_offload_d2  && draw2 $FAM e4b fused_attn4_m_offload fused $EOAL "$MID" $REV 1 field $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 900 $FAM/e4b/reference_m_offload && arm   $FAM e4b reference_attn4_m_offload reference $ROAL "$MID" $REV 1 field $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/e4b/fused_m             && arm   $FAM e4b fused_attn4_m fused $ERAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  # TC3-PREREG (12 GB): the mb1 secondary -- micro-batch 1 x accum 8, the same tokens per step, as TC1's -- for the e4b offload pair when the field recipe OOMed
  # on this card (fp32 adapters + their grads + 8-bit Adam states on 642 M parameters alone are ~6.5 GB), one draw each; then the as-shipped e4b under offload
  # (bf16 expert adapters, N(0,1/r) init) as a labelled FIT row: never a position, the question a 12 GB owner asks
  local so; so=$(status_of $FAM e4b fused_attn4_m_offload)
  if [ "$so" = oom ]; then
    echo "SECONDARY $FAM: e4b fused_attn4_m_offload OOMed at the field recipe -> the mb1 offload pair" | tee -a summary.txt
    can_run 600 $FAM/e4b/fused_m_offload_mb1     && arm $FAM e4b fused_attn4_m_offload_mb1 fused $EOAL "$MID" $REV 1 mb1 $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
    can_run 900 $FAM/e4b/reference_m_offload_mb1 && arm $FAM e4b reference_attn4_m_offload_mb1 reference $ROAL "$MID" $REV 1 mb1 $TOK $TS --attn-4bit 1 $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  fi
  can_run 600 $FAM/e4b/shipped_offload       && arm   $FAM e4b fused_attn4_shipped_offload fused $EOAL "$MID" $REV 1 field $TOK $TS --attn-4bit 1 $NATIVE ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/unsloth/m_mb1           && UNS_VENV=t28 arm $FAM unsloth ckpt_unsloth_m_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS $UNS --unsloth-moe-backend default $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/hf/m_mb1                && arm   $FAM hf hf_peft_m_mb1 hf $HAL "$MID" $REV 0 mb1 $TOK $TS $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  can_run 600 $FAM/axolotl/m               && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH ${NOTE:+--note} ${NOTE:+"$NOTE"}
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc2_small_family FAM MID REV FETCH_AL E4B_AL UNS_AL HF_AL AX_AL REF_AL MODE UT2 -- lane TC2's box A token (tc2small; TC2-PREREG-draft "Arms per family"), recipe `small`
# (N 60, 48 held-out rows every 20 steps), one process per arm, in THIS order:
#   MODE normal (granite, olmoe):  1 e4b/fused_attn4_m  2 hf/hf_peft_m  3 e4b/reference_attn4_m  4 e4b/fused_attn4_m_d2  5 hf/hf_peft_m_d2
#      6 unsloth/ckpt_unsloth_m (venv-unsloth grouped_mm, the notebooks' seven)  7 unsloth/ckpt_unsloth_m_experts (UT2, granite only: tp4 amendment 4's second arm --
#      expected attention-only or a bf16-expert PEFT fold, recorded)  8 hf/hf_peft_m_t214 (HF_VENV=t214: the axolotl venv's torch 2.14, experts_implementation=grouped_mm
#      passed if the installed transformers accepts it, what dispatched recorded)  9 axolotl/ckpt_axolotl_m  10 axolotl/ckpt_axolotl_best  11 e4b/fused_attn4_shipped
#   MODE gptoss (as tp4's gptoss MODE): e4b/fused_attn4_m is a REFUSED stub citing tp1/tp2 and e4b/reference_attn4_m a REFUSED stub citing tp4's bias rule -- both written
#      FIRST and refreshed by attn_only_m's own probes; then e4b/attn_only_m (x2: attn_only_m, attn_only_m_d2; --attn-4bit 0, tp4's arm) ; unsloth/ckpt_unsloth_m
#      (--unsloth-load-in-4bit 1: the bnb-4bit load, per-expert Linear4bit on this family) ; unsloth/ckpt_unsloth_mxfp4 (--unsloth-load-in-4bit 0: the 16-bit load that
#      keeps the MXFP4 experts packed; venv-unsloth grouped_mm; x2) ; hf/hf_peft_m ; axolotl/ckpt_axolotl_m. The trainable counts of e4b's attention-only arms and the
#      others' full-expert arms differ BY REGISTRATION: the reducer prints a "no common adapter set" line, never a ratio, on this family.
#   Every matched arm: --adapter-dtype fp32 --lora-init matched:$MATCHED_SEED; double-quant HF ON (the arm default), Unsloth / axolotl OFF (their arm defaults), as TC1.
tc2_small_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 HAL=$7 AAL=$8 RAL=$9 MODE=${10} UT2=${11:-}
  # the small instrument governs EVERY row of this family, the stubs included: `local` shadows the field-recipe globals for stubw / arm / tc1_prepare
  # (bash's dynamic scoping), so a stub written here carries steps=60 like the receipts beside it, and the reducer reads one N per family
  local STEPS=$SMALL_STEPS EVAL_N=$SMALL_EVAL_N EVAL_EVERY=$SMALL_EVAL_EVERY
  local ALL
  if [ "$MODE" = gptoss ]; then
    ALL="e4b:fused_attn4_m:fused e4b:attn_only_m:attn_only e4b:attn_only_m_d2:attn_only unsloth:ckpt_unsloth_m:unsloth unsloth:ckpt_unsloth_mxfp4:unsloth unsloth:ckpt_unsloth_mxfp4_d2:unsloth hf:hf_peft_m:hf axolotl:ckpt_axolotl_m:axolotl e4b:reference_attn4_m:reference"
  else
    ALL="e4b:fused_attn4_m:fused hf:hf_peft_m:hf e4b:reference_attn4_m:reference e4b:fused_attn4_m_d2:fused hf:hf_peft_m_d2:hf unsloth:ckpt_unsloth_m:unsloth ${UT2:+unsloth:ckpt_unsloth_m_experts:unsloth }hf:hf_peft_m_t214:hf axolotl:ckpt_axolotl_m:axolotl axolotl:ckpt_axolotl_best:axolotl e4b:fused_attn4_shipped:fused"
  fi
  say "===== TC2 small family $FAM ($MID @ $REV; mode=$MODE; matched seed $MATCHED_SEED; N $SMALL_STEPS eval every $SMALL_EVAL_EVERY on $SMALL_EVAL_N rows; alarms e4b $EAL unsloth $UAL hf $HAL axolotl $AAL reference $RAL; unsloth targets 2: ${UT2:-none})"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" $SMALL_EVAL_N || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"      # the matched set
  local NATIVE="--adapter-dtype native --lora-init native"                 # the shipped arm: as load_moe_4bit_streaming + add_attention_lora build it
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"                   # the notebooks' recipe (tp4's arm); double-quant OFF is the arm's default
  if [ "$MODE" = gptoss ]; then
    stubw $FAM e4b fused_attn4_m fused refused "SKIPPED as REFUSED: tp1 (P36) + tp2 (P40) rows cited -- enable_fast_train(dgrad=True) patched 0 modules on gpt-oss (experts built bare, no ExpertsLoRA: 'GPT-OSS-aware training LoRA is a separate change', loader.py); attn_only_m is the secondary row and refreshes this stub with its own probe" '{"cited": "tp1,tp2", "n_patched": 0}'
    stubw $FAM e4b reference_attn4_m reference refused "SKIPPED as REFUSED: tp4's gptoss MODE ran no reference arm -- TRAIN_ATTN_4BIT refuses gpt-oss's bias-carrying attention projections (quantize_attention_projections_4bit raises SystemExit on a bias, #435); attn_only_m's attn4 probe refreshes this stub with what it measured" '{"cited": "tp4", "n_patched": 0}'
    can_run 600 $FAM/e4b/attn_only_m    && arm   $FAM e4b attn_only_m attn_only $EAL "$MID" $REV 0 small $TOK $TS --attn-4bit 0 $MATCH
    can_run 600 $FAM/e4b/attn_only_m_d2 && draw2 $FAM e4b attn_only_m attn_only $EAL "$MID" $REV 0 small $TOK $TS --attn-4bit 0 $MATCH
    can_run 600 $FAM/unsloth/m          && arm   $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 small $TOK $TS $UNS --unsloth-moe-backend grouped_mm --unsloth-load-in-4bit 1 $MATCH
    can_run 600 $FAM/unsloth/mxfp4      && arm   $FAM unsloth ckpt_unsloth_mxfp4 unsloth $UAL "$MID" $REV 0 small $TOK $TS $UNS --unsloth-moe-backend grouped_mm --unsloth-load-in-4bit 0 $MATCH
    can_run 600 $FAM/unsloth/mxfp4_d2   && draw2 $FAM unsloth ckpt_unsloth_mxfp4 unsloth $UAL "$MID" $REV 0 small $TOK $TS $UNS --unsloth-moe-backend grouped_mm --unsloth-load-in-4bit 0 $MATCH
    can_run 600 $FAM/hf/m               && arm   $FAM hf hf_peft_m hf $HAL "$MID" $REV 0 small $TOK $TS $MATCH
    can_run 600 $FAM/axolotl/m          && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 small $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH
  else
    can_run 600 $FAM/e4b/fused_m        && arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 small $TOK $TS --attn-4bit 1 $MATCH
    can_run 600 $FAM/hf/m               && arm   $FAM hf hf_peft_m hf $HAL "$MID" $REV 0 small $TOK $TS $MATCH
    can_run 900 $FAM/e4b/reference_m    && arm   $FAM e4b reference_attn4_m reference $RAL "$MID" $REV 0 small $TOK $TS --attn-4bit 1 $MATCH
    can_run 600 $FAM/e4b/fused_m_d2     && draw2 $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 small $TOK $TS --attn-4bit 1 $MATCH
    can_run 600 $FAM/hf/m_d2            && draw2 $FAM hf hf_peft_m hf $HAL "$MID" $REV 0 small $TOK $TS $MATCH
    can_run 600 $FAM/unsloth/m          && arm   $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 small $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
    [ -n "$UT2" ] && can_run 600 $FAM/unsloth/m_experts && arm $FAM unsloth ckpt_unsloth_m_experts unsloth $UAL "$MID" $REV 0 small $TOK $TS --grad-ckpt unsloth --unsloth-targets "$UT2" --unsloth-moe-backend grouped_mm $MATCH
    can_run 600 $FAM/hf/m_t214          && HF_VENV=t214 arm $FAM hf hf_peft_m_t214 hf $HAL "$MID" $REV 0 small $TOK $TS --hf-experts-implementation grouped_mm $MATCH
    can_run 600 $FAM/axolotl/m          && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 small $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH
    can_run 600 $FAM/axolotl/best       && arm   $FAM axolotl ckpt_axolotl_best axolotl $AAL "$MID" $REV 0 small $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json --axolotl-best 1 --adapter-dtype fp32 --lora-init native
    can_run 600 $FAM/e4b/shipped        && arm   $FAM e4b fused_attn4_shipped fused $EAL "$MID" $REV 0 small $TOK $TS --attn-4bit 1 $NATIVE
  fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc2_big_family FAM MID REV FETCH_AL E4B_AL UNS_AL HF_AL AX_AL REF_AL OFFLOAD UT UT2 UP2 -- lane TC2's box B token (tc2big), the field recipe (N 20, 8 rows at 0 and N),
# one process per arm, in THIS order:  1 e4b/fused_attn4_m  2 unsloth/ckpt_unsloth_m  3 e4b/fused_attn4_m_d2  4 unsloth/ckpt_unsloth_m_d2 (the primary pair, two draws)
#   5 unsloth/ckpt_unsloth_m_experts (UT2 as target_modules + UP2 as explicit PEFT target_parameters; qwen3_5 only: the routed expert stacks named beside q/k/v/o)
#   6 hf/hf_peft_m  7 axolotl/ckpt_axolotl_m  8 axolotl/ckpt_axolotl_best  9 e4b/fused_attn4_shipped  10 e4b/reference_attn4_m (LAST: its alarm is the longest)
#   then the _mb1 secondary pair for any framework whose primary matched arm OOMed, as tc1_family. OFFLOAD reaches the e4b arms only (mixtral: --offload 1, as tp4;
#   Unsloth / HF / axolotl resident, as tp4); the Unsloth arms run in venv-unsloth with --unsloth-moe-backend grouped_mm (TC1's comparator configuration).
#   TC2 amendment 8: TC2_PRIMARY_RECIPE (unset = field) is the recipe of the four primary arms (1-4; mb1 = micro-batch 1 x accum 8, the same tokens per
#   step, under the same tags, so the pair stays the reducer's ordinary matched pair; HF / axolotl keep the field recipe), and a primary pair already at
#   mb1 has no _mb1 secondary; TC2_E4B_ARM_ENV (unset = nothing) is handed to every e4b arm as its TC1_ARM_EXTRA_ENV, never to another framework's.
tc2_big_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 HAL=$7 AAL=$8 RAL=$9 OFF=${10} UT=${11} UT2=${12:-} UP2=${13:-}
  local ALL="e4b:fused_attn4_m:fused unsloth:ckpt_unsloth_m:unsloth e4b:fused_attn4_m_d2:fused unsloth:ckpt_unsloth_m_d2:unsloth ${UT2:+unsloth:ckpt_unsloth_m_experts:unsloth }hf:hf_peft_m:hf axolotl:ckpt_axolotl_m:axolotl axolotl:ckpt_axolotl_best:axolotl e4b:fused_attn4_shipped:fused e4b:reference_attn4_m:reference"
  say "===== TC2 big family $FAM ($MID @ $REV; offload=$OFF unsloth_targets=$UT targets2=${UT2:-none} target_parameters=${UP2:-none}; matched seed $MATCHED_SEED; alarms e4b $EAL unsloth $UAL hf $HAL axolotl $AAL reference $RAL)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  # TC2 amendment 4: TC2_UNS_TARGET_PARAMS gives the matched Unsloth arms (both draws) the family's expert target parameters -- unset everywhere else
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT${TC2_UNS_TARGET_PARAMS:+ --unsloth-target-parameters $TC2_UNS_TARGET_PARAMS}"
  local UP2ARG=""; [ -n "$UP2" ] && UP2ARG="--unsloth-target-parameters $UP2"
  # TC2 amendment 8: the primary pair's recipe and the e4b arms' extra environment -- both unset (field, nothing) on every other token
  local PRIM=${TC2_PRIMARY_RECIPE:-field} EENV=${TC2_E4B_ARM_ENV:-}
  [ "$PRIM" = field ] || echo "PRIMARY $FAM: the four primary arms run recipe $PRIM (TC2_PRIMARY_RECIPE); e4b arms' extra env: ${EENV:-none}" | tee -a summary.txt
  can_run 600 $FAM/e4b/fused_m     && TC1_ARM_EXTRA_ENV="$EENV" arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV $OFF $PRIM $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m       && arm   $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 $PRIM $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/e4b/fused_m_d2  && TC1_ARM_EXTRA_ENV="$EENV" draw2 $FAM e4b fused_attn4_m fused $EAL "$MID" $REV $OFF $PRIM $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_d2    && draw2 $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 $PRIM $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  [ -n "$UT2" ] && can_run 600 $FAM/unsloth/m_experts && arm $FAM unsloth ckpt_unsloth_m_experts unsloth $UAL "$MID" $REV 0 field $TOK $TS --grad-ckpt unsloth --unsloth-targets "$UT2" $UP2ARG --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/hf/m            && arm   $FAM hf hf_peft_m hf $HAL "$MID" $REV 0 field $TOK $TS $MATCH
  can_run 600 $FAM/axolotl/m       && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH
  can_run 600 $FAM/axolotl/best    && arm   $FAM axolotl ckpt_axolotl_best axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json --axolotl-best 1 --adapter-dtype fp32 --lora-init native
  can_run 600 $FAM/e4b/shipped     && TC1_ARM_EXTRA_ENV="$EENV" arm   $FAM e4b fused_attn4_shipped fused $EAL "$MID" $REV $OFF field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 900 $FAM/e4b/reference_m && TC1_ARM_EXTRA_ENV="$EENV" arm   $FAM e4b reference_attn4_m reference $RAL "$MID" $REV $OFF field $TOK $TS --attn-4bit 1 $MATCH
  # the secondary pair (as tc1_family): micro-batch 1 x accum 8 -- same tokens per step -- for any framework whose primary matched arm OOMed
  local se su sh; se=$(status_of $FAM e4b fused_attn4_m); su=$(status_of $FAM unsloth ckpt_unsloth_m); sh=$(status_of $FAM hf hf_peft_m)
  if [ "$PRIM" != field ]; then        # TC2 amendment 8: the primary pair already ran micro-batch 1 -- an _mb1 pair would duplicate it
    echo "SECONDARY $FAM: none -- the primary pair ran recipe $PRIM already (e4b=$se unsloth=$su hf=$sh)" | tee -a summary.txt
  elif [ "$se" = oom ] || [ "$su" = oom ] || [ "$sh" = oom ]; then
    echo "SECONDARY $FAM: a primary matched arm OOMed (e4b=$se unsloth=$su hf=$sh) -> mb1 pair" | tee -a summary.txt
    can_run 600 $FAM/e4b/fused_m_mb1 && TC1_ARM_EXTRA_ENV="$EENV" arm $FAM e4b fused_attn4_m_mb1 fused $EAL "$MID" $REV $OFF mb1 $TOK $TS --attn-4bit 1 $MATCH
    can_run 600 $FAM/unsloth/m_mb1   && arm $FAM unsloth ckpt_unsloth_m_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
    [ "$sh" = oom ] && can_run 600 $FAM/hf/m_mb1 && arm $FAM hf hf_peft_m_mb1 hf $HAL "$MID" $REV 0 mb1 $TOK $TS $MATCH
  fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc2_small_box / tc2_big_box: lane TC2's two tokens, each family in the registered order with tp4's per-family ceilings (bench/tp4/tp4_run.sh's table: FETCH FUSED UNS HF REF;
# axolotl = the hf ceiling + 900; TC2-PREREG-draft "Alarms"). gptoss's REF column is tp4's (its reference arm is a stub on this family).
#                   FAM      MID                                        REV                                      FETCH E4B  UNS  HF   AX   REF  MODE   UT2
tc2_small_box(){
  tc2_small_family granite  ibm-granite/granite-3.1-3b-a800m-instruct a02780686e08a03fe0d2679a293b5c74a90efa89 1800 1800 1800 1800 2700 2400 normal "$UT_GRANITE2"
  tc2_small_family olmoe    allenai/OLMoE-1B-7B-0924-Instruct         7f1c97f440f06ce36705e4f2b843edb5925f4498 2400 2400 2400 2400 3300 3000 normal ""
  tc2_small_family gptoss   openai/gpt-oss-20b                        6cee5e81ee83917806bbde320786a8fb61efebee 3000 3600 2400 2400 3300 3600 gptoss ""
}
#                   FAM      MID                                        REV                                      FETCH E4B  UNS  HF   AX   REF  OFF UT     UT2    UP2
tc2_big_box(){
  tc2_big_family   qwen3_5  Qwen/Qwen3.6-35B-A3B                      995ad96eacd98c81ed38be0c5b274b04031597b0 6000 3600 3600 1800 2700 5400 0 "$UT4" "$UT4" "$UP_QWEN3_5"
  tc2_big_family   mixtral  mistralai/Mixtral-8x7B-Instruct-v0.1      eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 5400 2400 1800 2700 6000 1 "$UT7" ""     ""
}
# TC2 amendment 2 (2026-10-02): the Mixtral redraw token (TC1_FAMILIES=tc2mixtral on TC1_BOX=B). Box B's Unsloth matched arms ran out their
# 2,400 s alarm inside load on both draws (rc 142, no step taken), so P5's pair -- e4b under expert offload against Unsloth resident, two draws
# each -- and the reference arm run once more on one box with the Unsloth alarm at 7,200 s. The rows box B already holds (HF, both axolotl arms,
# e4b as shipped) are not re-run: they are skipped here as not_run stubs, the way TC1_SKIP would skip them. Nothing else in the family moves.
tc2_mixtral_redraw(){
  SKIP="$SKIP mixtral/hf/hf_peft_m mixtral/axolotl/ckpt_axolotl_m mixtral/axolotl/ckpt_axolotl_best mixtral/e4b/fused_attn4_shipped"
  tc2_big_family   mixtral  mistralai/Mixtral-8x7B-Instruct-v0.1      eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 5400 7200 1800 2700 6000 1 "$UT7" ""     ""
}
# TC2 amendment 4 (2026-10-02): Qwen3.6-35B-A3B's matched set (926,187,520 fp32 adapters) OOMs e4b's fused path resident on 32 GB
# at both micro-batches (tc1-5090-27), while Unsloth trains it resident given the family's expert target parameters. This token asks
# the registered next row: e4b under expert offload (--offload 1) x2 and the e4b reference under offload, against Unsloth's matched
# arm WITH the expert targets x2 (so the pair is the reducer's ordinary matched pair). HF, both axolotl arms and e4b as shipped are
# not re-run (tc1-5090-27 holds them) and appear as not_run stubs. Nothing else in the family moves.
tc2_qwen35_offload(){
  SKIP="$SKIP qwen3_5/hf/hf_peft_m qwen3_5/axolotl/ckpt_axolotl_m qwen3_5/axolotl/ckpt_axolotl_best qwen3_5/e4b/fused_attn4_shipped"
  TC2_UNS_TARGET_PARAMS="$UP_QWEN3_5"
  tc2_big_family   qwen3_5  Qwen/Qwen3.6-35B-A3B                      995ad96eacd98c81ed38be0c5b274b04031597b0 6000 5400 3600 1800 2700 7200 1 "$UT4" ""     ""
  TC2_UNS_TARGET_PARAMS=""
}
# TC2 amendment 6 (2026-10-04): both big families with EVERY e4b arm RESIDENT (--offload 0) on the 32 GB card, after TC1 amendments
# 10-15 and the training-memory changes since 2026-10-02 (the lean LoRA delta on by default; _ScatterCombine saving the bf16 `down`).
# On 2026-10-02 e4b's fused path OOMed resident on Qwen3.6 (tc1-5090-27) and Mixtral only ever ran under expert offload, while Unsloth
# trained both resident. The pair is the reducer's ordinary matched pair, both resident; a primary arm that OOMs falls to the _mb1 pair
# as in every family. HF, both axolotl arms and e4b as shipped are not re-run (box B and tc1-5090-27 hold them). Mixtral first (the
# nearer fit), then Qwen3.6 with the matched Unsloth arms given the family's expert target parameters (amendment 4's pair).
tc2_resident(){
  SKIP="$SKIP mixtral/hf/hf_peft_m mixtral/axolotl/ckpt_axolotl_m mixtral/axolotl/ckpt_axolotl_best mixtral/e4b/fused_attn4_shipped"
  SKIP="$SKIP qwen3_5/hf/hf_peft_m qwen3_5/axolotl/ckpt_axolotl_m qwen3_5/axolotl/ckpt_axolotl_best qwen3_5/e4b/fused_attn4_shipped"
  tc2_big_family   mixtral  mistralai/Mixtral-8x7B-Instruct-v0.1      eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 3600 3600 1800 2700 5400 0 "$UT7" ""     ""
  TC2_UNS_TARGET_PARAMS="$UP_QWEN3_5"
  tc2_big_family   qwen3_5  Qwen/Qwen3.6-35B-A3B                      995ad96eacd98c81ed38be0c5b274b04031597b0 6000 3600 3600 1800 2700 5400 0 "$UT4" ""     ""
  TC2_UNS_TARGET_PARAMS=""
}
# TC2 amendment 8 (2026-10-04), box M: Mixtral-8x7B alone at the field recipe, every e4b arm RESIDENT at e4b's DEFAULT settings -- this token
# sets no e4b environment, so grouped-nf4-gemm's `auto` takes its dense route for Mixtral's calls off sm_90 (gnf4#463; TC1 amendment 22 read
# it at 0.651x the fused kernels' step). Unsloth resident x2 and e4b x2, then the e4b reference; a primary arm that OOMs falls to the _mb1
# pair as in every family. HF, both axolotl arms and e4b as shipped are not re-run (box B and amendment 6's box hold them): not_run stubs.
tc2_mixtral_resident(){
  SKIP="$SKIP mixtral/hf/hf_peft_m mixtral/axolotl/ckpt_axolotl_m mixtral/axolotl/ckpt_axolotl_best mixtral/e4b/fused_attn4_shipped"
  tc2_big_family   mixtral  mistralai/Mixtral-8x7B-Instruct-v0.1      eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 3600 3600 1800 2700 5400 0 "$UT7" ""     ""
}
# TC2 amendment 8 (2026-10-04), box Q: Qwen3.6-35B-A3B alone, resident, the PRIMARY pair at micro-batch 1 x accum 8 (the same tokens per
# step) under its primary tags, two draws a side: e4b fused_attn4_m with E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1 (e4b arms only: the absmax
# double-quantized and the non-routed projections in NF4, the comparator's bytes -- amendment 7's box F, whose single mb1 draws read 9.24
# vs 18.49 s/step) against Unsloth ckpt_unsloth_m with the family's expert target parameters. The e4b reference (it OOMed resident at
# micro-batch 1 on box F), HF, both axolotl arms and e4b as shipped are not_run stubs; no _mb1 secondary runs. Every knob is reset after.
# The Unsloth arms' --expect-trainable is looked up on fused_attn4_m_mb1 (arm's mb1 recipe), which this box never writes, so they run
# without it; the reducer still reads their trainable count against e4b's fused_attn4_m (R11).
tc2_qwen35_mb1(){
  SKIP="$SKIP qwen3_5/e4b/reference_attn4_m qwen3_5/hf/hf_peft_m qwen3_5/axolotl/ckpt_axolotl_m qwen3_5/axolotl/ckpt_axolotl_best qwen3_5/e4b/fused_attn4_shipped"
  TC2_UNS_TARGET_PARAMS="$UP_QWEN3_5"
  TC2_PRIMARY_RECIPE=mb1
  TC2_E4B_ARM_ENV="E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1"
  tc2_big_family   qwen3_5  Qwen/Qwen3.6-35B-A3B                      995ad96eacd98c81ed38be0c5b274b04031597b0 6000 3600 3600 1800 2700 5400 0 "$UT4" ""     ""
  TC2_E4B_ARM_ENV=""
  TC2_PRIMARY_RECIPE=""
  TC2_UNS_TARGET_PARAMS=""
}
# tc1_nativebest_family FAM MID REV FETCH_AL E4B_AL UNS_AL AX_AL -- TC1 amendment 5 (2026-10-02): each framework's NATIVE-BEST
# configuration on one box, two interleaved draws each -- e4b as shipped (bf16 expert adapters, its own init), axolotl's scattermoe
# native-best, Unsloth's native-best (grouped_mm, speed tilt, its own init) -- plus e4b's matched fused arm as the box's anchor for
# the validity predicates. Not matched work: the adapter precision and init are each framework's own, said on every row.
tc1_nativebest_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 AAL=$7
  local ALL="e4b:fused_attn4_shipped:fused axolotl:ckpt_axolotl_best:axolotl unsloth:ckpt_unsloth_best:unsloth e4b:fused_attn4_shipped_d2:fused axolotl:ckpt_axolotl_best_d2:axolotl unsloth:ckpt_unsloth_best_d2:unsloth e4b:fused_attn4_m:fused"
  say "===== NATIVE-BEST family $FAM ($MID @ $REV; each framework as its users run it, two draws each; e4b fused_m the anchor)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local AXB="--axolotl-dataset $W/data/ds_alpaca.json --axolotl-best 1 --adapter-dtype fp32 --lora-init native"
  local UNB="$UNS --unsloth-moe-backend grouped_mm --unsloth-speed-tilt 1 --adapter-dtype fp32 --lora-init native"
  can_run 600 $FAM/e4b/shipped      && arm   $FAM e4b fused_attn4_shipped fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/axolotl/best     && arm   $FAM axolotl ckpt_axolotl_best axolotl $AAL "$MID" $REV 0 field $TOK $TS $AXB
  can_run 600 $FAM/unsloth/best     && arm   $FAM unsloth ckpt_unsloth_best unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNB
  can_run 600 $FAM/e4b/shipped_d2   && draw2 $FAM e4b fused_attn4_shipped fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/axolotl/best_d2  && draw2 $FAM axolotl ckpt_axolotl_best axolotl $AAL "$MID" $REV 0 field $TOK $TS $AXB
  can_run 600 $FAM/unsloth/best_d2  && draw2 $FAM unsloth ckpt_unsloth_best unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNB
  can_run 600 $FAM/e4b/fused_m      && arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_nativebest200_family FAM MID REV FETCH_AL E4B_AL AX_AL ANCH_AL -- TC1 amendment 8 (2026-10-02): e4b as shipped against axolotl's
# scattermoe native-best over 200 steps of the field recipe (the TC1b curve recipe: held-out every 40 steps on 16 rows), two interleaved
# draws each, then e4b's matched fused arm over the same 200 steps as the box's anchor. tc1-5090-34 found the scattermoe arm paying large
# per-process warm-up costs (216 s at step 1, 61-81 s at steps 3 and 6, spikes up to step 20) that its 11..20 window cannot separate from
# its steady step; P14 reads steps 101..200. Unsloth is not installed on this box (no Unsloth arm).
tc1_nativebest200_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 AAL=$6 MAL=$7
  local ALL="e4b:fused_attn4_shipped_200:fused axolotl:ckpt_axolotl_best_200:axolotl e4b:fused_attn4_shipped_200_d2:fused axolotl:ckpt_axolotl_best_200_d2:axolotl e4b:fused_attn4_m_200:fused"
  say "===== NATIVE-BEST 200 family $FAM ($MID @ $REV; e4b shipped vs axolotl scattermoe over $CURVE_STEPS steps, two draws each; e4b fused_m_200 the anchor)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" $CURVE_EVAL_N || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local AXB="--axolotl-dataset $W/data/ds_alpaca.json --axolotl-best 1 --adapter-dtype fp32 --lora-init native"
  can_run 900 $FAM/e4b/shipped_200     && arm   $FAM e4b fused_attn4_shipped_200 fused $EAL "$MID" $REV 0 curve $TOK $TS --attn-4bit 1 $NATIVE
  can_run 900 $FAM/axolotl/best_200    && arm   $FAM axolotl ckpt_axolotl_best_200 axolotl $AAL "$MID" $REV 0 curve $TOK $TS $AXB
  can_run 900 $FAM/e4b/shipped_200_d2  && draw2 $FAM e4b fused_attn4_shipped_200 fused $EAL "$MID" $REV 0 curve $TOK $TS --attn-4bit 1 $NATIVE
  can_run 900 $FAM/axolotl/best_200_d2 && draw2 $FAM axolotl ckpt_axolotl_best_200 axolotl $AAL "$MID" $REV 0 curve $TOK $TS $AXB
  can_run 900 $FAM/e4b/m_200           && arm   $FAM e4b fused_attn4_m_200 fused $MAL "$MID" $REV 0 curve $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_syncab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 10 (2026-10-03, #945): e4b against itself on one card. The legacy
# grouping with grouped-nf4-gemm's pageable index copies (E4B_GROUPING=legacy GNF4_PINNED_RING=0: 13 host syncs per MoE layer pass)
# against the single-read grouping with the pinned ring (E4B_GROUPING=single GNF4_PINNED_RING=1: 1 sync), on the shipped arm and the
# matched arm, two draws each in ABBA order so neither side always runs first. The values are identical by construction; the
# question is the step time. GNF4_SHA must carry the ring (grouped-nf4-gemm#438); the receipts record which path each arm ran.
tc1_syncab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_legacy:fused e4b:fused_attn4_shipped_sync1:fused e4b:fused_attn4_m_legacy:fused e4b:fused_attn4_m_sync1:fused e4b:fused_attn4_m_sync1_d2:fused e4b:fused_attn4_m_legacy_d2:fused e4b:fused_attn4_shipped_sync1_d2:fused e4b:fused_attn4_shipped_legacy_d2:fused"
  say "===== SYNC A/B family $FAM ($MID @ $REV; legacy grouping + pageable copies vs single-read grouping + pinned ring, #945)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local LEG="E4B_GROUPING=legacy GNF4_PINNED_RING=0" NEW="E4B_GROUPING=single GNF4_PINNED_RING=1"
  can_run 600 $FAM/e4b/shipped_legacy    && TC1_ARM_EXTRA_ENV="$LEG" arm   $FAM e4b fused_attn4_shipped_legacy fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_sync1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_shipped_sync1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_legacy          && TC1_ARM_EXTRA_ENV="$LEG" arm   $FAM e4b fused_attn4_m_legacy fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_sync1           && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_sync1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_sync1_d2        && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_sync1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_legacy_d2       && TC1_ARM_EXTRA_ENV="$LEG" draw2 $FAM e4b fused_attn4_m_legacy fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_sync1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_shipped_sync1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_legacy_d2 && TC1_ARM_EXTRA_ENV="$LEG" draw2 $FAM e4b fused_attn4_shipped_legacy fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_leanab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 13 (2026-10-03, #945): grouped-nf4-gemm's padded LoRA delta with its
# trimmed body (gnf4#440: flat row index, scatter-backward gather, no zero fill, scaling on the gathered rows and skipped at 1) against
# the previous body (NF4_QLORA_LEAN_DELTA=0), both on the post-#945 sync path, on the shipped arm and the matched arm, two draws each in
# ABBA order. Values are identical by construction; the question is the step time. GNF4_SHA must carry the switch (gnf4#440); the
# receipts record which body each arm ran.
tc1_leanab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_lean0:fused e4b:fused_attn4_shipped_lean1:fused e4b:fused_attn4_m_lean0:fused e4b:fused_attn4_m_lean1:fused e4b:fused_attn4_m_lean1_d2:fused e4b:fused_attn4_m_lean0_d2:fused e4b:fused_attn4_shipped_lean1_d2:fused e4b:fused_attn4_shipped_lean0_d2:fused"
  say "===== LEAN-DELTA A/B family $FAM ($MID @ $REV; gnf4's previous padded LoRA delta vs its trimmed body, #945)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="NF4_QLORA_LEAN_DELTA=0" NEW="NF4_QLORA_LEAN_DELTA=1"
  can_run 600 $FAM/e4b/shipped_lean0    && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_shipped_lean0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_lean1    && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_shipped_lean1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_lean0          && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_lean0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_lean1          && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_lean1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_lean1_d2       && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_lean1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_lean0_d2       && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_lean0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_lean1_d2 && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_shipped_lean1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_lean0_d2 && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_shipped_lean0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_tileab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 14 (2026-10-03, #945): grouped-nf4-gemm's prefill M-tile height keyed
# on the largest group (GNF4_PREFILL_TILE_RULE=max, the default) against the cost rule over the actual group sizes (=cost, gnf4#441), on
# the post-#945 sync path with the trimmed LoRA delta, the shipped arm and the matched arm, two draws each in ABBA order. Outputs are
# identical by construction; the question is the step time. GNF4_SHA must carry the rule (gnf4#441); the receipts record which rule each
# arm ran and the tile heights it launched.
tc1_tileab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_tilemax:fused e4b:fused_attn4_shipped_tilecost:fused e4b:fused_attn4_m_tilemax:fused e4b:fused_attn4_m_tilecost:fused e4b:fused_attn4_m_tilecost_d2:fused e4b:fused_attn4_m_tilemax_d2:fused e4b:fused_attn4_shipped_tilecost_d2:fused e4b:fused_attn4_shipped_tilemax_d2:fused"
  say "===== TILE-RULE A/B family $FAM ($MID @ $REV; gnf4's max-keyed prefill M-tile vs the cost rule, #945)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="GNF4_PREFILL_TILE_RULE=max" NEW="GNF4_PREFILL_TILE_RULE=cost"
  can_run 600 $FAM/e4b/shipped_tilemax     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_shipped_tilemax fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tilecost    && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_shipped_tilecost fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_tilemax           && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_tilemax fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_tilecost          && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_tilecost fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_tilecost_d2       && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_tilecost fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_tilemax_d2        && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_tilemax fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_tilecost_d2 && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_shipped_tilecost fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tilemax_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_shipped_tilemax fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_rmsab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 15 (2026-10-03, #945): the frozen RMSNorms through the Hugging Face composite
# (E4B_FUSED_RMSNORM=0, the default) against e4b's fused one-launch-each-way kernel (=1, #961), on the trimmed LoRA delta and the post-#945
# sync path, the shipped arm and the matched arm, two draws each in ABBA order. NOT bit-identical (row reductions in another order), so
# the box reads held-out loss beside the step time. The receipts record whether the fusion was requested, patched and called.
tc1_rmsab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_rms0:fused e4b:fused_attn4_shipped_rms1:fused e4b:fused_attn4_m_rms0:fused e4b:fused_attn4_m_rms1:fused e4b:fused_attn4_m_rms1_d2:fused e4b:fused_attn4_m_rms0_d2:fused e4b:fused_attn4_shipped_rms1_d2:fused e4b:fused_attn4_shipped_rms0_d2:fused"
  say "===== FUSED-RMSNORM A/B family $FAM ($MID @ $REV; the HF RMSNorm composite vs e4b's fused training kernel, #945)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="E4B_FUSED_RMSNORM=0" NEW="E4B_FUSED_RMSNORM=1"
  can_run 600 $FAM/e4b/shipped_rms0    && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_shipped_rms0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_rms1    && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_shipped_rms1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_rms0          && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_rms0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_rms1          && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_rms1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_rms1_d2       && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_rms1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_rms0_d2       && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_rms0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_rms1_d2 && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_shipped_rms1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_rms0_d2 && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_shipped_rms0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_reuseab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 20 (2026-10-04, #945): grouped-nf4-gemm's host reuse inside one MoE layer
# pass off (GNF4_HOST_REUSE=0, the default) against on (=1, gnf4#444: repeated index uploads, the down LoRA delta's plan and the adapters'
# sorted gather backward), on every current default (post-#945 sync path, trimmed delta, cost tile rule, fused RMSNorm), the shipped arm
# and the matched arm, two draws each in ABBA order. Values are identical by construction; the question is the step time. GNF4_SHA must
# carry the flag (gnf4#444); the receipts record the flag in force and the process's hit counts.
tc1_reuseab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_reuse0:fused e4b:fused_attn4_shipped_reuse1:fused e4b:fused_attn4_m_reuse0:fused e4b:fused_attn4_m_reuse1:fused e4b:fused_attn4_m_reuse1_d2:fused e4b:fused_attn4_m_reuse0_d2:fused e4b:fused_attn4_shipped_reuse1_d2:fused e4b:fused_attn4_shipped_reuse0_d2:fused"
  say "===== HOST-REUSE A/B family $FAM ($MID @ $REV; gnf4's per-pass host reuse off vs on, #945)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="GNF4_HOST_REUSE=0" NEW="GNF4_HOST_REUSE=1"
  can_run 600 $FAM/e4b/shipped_reuse0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_shipped_reuse0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_reuse1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_shipped_reuse1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_reuse0           && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_reuse0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_reuse1           && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_reuse1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_reuse1_d2        && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_reuse1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_reuse0_d2        && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_reuse0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_reuse1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_shipped_reuse1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_reuse0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_shipped_reuse0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_keepab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 21 (2026-10-04, #945): e4b's whole-layer gradient checkpointing (the
# default) against keeping the MoE activations of the last n decoder layers (E4B_MOE_KEEP_LAYERS=n with grouped-nf4-gemm's
# NF4_QLORA_COMPACT_DELTA=1, e4b#1007 / gnf4#445): n = 32 on the shipped arm, 16 on the matched arm (sized to the 5090's headroom),
# two draws each in ABBA order. Gradients are identical by construction; the questions are the step time and the peak memory.
tc1_keepab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_keep0:fused e4b:fused_attn4_shipped_keep1:fused e4b:fused_attn4_m_keep0:fused e4b:fused_attn4_m_keep1:fused e4b:fused_attn4_m_keep1_d2:fused e4b:fused_attn4_m_keep0_d2:fused e4b:fused_attn4_shipped_keep1_d2:fused e4b:fused_attn4_shipped_keep0_d2:fused"
  say "===== MOE-KEEP A/B family $FAM ($MID @ $REV; whole-layer checkpointing vs keeping the last n layers' MoE activations, #945)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="E4B_MOE_KEEP_LAYERS=0" SHIP_NEW="E4B_MOE_KEEP_LAYERS=32 NF4_QLORA_COMPACT_DELTA=1" M_NEW="E4B_MOE_KEEP_LAYERS=16 NF4_QLORA_COMPACT_DELTA=1"
  can_run 600 $FAM/e4b/shipped_keep0     && TC1_ARM_EXTRA_ENV="$OLD"      arm   $FAM e4b fused_attn4_shipped_keep0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_keep1     && TC1_ARM_EXTRA_ENV="$SHIP_NEW" arm   $FAM e4b fused_attn4_shipped_keep1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_keep0           && TC1_ARM_EXTRA_ENV="$OLD"      arm   $FAM e4b fused_attn4_m_keep0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_keep1           && TC1_ARM_EXTRA_ENV="$M_NEW"    arm   $FAM e4b fused_attn4_m_keep1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_keep1_d2        && TC1_ARM_EXTRA_ENV="$M_NEW"    draw2 $FAM e4b fused_attn4_m_keep1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_keep0_d2        && TC1_ARM_EXTRA_ENV="$OLD"      draw2 $FAM e4b fused_attn4_m_keep0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_keep1_d2  && TC1_ARM_EXTRA_ENV="$SHIP_NEW" draw2 $FAM e4b fused_attn4_shipped_keep1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_keep0_d2  && TC1_ARM_EXTRA_ENV="$OLD"      draw2 $FAM e4b fused_attn4_shipped_keep0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_denseab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 22 (2026-10-04): grouped-nf4-gemm's dense training GEMM route
# (GNF4_TRAIN_GEMM=dense, gnf4#459: one present expert dequantized at a time, its GEMM through torch.mm) against its fused 4-bit kernels
# (=fused), on the matched arm only (fp32 adapters, matched init), two draws a side in ABBA order, every other setting the default.
# GNF4_SHA must carry the route (gnf4#459); each receipt's route_ab record names the route in force and the dense_fwd / dense_dgrad counts.
tc1_denseab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_dense0:fused e4b:fused_attn4_m_dense1:fused e4b:fused_attn4_m_dense1_d2:fused e4b:fused_attn4_m_dense0_d2:fused"
  say "===== DENSE-ROUTE A/B family $FAM ($MID @ $REV; gnf4's fused 4-bit kernels vs its dense route, matched arm, amendment 22)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local OLD="GNF4_TRAIN_GEMM=fused" NEW="GNF4_TRAIN_GEMM=dense"
  can_run 600 $FAM/e4b/m_dense0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_dense0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dense1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_dense1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dense1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_dense1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dense0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_dense0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_mixtral_denseab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 22 (2026-10-04): the same A/B on Mixtral-8x7B-Instruct at TC2's pin
# and field recipe (N 20, 8 held-out rows at 0 and N), prepared as tc2_big_family prepares mixtral (tc1_prepare: the pinned Alpaca text,
# the alpaca template, seq $SEQ, $EVAL_N held-out rows -- so the tokens and held-out rows are TC2's own), every e4b arm RESIDENT (--offload 0)
# with E4B_ABSMAX_DQ=1 on BOTH sides: the double-quantized expert absmax leaves room for the dense route's one-expert transient (235 MB at
# gate_up). Each receipt records absmax_dq beside its route_ab record.
tc1_mixtral_denseab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_dense0:fused e4b:fused_attn4_m_dense1:fused e4b:fused_attn4_m_dense1_d2:fused e4b:fused_attn4_m_dense0_d2:fused"
  say "===== DENSE-ROUTE A/B family $FAM ($MID @ $REV; resident, E4B_ABSMAX_DQ=1 both sides; gnf4's fused 4-bit kernels vs its dense route, matched arm, amendment 22)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local OLD="GNF4_TRAIN_GEMM=fused E4B_ABSMAX_DQ=1" NEW="GNF4_TRAIN_GEMM=dense E4B_ABSMAX_DQ=1"
  can_run 600 $FAM/e4b/m_dense0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_dense0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dense1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_dense1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dense1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_dense1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dense0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_dense0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_memcensus_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 23 (2026-10-04): where e4b's resident training memory goes, against
# Unsloth's. Qwen3-30B-A3B at the pin, TC1's tokens, the matched set at the mb1 recipe (micro-batch 1 x accum 8: the same tokens per step), one
# draw per arm, IN THIS ORDER, each with tc1_arm.py's memory census on (--mem-census 1): e4b fused_attn4_m_mb1 (defaults: the fp32 expert
# absmax), e4b fused_attn4_m_mb1_dq (E4B_ABSMAX_DQ=1: #1040's double-quantized absmax), Unsloth ckpt_unsloth_m_mb1 (TC1's qwen3 Unsloth arm:
# the notebooks' seven targets, grouped_mm, venv-unsloth). No speed is read: the census slows the step. The token is in neither NEED_UNSLOTH=0
# list, so the box builds the Unsloth venvs as for every token that runs an Unsloth arm.
tc1_memcensus_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_mb1:fused e4b:fused_attn4_m_mb1_dq:fused unsloth:ckpt_unsloth_m_mb1:unsloth"
  say "===== MEMORY CENSUS family $FAM ($MID @ $REV; e4b fp32 absmax, e4b double-quantized absmax, Unsloth; micro-batch 1 x accum 8; --mem-census 1; amendment 23)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"                   # TC1's qwen3 Unsloth arm (tc1_family); double-quant OFF is the arm's default
  local CEN="--mem-census 1"
  can_run 600 $FAM/e4b/m_mb1     && arm $FAM e4b fused_attn4_m_mb1 fused $EAL "$MID" $REV 0 mb1 $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/e4b/m_mb1_dq  && TC1_ARM_EXTRA_ENV="E4B_ABSMAX_DQ=1" arm $FAM e4b fused_attn4_m_mb1_dq fused $EAL "$MID" $REV 0 mb1 $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/unsloth/m_mb1 && arm $FAM unsloth ckpt_unsloth_m_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $CEN
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_memc4k_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 47 (2026-10-06): amendment 23's memory census on amendment 39's packed
# rows (4,096 real tokens, micro-batch 1 x accum 4, TC1_PACK=1), where e4b peaked 7.6 GB above Unsloth (amendment 43). One draw per arm, IN THIS
# ORDER, each with tc1_arm.py's census on (--mem-census 1), every arm in venv-unsloth: e4b fused_attn4_m_p4 (the library's defaults: the fp32
# expert absmax, the padded LoRA delta, the chunked LM loss as `auto`), e4b fused_attn4_m_p4_lev (E4B_ABSMAX_DQ=1 + NF4_QLORA_COMPACT_DELTA=1:
# e4b's two memory levers), Unsloth ckpt_unsloth_m_p4 (TC1's qwen3 Unsloth arm, grouped_mm). No speed is read: the census slows the step.
tc1_memc4k_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_p4:fused e4b:fused_attn4_m_p4_lev:fused unsloth:ckpt_unsloth_m_p4:unsloth"
  say "===== PACKED MEMORY CENSUS family $FAM ($MID @ $REV; e4b defaults, e4b absmax-dq + compact delta, Unsloth; packed 4,096-token rows; --mem-census 1; amendment 47)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local CEN="--mem-census 1"
  can_run 600 $FAM/e4b/m_p4      && E4B_VENV=t212 arm $FAM e4b fused_attn4_m_p4 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/e4b/m_p4_lev  && TC1_ARM_EXTRA_ENV="E4B_ABSMAX_DQ=1 NF4_QLORA_COMPACT_DELTA=1" E4B_VENV=t212 arm $FAM e4b fused_attn4_m_p4_lev fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/unsloth/m_p4  && arm $FAM unsloth ckpt_unsloth_m_p4 unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $CEN
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_bmmab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 24 (2026-10-04): the RTX 5090's fp32 bmm host cost. First a replay, no model:
# bmm_bench.py (staged with routecalls-qwen3.json by TC1_EXTRA_STAGE) under venv-e4b (torch 2.8.0+cu128) and, when TC1's t212 install held,
# venv-unsloth (torch 2.12.1+cu130) -> BMMBENCH-t28.json / BMMBENCH-t212.json. Then the training A/B on TC1's qwen3 tokens and recipe: the
# matched arm and the shipped arm, each _tv0 (venv-e4b) vs _tv1 (venv-unsloth + e4b and gnf4 at the box's pins, TC1's t212 row), two
# draws a side in ABBA order. Every other setting is the default.
tc1_bmmab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_tv0:fused e4b:fused_attn4_m_tv1:fused e4b:fused_attn4_m_tv1_d2:fused e4b:fused_attn4_m_tv0_d2:fused e4b:fused_attn4_shipped_tv0:fused e4b:fused_attn4_shipped_tv1:fused e4b:fused_attn4_shipped_tv1_d2:fused e4b:fused_attn4_shipped_tv0_d2:fused"
  say "===== BMM A/B family $FAM ($MID @ $REV; the padded LoRA delta's bmm replay, then venv-e4b vs venv-unsloth on the matched and shipped arms, amendment 24)"
  local f ok=1; for f in bmm_bench.py routecalls-qwen3.json; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; echo "BMMBENCH STAGE MISSING $f" | tee -a summary.txt; ok=0; }; done
  if [ $ok = 1 ] && can_run 600 $FAM/bmmbench; then
    perl -e "alarm 900; exec @ARGV" $PY_E4B -u $W/bmm_bench.py $W/routecalls-qwen3.json $W/BMMBENCH-t28.json --label t28 > logs/run_${FAM}_bmmbench_t28.log 2>&1
    echo "BMMBENCH t28 rc=$?" | tee -a summary.txt; grep '^BMM ' logs/run_${FAM}_bmmbench_t28.log | sed 's/^/    /' | tee -a summary.txt
    if [ "$T212_OK" = 1 ]; then
      perl -e "alarm 900; exec @ARGV" $PY_UNS -u $W/bmm_bench.py $W/routecalls-qwen3.json $W/BMMBENCH-t212.json --label t212 > logs/run_${FAM}_bmmbench_t212.log 2>&1
      echo "BMMBENCH t212 rc=$?" | tee -a summary.txt; grep '^BMM ' logs/run_${FAM}_bmmbench_t212.log | sed 's/^/    /' | tee -a summary.txt
    else
      echo "BMMBENCH t212 NOT RUN: ${T212_REASON:-the t212 install did not hold}" | tee -a summary.txt
    fi
  fi
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  can_run 600 $FAM/e4b/m_tv0         && arm                 $FAM e4b fused_attn4_m_tv0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_tv1         && E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_tv1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_tv1_d2      && E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_tv1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_tv0_d2      && draw2               $FAM e4b fused_attn4_m_tv0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_tv0   && arm                 $FAM e4b fused_attn4_shipped_tv0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tv1   && E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_tv1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tv1_d2 && E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_tv1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tv0_d2 && draw2              $FAM e4b fused_attn4_shipped_tv0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_samestack_family FAM MID REV FETCH_AL E4B_AL UNS_AL REF_AL -- TC1 amendment 25 (2026-10-04): TC1's matched set with both frameworks on one
# stack -- e4b's anchor, its second draw and its reference in venv-unsloth (torch 2.12.1+cu130, transformers 5.5.0: TC1's t212 install), Unsloth as
# TC1 -- and e4b's matched arm in venv-e4b (the field image) as _t28, two draws, the environment pair read ABBA around Unsloth's draws.
tc1_samestack_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 RAL=$7
  local ALL="e4b:fused_attn4_m:fused unsloth:ckpt_unsloth_m:unsloth e4b:reference_attn4_m:reference e4b:fused_attn4_m_t28:fused e4b:fused_attn4_m_t28_d2:fused unsloth:ckpt_unsloth_m_d2:unsloth e4b:fused_attn4_m_d2:fused"
  say "===== SAME-STACK family $FAM ($MID @ $REV; e4b and Unsloth both in venv-unsloth, e4b's field-image arm beside it; amendment 25)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  can_run 600 $FAM/e4b/fused_m        && E4B_VENV=t212 arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m          && arm                 $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 900 $FAM/e4b/reference_m    && E4B_VENV=t212 arm   $FAM e4b reference_attn4_m reference $RAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/fused_m_t28    && arm                 $FAM e4b fused_attn4_m_t28 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/fused_m_t28_d2 && draw2               $FAM e4b fused_attn4_m_t28 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_d2       && draw2               $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/e4b/fused_m_d2     && E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_ompab_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 45 (2026-10-05): OMP_NUM_THREADS at the host's physical cores (om0,
# every TC1 box so far) vs at the container's CPU allotment (om1, $ALLOT above), e4b's matched arm and Unsloth's, both in venv-unsloth, two
# draws a side in ABBA order. A host whose allotment is not below its physical cores refused at setup (rc 18) before any install.
tc1_ompab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_om0:fused e4b:fused_attn4_m_om1:fused unsloth:ckpt_unsloth_m_om0:unsloth unsloth:ckpt_unsloth_m_om1:unsloth unsloth:ckpt_unsloth_m_om1_d2:unsloth unsloth:ckpt_unsloth_m_om0_d2:unsloth e4b:fused_attn4_m_om1_d2:fused e4b:fused_attn4_m_om0_d2:fused"
  say "===== THREADS A/B family $FAM ($MID @ $REV; OMP_NUM_THREADS $PHYS (physical cores) vs $ALLOT (the container's allotment), e4b and Unsloth in venv-unsloth, amendment 45)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local OLD="OMP_NUM_THREADS=$PHYS" NEW="OMP_NUM_THREADS=$ALLOT"
  can_run 600 $FAM/e4b/m_om0         && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_om0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_om1         && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_om1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_om0     && TC1_ARM_EXTRA_ENV="$OLD" arm                 $FAM unsloth ckpt_unsloth_m_om0 unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/unsloth/m_om1     && TC1_ARM_EXTRA_ENV="$NEW" arm                 $FAM unsloth ckpt_unsloth_m_om1 unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/unsloth/m_om1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2               $FAM unsloth ckpt_unsloth_m_om1 unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/unsloth/m_om0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2               $FAM unsloth ckpt_unsloth_m_om0 unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/e4b/m_om1_d2      && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_om1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_om0_d2      && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_om0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_prebindab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 26 (2026-10-04): the prebound Triton launches (E4B_TRITON_PREBIND +
# GNF4_TRITON_PREBIND, e4b#1078 + grouped-nf4-gemm#468; same compiled kernels, values identical) off vs on, the shipped and the matched arm, two
# draws a side in ABBA order, venv-e4b (torch 2.8.0+cu128, triton 3.4: a version the prebound path covers), every other default.
tc1_prebindab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_pb0:fused e4b:fused_attn4_shipped_pb1:fused e4b:fused_attn4_m_pb0:fused e4b:fused_attn4_m_pb1:fused e4b:fused_attn4_m_pb1_d2:fused e4b:fused_attn4_m_pb0_d2:fused e4b:fused_attn4_shipped_pb1_d2:fused e4b:fused_attn4_shipped_pb0_d2:fused"
  say "===== PREBIND A/B family $FAM ($MID @ $REV; Triton launches prebound off vs on, e4b + grouped-nf4-gemm, amendment 26)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="E4B_TRITON_PREBIND=0 GNF4_TRITON_PREBIND=0" NEW="E4B_TRITON_PREBIND=1 GNF4_TRITON_PREBIND=1"
  can_run 600 $FAM/e4b/shipped_pb0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_shipped_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_pb1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_shipped_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_pb0           && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pb1           && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pb1_d2        && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pb0_d2        && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_pb1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_shipped_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_pb0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_shipped_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_prebind37_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 35 (2026-10-05): amendment 26's prebound-launch A/B under triton 3.7.1,
# which the prebound path covers since experts4bit-qlora#1108 and grouped-nf4-gemm#471. The same eight arms in the same ABBA order, every one in
# venv-unsloth with e4b and grouped-nf4-gemm at the box's pins (TC1's t212 install: torch 2.12.1+cu130, transformers 5.5.0, triton 3.7.1).
tc1_prebind37_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_pb0:fused e4b:fused_attn4_shipped_pb1:fused e4b:fused_attn4_m_pb0:fused e4b:fused_attn4_m_pb1:fused e4b:fused_attn4_m_pb1_d2:fused e4b:fused_attn4_m_pb0_d2:fused e4b:fused_attn4_shipped_pb1_d2:fused e4b:fused_attn4_shipped_pb0_d2:fused"
  say "===== PREBIND A/B on triton 3.7 family $FAM ($MID @ $REV; Triton launches prebound off vs on in venv-unsloth, e4b + grouped-nf4-gemm, amendment 35)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="E4B_TRITON_PREBIND=0 GNF4_TRITON_PREBIND=0" NEW="E4B_TRITON_PREBIND=1 GNF4_TRITON_PREBIND=1"
  can_run 600 $FAM/e4b/shipped_pb0     && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_pb1     && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_pb0           && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pb1           && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pb1_d2        && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pb0_d2        && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_pb1_d2  && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_pb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_pb0_d2  && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_pb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_compactab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 36 (2026-10-05): grouped-nf4-gemm's compact padded LoRA delta
# (NF4_QLORA_COMPACT_DELTA=1, gnf4#445: saves its input, not its padded block; same values) off vs on, the shipped and the matched arm, two
# draws a side in ABBA order, every one in venv-unsloth with e4b and grouped-nf4-gemm at the box's pins (TC1's t212 install), every other default.
tc1_compactab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_cd0:fused e4b:fused_attn4_shipped_cd1:fused e4b:fused_attn4_m_cd0:fused e4b:fused_attn4_m_cd1:fused e4b:fused_attn4_m_cd1_d2:fused e4b:fused_attn4_m_cd0_d2:fused e4b:fused_attn4_shipped_cd1_d2:fused e4b:fused_attn4_shipped_cd0_d2:fused"
  say "===== COMPACT-DELTA A/B family $FAM ($MID @ $REV; the padded LoRA delta saving its block vs its input, venv-unsloth, amendment 36)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="NF4_QLORA_COMPACT_DELTA=0" NEW="NF4_QLORA_COMPACT_DELTA=1"
  can_run 600 $FAM/e4b/shipped_cd0     && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_cd0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_cd1     && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_cd1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_cd0           && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_cd0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_cd1           && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_cd1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_cd1_d2        && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_cd1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_cd0_d2        && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_cd0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_cd1_d2  && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_cd1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_cd0_d2  && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_cd0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_chunkab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 41 (2026-10-05): e4b's chunked LM loss (E4B_CHUNKED_LM_LOSS, #1142: the
# loss over token chunks, the full-vocabulary logits never materialised) off vs on at the field recipe, the shipped and the matched arm, two
# draws a side in ABBA order, every arm in venv-unsloth (TC1's t212 install), every other default -- its default decision.
tc1_chunkab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_ce0:fused e4b:fused_attn4_shipped_ce1:fused e4b:fused_attn4_m_ce0:fused e4b:fused_attn4_m_ce1:fused e4b:fused_attn4_m_ce1_d2:fused e4b:fused_attn4_m_ce0_d2:fused e4b:fused_attn4_shipped_ce1_d2:fused e4b:fused_attn4_shipped_ce0_d2:fused"
  say "===== CHUNKED-LOSS A/B family $FAM ($MID @ $REV; e4b's LM loss stock vs chunked, venv-unsloth, amendment 41)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="E4B_CHUNKED_LM_LOSS=0" NEW="E4B_CHUNKED_LM_LOSS=1"
  can_run 600 $FAM/e4b/shipped_ce0     && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_ce0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_ce1     && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_ce1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_ce0           && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_ce0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_ce1           && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_ce1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_ce1_d2        && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_ce1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_ce0_d2        && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_ce0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_ce1_d2  && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_ce1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_ce0_d2  && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_ce0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_chunkauto_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 44 (2026-10-05): amendment 41's box with E4B_CHUNKED_LM_LOSS=auto (#1178:
# chunk a training forward only when its stock fp32 logits would reach 1 GiB) in place of =1 -- off vs auto at the field recipe, the shipped
# and the matched arm, two draws a side in ABBA order, every arm in venv-unsloth (TC1's t212 install), every other default.
tc1_chunkauto_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_ca0:fused e4b:fused_attn4_shipped_ca1:fused e4b:fused_attn4_m_ca0:fused e4b:fused_attn4_m_ca1:fused e4b:fused_attn4_m_ca1_d2:fused e4b:fused_attn4_m_ca0_d2:fused e4b:fused_attn4_shipped_ca1_d2:fused e4b:fused_attn4_shipped_ca0_d2:fused"
  say "===== CHUNKED-LOSS AUTO A/B family $FAM ($MID @ $REV; e4b's LM loss stock vs auto, venv-unsloth, amendment 44)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="E4B_CHUNKED_LM_LOSS=0" NEW="E4B_CHUNKED_LM_LOSS=auto"
  can_run 600 $FAM/e4b/shipped_ca0     && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_ca0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_ca1     && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_ca1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_ca0           && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_ca0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_ca1           && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_ca1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_ca1_d2        && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_ca1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_ca0_d2        && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_ca0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_ca1_d2  && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_ca1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_ca0_d2  && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_ca0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_padbk_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 48 (2026-10-06): grouped-nf4-gemm's bucketed LoRA-delta padding
# (NF4_QLORA_PAD_BUCKETS=1, grouped-nf4-gemm#490: each bucket of groups within 2x in rows padded to its own widest, not every group to the
# hottest) off vs on, on amendment 39's packed rows -- the shipped and the matched arm, two draws a side in ABBA order, every arm in
# venv-unsloth at e4b's defaults (the chunked LM loss as `auto` chunks these rows).
tc1_padbk_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_pk0:fused e4b:fused_attn4_shipped_pk1:fused e4b:fused_attn4_m_pk0:fused e4b:fused_attn4_m_pk1:fused e4b:fused_attn4_m_pk1_d2:fused e4b:fused_attn4_m_pk0_d2:fused e4b:fused_attn4_shipped_pk1_d2:fused e4b:fused_attn4_shipped_pk0_d2:fused"
  say "===== PAD-BUCKETS A/B family $FAM ($MID @ $REV; grouped-nf4-gemm's LoRA delta one padded block vs buckets, packed rows, venv-unsloth, amendment 48)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="NF4_QLORA_PAD_BUCKETS=0" NEW="NF4_QLORA_PAD_BUCKETS=1"
  can_run 600 $FAM/e4b/shipped_pk0     && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_pk0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_pk1     && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_pk1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_pk0           && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_pk0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pk1           && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_pk1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pk1_d2        && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_pk1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_pk0_d2        && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_pk0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_pk1_d2  && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_pk1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_pk0_d2  && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_pk0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_padbk28_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 52 (2026-10-06): amendment 48's packed-row bucketing A/B in the FIELD
# IMAGE's environment (venv-e4b: torch 2.8.0+cu128, triton 3.4), where amendment 51 read e4b at its defaults 0.739 of its torch-2.12 step.
tc1_padbk28_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_k0:fused e4b:fused_attn4_shipped_k1:fused e4b:fused_attn4_m_k0:fused e4b:fused_attn4_m_k1:fused e4b:fused_attn4_m_k1_d2:fused e4b:fused_attn4_m_k0_d2:fused e4b:fused_attn4_shipped_k1_d2:fused e4b:fused_attn4_shipped_k0_d2:fused"
  say "===== PAD-BUCKETS A/B (torch 2.8) family $FAM ($MID @ $REV; one padded block vs buckets, packed rows, venv-e4b, amendment 52)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="NF4_QLORA_PAD_BUCKETS=0" NEW="NF4_QLORA_PAD_BUCKETS=1"
  can_run 600 $FAM/e4b/shipped_k0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_shipped_k0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_k1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_shipped_k1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_k0           && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_k0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_k1           && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_k1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_k1_d2        && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_k1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_k0_d2        && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_k0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_k1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_shipped_k1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_k0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_shipped_k0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_prof28_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 53 (2026-10-06): where the field image's torch 2.8 spends its extra time
# on packed rows at e4b's defaults. The TC1 profile instrument (3 warm + 3 profiled steps, before the timed window's steps 11..N) on the
# matched arm in venv-unsloth (torch 2.12) and venv-e4b (torch 2.8), both at their defaults (buckets `auto`), and in venv-e4b with the
# single block (NF4_QLORA_PAD_BUCKETS=0) beside; two draws each in A B C C B A order. No per-micro-batch timing: its syncs would drain the
# queue the profile is measuring.
tc1_prof28_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_q212:fused e4b:fused_attn4_m_q28:fused e4b:fused_attn4_m_q28k0:fused e4b:fused_attn4_m_q28k0_d2:fused e4b:fused_attn4_m_q28_d2:fused e4b:fused_attn4_m_q212_d2:fused"
  say "===== PROFILE (torch 2.12 vs 2.8) family $FAM ($MID @ $REV; e4b's matched arm at its defaults, packed rows, the profile instrument, amendment 53)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local PROF="--profile-steps $PROFILE_STEPS --profile-warm $PROFILE_WARM"
  local K0="NF4_QLORA_PAD_BUCKETS=0"
  can_run 600 $FAM/e4b/m_q212        && E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_q212 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_q28                          && arm   $FAM e4b fused_attn4_m_q28 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_q28k0       && TC1_ARM_EXTRA_ENV="$K0" arm   $FAM e4b fused_attn4_m_q28k0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_q28k0_d2    && TC1_ARM_EXTRA_ENV="$K0" draw2 $FAM e4b fused_attn4_m_q28k0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_q28_d2                       && draw2 $FAM e4b fused_attn4_m_q28 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_q212_d2     && E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_q212 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_ladder28_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 54 (2026-10-06): amendment 53 put 60 % of torch 2.8's extra time at
# e4b's defaults in host time inside the bucketed delta's batched matmuls. Two remedies against the defaults, on the matched arm in venv-e4b
# (torch 2.8) on packed rows, with the profile instrument: c0 the defaults; c1 cuBLASLt's heuristics cache raised to 262,144 entries (no
# code change); c2 grouped-nf4-gemm's bucket ladder (NF4_QLORA_PAD_BUCKETS_LADDER=1, #498). Two draws each in A B C C B A order.
tc1_ladder28_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_c0:fused e4b:fused_attn4_m_c1:fused e4b:fused_attn4_m_c2:fused e4b:fused_attn4_m_c2_d2:fused e4b:fused_attn4_m_c1_d2:fused e4b:fused_attn4_m_c0_d2:fused"
  say "===== LADDER (torch 2.8) family $FAM ($MID @ $REV; e4b's matched arm at its defaults vs the cuBLASLt cache vs the bucket ladder, packed rows, venv-e4b, amendment 54)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local PROF="--profile-steps $PROFILE_STEPS --profile-warm $PROFILE_WARM"
  local C1="CUBLASLT_HEURISTICS_CACHE_CAPACITY=262144" C2="NF4_QLORA_PAD_BUCKETS_LADDER=1"
  can_run 600 $FAM/e4b/m_c0                              && arm   $FAM e4b fused_attn4_m_c0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_c1    && TC1_ARM_EXTRA_ENV="$C1" arm   $FAM e4b fused_attn4_m_c1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_c2    && TC1_ARM_EXTRA_ENV="$C2" arm   $FAM e4b fused_attn4_m_c2 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_c2_d2 && TC1_ARM_EXTRA_ENV="$C2" draw2 $FAM e4b fused_attn4_m_c2 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_c1_d2 && TC1_ARM_EXTRA_ENV="$C1" draw2 $FAM e4b fused_attn4_m_c1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  can_run 600 $FAM/e4b/m_c0_d2                           && draw2 $FAM e4b fused_attn4_m_c0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_memc4kb_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 55 (2026-10-06): amendment 47's packed-row memory census at the
# CURRENT defaults (bucketed padding `auto` since grouped-nf4-gemm#492) -- e4b defaults, e4b with the double-quantized expert absmax, Unsloth;
# one draw each, --mem-census 1, every arm in venv-unsloth. No speed is read.
tc1_memc4kb_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_p4d:fused e4b:fused_attn4_m_p4d_dq:fused unsloth:ckpt_unsloth_m_p4d:unsloth"
  say "===== PACKED MEMORY CENSUS (current defaults) family $FAM ($MID @ $REV; e4b defaults, e4b absmax-dq, Unsloth; --mem-census 1; amendment 55)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local CEN="--mem-census 1"
  can_run 600 $FAM/e4b/m_p4d     && E4B_VENV=t212 arm $FAM e4b fused_attn4_m_p4d fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/e4b/m_p4d_dq  && TC1_ARM_EXTRA_ENV="E4B_ABSMAX_DQ=1" E4B_VENV=t212 arm $FAM e4b fused_attn4_m_p4d_dq fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/unsloth/m_p4d && arm $FAM unsloth ckpt_unsloth_m_p4d unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $CEN
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_dqpack_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 56 (2026-10-06): the double-quantized expert absmax as a library default
# on packed rows -- e4b's matched arm with the fp32 absmax (a0) against E4B_ABSMAX_DQ=1 (a1), two draws a side in ABBA order, and Unsloth's matched
# arm (one draw), every arm in venv-unsloth with --phase-peaks 1 so each run's peak is split into setup / eval / train.
tc1_dqpack_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_a0:fused e4b:fused_attn4_m_a1:fused e4b:fused_attn4_m_a1_d2:fused e4b:fused_attn4_m_a0_d2:fused unsloth:ckpt_unsloth_m_pp:unsloth"
  say "===== ABSMAX-DQ ON PACKED ROWS family $FAM ($MID @ $REV; e4b fp32 vs double-quantized absmax, Unsloth; --phase-peaks 1; amendment 56)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local PP="--phase-peaks 1" DQ="E4B_ABSMAX_DQ=1"
  can_run 600 $FAM/e4b/m_a0      && E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_a0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_a1      && TC1_ARM_EXTRA_ENV="$DQ" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_a1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_a1_d2   && TC1_ARM_EXTRA_ENV="$DQ" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_a1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_a0_d2   && E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_a0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/unsloth/m_pp  && arm $FAM unsloth ckpt_unsloth_m_pp unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $PP
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_memc4kt_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 57 (2026-10-07): the memory census of the TRAINING phase on packed
# rows. The box runs with TC1_EVAL_EVERY above TC1_STEPS, so no evaluation runs inside the census window (the final held-out evaluation
# comes after the census closes): e4b with the fp32 absmax (E4B_ABSMAX_DQ=0) and with it double-quantized (=1), and Unsloth; one draw each,
# --mem-census 1, every arm in venv-unsloth. No speed is read.
tc1_memc4kt_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_p4t:fused e4b:fused_attn4_m_p4t_dq:fused unsloth:ckpt_unsloth_m_p4t:unsloth"
  say "===== PACKED MEMORY CENSUS (training phase) family $FAM ($MID @ $REV; e4b fp32 absmax, e4b absmax-dq, Unsloth; --mem-census 1; amendment 57)"
  if [ "$EVAL_EVERY" -le "$STEPS" ]; then            # an in-loop evaluation would sit inside the census window: refuse, never mismeasure
    local t fw tag arm; for t in $ALL; do IFS=: read -r fw tag arm <<< "$t"
      stubw $FAM $fw $tag $arm not_run "amendment 57 needs TC1_EVAL_EVERY ($EVAL_EVERY) above TC1_STEPS ($STEPS): no evaluation inside the census"; done
    echo "$(echo $FAM | tr a-z A-Z) REFUSED eval_every=$EVAL_EVERY steps=$STEPS" | tee -a summary.txt; return 0
  fi
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local CEN="--mem-census 1"
  can_run 600 $FAM/e4b/m_p4t     && TC1_ARM_EXTRA_ENV="E4B_ABSMAX_DQ=0" E4B_VENV=t212 arm $FAM e4b fused_attn4_m_p4t fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/e4b/m_p4t_dq  && TC1_ARM_EXTRA_ENV="E4B_ABSMAX_DQ=1" E4B_VENV=t212 arm $FAM e4b fused_attn4_m_p4t_dq fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $CEN
  can_run 600 $FAM/unsloth/m_p4t && arm $FAM unsloth ckpt_unsloth_m_p4t unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $CEN
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_ckptoff_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 58 (2026-10-07): checkpoint inputs in pinned host memory
# (E4B_CKPT_OFFLOAD=1, engines/ckpt_offload.py) off (o0) vs on (o1) on packed rows, e4b's matched arm at its defaults otherwise, two draws a
# side in ABBA order, then Unsloth's matched arm (one draw); every arm in venv-unsloth with --phase-peaks 1.
tc1_ckptoff_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_o0:fused e4b:fused_attn4_m_o1:fused e4b:fused_attn4_m_o1_d2:fused e4b:fused_attn4_m_o0_d2:fused unsloth:ckpt_unsloth_m_oo:unsloth"
  say "===== CHECKPOINT OFFLOAD family $FAM ($MID @ $REV; e4b E4B_CKPT_OFFLOAD 0 vs 1, Unsloth; packed rows; --phase-peaks 1; amendment 58)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local PP="--phase-peaks 1" OFF="E4B_CKPT_OFFLOAD=0" ON="E4B_CKPT_OFFLOAD=1"
  can_run 600 $FAM/e4b/m_o0      && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_o0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_o1      && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_o1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_o1_d2   && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_o1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_o0_d2   && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_o0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/unsloth/m_oo  && arm $FAM unsloth ckpt_unsloth_m_oo unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $PP
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_evalce_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 60 (2026-10-07): the held-out loss stock (e0) vs from the logits in
# fp32 chunks (E4B_CHUNKED_EVAL_LOSS=1, e1) on packed rows -- e4b's matched arm with E4B_CKPT_OFFLOAD=1 and its defaults otherwise, two draws a
# side in ABBA order, then Unsloth's matched arm (one draw); every arm in venv-unsloth with --phase-peaks 1.
tc1_evalce_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_e0:fused e4b:fused_attn4_m_e1:fused e4b:fused_attn4_m_e1_d2:fused e4b:fused_attn4_m_e0_d2:fused unsloth:ckpt_unsloth_m_ee:unsloth"
  say "===== CHUNKED EVAL LOSS family $FAM ($MID @ $REV; e4b E4B_CHUNKED_EVAL_LOSS 0 vs 1 with E4B_CKPT_OFFLOAD=1, Unsloth; packed rows; --phase-peaks 1; amendment 60)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local PP="--phase-peaks 1" OFF="E4B_CKPT_OFFLOAD=1 E4B_CHUNKED_EVAL_LOSS=0" ON="E4B_CKPT_OFFLOAD=1 E4B_CHUNKED_EVAL_LOSS=1"
  can_run 600 $FAM/e4b/m_e0      && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_e0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_e1      && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_e1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_e1_d2   && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_e1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_e0_d2   && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_e0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/unsloth/m_ee  && arm $FAM unsloth ckpt_unsloth_m_ee unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $PP
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_combck_family FAM MID REV FETCH_AL E4B_AL UNS_AL -- TC1 amendment 61 (2026-10-07): the routed-expert combine whole (c0,
# E4B_COMBINE_CHUNK=0) vs over row chunks (c1, the default since #1304) on packed rows -- e4b's matched arm with E4B_CKPT_OFFLOAD=1 and its
# defaults otherwise, two draws a side in ABBA order, then Unsloth's matched arm (one draw); every arm in venv-unsloth with --phase-peaks 1.
tc1_combck_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6
  local ALL="e4b:fused_attn4_m_c0:fused e4b:fused_attn4_m_c1:fused e4b:fused_attn4_m_c1_d2:fused e4b:fused_attn4_m_c0_d2:fused unsloth:ckpt_unsloth_m_cc:unsloth"
  say "===== COMBINE ROW CHUNKS family $FAM ($MID @ $REV; e4b E4B_COMBINE_CHUNK 0 vs default with E4B_CKPT_OFFLOAD=1, Unsloth; packed rows; --phase-peaks 1; amendment 61)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  local PP="--phase-peaks 1" OFF="E4B_CKPT_OFFLOAD=1 E4B_COMBINE_CHUNK=0" ON="E4B_CKPT_OFFLOAD=1"
  can_run 600 $FAM/e4b/m_c0      && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_c0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_c1      && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_c1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_c1_d2   && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_c1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_c0_d2   && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_c0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/unsloth/m_cc  && arm $FAM unsloth ckpt_unsloth_m_cc unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $PP
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_ckptre_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 62 (2026-10-07): why amendment 59's field-recipe step got faster. The shipped
# arm at TC1's field recipe with Hugging Face's checkpoint (r0, E4B_CKPT_OFFLOAD=0), the reentrant checkpoint alone (rr, =reentrant) and the
# reentrant checkpoint with its inputs in pinned host memory (r1, =1); two draws a side (r0 rr r1 r1 rr r0), every arm in venv-unsloth.
tc1_ckptre_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_r0:fused e4b:fused_attn4_shipped_rr:fused e4b:fused_attn4_shipped_r1:fused e4b:fused_attn4_shipped_r1_d2:fused e4b:fused_attn4_shipped_rr_d2:fused e4b:fused_attn4_shipped_r0_d2:fused"
  say "===== FIELD CHECKPOINT FLAVOUR family $FAM ($MID @ $REV; shipped arm, E4B_CKPT_OFFLOAD 0 vs reentrant vs 1, field recipe, venv-unsloth, amendment 62)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local NATIVE="--adapter-dtype native --lora-init native"
  local PP="--phase-peaks 1" R0="E4B_CKPT_OFFLOAD=0" RR="E4B_CKPT_OFFLOAD=reentrant" R1="E4B_CKPT_OFFLOAD=1"
  can_run 600 $FAM/e4b/shipped_r0     && TC1_ARM_EXTRA_ENV="$R0" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_r0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/shipped_rr     && TC1_ARM_EXTRA_ENV="$RR" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_rr fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/shipped_r1     && TC1_ARM_EXTRA_ENV="$R1" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_r1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/shipped_r1_d2  && TC1_ARM_EXTRA_ENV="$R1" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_r1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/shipped_rr_d2  && TC1_ARM_EXTRA_ENV="$RR" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_rr fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/shipped_r0_d2  && TC1_ARM_EXTRA_ENV="$R0" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_r0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_ckptofff_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 59 (2026-10-07): amendment 58's switch (E4B_CKPT_OFFLOAD=1, checkpoint
# inputs in pinned host memory) off (f0) vs on (f1) at TC1's FIELD recipe, where each layer's input is a few hundred tokens -- the shipped and the
# matched arm, two draws a side in ABBA order, every arm in venv-unsloth at e4b's defaults otherwise with --phase-peaks 1: the read before any default.
tc1_ckptofff_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_f0:fused e4b:fused_attn4_shipped_f1:fused e4b:fused_attn4_m_f0:fused e4b:fused_attn4_m_f1:fused e4b:fused_attn4_m_f1_d2:fused e4b:fused_attn4_m_f0_d2:fused e4b:fused_attn4_shipped_f1_d2:fused e4b:fused_attn4_shipped_f0_d2:fused"
  say "===== FIELD CHECKPOINT OFFLOAD family $FAM ($MID @ $REV; E4B_CKPT_OFFLOAD 0 vs 1, field recipe, venv-unsloth, --phase-peaks 1, amendment 59)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local PP="--phase-peaks 1" OFF="E4B_CKPT_OFFLOAD=0" ON="E4B_CKPT_OFFLOAD=1"
  can_run 600 $FAM/e4b/shipped_f0     && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_f0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/shipped_f1     && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_f1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/m_f0           && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_f0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_f1           && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_f1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_f1_d2        && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_f1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/m_f0_d2        && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_f0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PP
  can_run 600 $FAM/e4b/shipped_f1_d2  && TC1_ARM_EXTRA_ENV="$ON"  E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_f1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  can_run 600 $FAM/e4b/shipped_f0_d2  && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_f0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PP
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_fieldbk_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 49 (2026-10-06): grouped-nf4-gemm's bucketed LoRA-delta padding
# (NF4_QLORA_PAD_BUCKETS=1, grouped-nf4-gemm#490) off vs on at TC1's FIELD recipe (amendment 48 read it on packed rows) -- the shipped and the
# matched arm, two draws a side in ABBA order, every arm in venv-unsloth at e4b's defaults: whether the short rows pay for the extra launches.
tc1_fieldbk_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_fb0:fused e4b:fused_attn4_shipped_fb1:fused e4b:fused_attn4_m_fb0:fused e4b:fused_attn4_m_fb1:fused e4b:fused_attn4_m_fb1_d2:fused e4b:fused_attn4_m_fb0_d2:fused e4b:fused_attn4_shipped_fb1_d2:fused e4b:fused_attn4_shipped_fb0_d2:fused"
  say "===== FIELD PAD-BUCKETS A/B family $FAM ($MID @ $REV; grouped-nf4-gemm's LoRA delta one padded block vs buckets, field recipe, venv-unsloth, amendment 49)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="NF4_QLORA_PAD_BUCKETS=0" NEW="NF4_QLORA_PAD_BUCKETS=1"
  can_run 600 $FAM/e4b/shipped_fb0     && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_fb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_fb1     && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_fb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_fb0           && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_fb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_fb1           && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_fb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_fb1_d2        && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_fb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_fb0_d2        && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_fb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_fb1_d2  && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_fb1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_fb0_d2  && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_fb0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_fieldauto_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 50 (2026-10-06): NF4_QLORA_PAD_BUCKETS=auto (grouped-nf4-gemm#491:
# bucket a call only when it carries >= 16,384 routed rows) against 0 at TC1's FIELD recipe, whose calls carry at most 9,040 -- the shipped and
# the matched arm, two draws a side in ABBA order, every arm in venv-unsloth at e4b's defaults: auto must never bucket here.
tc1_fieldauto_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_shipped_fa0:fused e4b:fused_attn4_shipped_fa1:fused e4b:fused_attn4_m_fa0:fused e4b:fused_attn4_m_fa1:fused e4b:fused_attn4_m_fa1_d2:fused e4b:fused_attn4_m_fa0_d2:fused e4b:fused_attn4_shipped_fa1_d2:fused e4b:fused_attn4_shipped_fa0_d2:fused"
  say "===== FIELD PAD-BUCKETS AUTO family $FAM ($MID @ $REV; NF4_QLORA_PAD_BUCKETS 0 vs auto, field recipe, venv-unsloth, amendment 50)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OLD="NF4_QLORA_PAD_BUCKETS=0" NEW="NF4_QLORA_PAD_BUCKETS=auto"
  can_run 600 $FAM/e4b/shipped_fa0     && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_fa0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_fa1     && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_shipped_fa1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/m_fa0           && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_fa0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_fa1           && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_fa1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_fa1_d2        && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_fa1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_fa0_d2        && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_fa0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_fa1_d2  && TC1_ARM_EXTRA_ENV="$NEW" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_fa1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_fa0_d2  && TC1_ARM_EXTRA_ENV="$OLD" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_shipped_fa0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_decoded_gate -- TC1 amendment 46 (2026-10-05): the decoded A/B's FIRST step, once per box, before any of its arms. grouped-nf4-gemm's
# compiled correctness tests for GNF4_TRAIN_GEMM=decoded run on this card, from a checkout of grouped-nf4-gemm at GNF4_SHA (the installed
# package's commit, which the tripwire pins), in venv-e4b (the arms' venv): kernel/test_nf4_route.py's decoded, cap and dequant tests -- RD1's
# fp32-reference gate (the route's error <= 2x the dense route's, every call), capped == uncapped bit for bit, the cap's bound on the decode
# transient, dequant_groups bit-equal to dequant_ref -- and kernel/test_nf4_route_decision.py (`auto` never answers `decoded`). A failed test,
# or a gate that cannot run, refuses the box before any timing: BOX_REFUSED decoded-gate, exit 19. 19 is not one of the host-limited codes
# adertha admits as machine evidence (13, 14, 17, 18): a kernel defect names no machine. $W/decgate.json is the record (P107). Each run must
# also have PASSED every test DECGATE_REQUIRED names for it: a grouped-nf4-gemm without the route has none of them, and -k alone would then
# select only the dequant tests and pass (rehearsed on gnf4 054be19, before #487) -- a gate that never ran the code it gates.
DECGATE_K="decoded or cap_bounds or dequant_groups"
DECGATE_REQUIRED="route:test_decoded_route_passes_the_rd1_gate_against_the_dense_route,test_decoded_route_matches_the_fused_kernels_on_any_card,test_the_cap_bounds_the_decode_transient,test_dequant_groups_is_bit_equal_to_dequant_ref decision:test_decoded_is_taken_only_when_asked_for"
tc1_decoded_gate(){
  [ -f $W/decgate.json ] && grep -q '"passed": true' $W/decgate.json && return 0
  say "===== DECODED-ROUTE GATE (amendment 46): grouped-nf4-gemm @$GNF4_SHA's compiled tests for the decoded route on this card, before any arm"
  local SRC=$W/gnf4-src REASON=""
  rm -rf $SRC; : > logs/decgate_rc.txt
  if ! perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $SRC > logs/decgate_clone.log 2>&1 \
     || ! git -C $SRC checkout -q "$GNF4_SHA" >> logs/decgate_clone.log 2>&1; then
    REASON="the grouped-nf4-gemm checkout at $GNF4_SHA failed (logs/decgate_clone.log)"
  elif ! $PY_E4B -m pytest --version > /dev/null 2>&1 \
       && ! perl -e 'alarm 600; exec @ARGV' $PY_E4B -m pip install -q --no-input pytest > logs/decgate_pytest.log 2>&1; then
    REASON="pytest could not be installed into venv-e4b (logs/decgate_pytest.log)"
  fi
  if [ -z "$REASON" ]; then
    (cd $SRC/kernel && env -u TRITON_INTERPRET perl -e 'alarm 1500; exec @ARGV' $PY_E4B -m pytest -q -p no:cacheprovider test_nf4_route.py \
       -k "$DECGATE_K" --junitxml=$W/logs/decgate_route.xml) > logs/decgate_route.log 2>&1
    echo "decgate route rc=$?" >> logs/decgate_rc.txt
    (cd $SRC/kernel && env -u TRITON_INTERPRET perl -e 'alarm 600; exec @ARGV' $PY_E4B -m pytest -q -p no:cacheprovider test_nf4_route_decision.py \
       --junitxml=$W/logs/decgate_decision.xml) > logs/decgate_decision.log 2>&1
    echo "decgate decision rc=$?" >> logs/decgate_rc.txt
  fi
  local HEAD_AT; HEAD_AT=$(git -C $SRC rev-parse HEAD 2>/dev/null)
  TC1_W="$W" DECGATE_REASON="$REASON" GNF4_SHA="$GNF4_SHA" DECGATE_K="$DECGATE_K" DECGATE_REQUIRED="$DECGATE_REQUIRED" DECGATE_HEAD="$HEAD_AT" \
    $PY_E4B - <<'PYG' 2>&1 | tee -a summary.txt
import json, os, re, xml.etree.ElementTree as ET
W = os.environ["TC1_W"]
reason = os.environ.get("DECGATE_REASON") or ""
rcs = {}
for ln in open(os.path.join(W, "logs", "decgate_rc.txt")):
    m = re.match(r"decgate (\w+) rc=(\d+)", ln)
    if m:
        rcs[m.group(1)] = int(m.group(2))
required = {k: v.split(",") for k, v in (w.split(":", 1) for w in os.environ["DECGATE_REQUIRED"].split())}
runs = []
for key, name in (("route", f'test_nf4_route.py -k "{os.environ["DECGATE_K"]}"'), ("decision", "test_nf4_route_decision.py")):
    if key not in rcs:
        continue
    t = fl = er = sk = 0
    ok_names = set()
    x = os.path.join(W, "logs", f"decgate_{key}.xml")
    if os.path.exists(x):
        root = ET.parse(x).getroot()
        for su in ([root] if root.tag == "testsuite" else list(root.iter("testsuite"))):
            t += int(su.get("tests", 0)); fl += int(su.get("failures", 0)); er += int(su.get("errors", 0)); sk += int(su.get("skipped", 0))
        for tc in root.iter("testcase"):
            if not any(c.tag in ("failure", "error", "skipped") for c in tc):
                ok_names.add(tc.get("name", "").split("[")[0])
    runs.append({"name": name, "rc": rcs[key], "tests": t, "failures": fl, "errors": er, "skipped": sk,
                 "missing": [n for n in required.get(key, []) if n not in ok_names]})
ran = not reason and len(runs) == 2
passed = ran and all(r["rc"] == 0 and r["tests"] > 0 and not (r["failures"] or r["errors"] or r["skipped"] or r["missing"]) for r in runs)
try:
    import torch, triton
    gpu, tv, trv = torch.cuda.get_device_name(), torch.__version__, triton.__version__
except Exception as e:
    gpu = tv = trv = f"unavailable: {type(e).__name__}"
rec = {"ran": ran, "passed": passed, "gnf4_sha": os.environ.get("GNF4_SHA"), "checkout_head": os.environ.get("DECGATE_HEAD") or None, "gpu": gpu,
       "torch": tv, "triton": trv, "runs": runs, "reason": reason or None}
json.dump(rec, open(os.path.join(W, "decgate.json"), "w"), indent=1)
print(f"DECODED GATE {'PASSED' if passed else 'NOT PASSED'} on {gpu} (torch {tv}, triton {trv}): "
      + ("; ".join(f"{r['name']} rc {r['rc']}: {r['tests']} tests, {r['failures']} failed, {r['errors']} errors, {r['skipped']} skipped"
                   + (f", required tests not passed: {r['missing']}" if r["missing"] else "") for r in runs) or reason))
PYG
  # the checkout is not evidence (decgate.json, logs/decgate_* are): gone before any arm, so no partial pull or fetch carries its files --
  # its docs/receipts-ab/receipt.json once failed adertha's reconciler for every launch on the account (tc1dec-5090-4, 2026-10-06)
  rm -rf $SRC
  grep -q '"passed": true' $W/decgate.json 2>/dev/null && return 0
  say "BOX REFUSED: the decoded route's correctness gate did not pass on this card ($W/decgate.json) -- no arm runs"
  echo "BOX_REFUSED decoded-gate" | tee -a summary.txt
  finish 19; }
# tc1_decodedab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 46 (2026-10-05): grouped-nf4-gemm's decoded training route
# (GNF4_TRAIN_GEMM=decoded, gnf4#487: per chunk of present groups, dequant_groups + one grouped bf16 GEMM launch, GNF4_DECODED_MAX_BYTES at its
# 256 MiB default) against its fused 4-bit kernels (=fused), on the matched arm only, two draws a side in ABBA order, every arm resident in
# venv-e4b (RD1's software), every other setting the default. tc1_decoded_gate runs first, once per box. olmoedecab is OLMoE at TC2's pin
# through tc1_prepare (TC1's field recipe); qwen3decab is TC1's qwen3 tokens. Each receipt's route_ab record names the route in force and the
# decoded_fwd / decoded_dgrad counts.
tc1_decodedab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  tc1_decoded_gate
  local ALL="e4b:fused_attn4_m_dec0:fused e4b:fused_attn4_m_dec1:fused e4b:fused_attn4_m_dec1_d2:fused e4b:fused_attn4_m_dec0_d2:fused"
  say "===== DECODED-ROUTE A/B family $FAM ($MID @ $REV; gnf4's fused 4-bit kernels vs its decoded route, matched arm, venv-e4b, amendment 46)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local OLD="GNF4_TRAIN_GEMM=fused" NEW="GNF4_TRAIN_GEMM=decoded"
  can_run 600 $FAM/e4b/m_dec0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_dec0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dec1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_dec1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dec1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_dec1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dec0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_dec0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_dqab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 28 (2026-10-04): e4b's expert absmax fp32 (E4B_ABSMAX_DQ=0, the default) vs
# double-quantized (=1, #1040), the matched arm, resident, two draws a side in ABBA order. The same function serves both tokens: qwen3dqab
# (Qwen3-30B-A3B, TC1's tokens) and mixtraldqab (Mixtral-8x7B-Instruct at TC2's pin, prepared as tc2_big_family prepares mixtral).
tc1_dqab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_dq0:fused e4b:fused_attn4_m_dq1:fused e4b:fused_attn4_m_dq1_d2:fused e4b:fused_attn4_m_dq0_d2:fused"
  say "===== ABSMAX-DQ A/B family $FAM ($MID @ $REV; the expert absmax fp32 vs double-quantized, matched arm, resident, amendment 28)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local OLD="E4B_ABSMAX_DQ=0" NEW="E4B_ABSMAX_DQ=1"
  can_run 600 $FAM/e4b/m_dq0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_dq0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dq1     && TC1_ARM_EXTRA_ENV="$NEW" arm   $FAM e4b fused_attn4_m_dq1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dq1_d2  && TC1_ARM_EXTRA_ENV="$NEW" draw2 $FAM e4b fused_attn4_m_dq1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_dq0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_dq0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_tritonab_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 32 (2026-10-05): one variable, Triton. venv-e4b (torch 2.8.0+cu128) with
# its own triton 3.4 (side tr0) vs triton 3.7.1 (side tr1: installed alone into $W/venv-triton37 and put first on the arm's PYTHONPATH), the
# matched and the shipped arm, two draws a side in ABBA order. The prebound launches are off on both sides (they cover triton 3.4 / 3.6
# only, so they would otherwise run on tr0 alone). A failed triton 3.7.1 install leaves every tr1 arm an install_failed row.
tc1_tritonab_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_tr0:fused e4b:fused_attn4_m_tr1:fused e4b:fused_attn4_m_tr1_d2:fused e4b:fused_attn4_m_tr0_d2:fused e4b:fused_attn4_shipped_tr0:fused e4b:fused_attn4_shipped_tr1:fused e4b:fused_attn4_shipped_tr1_d2:fused e4b:fused_attn4_shipped_tr0_d2:fused"
  say "===== TRITON A/B family $FAM ($MID @ $REV; venv-e4b with triton 3.4 vs 3.7.1, matched and shipped arms, amendment 32)"
  local TR37=$W/venv-triton37 TROK=1 TRWHY=""
  $PY_BASE -m uv --version > logs/pip_triton37.log 2>&1 || $PY_BASE -m pip install -q --no-input uv >> logs/pip_triton37.log 2>&1
  perl -e 'alarm 900; exec @ARGV' $PY_BASE -m uv pip install --python $PY_E4B --target $TR37 --no-deps "triton==3.7.1" >> logs/pip_triton37.log 2>&1 || TROK=0
  if [ $TROK = 1 ] && ! PYTHONPATH=$TR37 $PY_E4B -c "import triton, torch; assert triton.__version__.startswith('3.7'), triton.__version__; print('triton', triton.__version__, 'torch', torch.__version__)" >> logs/pip_triton37.log 2>&1; then TROK=0; fi
  [ $TROK = 1 ] || TRWHY="triton 3.7.1 did not install or import beside torch 2.8 (logs/pip_triton37.log): $(tail -2 logs/pip_triton37.log | tr '\n' ' ' | cut -c1-200)"
  echo "TRITON37 ok=$TROK $(tail -1 logs/pip_triton37.log | cut -c1-120)" | tee -a summary.txt
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local OFF="E4B_TRITON_PREBIND=0 GNF4_TRITON_PREBIND=0"
  local OLD="$OFF" NEW="$OFF PYTHONPATH=$TR37"
  t1(){ local how=$1 tag=$2; shift 2
    if [ $TROK = 1 ]; then TC1_ARM_EXTRA_ENV="$NEW" $how $FAM e4b $tag fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 "$@"
    else stubw $FAM e4b $([ $how = draw2 ] && echo ${tag}_d2 || echo $tag) fused install_failed "$TRWHY"; fi; }
  can_run 600 $FAM/e4b/m_tr0           && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_m_tr0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_tr1           && t1 arm   fused_attn4_m_tr1 $MATCH
  can_run 600 $FAM/e4b/m_tr1_d2        && t1 draw2 fused_attn4_m_tr1 $MATCH
  can_run 600 $FAM/e4b/m_tr0_d2        && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_m_tr0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/shipped_tr0     && TC1_ARM_EXTRA_ENV="$OLD" arm   $FAM e4b fused_attn4_shipped_tr0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tr1     && t1 arm   fused_attn4_shipped_tr1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tr1_d2  && t1 draw2 fused_attn4_shipped_tr1 $NATIVE
  can_run 600 $FAM/e4b/shipped_tr0_d2  && TC1_ARM_EXTRA_ENV="$OLD" draw2 $FAM e4b fused_attn4_shipped_tr0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_envsplit_family FAM MID REV FETCH_AL E4B_AL -- TC1 amendment 34 (2026-10-05): amendment 24's environment gain split on the 5090. The matched
# arm in three environments, two draws each in ABC CBA order: e0 venv-e4b (torch 2.8.0, transformers 5.18.0, triton 3.4), e1 venv-e4b-tf55
# (built here like venv-e4b, transformers 5.5.0: the one variable against e0), e2 venv-unsloth + e4b (torch 2.12.1, transformers 5.5.0,
# triton 3.7.1: torch + triton against e1). The prebound launches are off on every side (they cover triton 3.4 / 3.6, not 3.7).
tc1_envsplit_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5
  local ALL="e4b:fused_attn4_m_e0:fused e4b:fused_attn4_m_e1:fused e4b:fused_attn4_m_e2:fused e4b:fused_attn4_m_e2_d2:fused e4b:fused_attn4_m_e1_d2:fused e4b:fused_attn4_m_e0_d2:fused"
  say "===== ENV SPLIT family $FAM ($MID @ $REV; e0 venv-e4b, e1 venv-e4b + transformers 5.5.0, e2 venv-unsloth; matched arm; amendment 34)"
  TF55_OK=0; TF55_REASON=""
  if $PY_BASE -m venv --system-site-packages $W/venv-e4b-tf55 > logs/venv_e4b_tf55.log 2>&1; then
    pip_fresh $W/venv-e4b-tf55/bin/python e4b_tf55
    if perl -e 'alarm 2400; exec @ARGV' $W/venv-e4b-tf55/bin/python -m pip install -q --no-input --prefer-binary \
         "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
         "transformers==5.5.0" "bitsandbytes==$BNB_VER" "peft==$PEFT_VER" accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken >> logs/venv_e4b_tf55.log 2>&1 \
       && $W/venv-e4b-tf55/bin/python -c "import torch, transformers, experts4bit_qlora; assert transformers.__version__ == '5.5.0', transformers.__version__; print('venv-e4b-tf55 torch', torch.__version__, 'transformers', transformers.__version__, 'e4b', experts4bit_qlora.__version__)" >> logs/venv_e4b_tf55.log 2>&1; then
      TF55_OK=1
    else TF55_REASON="venv-e4b-tf55 (transformers 5.5.0) did not install or import (logs/venv_e4b_tf55.log): $(tail -2 logs/venv_e4b_tf55.log | tr '\n' ' ' | cut -c1-200)"; fi
  else TF55_REASON="venv-e4b-tf55 could not be created (logs/venv_e4b_tf55.log)"; fi
  echo "VENV-E4B-TF55 ok=$TF55_OK $(tail -1 logs/venv_e4b_tf55.log | cut -c1-120)" | tee -a summary.txt
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local OFF="E4B_TRITON_PREBIND=0 GNF4_TRITON_PREBIND=0"
  can_run 600 $FAM/e4b/m_e0     && TC1_ARM_EXTRA_ENV="$OFF" arm                 $FAM e4b fused_attn4_m_e0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_e1     && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=tf55 arm   $FAM e4b fused_attn4_m_e1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_e2     && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 arm   $FAM e4b fused_attn4_m_e2 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_e2_d2  && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=t212 draw2 $FAM e4b fused_attn4_m_e2 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_e1_d2  && TC1_ARM_EXTRA_ENV="$OFF" E4B_VENV=tf55 draw2 $FAM e4b fused_attn4_m_e1 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/m_e0_d2  && TC1_ARM_EXTRA_ENV="$OFF" draw2               $FAM e4b fused_attn4_m_e0 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_routebench_family FAM ALARM -- TC1c amendment 3 (2026-10-04): a kernel-route replay, not a training run. grouped-nf4-gemm's
# fused NF4 grouped GEMM (forward and dgrad) against a whole-stack bitsandbytes dequantize_4bit + torch._grouped_mm on the
# recorded real-router calls of e4b's training step (routecalls-qwen3.json, staged by TC1_EXTRA_STAGE with route_bench.py; no
# model is downloaded). Writes ROUTEBENCH.json -- a name the reducer's *_*_*.json receipt glob never matches.
tc1_routebench_family(){ local FAM=$1 AL=$2
  say "===== ROUTE BENCH $FAM (fused NF4 grouped GEMM vs dequantize_4bit + torch._grouped_mm, recorded real-router calls)"
  local f; for f in route_bench.py routecalls-qwen3.json; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; echo "ROUTEBENCH STAGE MISSING $f" | tee -a summary.txt; return 0; }; done
  can_run 900 $FAM/routebench || return 0
  env TC1_BOX_CLASS="$BOX_CLASS" perl -e "alarm $AL; exec @ARGV" $PY_E4B -u $W/route_bench.py $W/routecalls-qwen3.json $W/ROUTEBENCH.json --reps 10 > logs/run_${FAM}_routebench.log 2>&1
  local rc=$?
  echo "ROUTEBENCH rc=$rc $([ -s $W/ROUTEBENCH.json ] && $PY_E4B -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({"gpu": d["gpu"], "grouped_mm": d["grouped_mm_supported"], "by_op": d["by_op"]}))' $W/ROUTEBENCH.json)" | tee -a summary.txt
  tail -6 logs/run_${FAM}_routebench.log | sed "s/^/    /"
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt; }
# tc1_fusedsweep_family FAM ALARM -- TC1c amendment 5 (2026-10-04): a kernel config replay, not a training run. grouped-nf4-gemm's fused
# forward (prefill_variant, prefill_groups, BLOCK_N, warps, stages) and dgrad (BLOCK_M, BLOCK_N, BLOCK_K, warps) configs on the recorded
# real-router calls (routecalls-qwen3.json, staged with fused_sweep.py by TC1_EXTRA_STAGE; no model). Writes FUSEDSWEEP.json.
tc1_fusedsweep_family(){ local FAM=$1 AL=$2
  say "===== FUSED SWEEP $FAM (grouped-nf4-gemm's fused forward and dgrad configs on recorded real-router calls)"
  local f; for f in fused_sweep.py routecalls-qwen3.json; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; echo "FUSEDSWEEP STAGE MISSING $f" | tee -a summary.txt; return 0; }; done
  can_run 900 $FAM/fusedsweep || return 0
  env TC1_BOX_CLASS="$BOX_CLASS" perl -e "alarm $AL; exec @ARGV" $PY_E4B -u $W/fused_sweep.py $W/routecalls-qwen3.json $W/FUSEDSWEEP.json --reps 5 > logs/run_${FAM}_fusedsweep.log 2>&1
  local rc=$?
  echo "FUSEDSWEEP rc=$rc $(grep -E '^(fwd|dgrad) default' logs/run_${FAM}_fusedsweep.log | tr '\n' ' ')" | tee -a summary.txt
  tail -6 logs/run_${FAM}_fusedsweep.log | sed "s/^/    /"
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt; }
# tc1_prof945_family FAM MID REV FETCH_AL PROF_AL -- TC1 amendment 12 (2026-10-03, #945): where e4b's fused training step goes once
# the host syncs are gone -- the TC1 profile instrument (3 warm + 3 profiled steps, dmon beside) on the shipped and the matched arm with
# the single-read grouping + pinned ring, and the matched arm with the legacy grouping + pageable copies as the before-picture.
tc1_prof945_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 PAL=$5
  local ALL="e4b:fused_attn4_shipped_prof:fused e4b:fused_attn4_m_prof:fused e4b:fused_attn4_m_prof_legacy:fused"
  say "===== PROFILE family $FAM ($MID @ $REV; e4b after #945 vs the legacy path, the TC1 profile instrument)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local PROF="--log-every 1 --microbatch-timing 1 --profile-steps $PROFILE_STEPS --profile-warm $PROFILE_WARM"
  local LEG="E4B_GROUPING=legacy GNF4_PINNED_RING=0" NEW="E4B_GROUPING=single GNF4_PINNED_RING=1"
  local dp; dp=$(dmon_start ${FAM}_e4b_fused_attn4_shipped_prof)
  can_run 600 $FAM/e4b/shipped_prof   && TC1_ARM_EXTRA_ENV="$NEW" arm $FAM e4b fused_attn4_shipped_prof fused $PAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE $PROF
  dmon_stop $dp; dp=$(dmon_start ${FAM}_e4b_fused_attn4_m_prof)
  can_run 600 $FAM/e4b/m_prof         && TC1_ARM_EXTRA_ENV="$NEW" arm $FAM e4b fused_attn4_m_prof fused $PAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  dmon_stop $dp; dp=$(dmon_start ${FAM}_e4b_fused_attn4_m_prof_legacy)
  can_run 600 $FAM/e4b/m_prof_legacy  && TC1_ARM_EXTRA_ENV="$LEG" arm $FAM e4b fused_attn4_m_prof_legacy fused $PAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  dmon_stop $dp
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_axolotl_family FAM MID REV FETCH_AL E4B_AL HF_AL AX_AL -- TC1-PREREG amendment 3 (2026-10-02): the axolotl rows re-asked on their own box, each a
# position within this box against the e4b fused_m it runs first; two draws of the matched pair so the matched position carries its cross-draw interval:
#   e4b/fused_attn4_m  axolotl/ckpt_axolotl_m  e4b/fused_attn4_m_d2  axolotl/ckpt_axolotl_m_d2  axolotl/ckpt_axolotl_best (scattermoe, native init)
#   hf/hf_peft_m_mb1_t214 (the HF arm on the axolotl venv's torch 2.14 at mb1 x accum 8, experts_implementation=grouped_mm; its gate -- the judged
#   family's hf/hf_peft_m OOMed -- is TC1's registered record (tc1-5090-16: OOM at mb2 and at mb1), so it runs here unconditionally)
tc1_axolotl_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 HAL=$6 AAL=$7
  local ALL="e4b:fused_attn4_m:fused axolotl:ckpt_axolotl_m:axolotl e4b:fused_attn4_m_d2:fused axolotl:ckpt_axolotl_m_d2:axolotl axolotl:ckpt_axolotl_best:axolotl hf:hf_peft_m_mb1_t214:hf"
  say "===== AXOLOTL family $FAM ($MID @ $REV; TC1-PREREG amendment 3: the axolotl rows, two draws, against this box's own e4b fused_m; alarms e4b $EAL hf $HAL axolotl $AAL)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  can_run 600 $FAM/e4b/fused_m     && arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/axolotl/m       && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH
  can_run 600 $FAM/e4b/fused_m_d2  && draw2 $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/axolotl/m_d2    && draw2 $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH
  can_run 600 $FAM/axolotl/best    && arm   $FAM axolotl ckpt_axolotl_best axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json --axolotl-best 1 --adapter-dtype fp32 --lora-init native
  can_run 600 $FAM/hf/mb1_t214     && HF_VENV=t214 arm $FAM hf hf_peft_m_mb1_t214 hf $HAL "$MID" $REV 0 mb1 $TOK $TS --hf-experts-implementation grouped_mm $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_curve_family FAM MID REV FETCH_AL E4B_AL UNS_AL ANCHOR_AL SCALE_AL -- lane TC1b (TC1B-PREREG "Arms, in this order"), one process per arm:
#   1 e4b/fused_attn4_m_200  2 unsloth/ckpt_unsloth_m_200 (the matched pair at N 200: fp32 adapters, matched init, grouped_mm in venv-unsloth)
#   3 e4b/fused_attn4_shipped_200 (as the loader builds it: --adapter-dtype native --lora-init native)
#   4 e4b/fused_attn4_p38 + unsloth/ckpt_unsloth_p38 (tp4's anchor pair at tp2/P38's fixture, clinical tokens) + unsloth/ckpt_unsloth_p38_t28
#   5 e4b/fused_attn4_m_t1 + unsloth/ckpt_unsloth_m_t1 (micro-batch 1 x accum 1)   6 e4b/fused_attn4_m_r64 + unsloth/ckpt_unsloth_m_r64 (r 64 / alpha 64)
# The matched pair runs first so a deadline cannot eat it; the family's tokens file carries CURVE_EVAL_N held-out rows (arms 5-6 take the first 8).
tc1_curve_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 AAL=$7 SAL=$8
  local ALL="e4b:fused_attn4_m_200:fused unsloth:ckpt_unsloth_m_200:unsloth e4b:fused_attn4_shipped_200:fused e4b:fused_attn4_p38:fused unsloth:ckpt_unsloth_p38:unsloth unsloth:ckpt_unsloth_p38_t28:unsloth e4b:fused_attn4_m_t1:fused unsloth:ckpt_unsloth_m_t1:unsloth e4b:fused_attn4_m_r64:fused unsloth:ckpt_unsloth_m_r64:unsloth"
  say "===== CURVE family $FAM (TC1b; $MID @ $REV; matched seed $MATCHED_SEED; N $CURVE_STEPS eval every $CURVE_EVAL_EVERY on $CURVE_EVAL_N rows; alarms e4b $EAL unsloth $UAL anchor $AAL t1/r64 $SAL)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" $CURVE_EVAL_N || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"      # the matched arms (1, 2, 5, 6)
  local NATIVE="--adapter-dtype native --lora-init native"                 # arm 3: as load_moe_4bit_streaming + add_attention_lora build it
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"                   # the notebooks' recipe (tp4's arm); double-quant OFF is the arm's default on the matched arms
  # tp4's anchor arms byte-for-byte (bench/tp4/tp4_run.sh anchor_pair + tp4_arm.py @ 10ce711d): tp4_arm.py cast every NON-e4b arm's adapters to fp32 and
  # left e4b's as the loader built them (bf16 expert adapters), native init on both, no double-quant knob (Unsloth's loader default), dgrad=1 on both drivers.
  # The one per-arm environment difference from tp4_run.sh is OMP_NUM_THREADS=$PHYS (phase 3 F19, set on every TC1 arm); tp4 left it unset. Recorded in arm_facts.
  local ANCHOR_E4B="--adapter-dtype native --lora-init native" ANCHOR_UNS="--adapter-dtype fp32 --lora-init native --unsloth-double-quant default"
  can_run 600 $FAM/e4b/m_200        && arm $FAM e4b fused_attn4_m_200 fused $EAL "$MID" $REV 0 curve $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_200    && arm $FAM unsloth ckpt_unsloth_m_200 unsloth $UAL "$MID" $REV 0 curve $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/e4b/shipped_200  && arm $FAM e4b fused_attn4_shipped_200 fused $EAL "$MID" $REV 0 curve $TOK $TS --attn-4bit 1 $NATIVE
  # the anchor pair: tp2's text, tokenised as tp4 did (clinical template, seq 512, the first A_EVAL_N held-out rows)
  local ATOK=$W/tokens_${FAM}_p38.json ATS CLIN_SHA; CLIN_SHA=$($PY_E4B -c "import json; print(json.load(open('$W/ds_manifest.json'))['clinical']['sha256'])")
  if tokenise $FAM "$MID" $REV clinical $A_SEQ $W/data/ds_clinical.json $CLIN_SHA $ATOK $A_EVAL_N; then
    ATS=$(tok_sha $ATOK); echo "TOKENS ${FAM}_p38 clinical sha=$ATS train_only_sha=$(train_sha $ATOK) (tp2's tokens_qwen3 / tp4's tokens_qwen3_p38 sha for the cross-check: their receipts record it)" | tee -a summary.txt
    can_run 600 $FAM/e4b/p38         && arm $FAM e4b fused_attn4_p38 fused $AAL "$MID" $REV 0 anchor $ATOK $ATS --attn-4bit 1 $ANCHOR_E4B
    can_run 600 $FAM/unsloth/p38     && arm $FAM unsloth ckpt_unsloth_p38 unsloth $AAL "$MID" $REV 0 anchor $ATOK $ATS $UNS --unsloth-moe-backend grouped_mm $ANCHOR_UNS \
      --note "tp4's anchor arm EXCEPT the venv: venv-unsloth (unsloth[cu130-torch2121], torch 2.12.1+cu130) with UNSLOTH_MOE_BACKEND=grouped_mm; tp4 ran venv-unsloth-t28 (torch 2.8.0+cu128) with the loader-default backend -- ckpt_unsloth_p38_t28 is that arm"
    can_run 600 $FAM/unsloth/p38_t28 && UNS_VENV=t28 arm $FAM unsloth ckpt_unsloth_p38_t28 unsloth $AAL "$MID" $REV 0 anchor $ATOK $ATS $UNS --unsloth-moe-backend default $ANCHOR_UNS \
      --note "byte-for-byte tp4's anchor arm: venv-unsloth-t28 (unsloth[cu128-torch280] on the image's torch 2.8.0), the loader-default MoE backend"
  else
    tail -3 logs/prepare_${FAM}_clinical.log; echo "${FAM}_p38: TOKENS FAIL" | tee -a summary.txt
    local why="tokenise failed (logs/prepare_${FAM}_clinical.log): $(tail -1 logs/prepare_${FAM}_clinical.log | cut -c1-200)"
    stubw $FAM e4b fused_attn4_p38 fused harness_error "$why"; stubw $FAM unsloth ckpt_unsloth_p38 unsloth harness_error "$why"; stubw $FAM unsloth ckpt_unsloth_p38_t28 unsloth harness_error "$why"
  fi
  # the tokens-per-step scaling pair (micro-batch 1 x accum 1) and the rank pair (r 64 / alpha 64): the field recipe otherwise, 8 rows at 0 and N
  can_run 600 $FAM/e4b/m_t1         && arm $FAM e4b fused_attn4_m_t1 fused $SAL "$MID" $REV 0 t1 $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_t1     && arm $FAM unsloth ckpt_unsloth_m_t1 unsloth $SAL "$MID" $REV 0 t1 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/e4b/m_r64        && arm $FAM e4b fused_attn4_m_r64 fused $SAL "$MID" $REV 0 r64 $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_r64    && arm $FAM unsloth ckpt_unsloth_m_r64 unsloth $SAL "$MID" $REV 0 r64 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_family FAM MID REV FETCH_AL E4B_AL UNS_AL HF_AL AX_AL REF_AL PROF_AL -- the JUDGED family (phase 3 I [F4]), one process per arm, in this order:
#   1 e4b/fused_attn4_m  2 unsloth/ckpt_unsloth_m  3 e4b/reference_attn4_m (THIRD: the e4b-side control sits beside the pair it controls)
#   4 e4b/fused_attn4_m_d2  5 unsloth/ckpt_unsloth_m_d2  6 hf/hf_peft_m  7 axolotl/ckpt_axolotl_m
#   8 e4b/fused_attn4_m_prof  9 unsloth/ckpt_unsloth_prof (3 warm + 3 profiled, dmon beside each)  then the _mb1 pair when a primary matched arm OOMed.
# The labelled / native-best rows run under the `qwen3native` token (tc1_native_family) on their own box.
tc1_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 HAL=$7 AAL=$8 RAL=$9 PAL=${TC1_PROF_ALARM:-${10}}
  local ALL="e4b:fused_attn4_m:fused unsloth:ckpt_unsloth_m:unsloth e4b:reference_attn4_m:reference e4b:fused_attn4_m_d2:fused unsloth:ckpt_unsloth_m_d2:unsloth hf:hf_peft_m:hf axolotl:ckpt_axolotl_m:axolotl e4b:fused_attn4_m_prof:fused unsloth:ckpt_unsloth_prof:unsloth"
  say "===== family $FAM ($MID @ $REV; matched seed $MATCHED_SEED; alarms e4b $EAL unsloth $UAL hf $HAL axolotl $AAL reference $RAL prof $PAL)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"      # the matched set
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"                   # the notebooks' recipe (tp4's arm); double-quant OFF is the arm's default
  local PROF="--log-every 1 --microbatch-timing 1 --profile-steps $PROFILE_STEPS --profile-warm $PROFILE_WARM"
  can_run 600 $FAM/e4b/fused_m     && arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m       && arm   $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 900 $FAM/e4b/reference_m && arm   $FAM e4b reference_attn4_m reference $RAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/fused_m_d2  && draw2 $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_d2    && draw2 $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/hf/m            && arm   $FAM hf hf_peft_m hf $HAL "$MID" $REV 0 field $TOK $TS $MATCH
  can_run 600 $FAM/axolotl/m       && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH
  local dp; dp=$(dmon_start ${FAM}_e4b_fused_attn4_m_prof)
  can_run 600 $FAM/e4b/prof        && arm   $FAM e4b fused_attn4_m_prof fused $PAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  dmon_stop $dp
  dp=$(dmon_start ${FAM}_unsloth_ckpt_unsloth_prof)
  can_run 600 $FAM/unsloth/prof    && arm   $FAM unsloth ckpt_unsloth_prof unsloth $PAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $PROF
  dmon_stop $dp
  for f in $W/logs/dmon_*.txt; do [ -s "$f" ] && echo "DMON $(basename $f) $(wc -l < $f) samples" >> summary.txt; done
  # the secondary pair (TC1-PREREG "Arms"): micro-batch 1 x accum 8 -- same tokens per step -- for any framework whose primary matched arm OOMed, as tp4
  local se su sh; se=$(status_of $FAM e4b fused_attn4_m); su=$(status_of $FAM unsloth ckpt_unsloth_m); sh=$(status_of $FAM hf hf_peft_m)
  if [ "$se" = oom ] || [ "$su" = oom ] || [ "$sh" = oom ]; then
    echo "SECONDARY $FAM: a primary matched arm OOMed (e4b=$se unsloth=$su hf=$sh) -> mb1 pair" | tee -a summary.txt
    can_run 600 $FAM/e4b/fused_m_mb1 && arm $FAM e4b fused_attn4_m_mb1 fused $EAL "$MID" $REV 0 mb1 $TOK $TS --attn-4bit 1 $MATCH
    can_run 600 $FAM/unsloth/m_mb1   && arm $FAM unsloth ckpt_unsloth_m_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
    [ "$sh" = oom ] && can_run 600 $FAM/hf/m_mb1 && arm $FAM hf hf_peft_m_mb1 hf $HAL "$MID" $REV 0 mb1 $TOK $TS $MATCH
  fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_native_family FAM MID REV FETCH_AL E4B_AL UNS_AL HF_AL AX_AL -- the LABELLED / native-best rows on their own box (phase 3 I/J), each a position
# within this box against the e4b fused_m it runs first:
#   e4b/fused_attn4_m (once)  unsloth/ckpt_unsloth_best  unsloth/ckpt_unsloth_t28  unsloth/ckpt_unsloth_triton  e4b/fused_attn4_shipped
#   e4b/fused_attn4_m_nodgrad (--dgrad 0: enable_fast_train's default)  e4b/fused_attn4_m_t212 (e4b + gnf4 on torch 2.12.1+cu130)
#   axolotl/ckpt_axolotl_best  hf/hf_peft_m_mb1_t214 (the HF arm on the axolotl venv's torch 2.14, experts_implementation=grouped_mm; only when
#   the judged family's hf_peft_m on THIS box OOMed -- otherwise a not_run row saying the gate could not be read)
tc1_native_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 HAL=$7 AAL=$8
  local ALL="e4b:fused_attn4_m:fused unsloth:ckpt_unsloth_best:unsloth unsloth:ckpt_unsloth_t28:unsloth unsloth:ckpt_unsloth_triton:unsloth e4b:fused_attn4_shipped:fused e4b:fused_attn4_m_nodgrad:fused e4b:fused_attn4_m_t212:fused axolotl:ckpt_axolotl_best:axolotl hf:hf_peft_m_mb1_t214:hf"
  say "===== NATIVE family $FAM ($MID @ $REV; the labelled rows, each against this box's own e4b fused_m)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  can_run 600 $FAM/e4b/fused_m     && arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/best    && arm   $FAM unsloth ckpt_unsloth_best unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm --unsloth-speed-tilt 1 --adapter-dtype fp32 --lora-init native
  can_run 600 $FAM/unsloth/t28     && UNS_VENV=t28 arm $FAM unsloth ckpt_unsloth_t28 unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend default $MATCH
  can_run 600 $FAM/unsloth/triton  && arm   $FAM unsloth ckpt_unsloth_triton unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend unsloth_triton $MATCH
  can_run 600 $FAM/e4b/shipped     && arm   $FAM e4b fused_attn4_shipped fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/nodgrad     && arm   $FAM e4b fused_attn4_m_nodgrad fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 --dgrad 0 $MATCH
  can_run 600 $FAM/e4b/t212        && E4B_VENV=t212 arm $FAM e4b fused_attn4_m_t212 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/axolotl/best    && arm   $FAM axolotl ckpt_axolotl_best axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json --axolotl-best 1 --adapter-dtype fp32 --lora-init native
  local sh; sh=$(status_of qwen3 hf hf_peft_m)
  if [ "$sh" = oom ]; then
    can_run 600 $FAM/hf/mb1_t214   && HF_VENV=t214 arm $FAM hf hf_peft_m_mb1_t214 hf $HAL "$MID" $REV 0 mb1 $TOK $TS --hf-experts-implementation grouped_mm $MATCH
  else
    stubw $FAM hf hf_peft_m_mb1_t214 hf not_run "runs only when the judged family's hf/hf_peft_m on this box OOMed; its status here is '$sh'"
  fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# ---------------------------------------------------------------- the plan (TC1-PREREG "Model", "Alarms": e4b 3600, Unsloth 3600, HF 1800, axolotl 2700, reference 5400, profiled 2400; fetch as tp4's qwen3)
#      FAM    MID                 REV                                       FETCH E4B  UNS  HF   AX   REF  PROF
for FAM in $FAMILIES; do case "$FAM" in
  qwen3)       tc1_family        qwen3       Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 1800 2700 5400 2400;;
  qwen3native) tc1_native_family qwen3native Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 1800 2700;;
  #                                                                                                        FETCH E4B  UNS  AX     (TC1 amendment 5: native-best vs native-best)
  qwen3nativebest) tc1_nativebest_family qwen3nativebest Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 2700;;
  qwen3prof945) tc1_prof945_family qwen3prof945 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 12 (#945)
  qwen3syncab) tc1_syncab_family qwen3syncab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 10 (#945)
  qwen3leanab) tc1_leanab_family qwen3leanab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 13 (#945)
  qwen3tileab) tc1_tileab_family qwen3tileab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 14 (#945)
  qwen3rmsab) tc1_rmsab_family qwen3rmsab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 15 (#945)
  qwen3reuseab) tc1_reuseab_family qwen3reuseab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 20 (#945)
  qwen3keepab) tc1_keepab_family qwen3keepab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 21 (#945)
  qwen3denseab) tc1_denseab_family qwen3denseab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 22
  qwen3bmmab)  tc1_bmmab_family  qwen3bmmab  Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 24: bmm replay + venv-e4b vs venv-unsloth
  qwen3samestackh2) tc1_samestack_family qwen3samestackh2 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 5400;;   # TC1 amendment 42: amendment 33's box on a second host
  qwen3samestack) tc1_samestack_family qwen3samestack Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 5400;;   # TC1 amendment 25: both frameworks on one stack
  # TC1 amendment 39: amendment 25's family in the packed 4,096-token regime (the box runs TC1_PACK=1 TC1_SEQ=4096 TC1_MB=1 TC1_ACCUM=4 TC1_STEPS=30;
  # refused above otherwise). Alarms for 30 steps of a 15-40 s step: ~150-250 s of prologue (load ~60-90 s, C1 ~60-80 s, eval0 now 8 x 4,096
  # tokens), 30 x 40 = 1,200 s of training, three held-out passes (0, 20, 30), ~60-90 s of epilogue: ~1,800 s at the top of the range. e4b
  # 3600 (2x that); Unsloth 5400 (its field step was 2.2x e4b's: at that ratio on a 40 s e4b step it needs ~3,300 s, and an alarm must not
  # turn a reading outside P84's band into UNTESTED); the reference (skipped on the registered box) 7200, its per-expert loop at 4,096 tokens.
  qwen3samestack4kce2) tc1_samestack_family qwen3samestack4kce2 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 5400 7200;;   # TC1 amendment 43: amendment 40, the LoRA loop a recorded route
  qwen3samestack4kd) tc1_samestack_family qwen3samestack4kd Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 5400 7200;;   # TC1 amendment 51: the packed same-stack position at e4b's defaults (chunked loss auto, buckets auto)
  qwen3samestack4kce) tc1_samestack_family qwen3samestack4kce Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 5400 7200;;   # TC1 amendment 40: amendment 39 with E4B_CHUNKED_LM_LOSS (TC1_E4B_ENV)
  qwen3samestack4k) tc1_samestack_family qwen3samestack4k Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 5400 7200;;
  qwen3samestackh100) tc1_samestack_family qwen3samestackh100 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 5400;;   # TC1c amendment 9: amendment 25 on an H100 NVL
  mixtralsamestack) tc1_samestack_family mixtralsamestack mistralai/Mixtral-8x7B-Instruct-v0.1 eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 3600 3600 5400;;   # TC2 amendment 9: amendment 25 on Mixtral, resident, defaults
  qwen3prebindab) tc1_prebindab_family qwen3prebindab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 26: prebound Triton launches off vs on
  qwen3prebind37) tc1_prebind37_family qwen3prebind37 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 35: amendment 26 on triton 3.7.1 (venv-unsloth)
  qwen3compactab) tc1_compactab_family qwen3compactab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 36: the compact padded LoRA delta off vs on (venv-unsloth)
  qwen3compactab2) tc1_compactab_family qwen3compactab2 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 37: amendment 36 with gnf4#473's backward, another host
  qwen3compactab3) tc1_compactab_family qwen3compactab3 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 38: the compact delta's default decision, a third host
  mixtralcompactab) tc1_compactab_family mixtralcompactab mistralai/Mixtral-8x7B-Instruct-v0.1 eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 3600;;   # TC1 amendment 38: the same on Mixtral, resident, defaults (shipped arms skipped)
  qwen3chunkab) tc1_chunkab_family qwen3chunkab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 41: e4b's chunked LM loss off vs on (venv-unsloth)
  qwen3chunkauto) tc1_chunkauto_family qwen3chunkauto Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 44: e4b's chunked LM loss off vs auto (venv-unsloth)
  qwen3ompab) tc1_ompab_family qwen3ompab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 5400;;   # TC1 amendment 45: OMP_NUM_THREADS physical cores vs the container's allotment
  olmoedecab) tc1_decodedab_family olmoedecab allenai/OLMoE-1B-7B-0924-Instruct 7f1c97f440f06ce36705e4f2b843edb5925f4498 2400 2400;;   # TC1 amendment 46: fused vs decoded (TC2's olmoe pin)
  qwen3decab) tc1_decodedab_family qwen3decab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 46: fused vs decoded
  qwen3dqab)   tc1_dqab_family   qwen3dqab   Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 28: absmax fp32 vs double-quantized
  mixtraldqab) tc1_dqab_family   mixtraldqab mistralai/Mixtral-8x7B-Instruct-v0.1 eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 3600;;   # TC1 amendment 28 (TC2's mixtral pin, fetch 7200, e4b 3600)
  qwen3tritonab) tc1_tritonab_family qwen3tritonab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 32: triton 3.4 vs 3.7.1 in venv-e4b
  qwen3envsplit) tc1_envsplit_family qwen3envsplit Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 34: transformers vs torch+triton
  mixtraldenseab) tc1_mixtral_denseab_family mixtraldenseab mistralai/Mixtral-8x7B-Instruct-v0.1 eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 3600;;   # TC1 amendment 22 (TC2's mixtral pin, fetch 7200, e4b 3600)
  qwen3memcensus) tc1_memcensus_family qwen3memcensus Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600;;   # TC1 amendment 23 (fetch 5400, e4b 3600, Unsloth 3600)
  qwen3memc4k) tc1_memc4k_family qwen3memc4k Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 5400 5400;;   # TC1 amendment 47: the memory census on packed 4,096-token rows
  qwen3memc4kb) tc1_memc4kb_family qwen3memc4kb Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 5400 5400;;   # TC1 amendment 55: amendment 47's census at the current defaults
  qwen3memc4kt) tc1_memc4kt_family qwen3memc4kt Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 5400 5400;;   # TC1 amendment 57: the census of the training phase on packed rows
  qwen3dqpack) tc1_dqpack_family qwen3dqpack Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 5400 5400;;   # TC1 amendment 56: the double-quantized absmax on packed rows, phase peaks
  qwen3ckptoff) tc1_ckptoff_family qwen3ckptoff Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 5400 5400;;   # TC1 amendment 58: checkpoint inputs in pinned host memory, off vs on, packed rows
  qwen3evalce) tc1_evalce_family qwen3evalce Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 5400 5400;;   # TC1 amendment 60: the held-out loss from the logits in chunks, packed rows
  qwen3combck) tc1_combck_family qwen3combck Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 5400 5400;;   # TC1 amendment 61: the combine whole vs over row chunks, packed rows
  qwen3ckptofff) tc1_ckptofff_family qwen3ckptofff Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 59: checkpoint inputs in pinned host memory, off vs on, field recipe
  qwen3ckptre) tc1_ckptre_family qwen3ckptre Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 62: the field speed-up's cause -- checkpoint flavour vs the copies
  qwen3padbk) tc1_padbk_family qwen3padbk Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 48: the LoRA delta's bucketed padding on packed rows
  qwen3padbk28) tc1_padbk28_family qwen3padbk28 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 52: amendment 48's bucketing A/B in venv-e4b (torch 2.8)
  qwen3prof28) tc1_prof28_family qwen3prof28 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 53: the matched arm profiled in torch 2.12 and 2.8 at e4b's defaults
  qwen3ladder28) tc1_ladder28_family qwen3ladder28 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 54: defaults vs the cuBLASLt cache vs the bucket ladder, torch 2.8
  qwen3fieldbk) tc1_fieldbk_family qwen3fieldbk Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 49: the LoRA delta's bucketed padding at the field recipe
  qwen3fieldauto) tc1_fieldauto_family qwen3fieldauto Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;   # TC1 amendment 50: the bucket gate at the field recipe
  routebench)  tc1_routebench_family routebench 1800;;   # TC1c amendment 3: a kernel-route replay (no model)
  fusedsweep)  tc1_fusedsweep_family fusedsweep 2400;;   # TC1c amendment 5: a fused-kernel config replay (no model)
  qwen3nativebest200) tc1_nativebest200_family qwen3nativebest200 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 4800 5400 4800;;   # TC1 amendment 8
  #                                                                                                        FETCH E4B  HF   AX     (amendment 3: the axolotl box)
  qwen3axolotl) tc1_axolotl_family qwen3axolotl Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 1800 2700;;
  #                                                                                                        FETCH E4B  UNS  ANCH SCALE   (TC1b alarms: e4b 200-step 4800, Unsloth 200-step 9000, anchor 1800 each, t1/r64 3600)
  qwen3curve)  tc1_curve_family  qwen3curve  Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 4800 9000 1800 3600;;
  tc2small)    tc2_small_box;;                 # lane TC2, box A: granite, olmoe, gptoss (tc2_small_box's table)
  tc2big)      tc2_big_box;;                   # lane TC2, box B: qwen3_5, mixtral (tc2_big_box's table)
  tc2mixtral)  tc2_mixtral_redraw;;            # lane TC2 amendment 2: Mixtral's P5 pair and reference redrawn with the Unsloth alarm at 7,200 s
  tc2qwen35off) tc2_qwen35_offload;;           # lane TC2 amendment 4: Qwen3.6's matched set, e4b under expert offload vs Unsloth resident with expert targets
  tc2resident) tc2_resident;;                  # lane TC2 amendment 6: Mixtral and Qwen3.6 with every e4b arm resident on the 32 GB card
  tc2mixtralres) tc2_mixtral_resident;;        # lane TC2 amendment 8, box M: Mixtral alone, every e4b arm resident at e4b's default settings
  tc2qwen35mb1) tc2_qwen35_mb1;;               # lane TC2 amendment 8, box Q: Qwen3.6 alone, resident, the primary pair at micro-batch 1
  # TC3 (TC3-PREREG-draft): the 24 GB RTX 4090 token (TC1_GPU_CLASS=4090) and the owned 12 GB RTX A2000 token (TC1_GPU_CLASS="RTX A2000", TC1_LOCAL_BOX=1)
  #                                                                                                        FETCH ERES EOFF MB1  UNS  HF   HOFF AX   ALO  AZ3  ROFF   (the draft's alarms)
  qwen3frontier)   tc1_frontier_family   qwen3frontier   Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 1200 3600 3600 3600 1800 3600 2700 3600 3600 5400;;
  #                                                                                                        FETCH ERES EOFF UNS  HF   AX   ROFF   (FETCH unused with a local snapshot)
  qwen3frontier12) tc1_frontier12_family qwen3frontier12 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 1200 7200 3600 1800 2700 14400;;
  *) say "unknown family token $FAM"; echo "UNKNOWN $FAM" >> summary.txt;;
esac; done
# ---------------------------------------------------------------- reduce, summarise, mark
say "reduce"
# TC2: the tc2small token's families run SMALL_STEPS, so the reducer reads N per family from the receipts (no --steps); every other token passes the field N
case " $FAMILIES " in *" tc2small "*) REDUCE_STEPS="";; *) REDUCE_STEPS="--steps $STEPS";; esac
$PY_E4B $W/tc1_reduce.py $W --md $W/RESULTS-tc1.md $REDUCE_STEPS > RESULTS.txt 2>&1; tail -40 RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
[ "$NEED_UNSLOTH" != 1 ] || [ "$UNS_OK" = 1 ] || echo "NO cu130 UNSLOTH COMPARATOR on this box (venv-unsloth: ${CU130_OK:-?} driver gate, install ok=$UNS_OK): its rows are refused/install_failed" | tee -a summary.txt
[ "$NEED_UNSLOTH" != 1 ] || [ "$UNS_T28_OK" = 1 ] || echo "NO torch-2.8 UNSLOTH ROW on this box (venv-unsloth-t28 did not install/import)" | tee -a summary.txt
[ "$AX_OK" = 1 ] || echo "NO AXOLOTL on this box: $AX_REASON" | tee -a summary.txt
case " $FAMILIES " in *" qwen3frontier "*) echo "AXOLOTL[deepspeed] extra installed=$AX_DS_OK (TC3 arm 10 is a refused row either way; its stub records the deepspeed version)" | tee -a summary.txt;; esac
finish 0
