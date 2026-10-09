#!/bin/bash
# bench/p127/p127_run.sh -- lane P127, BOX side (bench/p127/PREREG-p127.md; e4b#1313). Started detached by p127_drive.sh
# with the run's nonce; P127_RUN_NONCE first, then P127_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P127_SUCCESS.<nonce> only when the reducer ran (or, under P127_PROVE=1, when the proof passed).
#
# The launch-bound glue, before and after, on ONE RTX 5090: the engine built as the shipped server builds it at its
# defaults (E4B_PAGED_MAX_SEQS=16 fixed), in five arms of their own processes -- A1 B1 B2 A2 M1. A is e4b a8c01d42 +
# grouped-nf4-gemm v0.44.0; B is the launch commit + gnf4 at #527's merge; M is B with a round-toward-zero router-weight
# store. Each install is a clone of its own; the box switches the editable installs before each arm.
#   order    refusals; install + clones + the diff audit + tripwire; self-tests; premise; fetch; bake; prompts; arms;
#            reduce
# Knobs (recorded; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P127_GPU_CLASS P127_MIN_DISK_GB P127_MIN_RAM_GB P127_REHEARSAL P127_SHORT P127_LONG P127_REPS.
# P127_PROVE=1 is the PROVING RUN: everything above on Qwen3-30B-A3B itself at 8 / 24 tokens, 1 rep (the only family
# that engages every P127 path).
set -uo pipefail
W=/root/p127; mkdir -p $W/logs $W/src; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p127: $*"; }
NONCE=${P127_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P127_RUN_NONCE.tmp && mv $W/P127_RUN_NONCE.tmp $W/P127_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P127_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P127_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P127_RUN_ID P127_DEADLINE_EPOCH P127_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
# ---- the registered stacks (PREREG "Arms") and the diff audit's allow-lists (PREREG "The served-path diff audit")
E4B_A=a8c01d426bc3e489c8bd11c7c8ced091da557582     # the last main commit before #1448
GNF4_A=d1f64ba50afce94533e0166ba3332ef075aa43bb    # grouped-nf4-gemm v0.44.0
GNF4_B=d769d5022c0fb7a2ada847f69a3cba6e0f45c77f    # grouped-nf4-gemm at #527's merge (carries #526-#530)
E4B_B=$E4B_SHA                                     # the launch commit: this registration's merge
E4B_P127="ce3dfb5413a33c780a76eac7a8142a881515a444 1ddcb0aeeae63f32d57ab67fa3f5e48d8c370781 8ea9729608135fb47fa94835258eceecc68bb8b7"
E4B_INERT="2384d2c3083b615a0c8f117cb30799e0a59f6171 058f98eb419ff463dd19c21096ecdc6ce1eff1be f539896244bf418ac782ef4257f32d8fbd0c236b abe7d77223ed19a4f18f6f412877982862604213"
GNF4_P127="18f5bdaa491f4ff85c2c984c01499147aacd6267 f69adccf839b5e2dfb8b6eecd957232e1decb4f9 d0a2e56d903dad0385f1a86d7afe225caf363105 b64a39b5067db68e40b1bb365dbe95766c3d5d26 $GNF4_B"
GNF4_INERT="14b1f23d0befb5aef64aaa638863e69ef4418d04 e21a71242b361dd0a3f72cdf4152de94633891a8 7d4163b6c3827ae982c8826da28c3b89a636cdcc"
case "$GNF4_B $E4B_P127" in *__*) say "refusing: a registered SHA is still a placeholder"; finish 78;; esac
PROVE=${P127_PROVE:-0}
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                    # SC1's pin; the proof's too
if [ "$PROVE" = 1 ]; then
  SHORT_DEF=8; LONG_DEF=24; REPS_DEF=1; NEED_FETCH=1800; NEED_BAKE=1500; NEED_ARM=600
else
  SHORT_DEF=32; LONG_DEF=160; REPS_DEF=3; NEED_FETCH=2400; NEED_BAKE=1800; NEED_ARM=900
fi
GPU_CLASS=${P127_GPU_CLASS:-5090}; MIN_DISK_GB=${P127_MIN_DISK_GB:-150}; MIN_RAM_GB=${P127_MIN_RAM_GB:-60}
REHEARSAL=${P127_REHEARSAL:-0}; SHORT=${P127_SHORT:-$SHORT_DEF}; LONG=${P127_LONG:-$LONG_DEF}; REPS=${P127_REPS:-$REPS_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset: the default server is the subject (max_seqs is fixed below)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS E4B_FUSE_COMBINE E4B_INT4_TILE_PROGRAMS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS E4B_PAGED_DECODE_LOOKAHEAD \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS GNF4_GEMV_BW GNF4_PDL
: > summary.txt; echo "$P127_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b_a=$E4B_A e4b_b=$E4B_B gnf4_a=$GNF4_A gnf4_b=$GNF4_B model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE short=$SHORT long=$LONG reps=$REPS" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$SHORT" != "$SHORT_DEF" ] || [ "$LONG" != "$LONG_DEF" ] || [ "$REPS" != "$REPS_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p127_box.py p127_reduce.py p109_box.py k8_bake.py calib.json test_decode_graph_buckets.py test_t1_glue_host_casts.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p127/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk, the host RAM
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P127_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P127_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P127_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
# ---- the dependencies once; then each stack's clone at its commit (blob-less, with the history the audit reads)
say "install deps (transformers 5.17.0, bitsandbytes 0.50.2; torch held at $TORCH_PIN)"
pipx logs/pip_deps.log 1800 --prefer-binary "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece \
  safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_deps.log; say "PIP FAIL (deps)"; finish 9; }
clone(){ local url=$1 dir=$2
  [ -d $dir/.git ] || perl -e 'alarm 900; exec @ARGV' git clone -q --filter=blob:none "$url" $dir > logs/clone_$(basename $dir).log 2>&1; }
clone https://github.com/pjordanandrsn/experts4bit-qlora.git $W/src/e4b || { say "CLONE FAIL (e4b)"; finish 9; }
clone https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/src/gnf4 || { say "CLONE FAIL (gnf4)"; finish 9; }
for side in A B; do
  E=E4B_$side; G=GNF4_$side
  git -C $W/src/e4b worktree add -q --detach $W/src/e4b_$side ${!E} > /dev/null 2>&1 || { say "CHECKOUT FAIL e4b_$side ${!E}"; finish 9; }
  git -C $W/src/gnf4 worktree add -q --detach $W/src/gnf4_$side ${!G} > /dev/null 2>&1 || { say "CHECKOUT FAIL gnf4_$side ${!G}"; finish 9; }
done
# ---- the served-path diff audit (PREREG): every package commit between the two stacks is P127's or registered inert
E4B_LIST=$(git -C $W/src/e4b log --format=%H $E4B_A..$E4B_B -- experts4bit_qlora/)
GNF4_LIST=$(git -C $W/src/gnf4 log --format=%H $GNF4_A..$GNF4_B -- kernel/ gnf4_native/)
E4B_LIST="$E4B_LIST" GNF4_LIST="$GNF4_LIST" OK_E4B="$E4B_P127 $E4B_INERT" OK_GNF4="$GNF4_P127 $GNF4_INERT" python - <<'PYA' || { say "AUDIT FAIL"; finish 31; }
import json, os
unl = [("e4b", c) for c in os.environ["E4B_LIST"].split() if c not in os.environ["OK_E4B"].split()]
unl += [("gnf4", c) for c in os.environ["GNF4_LIST"].split() if c not in os.environ["OK_GNF4"].split()]
json.dump({"ok": not unl, "unlisted": unl, "e4b": os.environ["E4B_LIST"].split(), "gnf4": os.environ["GNF4_LIST"].split()},
          open("/root/p127/audit.json", "w"), indent=1)
print("AUDIT", "ok" if not unl else f"REFUSED unlisted={unl}")
raise SystemExit(0 if not unl else 31)
PYA
grep -a "" audit.json | head -3 | tee -a summary.txt
# the install switch: each arm's e4b and gnf4 clones, editable, nothing else moves
use(){ pipx logs/pip_use_$1.log 600 --no-deps -e $W/src/e4b_$1 -e $W/src/gnf4_$1 || { tail -3 logs/pip_use_$1.log; say "PIP FAIL (use $1)"; finish 9; }; }
use B
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, inspect, importlib.metadata as md
import experts4bit_qlora as e, int4_b32, nf4_grouped, transformers, torch, triton
assert "/src/e4b_B/" in e.__file__ and "/src/gnf4_B/" in int4_b32.__file__, (e.__file__, int4_b32.__file__)
assert transformers.__version__ == "5.17.0", transformers.__version__
assert "weights_dtype" in inspect.signature(int4_b32.router_epilogue).parameters, "B's gnf4 lacks #526"
assert hasattr(int4_b32, "rope_norm_qk") and "residual" in inspect.signature(int4_b32.combine_rows).parameters
assert torch.int64 in nf4_grouped.EXPERT_ID_DTYPES and "gather_div" in inspect.signature(nf4_grouped.gemm_4bit_grouped).parameters
from experts4bit_qlora.engines import glue_r2
assert hasattr(glue_r2, "license_moe_residual"), "the launch commit lacks the residual PR"
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401
open("/root/p127/versions.txt", "a").write(f"e4b B {e.__version__}\ngnf4 B {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, md.version("grouped-nf4-gemm"))
PYT
cat versions.txt | tee -a summary.txt
python $W/p127_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p127_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card at B's stack, before anything is fetched
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py -q -rs -p no:cacheprovider) > logs/premise_graphs.log 2>&1
rc1=$?; L1=$(tail -1 logs/premise_graphs.log)
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_t1_glue_host_casts.py -q -rs -p no:cacheprovider -k real_kernels) > logs/premise_kernels.log 2>&1
rc2=$?; L2=$(tail -1 logs/premise_kernels.log)
{ echo "premise graphs rc=$rc1: $L1"; echo "premise kernels rc=$rc2: $L2"; } | tee -a summary.txt
if [ "$rc1" = 0 ] && echo "$L1" | grep -q "7 passed" && ! echo "$L1" | grep -q skipped \
   && [ "$rc2" = 0 ] && echo "$L2" | grep -q "1 passed" && ! echo "$L2" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed" > REFUSAL; finish 25
fi
# ---- the checkpoint, its NF4 arena (P39's k8_bake.py, as SC1 bakes it) and P109's prompts
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
perl -e "alarm 1200; exec @ARGV" python $W/p127_box.py --prompts-only --model "$MODEL" --revision "$REV" --out $W/prompts.json > logs/prompts.log 2>&1 \
  || { tail -4 logs/prompts.log; say "PROMPTS FAIL"; finish 19; }
grep -a "^P127_PROMPTS" logs/prompts.log | tee -a summary.txt
# ---- the five arms: A1 B1 B2 A2 M1, each a fresh process on its own install (M: identity pass only)
# E4B_INT4_TILE_PROGRAMS=1: #1476 made `auto` the default after A; inert at this subject (PREREG audit), pinned anyway
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=16 E4B_INT4_TILE_PROGRAMS=1"
for TAG in A1 B1 B2 A2 M1; do
  ARM=${TAG:0:1}; SIDE=$ARM; [ "$ARM" = M ] && SIDE=B
  E=E4B_$SIDE; G=GNF4_$SIDE; EXTRA=""; [ "$ARM" = M ] && EXTRA="--identity-only"
  can_run $NEED_ARM "arm $TAG" || finish 40
  use $SIDE
  AL=$(step_alarm 2400); say "arm $TAG on stack $SIDE (alarm=$AL)"
  # shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
  env PYTHONPATH= $ENGINE_ENV P127_ARM=$ARM E4B_SHA=${!E} GNF4_SHA=${!G} \
    perl -e "alarm $AL; exec @ARGV" python $W/p127_box.py --prompts $W/prompts.json --out $W/arm_$TAG.json --tag $TAG \
      --short $SHORT --long $LONG --reps $REPS $EXTRA > logs/arm_$TAG.log 2>&1
  rc=$?
  { echo -n "arm $TAG rc=$rc "; grep -a "^P127_ARM" logs/arm_$TAG.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
  [ "$rc" = 0 ] || { tail -6 logs/arm_$TAG.log | cut -c1-300 | tee -a summary.txt; say "ARM $TAG FAILED (rc=$rc) -- the reducer will VOID"; }
done
say "reduce"; python $W/p127_reduce.py --dir $W --out $W/verdict.json --e4b-a $E4B_A --e4b-b $E4B_B --gnf4-a $GNF4_A \
  --gnf4-b $GNF4_B --revision $REV 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; print(json.load(open('$W/verdict.json'))['verdict'])")
  [ "$V" != VOID ] || { say "PROVE: the reducer VOIDed the proof's arms -- not proved"; finish 27; }
  echo "PROVE -- the proving run: the whole box on $MODEL at $SHORT/$LONG; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
