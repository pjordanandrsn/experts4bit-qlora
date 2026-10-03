#!/bin/bash
# bench/p100/p100_run.sh -- lane P100, BOX side (bench/p100/PREREG-p100.md; e4b#916). Started detached by p100_drive.sh
# with the run's nonce; P100_RUN_NONCE first, then P100_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P100_SUCCESS.<nonce> only when the reducer ran.
#
# Confirm, with no code change, where the paged prefill of Qwen3-30B-A3B on the int4 expert store spends its time
# (e4b#916): SC1 box B read TTFT 2.08 s at 512 tokens and 16.6 s at 4096 (chunk 512). On ONE RTX 5090, at the launch
# commit, each arm a fresh engine in its own process:
#   c512_t512    TTFT-512  at E4B_PAGED_CHUNK_TOKENS=512  (SC1's arm, one chunk) + the dispatch census + the kernel count
#   c2048_t4096  TTFT-4096 at chunk 2048 (two chunks)    + the dispatch census
#   prof         step_decomp.py --cprofile-out at --batch 1, int4 experts, one 512-token chunk + the dispatch census
#   c512_t4096   TTFT-4096 at chunk 512 (SC1's arm, eight chunks) + the dispatch census
#   c1024_t4096  TTFT-4096 at chunk 1024 (four chunks)   + the dispatch census
# The TTFT arms are bench/sc1/sc1_e4b_sched.py --ttft at its registered bytes, with SC1's stack env, through
# p100_box.py (which counts only in an extra request AFTER the timed ones). The token ids are SC1's (sc1_prompts.py);
# the runner refuses if their digests differ from SC1's reading.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P100_GPU_CLASS P100_MIN_DISK_GB P100_REHEARSAL P100_ARMS. The guard is 1 h, so there is no proving run.
set -uo pipefail
W=/root/p100; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p100: $*"; }
NONCE=${P100_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P100_RUN_NONCE.tmp && mv $W/P100_RUN_NONCE.tmp $W/P100_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P100_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P100_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P100_RUN_ID P100_DEADLINE_EPOCH P100_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1, e4b CI's pin and SC1 box B's
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
SHA_B1=a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3        # SC1 box B prompts_b1.json
SHA_B1_4096=cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd   # SC1 box B prompts_b1_4096.json
GPU_CLASS=${P100_GPU_CLASS:-5090}; MIN_DISK_GB=${P100_MIN_DISK_GB:-110}; REHEARSAL=${P100_REHEARSAL:-0}
ARMS=${P100_ARMS:-c512_t512 c2048_t4096 prof c512_t4096 c1024_t4096}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever starts unset; each arm sets SC1's stack explicitly
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_PLACEMENT E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_INT4_ARTIFACT_DIR E4B_INT4_EXPECTED_FINGERPRINT PYTHONPATH
# SC1's stack, byte for byte (bench/sc1/sc1_run.sh SPEEDENV / FOLDS / ROUTEENV; the int4_sched arms SC1 box B's TTFT ran)
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
: > summary.txt; echo "$P100_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB arms='$ARMS'" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 110 ] || [ "$ARMS" != "c512_t512 c2048_t4096 prof c512_t4096 c1024_t4096" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p100_box.py p100_reduce.py sc1_e4b_sched.py sc1_prompts.py step_decomp.py p98_bake.py calib.json p42_usercustomize.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p100/staged.sha256"; finish 9; }
# P42's hook goes under hook/ (step_decomp's levers); it is staged under another name so no process imports it by accident
mkdir -p $W/hook && cp $W/p42_usercustomize.py $W/hook/usercustomize.py || { say "STAGE FAIL: hook"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the 61 GB checkpoint, its NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P100_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P100_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P100_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
# The tripwire: the commits, and the code under test present AS the trace read it -- so a reading is about this path.
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, site, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora import serve_paged
src = inspect.getsource(hr._fused_over_stack)
assert 'w = dequant_int4_ref(st["packed"][e_],' in src, "the int4 store's host-grouped loop is not the code the trace read"
assert {"local_ids", "int4_stores", "singleton_groups", "device_grouping"} <= set(inspect.signature(hr._fused_over_stack).parameters)
assert "cfg.max_seqs > 1" in inspect.getsource(serve_paged._batched_graph_grouping), "max_seqs == 1 no longer leaves DEVICE_GROUPING off"
assert hr.DEVICE_GROUPING == [False], hr.DEVICE_GROUPING
import int4_pack_ref, fp8_paged_attn  # noqa: F401
assert callable(int4_pack_ref.dequant_int4_ref)
assert site.ENABLE_USER_SITE, "usercustomize would not load: step_decomp's levers (P42's hook) need the user site"
import experts4bit_qlora as e, torch, triton
open("/root/p100/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p100_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p100_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
# ---- the checkpoint (pinned), the NF4 arena (bench/p98/p98_bake.py, revision-pinned), SC1's prompt files
can_run 2400 fetch_qwen3 || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(arm_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch_qwen3.log 2>&1 || { tail -2 logs/fetch_qwen3.log; say "DL FAIL"; finish 11; }
tail -1 logs/fetch_qwen3.log | tee -a summary.txt
can_run 1800 bake_qwen3 || finish 40
say "bake qwen3"; mkdir -p $W/work_qwen3
perl -e "alarm $(arm_alarm 1800); exec @ARGV" python $W/p98_bake.py --model "$MODEL" --revision "$REV" --work $W/work_qwen3 > logs/bake_qwen3.log 2>&1
rc=$?
{ echo -n "bake qwen3 rc=$rc "; python -c "import json; r=json.load(open('$W/work_qwen3/bake.json')); print({k: r.get(k) for k in ('status','layers','experts','snapshot_gib','loaded_commit','err')})" 2>&1 | cut -c1-300; } | tee -a summary.txt
[ "$rc" = 0 ] && [ -e $W/work_qwen3/nf4.arena ] || { tail -5 logs/bake_qwen3.log; say "BAKE FAIL"; finish 12; }
say "prompt dump: sc1_prompts.py (step_decomp._k8_window) -> prompts_b1.json + prompts_b1_4096.json (and SC1's other files)"
SC1_HARNESS_DIR=$W perl -e "alarm $(arm_alarm 1200); exec @ARGV" python $W/sc1_prompts.py --model "$MODEL" --rev "$REV" --out $W > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT DUMP FAIL"; finish 19; }
grep -aE "^PROMPTS|^WINDOW" logs/prompts.log | tee -a summary.txt
for pair in "prompts_b1.json:$SHA_B1" "prompts_b1_4096.json:$SHA_B1_4096"; do
  f=${pair%%:*}; want=${pair#*:}
  got=$(python -c "import json; print(json.load(open('$W/$f'))['prompts_sha256'])" 2>/dev/null)
  [ "$got" = "$want" ] || { say "REFUSED: $f digest $got is not SC1 box B's $want -- not the token ids SC1 read"; echo "refused: prompts $f" > REFUSAL; finish 19; }
  echo "PROMPTS_MATCH_SC1 $f $got" | tee -a summary.txt
done
# ---- the arms, each a fresh engine in its own process
ttft_arm(){ local NAME=$1 C L PF MTS LC=0 AL S
  C=${NAME#c}; C=${C%%_*}; L=${NAME##*_t}
  case "$L" in 512) PF=$W/prompts_b1.json; MTS=2048;; 4096) PF=$W/prompts_b1_4096.json; MTS=4104;; *) say "unknown prompt $L"; return 26;; esac
  [ "$NAME" = c512_t512 ] && LC=1
  AL=$(arm_alarm 1200); can_run 600 "arm_$NAME" || return 40
  S=ttft_$NAME; say "arm $NAME: TTFT-$L at chunk $C (launch census $LC; alarm=$AL)"
  env PYTHONPATH= $ROUTEENV $SPEEDENV \
    E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work_qwen3/nf4.arena E4B_PAGED_CALIB=$W/calib.json \
    E4B_PAGED_PLACEMENT=all-vram E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=$MTS E4B_PAGED_CHUNK_TOKENS=$C \
    E4B_PAGED_GRAPHS=1 E4B_PAGED_BUCKETS=1 E4B_PAGED_FUSE_QKV=1 E4B_PAGED_TORCH_THREADS=8 \
    E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA SC1_ARM=$S SC1_BATCH=1 SC1_PROMPTS=$PF SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$P100_INSTANCE_ID \
    P100_CENSUS_OUT=$W/census_$NAME.json P100_LAUNCH_CENSUS=$LC \
    perl -e "alarm $AL; exec @ARGV" python $W/p100_box.py ttft > logs/arm_$NAME.log 2>&1
  local rc=$?
  { echo -n "arm $NAME rc=$rc "; grep -aE "^SC1SCHED |^P100_CENSUS " logs/arm_$NAME.log | cut -c1-300 | tr '\n' ' '; } >> summary.txt; echo >> summary.txt
  [ "$rc" = 0 ] || return 26; }
prof_arm(){ local AL; AL=$(arm_alarm 1500); can_run 900 arm_prof || return 40
  say "arm prof: step_decomp.py --cprofile-out at --batch 1, int4 experts (P42's hook), one 512-token chunk (alarm=$AL)"
  env PYTHONPATH=$W/hook $ROUTEENV $SPEEDENV P100_CENSUS_OUT=$W/census_prof.json \
    perl -e "alarm $AL; exec @ARGV" python $W/p100_box.py prof --model "$MODEL" --arena $W/work_qwen3/nf4.arena --calib $W/calib.json \
      --placement-override all-vram --amort off --batch 1 --prompt-len 512 --chunk 512 --gen-tokens 32 --fuse-qkv \
      --cprofile-out $W/cprofile_b1.txt --out $W/prof_b1.json > logs/arm_prof.log 2>&1
  local rc=$?
  { echo -n "arm prof rc=$rc "; grep -aE "^P100_CENSUS |CPROFILE_OUT|INT4EXP enabled|GEN HOOK" logs/arm_prof.log | cut -c1-200 | tr '\n' ' '; grep -a "dequant_int4_ref" $W/cprofile_b1.txt 2>/dev/null | head -1 | tr -s ' '; } >> summary.txt; echo >> summary.txt
  [ "$rc" = 0 ] || return 26; }
for N in $ARMS; do
  case "$N" in prof) prof_arm; rec $?;; c*_t*) ttft_arm "$N"; rec $?;; *) say "unknown arm $N"; rec 26;; esac
done
say "reduce"; python $W/p100_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish "$rc_any"
