#!/bin/bash
# bench/p82/p82_run.sh -- lane P82, BOX side (bench/p82/PREREG-p82.md; e4b#511, e4b#674, e4b#777). Derived from
# bench/p82/p82_run.sh (preflight, install, tripwire, fetch, bake, the build dumping both packs, the five arms) with three
# changes: every process runs the licensed build's fp32 router weights (E4B_ROUTER_EPI_CAST=0, a registered knob here),
# and the runner stamps what the router module reads under each process's env into a sidecar (<receipt>.router.json);
# the tripwire refuses a cut without #777's bucket-1 routing or grouped-nf4-gemm#413's append; and two K8 arms after
# the five read wikitext through BOTH packs loaded by fingerprint, once with the fp32 router (K32) and once with the
# shipped cast default (K16), so the cast's share of #674's K8 gap is read on identical bytes on one box.
# step_decomp.py and the hook are bench/p82's (referenced; tests/test_p82_staged_pin.py pins them), so the dynb
# receipts carry "lane": "P82" and the banner P82_DYNB: that names the harness copy, not this lane.
# Started detached by p82_drive.sh with the run's nonce; P82_RUN_NONCE first, then P82_EXIT_CODE.<nonce> + TP_DONE.<nonce>
# on every exit and P82_SUCCESS.<nonce> only when the reducer ran over five receipts.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P82_MODEL P82_REVISION P82_LICENSED_FP P82_GPU_CLASS P82_MIN_DISK_GB P82_SEG P82_CALIB_NSEQ P82_HESSIAN_BUDGET_GB
# P82_BUILD_PPL_STEPS P82_FIRST_CHUNK_S P82_SKIP_INSTALL P82_REHEARSAL.
# P82_PROVE=1 is the PROVING RUN: refusals, the install at the pins, the tripwire and an egress probe -- no model; exit 0.
set -uo pipefail
W=/root/p82; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p82: $*"; }
NONCE=${P82_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P82_RUN_NONCE.tmp && mv $W/P82_RUN_NONCE.tmp $W/P82_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P82_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P82_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P82_RUN_ID P82_DEADLINE_EPOCH P82_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; done
[ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] || { say "refusing: a pin is not a 40-char sha"; finish 78; }
# ---- registered defaults (PREREG-p82 "Instrument"); a rehearsal overrides them and says so
D_MODEL=Qwen/Qwen3-30B-A3B; D_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
D_FP=sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42     # P55x's licensed expert pack
MID=${P82_MODEL:-$D_MODEL}; REV=${P82_REVISION:-$D_REV}; LIC_FP=${P82_LICENSED_FP:-$D_FP}; GPU_CLASS=${P82_GPU_CLASS:-5090}
MIN_DISK_GB=${P82_MIN_DISK_GB:-200}; SEG=${P82_SEG:-32}; NSEQ=${P82_CALIB_NSEQ:-128}; HBUDGET=${P82_HESSIAN_BUDGET_GB:-24}
BUILD_STEPS=${P82_BUILD_PPL_STEPS:-2048}; SKIP_INSTALL=${P82_SKIP_INSTALL:-0}; REHEARSAL=${P82_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export PYTHONPATH=$W/hook${PYTHONPATH:+:$PYTHONPATH} E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=$HBUDGET
# the levers this lane does not read stay at their shipped defaults; the licensed stack's own switches are set per step
unset E4B_SERVE_EXP_INT4 E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB E4B_SERVE_ATTN_INT4 \
      E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_SERVE_ATTN_INT4_ARTIFACT E4B_SERVE_ATTN_INT4_FINGERPRINT E4B_SERVE_ATTN_INT4_DUMP E4B_INT4_ARTIFACT_DIR \
      E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_DUMP_ARTIFACT_DIR
# the licensed build's router arithmetic (P55x/P64/P70 on e4b <= 0.37.4, before 0.37.5 made the cast the default):
# fp32 routing weights. Every process inherits it; K16 alone removes it (below).
export E4B_ROUTER_EPI_CAST=0
: > summary.txt; echo "$P82_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS router_epi_cast=$E4B_ROUTER_EPI_CAST model=$MID rev=$REV licensed_fp=$LIC_FP gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB seg=$SEG calib_nseq=$NSEQ hessian_budget_gb=$HBUDGET build_ppl_steps=$BUILD_STEPS skip_install=$SKIP_INSTALL" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$MID" != "$D_MODEL" ] || [ "$REV" != "$D_REV" ] || [ "$LIC_FP" != "$D_FP" ] || [ "$GPU_CLASS" != 5090 ] \
   || [ "$SEG" != 32 ] || [ "$NSEQ" != 128 ] || [ "$HBUDGET" != 24 ] || [ "$BUILD_STEPS" != 2048 ] || [ "$SKIP_INSTALL" != 0 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p82_reduce.py step_decomp.py k8_bake.py calib.json hook/usercustomize.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p82/staged.sha256"; finish 9; }
python $W/p82_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is fetched: the card class and the disk the working set needs (host-limited: 13)
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the bf16 checkpoint + the NF4 arena; overlay is not machine disk)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard (P54's): never start a step that cannot finish 10 min before teardown
arm_alarm(){ local left=$(( P82_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install: e4b pinned + P37's toolchain pins (image python); gnf4 at the registered cut
export DEBIAN_FRONTEND=noninteractive
if [ "$SKIP_INSTALL" = 0 ]; then
  say "install e4b @$E4B_SHA (image python; P37 pins) + gnf4 @$GNF4_SHA"
  perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
    "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
  perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
else
  say "install SKIPPED (P82_SKIP_INSTALL=1): the importable packages are whatever the environment provides -- recorded below"
fi
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, sys, inspect, importlib.metadata as md
sys.path.insert(0, "/root/p82")
from experts4bit_qlora.engines.paged_runner import PagedModelRunner, bucket_for  # noqa: F401
assert hasattr(PagedModelRunner, "enable_decode_graphs"), "e4b cut lacks enable_decode_graphs (#757)"
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
assert "scratch_slots" in inspect.signature(Fp8PagedKV.__init__).parameters, "e4b cut lacks Fp8PagedKV(scratch_slots=)"
from experts4bit_qlora.engines import hot_residency as hr
assert hr.DEVICE_GROUPING[0] is False and hr.FORCE_SINGLETON_GROUPS[0] is False, "grouping flags are not at their shipped defaults"
import fp8_kv
assert hasattr(fp8_kv, "fp8_kv_append_bt1"), "gnf4 cut lacks the fused batch KV append the graphs need"
assert "_e4m3_group" in inspect.getsource(fp8_kv), "gnf4 cut lacks #413's append (quantize_kv_fp8's bytes; >= 0.33.7)"
from experts4bit_qlora.engines import paged_attention as pa
assert '"_g_sel", None) is not None' in inspect.getsource(pa.paged_attention_forward), \
    "e4b cut lacks #777 (a bound one-row bucket appends to its own slot)"
from experts4bit_qlora.engines import router_epilogue as rte
assert os.environ.get("E4B_ROUTER_EPI_CAST") == "0" and rte.CAST_WEIGHTS[0] is False, \
    f"the router is not at the licensed fp32 weights (env {os.environ.get('E4B_ROUTER_EPI_CAST')!r}, CAST_WEIGHTS {rte.CAST_WEIGHTS[0]!r})"
import step_decomp
assert hasattr(step_decomp, "_dynb_plan") and hasattr(step_decomp, "_dynb_stage"), "staged step_decomp lacks the P81 stage"
from experts4bit_qlora.engines.int4_attn_calib import dump_attn_int4_artifact, enable_serve_attn_int4_from_artifact  # noqa: F401  (#754)
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
for k in ("artifact_dir", "expected_fingerprint", "dump_artifact_dir"):
    assert k in inspect.signature(f).parameters, f"e4b cut lacks the #405 knob {k!r}"
import site
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
sys.path.insert(0, "/root/p82/hook")
import usercustomize  # noqa: F401
assert "enable_from_env" in open("/root/p82/hook/usercustomize.py").read(), "the staged hook is not v7"
import experts4bit_qlora as e, torch, triton, transformers
gnf4 = md.version("grouped-nf4-gemm")
open("/root/p82/versions.txt", "w").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']} (from {os.path.dirname(e.__file__)})\ngnf4 {gnf4} @{os.environ['GNF4_SHA']}\n"
                                          f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n"
                                          f"cc {torch.cuda.get_device_capability()}\nrouter_epi_cast_weights {rte.CAST_WEIGHTS[0]}\n")
print("tripwire OK: e4b", e.__version__, "gnf4", gnf4, "torch", torch.__version__, "triton", triton.__version__)
PYT
cat versions.txt | tee -a summary.txt
if [ "${P82_PROVE:-0}" = 1 ]; then
  echo "PROVE -- the proving run: no model, no arm" | tee -a summary.txt
  say "PROVE: HF CDN egress probe (50 MB range, 20 s cap; recorded, not a refusal)"
  python - <<'PYE' 2>&1 | tail -1 | tee -a summary.txt forensics.txt
import time, urllib.request
t = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request("https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors",
                                                      headers={"Range": "bytes=0-52428799"}), timeout=20)
    n = len(r.read())
    print(f"PROVE hf_cdn_mbps={n / (time.time() - t) / 1e6:.1f} bytes={n}")
except Exception as e:
    print(f"PROVE hf_cdn_probe_failed {type(e).__name__}: {str(e)[:120]}")
PYE
  : > PROVED; finish 0
fi
# ---- fetch (pinned) and bake (P39's k8_bake.py), as P70
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
bake_fail(){ say "BAKE FAIL ($1)"; python -c "import json; r=json.load(open('$W/work/bake.json')); print('BAKE_ERR', r.get('status'), r.get('step')); print(r.get('err')); print(r.get('tb'))" 2>/dev/null | tee -a logs/bake.log | tail -n 25 | cut -c1-400; finish 12; }
QA=$W/work/nf4.arena
say "bake NF4 arena"; mkdir -p $W/work
K8_MODEL="$MID" K8_WORK="$W/work" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; bake_fail rc; }
[ -e "$QA" ] || bake_fail "no arena"
grep -aE "arena|BAKE" logs/bake.log | tail -1 | tee -a summary.txt
# ---- the build (P70's, both packs dumped): the licensed recipe -- streamed calibrated int4 experts + calibrated int4
# attention + round-1/2 folds + router epilogue -- and its K8 reading, in ONE process whose attention bytes are dumped
# the router stamp (sidecar <receipt>.router.json): router_epilogue.CAST_WEIGHTS is read from E4B_ROUTER_EPI_CAST at
# import and nothing in step_decomp or the hook assigns it (tests/test_p82_staged_pin.py), so a process under the
# same env reads what the arm's process read. The receipt itself stays byte-for-byte as the harness wrote it.
stamp(){ local out=$1; shift
  env "$@" python - "$out" <<'PYS' || say "router stamp failed for $out"
import json, os, sys
from experts4bit_qlora.engines import router_epilogue as r
json.dump({"router_epi_cast_env": os.environ.get("E4B_ROUTER_EPI_CAST"), "router_epi_cast_weights": r.CAST_WEIGHTS[0],
           "router_epi_casts_softmax_topk": r._cast_for("softmax_topk"),
           "expected_expert_fingerprint": os.environ.get("E4B_INT4_EXPECTED_FINGERPRINT"),
           "expected_attention_fingerprint": os.environ.get("E4B_SERVE_ATTN_INT4_FINGERPRINT")}, open(sys.argv[1], "w"))
PYS
}
fp_of(){ python -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }
verifies(){ python -c "from experts4bit_qlora.engines.pack_manifest import verify_artifact as v, LAYOUT, ATTN_LAYOUT; v('$1', expected_model_revision='$REV', expected_layout=$2)" > logs/verify_$(basename $1).log 2>&1; }
first_chunk_watchdog(){ local pid=$1 log=$2 budget=${P82_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "$log" 2>/dev/null && { say "build: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
      say "HOST-LIMITED: no calibration chunk in ${budget}s -- killing the build"; echo "HOSTLIMITED build no calibration chunk in ${budget}s" >> summary.txt
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30
    fi
    sleep 20
  done; return 0; }
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
rm -rf $W/artifact1 $W/attn1
say "build: calibrate the licensed stack, dump the expert pack (artifact1) and the attention pack (attn1), K8 at $BUILD_STEPS steps"
env E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_CALIB_LAYERS_PER_PASS=10 E4B_CALIB_SOURCE=c4 \
    E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_DUMP=$W/attn1 $FOLDS \
  perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
    --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps $BUILD_STEPS \
    --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/build_ppl_wikitext.json > logs/build.log 2>&1 &
pid=$!; brc=0; first_chunk_watchdog "$pid" logs/build.log || brc=$?; wrc=0; wait "$pid" 2>/dev/null || wrc=$?; [ "$brc" = 0 ] && brc=$wrc
stamp $W/build_ppl_wikitext.router.json
grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" logs/build.log | tail -6 | cut -c1-300 | sed "s/^/    /"
grep -aE "K8_PPL" logs/build.log | tail -1 | sed "s/^/BUILD /" >> summary.txt
[ "$brc" = 30 ] && finish 30
verifies $W/artifact1 LAYOUT || { say "the build left no expert pack that verifies (rc=$brc)"; echo "BUILD FAILED rc=$brc (experts)" >> summary.txt; finish 20; }
verifies $W/attn1 ATTN_LAYOUT || { say "the build left no attention pack that verifies (rc=$brc)"; echo "BUILD FAILED rc=$brc (attention)" >> summary.txt; finish 20; }
[ "$brc" = 0 ] || { say "the build exited rc=$brc after dumping two packs that verify -- continuing on the packs"; echo "BUILD rc=$brc after verified dumps" >> summary.txt; }
FP_E=$(fp_of $W/artifact1); FP_A=$(fp_of $W/attn1)
[ -n "$FP_E" ] && [ -n "$FP_A" ] || { say "a pack has no fingerprint"; finish 20; }
echo "PACK experts $FP_E ($([ "$FP_E" = "$LIC_FP" ] && echo "= the licensed pack (P55x)" || echo "!= the licensed $LIC_FP"))" | tee -a summary.txt
echo "PACK attention $FP_A (layout int4_b32.attn.v1; $(python -c "import json; m=json.load(open('$W/attn1/manifest.json')); print(len(m['layers']), 'projections', m.get('groups'))"))" | tee -a summary.txt
printf '{"expert_fingerprint": "%s", "licensed_expert_fingerprint": "%s", "expert_licensed": %s, "attention_fingerprint": "%s"}\n' \
  "$FP_E" "$LIC_FP" "$([ "$FP_E" = "$LIC_FP" ] && echo true || echo false)" "$FP_A" > packs.json
rm -rf $W/work/nf4snap 2>/dev/null   # the bake's intermediate snapshot (~16 GB): the arena is what every later step reads (P70)
# ---- the arms, in the registered order; one process and one receipt each; both packs installed by fingerprint
LOAD="E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_CALIB_SOURCE=c4 E4B_INT4_ARTIFACT_DIR=$W/artifact1 E4B_INT4_EXPECTED_FINGERPRINT=$FP_E E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_ARTIFACT=$W/attn1 E4B_SERVE_ATTN_INT4_FINGERPRINT=$FP_A $FOLDS"
arm(){ local ARM=$1 MODE=$2
  say "arm $ARM (mode=$MODE seg=$SEG)"
  env $LOAD perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
      --placement-override all-vram --amort off --batch 16 --prompt-len 512 --no-fuse-qkv \
      --dynb-mode "$MODE" --dynb-grouping device --dynb-arm "$ARM" --dynb-seg "$SEG" --out $W/p82_$ARM.json > logs/arm_$ARM.log 2>&1
  local rc=$?
  if [ ! -s $W/p82_$ARM.json ]; then
    local oom=false; grep -aqE "OutOfMemoryError|CUDA out of memory" logs/arm_$ARM.log && oom=true
    python -c "import json,sys; json.dump({'arm':'$ARM','mode':'$MODE','status':'error','rc':$rc,'oom':$( [ $oom = true ] && echo True || echo False),'tail':sys.argv[1]}, open('$W/p82_$ARM.json','w'))" "$(tail -3 logs/arm_$ARM.log | tr '\n' ' ' | cut -c1-600)"
  fi
  { echo -n "arm $ARM rc=$rc "; grep -a "P81_DYNB\|TRACE MISMATCH\|REFUSED\|Error" logs/arm_$ARM.log | tail -1 | cut -c1-420; echo; } | tee -a summary.txt
  stamp $W/p82_$ARM.router.json $LOAD
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; }
arm A1 eager
arm B1 graph
arm B2 graph
arm A2 eager
arm P padded
# the decisive reduction first, so a K8 arm that runs out of guard cannot cost the verdict; re-run after them for the report
python $W/p82_reduce.py --dir $W --out $W/verdict.json --licensed-fp "$LIC_FP" 2>&1 | tee -a summary.txt
# ---- the K8 arms (reported for #674, never decisive): the build's wikitext K8 read again through BOTH packs loaded by
# fingerprint, with the fp32 router (K32) and with E4B_ROUTER_EPI_CAST unset (K16: 0.37.5's default casts softmax_topk)
k8arm(){ local NAME=$1 U=""; [ "$NAME" = K16 ] && U="-u E4B_ROUTER_EPI_CAST"
  local left=$(( P82_DEADLINE_EPOCH - $(date +%s) - 600 ))
  [ "$left" -ge 900 ] || { say "arm $NAME SKIPPED: ${left}s of guard left < 900s"; echo "arm $NAME SKIPPED (guard)" >> summary.txt; return 0; }
  say "arm $NAME (K8 wikitext, $BUILD_STEPS steps, both packs by fingerprint${U:+, router cast at the shipped default})"
  env $U $LOAD perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
      --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps $BUILD_STEPS \
      --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/k8_$NAME.json > logs/k8_$NAME.log 2>&1
  local rc=$?
  stamp $W/k8_$NAME.router.json $U $LOAD
  { echo -n "arm $NAME rc=$rc "; grep -a "K8_PPL\|REFUSED\|Error" logs/k8_$NAME.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; }
k8arm K32
k8arm K16
python $W/p82_reduce.py --dir $W --out $W/verdict.json --licensed-fp "$LIC_FP" 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish 0
