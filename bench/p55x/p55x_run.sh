#!/bin/bash
# bench/p55x/p55x_run.sh -- lane P55x, BOX side (bench/p55x/P55X-PREREG.md). Started detached by p55x_drive.sh
# with the run's nonce; writes P55X_RUN_NONCE first, then P55X_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit
# and P55X_SUCCESS.<nonce> only on clean completion.
#
# Build ONE calibrated int4 expert pack with the streamed 64k recipe, dump it as a hash-pinned artifact (#405),
# run the registered two-text K8 gate on THAT ARTIFACT (loaded by fingerprint, the path a user gets), and build
# it a second time to ask whether the recipe is byte-deterministic on one box. Nothing here changes a default,
# a threshold or a gate; the prereg's decision rule reads the receipts.
set -uo pipefail
W=/root/p55x; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p55x: $*"; }
NONCE=${P55X_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P55X_RUN_NONCE.tmp && mv $W/P55X_RUN_NONCE.tmp $W/P55X_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P55X_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P55X_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P55X_RUN_ID P55X_DEADLINE_EPOCH P55X_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39; export E4B_MODEL_ID=$MID
: > summary.txt; echo "$P55X_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p55x/staged.sha256"; finish 9; }

# ---- K0: refuse a box that cannot hold the working set, BEFORE any fetch (e4b#604/#605).
# A Vast instance's container overlay is not the machine disk the launcher ordered on: p48-gemma4layer
# came up with a 32 GB overlay on a host filtered for >= 320 GB and died ENOSPC mid-fetch, after K0 passed,
# for $0.49. This lane's working set is the bf16 checkpoint (~61 GB) + the NF4 arena (~16 GB) + TWO pack
# artifacts (~15.2 GiB each) ~= 108 GB, so the floor is 200.
MIN_DISK_GB=${P55X_MIN_DISK_GB:-200}
AVAIL_GB=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc '0-9')
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
df -h /root | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
python3 - <<PY > k0.json
import json; json.dump({"gpu": """$GPU_NAME""".strip(), "avail_disk_gb": ${AVAIL_GB:-0}, "min_disk_gb": $MIN_DISK_GB,
                        "upload_mb_s_probe": "measured by the controller before this script started"}, open("/dev/stdout","w"))
PY
case "$GPU_NAME" in *5090*) ;; *) say "STOP-0: GPU is '$GPU_NAME', not an RTX 5090 -- refusing before any fetch"; echo "REFUSED gpu_class $GPU_NAME" >> summary.txt; finish 13;; esac
[ "${AVAIL_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "STOP-0: /root has ${AVAIL_GB:-0} GB free, floor $MIN_DISK_GB (overlay is not machine disk) -- refusing before any fetch"; echo "REFUSED disk ${AVAIL_GB:-0}GB < ${MIN_DISK_GB}GB" >> summary.txt; finish 13; }
say "K0 OK: $GPU_NAME, ${AVAIL_GB} GB free on /root"

# ---- deadline guard: never start an arm that cannot finish 10 min before the launcher tears the box down
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P55X_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P55X_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
# Per-arm alarm derived from what is LEFT, not a literal: p39-box1b-4 died on a hardcoded 5400 s and burned
# 90 min with 4 h of paid wallclock unused.
arm_alarm(){ local left=$(( P55X_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 1800 ] && left=1800; [ "$left" -gt 18000 ] && left=18000; echo "$left"; }
# A deadline-derived alarm still lets a bad host spend the whole rental before saying so. A calibrating arm is
# killed early if chunk 1 does not land inside P55X_FIRST_CHUNK_S (~3x the 520 s a good host takes).
first_chunk_watchdog(){ local pid=$1 name=$2 budget=${P55X_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)
  case "$name" in *build*) ;; *) return 0;; esac
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "logs/run_$name.log" 2>/dev/null && { say "$name: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
      say "HOST-LIMITED: $name produced no calibration chunk in ${budget}s (a good host: ~520s) -- killing the arm"
      echo "HOSTLIMITED $name no calibration chunk in ${budget}s" >> summary.txt
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30
    fi
    sleep 20
  done
  return 0; }

# ---- install: e4b pinned + P37's toolchain pins; gnf4 at the cut e4b CI pins
export DEBIAN_FRONTEND=noninteractive
say "install e4b @$E4B_SHA (image python; P37 pins)"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
PYTHONPATH=$W/hook python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
sig = inspect.signature(f).parameters
for k in ("artifact_dir", "expected_fingerprint", "dump_artifact_dir", "assignment"):
    assert k in sig, f"e4b cut lacks the #405/#531 knob {k!r}"
# the load-by-fingerprint path this lane's gate runs on, and the payloads the fingerprint must cover
from experts4bit_qlora.engines.pack_manifest import (ASSIGNMENT_PATH, IDENTITY_PATH, read_assignment,  # noqa: F401
                                                     verify_artifact, compute_pack_fingerprint)        # noqa: F401
from experts4bit_qlora.engines.int4_experts import (dump_calibrated_artifact,                          # noqa: F401
                                                    enable_serve_experts_int4_from_artifact)           # noqa: F401
from experts4bit_qlora import k8_gate                                                                  # noqa: F401
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p55x/versions.txt", "a").write(
    f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')}\ntorch {torch.__version__}\n"
    f"triton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK: e4b", e.__version__, "torch", torch.__version__, "triton", triton.__version__)
PYT
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }

# ---- fetch (pinned) + bake, as P37/P39
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; import sys; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
say "bake NF4 arena"; mkdir -p $W/work_qwen3
K8_MODEL="$MID" K8_WORK="$W/work_qwen3" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL (no arena)"; finish 12; }

# ---- the recipe, stated rather than derived (prereg): 64k C4-validation tokens, 48 layers in 5 passes of 10,
# min_rows 32 and damping 0.01 at their defaults, GPU solve.
CAL="E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128 E4B_CALIB_LAYERS_PER_PASS=10"
fp_of(){ python -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }

# k8_arm NAME KIND(nf4|calibattn|rtnattn) SRC [env...] -> receipt qwen3_ppl_${NAME}_${SRC}.json
# KIND selects the attention treatment and the folds; p37c's k8() verbatim in flags otherwise.
k8_arm(){ local NAME=$1 KIND=$2 SRC=$3; shift 3
  local EXP=1 CA=0 RTN=0 G=1 R=1 E=1
  case "$KIND" in
    nf4)       EXP=0; CA=0; RTN=0; G=0; R=0; E=0 ;;
    calibattn) CA=1 ;;                 # bo6c's configuration: 192 GPTQ-calibrated attention projections
    rtnattn)   RTN=1 ;;                # round-to-nearest attention: no Hessian, no routing, pinnable
    *) say "k8_arm: unknown KIND $KIND"; return 2 ;;
  esac
  { echo "P55X arm=${NAME}_$SRC kind=$KIND at=$(date -u +%FT%TZ)"; } > logs/run_${NAME}_$SRC.log
  local t0; t0=$(date +%s)
  env "$@" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_SERVE_ATTN_INT4=$RTN E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
      --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 \
      --b1d-loop eager --no-fuse-qkv --ppl-source $SRC --out $W/qwen3_ppl_${NAME}_$SRC.json >> logs/run_${NAME}_$SRC.log 2>&1 &
  local pid=$! rc=0
  first_chunk_watchdog "$pid" "${NAME}_$SRC" || rc=$?
  local wrc=0; wait "$pid" 2>/dev/null || wrc=$?
  # `[ "$rc" = 0 ] && rc=$?` captured the TEST's status, not wait's, so every arm reported rc=0 and
  # p39-box3 walked past a RuntimeError (e4b#641). Read the exit code from the command that produced it.
  [ "$rc" = 0 ] && rc=$wrc
  grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" logs/run_${NAME}_$SRC.log | tail -4 | sed "s/^/    /"
  grep -aoE "INT4EXP (calibrated|loaded|assignment)[^|]{0,120}" logs/run_${NAME}_$SRC.log | tail -2 | sed "s/^/INT4EXP ${NAME}_$SRC /" >> summary.txt
  grep -aoE "ATTNINT4 [a-z]*:? ?[0-9]+ projections" logs/run_${NAME}_$SRC.log | tail -1 | sed "s/^/ATTNINT4 ${NAME}_$SRC /" >> summary.txt
  { echo -n "k8 $NAME src=$SRC kind=$KIND rc=$rc in $(( $(date +%s) - t0 ))s "; grep -aE "K8_PPL" logs/run_${NAME}_$SRC.log | tail -1 | cut -c1-240; echo; } >> summary.txt
  return $rc; }

rc_any=0; note(){ [ "$rc_any" = 0 ] && rc_any=$1; }

# ---- 1,2: the NF4 reference on this box (cheap, and every later delta is measured against it)
can_run 900 nf4_wikitext && { k8_arm nf4 nf4 wikitext || note $?; }
can_run 900 nf4_c4val1   && { k8_arm nf4 nf4 c4val1   || note $?; }

# ---- 3: build the pack and dump it as an artifact. It also scores wikitext from the LIVE stores, which is
# free here and gives the dump->load round trip a cross-check against arm 4's score of the same bytes.
can_run 4200 licbuild || finish 20
k8_arm licbuild calibattn wikitext $CAL E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact1 || { brc=$?
  [ "$brc" = 30 ] && { say "host-limited on the build; nothing measured about the pack"; finish 30; }
  say "the build arm failed (rc=$brc) -- there is no pack to gate"; finish 20; }
FP1=$(fp_of $W/artifact1); [ -n "$FP1" ] || { say "no artifact fingerprint after the build"; echo "NO_FINGERPRINT after build1" >> summary.txt; finish 20; }
echo "ARTIFACT1 $FP1" | tee -a summary.txt
du -sh $W/artifact1 2>/dev/null | sed "s/^/ARTIFACT1_SIZE /" >> summary.txt
# The controller watches for this marker and starts pulling the 15.2 GiB in the background, so the transfer
# overlaps the four gate arms and the second build instead of being paid for at the end (STOP-4).
printf '%s\n' "$FP1" > ARTIFACT_READY.$NONCE

# ---- 4,5 (primary) and 6,7 (secondary): the gate, on the ARTIFACT, loaded by fingerprint -- the path a user
# gets. A mismatch refuses and never rebuilds (#405), so these arms also prove the licensed load path works.
LOAD="E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128 E4B_INT4_ARTIFACT_DIR=$W/artifact1 E4B_INT4_EXPECTED_FINGERPRINT=$FP1"
can_run 1500 lic_wikitext && { k8_arm lic calibattn wikitext $LOAD || note $?; }
can_run 1500 lic_c4val1   && { k8_arm lic calibattn c4val1   $LOAD || note $?; }
can_run 1200 licrtn_wikitext && { k8_arm licrtn rtnattn wikitext $LOAD || note $?; }
can_run 1200 licrtn_c4val1   && { k8_arm licrtn rtnattn c4val1   $LOAD || note $?; }

# ---- the registered gate, applied by the library so every lane applies the same rule
python - <<PY > gate_verdict.json 2>> logs/gate.log
import json, os
from experts4bit_qlora.k8_gate import Arm, verdict
W = "$W"
def arm(name, src):
    p = f"{W}/qwen3_ppl_{name}_{src}.json"
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return Arm(ppl=float(d["ppl"]), text_sha=str(d["text_sha"]),
               steps=int(d.get("steps", d["tokens_scored"])), ppl_source=src)
out = {"budget": 0.05, "calibrated": True, "calibration_domain": "c4val1", "pack_fingerprint": "$FP1",
       "configurations": {}}
for cand in ("lic", "licrtn"):
    pairs, missing = [], []
    for src in ("wikitext", "c4val1"):
        b, c = arm("nf4", src), arm(cand, src)
        (pairs.append((b, c)) if (b and c) else missing.append(src))
    if not pairs:
        out["configurations"][cand] = {"verdict": "NOT RUN", "missing": missing}; continue
    try:
        ok, lines = verdict(pairs, calibrated=True, calibration_domain="c4val1")
    except Exception as e:
        out["configurations"][cand] = {"verdict": "ERROR", "error": repr(e)[:300], "missing": missing}; continue
    out["configurations"][cand] = {
        "verdict": "PASS" if ok else "FAIL", "incomplete": missing or None, "lines": lines,
        "texts": {src: {"nf4": arm("nf4", src).ppl, "cand": arm(cand, src).ppl,
                        "delta": arm(cand, src).ppl - arm("nf4", src).ppl}
                  for src in ("wikitext", "c4val1") if arm("nf4", src) and arm(cand, src)}}
# the dump -> load round trip: build1 scored wikitext from the live stores, arm 4 from the artifact
lb, lw = arm("licbuild", "wikitext"), arm("lic", "wikitext")
out["roundtrip_wikitext"] = ({"live_store_ppl": lb.ppl, "artifact_ppl": lw.ppl, "delta": lw.ppl - lb.ppl}
                             if (lb and lw) else None)
json.dump(out, open("/dev/stdout", "w"), indent=1)
PY
[ -s gate_verdict.json ] || { say "GATE VERDICT NOT WRITTEN"; note 41; }
python -c "import json;d=json.load(open('$W/gate_verdict.json'));print('GATE', ' '.join(f\"{k}={v.get('verdict')}\" for k,v in d['configurations'].items()))" 2>/dev/null | tee -a summary.txt

# ---- 8: the same recipe a second time. Same-box byte determinism has never been measured for this pack --
# bo6c's calib-deterministic row compared mean_nll and COUNTS for the 16k arm, before the fingerprint existed.
if can_run 4200 build2; then
  k8_arm licbuild2 calibattn wikitext $CAL E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact2 || { rc2=$?
    [ "$rc2" = 30 ] && { echo "HOSTLIMITED build2" >> summary.txt; } || { echo "BUILD2 FAILED rc=$rc2" >> summary.txt; note "$rc2"; }; }
  FP2=$(fp_of $W/artifact2)
  echo "ARTIFACT2 ${FP2:-<none>} $([ -n "$FP2" ] && { [ "$FP2" = "$FP1" ] && echo SAME_BYTES || echo DIFFERENT_BYTES; })" | tee -a summary.txt
  python - <<PY > determinism.json 2>> logs/gate.log
import json, os
from experts4bit_qlora.engines.pack_manifest import read_assignment
W = "$W"
def man(d):
    p = f"{W}/{d}/manifest.json"
    return json.load(open(p)) if os.path.exists(p) else None
def counts(m):
    c = (m or {}).get("calibrated_counts")
    return list(c) if c else None
m1, m2 = man("artifact1"), man("artifact2")
def mmh(d):
    try: return read_assignment(f"{W}/{d}")["method_map_hash"]
    except Exception: return None
out = {"fp1": (m1 or {}).get("pack_fingerprint"), "fp2": (m2 or {}).get("pack_fingerprint"),
       "counts1": counts(m1), "counts2": counts(m2),
       "method_map_hash1": mmh("artifact1"), "method_map_hash2": mmh("artifact2"),
       "row_count_vector_hash1": (m1 or {}).get("row_count_vector_hash"),
       "row_count_vector_hash2": (m2 or {}).get("row_count_vector_hash"),
       "calibration_token_stream_sha": (m1 or {}).get("calibration_token_stream_sha")}
out["bytes_identical"] = (out["fp1"] is not None and out["fp1"] == out["fp2"])
out["classification_identical"] = (out["method_map_hash1"] is not None
                                   and out["method_map_hash1"] == out["method_map_hash2"])
out["verdict"] = ("NOT RUN" if out["fp2"] is None else
                  "DETERMINISTIC" if out["bytes_identical"] else "NOT DETERMINISTIC")
json.dump(out, open("/dev/stdout", "w"), indent=1)
PY
  python -c "import json;d=json.load(open('$W/determinism.json'));print('DETERMINISM',d['verdict'],'counts',d['counts1'],d['counts2'])" 2>/dev/null | tee -a summary.txt
  # artifact2's payload bytes have served their purpose (the fingerprint is recorded); the manifest and the
  # small payloads stay for the receipt, the 15.2 GiB of layers go so the box is not near full at teardown.
  rm -f $W/artifact2/payloads/layer_* 2>/dev/null
  echo "ARTIFACT2_PAYLOADS_REMOVED (fingerprint recorded; bytes not retained -- artifact1 is the one kept)" >> summary.txt
fi

# ---- the terminal marker has to encode the OUTCOME, not that the script reached its last line
NARMS=$(ls $W/qwen3_ppl_*.json 2>/dev/null | wc -l | tr -d ' ')
echo "ARMS_WITH_RECEIPTS $NARMS" >> summary.txt
[ "$NARMS" -ge 6 ] || { say "only $NARMS K8 receipts (need >= 6: nf4 x2, lic x2, licrtn x2)"; note 42; }
[ -s "$W/artifact1/manifest.json" ] || { say "artifact1 has no manifest"; note 43; }
say "----- summary -----"; cat summary.txt
finish "$rc_any"
