#!/bin/bash
# bench/tp4/tp4_run.sh -- lane tp4, BOX side (bench/tp4/TP4-PREREG.md). Started detached by tp4_drive.sh with the run's
# nonce; writes TP4_RUN_NONCE first, then TP4_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and TP4_SUCCESS.<nonce>
# only when the plan completed (every arm reached a row and the reducer ran). Pattern: tp2_run.sh (per-family fetch /
# tokenise / arms / free, stubs for every non-run, versions + tripwires) + p39_run.sh (nonce handshake, deadline-derived
# alarms, finish()). A refusal / OOM / load fault / alarm is a ROW; nothing is coerced.
#
#   TP4_BOX=A  granite, olmoe, gpt-oss at the field recipe + the Qwen3 ANCHOR pair at tp2's fixture (+ the NOT_RUN rows)
#   TP4_BOX=B  qwen3, qwen3_5 (Qwen3.6-35B-A3B) at the field recipe
#   TP4_BOX=C  gemma4, mixtral at the field recipe (Mixtral e4b arms with expert offload -- the 32 GB card)
#
# Three frameworks: e4b (GitHub main @ E4B_SHA + grouped-nf4-gemm @ GNF4_SHA, venv-e4b), Unsloth (latest PyPI at launch,
# venv-unsloth), plain HF+PEFT+bnb (venv-e4b's transformers/peft/bitsandbytes -- the same torch, the same transformers).
set -uo pipefail
LANE=tp4; W=/root/$LANE; mkdir -p $W/logs $W/adapters $W/data; cd $W || exit 9
say(){ echo "[$(date -u +%FT%TZ)] tp4/box${TP4_BOX:-?}: $*"; }
NONCE=${TP4_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/TP4_RUN_NONCE.tmp && mv $W/TP4_RUN_NONCE.tmp $W/TP4_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > TP4_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > TP4_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in TP4_BOX TP4_RUN_ID TP4_DEADLINE_EPOCH TP4_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$TP4_BOX" in A|B|C) ;; *) say "refusing: TP4_BOX must be A, B or C"; finish 78;; esac
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PREREG=tp4/TP4-PREREG.md
export TP4_INSTANCE_ID
# ---------------------------------------------------------------- the registered fixture (TP4-PREREG "Fixture"): the Unsloth notebooks' recipe, verbatim
STEPS=${TP4_STEPS:-60}; SEQ=${TP4_SEQ:-2048}; MB=${TP4_MB:-2}; ACCUM=${TP4_ACCUM:-4}; R=${TP4_R:-16}; ALPHA=${TP4_ALPHA:-16}
LR=${TP4_LR:-2e-4}; WD=${TP4_WD:-0.001}; WARMUP=${TP4_WARMUP:-5}; SCHED=${TP4_SCHED:-linear}; OPTIM=${TP4_OPTIM:-adamw_8bit}; SEED=${TP4_SEED:-3407}
EVAL_EVERY=${TP4_EVAL_EVERY:-20}; EVAL_N=${TP4_EVAL_N:-48}; AUTOCAST=${TP4_AUTOCAST:-0}; TEMPLATE=alpaca
DS_ALPACA_SHA=${TP4_DS_ALPACA_SHA:-5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb}   # tp4_alpaca.py output, registered
# the anchor pair's fixture = tp2/P38 as run (TP4-PREREG "Anchor"): clinical text, seq 512, batch 1, r 8 / alpha 16, lr 1e-4, torch AdamW wd 0.01, constant, seed 0
A_STEPS=60; A_SEQ=512; A_MB=1; A_ACCUM=1; A_R=8; A_ALPHA=16; A_LR=1e-4; A_WD=0.01; A_WARMUP=0; A_SCHED=constant; A_OPTIM=adamw_torch; A_SEED=0; A_TEMPLATE=clinical
TF_VER=${TP4_TRANSFORMERS_VER:-5.17.0}; BNB_VER=${TP4_BNB_VER:-0.50.2}; PEFT_VER=${TP4_PEFT_VER:-0.20.0}
SKIP=${TP4_SKIP:-}; PIN_FALLBACK=${TP4_PIN_FALLBACK:-0}; GPU_CLASS=${TP4_GPU_CLASS:-5090}
case "$TP4_BOX" in
  A) FAMILIES=${TP4_FAMILIES:-"granite olmoe gptoss qwen3anchor notrun"};;
  B) FAMILIES=${TP4_FAMILIES:-"qwen3 qwen3_5"};;
  C) FAMILIES=${TP4_FAMILIES:-"gemma4 mixtral"};;
esac
: > summary.txt; echo "$TP4_INSTANCE_ID" > INSTANCE_ID
echo "FIXTURE field: template=$TEMPLATE steps=$STEPS seq=$SEQ micro_batch=$MB accum=$ACCUM r=$R alpha=$ALPHA lr=$LR wd=$WD warmup=$WARMUP sched=$SCHED optim=$OPTIM seed=$SEED eval_every=$EVAL_EVERY eval_n=$EVAL_N autocast=$AUTOCAST" | tee -a summary.txt
echo "FIXTURE anchor: template=$A_TEMPLATE steps=$A_STEPS seq=$A_SEQ micro_batch=$A_MB accum=$A_ACCUM r=$A_R alpha=$A_ALPHA lr=$A_LR wd=$A_WD sched=$A_SCHED optim=$A_OPTIM seed=$A_SEED" | tee -a summary.txt
echo "BOX $TP4_BOX families: $FAMILIES; e4b $E4B_SHA gnf4 $GNF4_SHA; run $TP4_RUN_ID instance $TP4_INSTANCE_ID deadline $TP4_DEADLINE_EPOCH" | tee -a summary.txt
# ---------------------------------------------------------------- staged pieces, box class, forensics
for f in tp4_arm.py tp4_reduce.py tp4_alpaca.py n9_datasets.py ds_manifest.json; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "BOX REFUSED: gpu '$GPU_NAME' is not the registered class ($GPU_CLASS)"; echo "BOX_REFUSED gpu=$GPU_NAME" >> summary.txt; finish 12;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,power.limit,clocks.max.sm --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt; grep MemTotal /proc/meminfo | tee -a forensics.txt; cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
python3 - "$TP4_BOX" "$TP4_RUN_ID" "$TP4_INSTANCE_ID" "$GPU_NAME" <<'PYB' > box.json
import json, os, subprocess, sys
box, run_id, iid, gpu = sys.argv[1:5]
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return f"ERR {e}"
print(json.dumps({"box": box, "run_id": run_id, "instance_id": iid, "gpu": gpu, "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
                  "cpu": sh("lscpu | grep 'Model name' | cut -d: -f2 | xargs"), "nproc": os.cpu_count(), "mem_total_kb": sh("grep MemTotal /proc/meminfo | awk '{print $2}'"),
                  "cgroup_memory_max": sh("cat /sys/fs/cgroup/memory.max 2>/dev/null"), "disk_root": sh("df -h /root | tail -1"), "hostname": sh("hostname"),
                  "registered_gpu_class": os.environ.get("TP4_GPU_CLASS", "5090"), "prereg": "tp4/TP4-PREREG.md"}, indent=1))
PYB
# ---------------------------------------------------------------- deadline-derived alarms (p39's rule): what is LEFT minus a fetch margin, never a literal that outlives the rental
left(){ echo $(( TP4_DEADLINE_EPOCH - $(date +%s) )); }
alarm_for(){ local want=$1 l; l=$(( $(left) - 900 )); [ "$l" -lt "$want" ] && want=$l; [ "$want" -lt 300 ] && want=300; echo "$want"; }
can_run(){ local need=$1 name=$2; [ $(( $(left) - 900 )) -ge "$need" ] && return 0; say "STOP-DEADLINE: $name needs ${need}s, $(left)s left -- skipped (host-limited)"; echo "SKIPPED $name host-limited deadline" >> summary.txt; return 1; }
# ---------------------------------------------------------------- installs
export DEBIAN_FRONTEND=noninteractive
PY_E4B=$W/venv-e4b/bin/python; PY_UNS=$W/venv-unsloth/bin/python
say "venv-e4b (system torch): e4b @$E4B_SHA + gnf4 @$GNF4_SHA + transformers==$TF_VER bitsandbytes==$BNB_VER peft==$PEFT_VER"
python -m venv --system-site-packages $W/venv-e4b || { say "VENV FAIL (e4b)"; finish 9; }
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
print("tp4 tripwire OK (e4b):", e.__version__, "@", ce[:12], "gnf4", md.version("grouped-nf4-gemm"), "@", cg[:12], "torch", torch.__version__, "triton", triton.__version__,
      "transformers", transformers.__version__, "bnb", bitsandbytes.__version__, "peft", peft.__version__)
open("/root/tp4/versions.txt", "a").write(f"e4b {e.__version__} @{ce} (GitHub main)\ngnf4 {md.version('grouped-nf4-gemm')} @{cg} (GitHub main)\ntorch(e4b/hf) {torch.__version__}\ntriton(e4b/hf) {triton.__version__}\n"
                                          f"transformers(e4b/hf) {transformers.__version__}\nbitsandbytes(e4b/hf) {bitsandbytes.__version__}\npeft(hf) {peft.__version__}\n")
PYT
tail -1 logs/tripwire_e4b.log
# Unsloth: latest PyPI at launch (recorded), its own venv, NO transformers/bnb/peft pins from us (P38 amendment 1); torchao removed on the ScalingType tripwire (P38 amendment 2)
UNS_VER=${TP4_UNSLOTH_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth/json',timeout=60))['info']['version'])" 2>/dev/null)}
ZOO_VER=${TP4_UNSLOTH_ZOO_VERSION:-$(python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('https://pypi.org/pypi/unsloth-zoo/json',timeout=60))['info']['version'])" 2>/dev/null)}
UNS_OK=1; [ -z "$UNS_VER" ] && { echo "UNSLOTH VERSION UNRESOLVED (set TP4_UNSLOTH_VERSION)"; UNS_OK=0; }
if [ "$UNS_OK" = 1 ]; then
  say "venv-unsloth: unsloth[cu128-torch280]==$UNS_VER unsloth_zoo==$ZOO_VER"
  python -m venv $W/venv-unsloth && perl -e 'alarm 2700; exec @ARGV' $PY_UNS -m pip install -q --no-input --no-cache-dir \
    "unsloth[cu128-torch280]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.log 2>&1
  rc=$?; echo "pip(unsloth) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_unsloth.log; echo "PIP FAIL (unsloth) -- other arms still run; Unsloth rows = install_failed"; UNS_OK=0; }
fi
cat > $W/tripwire_unsloth.py <<'PYU'
import importlib.metadata as md, torch, transformers, bitsandbytes, peft
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
print("tp4 tripwire OK (unsloth):", unsloth.__version__, "zoo", unsloth_zoo.__version__, "torch", torch.__version__, "triton", tri, "transformers", transformers.__version__,
      "bnb", bitsandbytes.__version__, "peft", peft.__version__, "torchao", tao, "moe_backend", select_moe_backend(), "separated_lora", _should_use_separated_lora(), "FastModel", fm)
open("/root/tp4/versions.txt", "a").write(f"unsloth {unsloth.__version__}\nunsloth_zoo {unsloth_zoo.__version__}\ntorch(unsloth) {torch.__version__}\ntriton(unsloth) {tri}\n"
                                          f"transformers(unsloth) {transformers.__version__}\nbitsandbytes(unsloth) {bitsandbytes.__version__}\npeft(unsloth) {peft.__version__}\ntorchao {tao}\nmoe_backend {select_moe_backend()}\n")
PYU
if [ "$UNS_OK" = 1 ]; then
  $PY_UNS $W/tripwire_unsloth.py > logs/tripwire_unsloth.log 2>&1; trc=$?
  if [ $trc -ne 0 ] && grep -q "ScalingType" logs/tripwire_unsloth.log; then
    say "tripwire(unsloth) failed on ScalingType -> removing torchao (P38 amendment 2)"; cp logs/tripwire_unsloth.log logs/tripwire_unsloth.attempt1.log
    $PY_UNS -m pip uninstall -y -q torchao > logs/pip_unsloth_torchao_removed.log 2>&1
    echo "AMENDMENT-CLASS (P38 amendment 2): torchao removed from venv-unsloth after the tripwire failed on ScalingType" | tee -a summary.txt
    $PY_UNS $W/tripwire_unsloth.py > logs/tripwire_unsloth.log 2>&1; trc=$?
  fi
  [ $trc -ne 0 ] && { echo "TRIPWIRE FAIL (unsloth) -- other arms still run; Unsloth rows = install_failed"; tail -5 logs/tripwire_unsloth.log; UNS_OK=0; }
  tail -1 logs/tripwire_unsloth.log
fi
# ---------------------------------------------------------------- the fixed texts: Alpaca (field recipe) + clinical (the anchor pair only)
say "dataset alpaca (tp4_alpaca.py: unsloth/alpaca-cleaned @ pinned revision, seed 3407, 1200/48)"
(cd $W/data && perl -e 'alarm 900; exec @ARGV' $PY_E4B $W/tp4_alpaca.py --out $W/data/ds_alpaca.json > $W/logs/dataset_alpaca.log 2>&1) || { tail -3 logs/dataset_alpaca.log; say "DATASET FAIL (alpaca)"; finish 13; }
tail -1 logs/dataset_alpaca.log
GOT=$(sha256sum $W/data/ds_alpaca.json | awk '{print $1}'); [ "$GOT" = "$DS_ALPACA_SHA" ] || { say "DATASET MISMATCH alpaca: $GOT != $DS_ALPACA_SHA"; finish 13; }
echo "DATASET alpaca sha=$DS_ALPACA_SHA" | tee -a summary.txt
if [ "$TP4_BOX" = A ]; then
  say "dataset clinical (n9_datasets.py, sha-verified against ds_manifest.json) for the anchor pair"
  (cd $W/data && $PY_E4B $W/n9_datasets.py $W/data > $W/logs/dataset_clinical.log 2>&1); tail -1 logs/dataset_clinical.log
  CLIN_SHA=$($PY_E4B -c "import json; print(json.load(open('$W/ds_manifest.json'))['clinical']['sha256'])")
  GOT=$(sha256sum $W/data/ds_clinical.json | awk '{print $1}'); [ "$GOT" = "$CLIN_SHA" ] || { say "DATASET MISMATCH clinical: $GOT != $CLIN_SHA"; finish 13; }
  echo "DATASET clinical sha=$CLIN_SHA" | tee -a summary.txt
fi
# ---------------------------------------------------------------- helpers
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
status_of(){ $PY_E4B -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','missing'))" "$W/${1}_${2}_${3}.json" 2>/dev/null || echo missing; }
# stubw FAM FW TAG ARM STATUS REASON [extra-json]: a row for an attempt that never reached the harness
stubw(){ $PY_E4B - "$W" "$STEPS" "$SEQ" "$ACCUM" "$MB" "$PREREG" "$@" <<'PYS'
import json, os, sys
W, steps, seq, accum, mb, prereg, fam, fw, tag, arm, status, reason = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], *sys.argv[7:13]
extra = json.loads(sys.argv[13]) if len(sys.argv) > 13 else {}
rec = {"framework": fw, "fam": fam, "arm": arm, "tag": tag, "status": status, "reason": reason[:800], "steps": steps, "seq": seq, "accum": accum, "micro_batch": mb,
       "written_by": "tp4_run.sh", "prereg": prereg, **extra}
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
    say "PIN MISMATCH $FAM: staged $GOT != pin $REV -> fetching the pinned revision (TP4_PIN_FALLBACK=1, an amendment)"
    perl -e "alarm $(alarm_for $AL); exec @ARGV" $PY_E4B -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MID', revision='$REV', allow_patterns=['*.safetensors', '*.json', 'tokenizer*', '*.model', '*.txt', 'merges.txt', 'vocab.json', '*.tiktoken', '*.jinja'], max_workers=4))" > logs/fetch_${FAM}_pinned.log 2>&1 \
      || { echo "$FAM: PINNED FETCH FAILED" | tee -a summary.txt; FETCH_REASON="main moved past the pin ($GOT != $REV) and the pinned fetch failed"; return 1; }
    mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main
    echo "AMENDMENT (TP4_PIN_FALLBACK): $FAM main=$GOT != pin=$REV; pinned snapshot fetched; refs/main := pin (cache pointer only)" | tee -a summary.txt
  else
    echo "PIN MISMATCH $FAM: staged=$GOT != pin=$REV -- main moved past the pin; the family is ABORTED with load_fault stubs (not coerced)" | tee -a summary.txt
    FETCH_REASON="staged snapshot $GOT != pinned revision $REV (main moved past the pin); family aborted, not coerced (set TP4_PIN_FALLBACK=1 as an amendment)"
    return 2
  fi
  df -h /root | tail -1; return 0; }
# expect_of FAM E4BTAG: the family's e4b trainable count (the primary receipt of the same recipe) for --expect-trainable
expect_of(){ $PY_E4B - "$W" "$1" "$2" <<'PYE' 2>/dev/null
import json, os, sys
W, fam, tag = sys.argv[1], sys.argv[2], sys.argv[3]
for t in (tag, "attn_only", "reference_attn4"):
    p = os.path.join(W, f"{fam}_e4b_{t}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        if r.get("status") == "ok" and r.get("trainable_params"):
            print(r["trainable_params"]); break
PYE
}
# arm FAM FW TAG ARM ALARM MID REV OFFLOAD RECIPE(field|anchor|mb1) TOK TOK_SHA [extra args...]: one process, one JSON, one alarm
arm(){ local FAM=$1 FW=$2 TAG=$3 ARM=$4 AL=$5 MID=$6 REV=$7 OFF=$8 RECIPE=$9 TOK=${10} TOK_SHA=${11}; shift 11
  { skip $FAM || skip $FAM/$FW/$TAG; } && { say "skip $FAM/$FW/$TAG"; stubw $FAM $FW $TAG $ARM not_run "skipped by TP4_SKIP"; return 0; }
  local PY=$PY_E4B; [ "$FW" = unsloth ] && PY=$PY_UNS
  if [ "$FW" = unsloth ] && [ "$UNS_OK" != 1 ]; then stubw $FAM $FW $TAG $ARM install_failed "unsloth venv did not install/import (logs/pip_unsloth.log, logs/tripwire_unsloth.log)"; return 0; fi
  local s=$STEPS q=$SEQ m=$MB ac=$ACCUM r=$R al=$ALPHA lr=$LR wd=$WD wu=$WARMUP sc=$SCHED op=$OPTIM sd=$SEED ex_tag=fused_attn4
  case "$RECIPE" in
    anchor) s=$A_STEPS; q=$A_SEQ; m=$A_MB; ac=$A_ACCUM; r=$A_R; al=$A_ALPHA; lr=$A_LR; wd=$A_WD; wu=$A_WARMUP; sc=$A_SCHED; op=$A_OPTIM; sd=$A_SEED; ex_tag=fused_attn4_p38;;
    mb1)    m=1; ac=$(( MB * ACCUM )); ex_tag=fused_attn4_mb1;;
  esac
  local EXP EXPARG=""; [ "$FW" != e4b ] && { EXP=$(expect_of $FAM $ex_tag); [ -n "$EXP" ] && EXPARG="--expect-trainable $EXP"; }
  local A; A=$(alarm_for $AL)
  say "arm $FAM/$FW/$TAG (arm=$ARM recipe=$RECIPE steps=$s seq=$q mb=$m accum=$ac r=$r lr=$lr optim=$op sched=$sc offload=$OFF alarm=$A expect_trainable=${EXP:-none} $*)"
  local sp; sp=$(vram_start ${FAM}_${FW}_$TAG)
  HF_HUB_OFFLINE=1 UNSLOTH_ENABLE_LOGGING=1 TP4_BOX_CLASS="RTX $GPU_CLASS" perl -e "alarm $A; exec @ARGV" $PY -u $W/tp4_arm.py --framework $FW --arm $ARM --tag $TAG --fam $FAM --model "$MID" --revision $REV \
      --steps $s --seq $q --micro-batch $m --accum $ac --autocast $AUTOCAST --lr $lr --r $r --alpha $al --seed $sd --offload $OFF \
      --optim $op --weight-decay $wd --lr-schedule $sc --warmup-steps $wu \
      --tokens $TOK --tokens-sha $TOK_SHA --eval-every $EVAL_EVERY --eval-n $EVAL_N --unsloth-loader FastLanguageModel $EXPARG \
      --prereg $PREREG --out $W --adapter-dir $W/adapters "$@" > logs/run_${FAM}_${FW}_$TAG.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/${FAM}_${FW}_$TAG.json ]; then stubw $FAM $FW $TAG $ARM alarm "arm alarm $A s (SIGALRM; the process could not write its own stub)"; fi
  grep -aE "^CELL |^LOAD OK|^ENGAGE|^STUB|^LOADER FALLBACK|Enabling LoRA on MoE|MoE bnb4bit|Error|error:" logs/run_${FAM}_${FW}_$TAG.log | tail -3 | cut -c1-300 | sed "s/^/    /"
  { echo -n "$FAM/$FW/$TAG rc=$rc "; grep -aE "^CELL " logs/run_${FAM}_${FW}_$TAG.log | tail -1 | cut -c1-400; echo; } >> summary.txt
  $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader
  rm -rf $W/adapters/* 2>/dev/null; }
free_family(){ rm -rf /root/.cache/huggingface/hub/models--$2; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
tokenise(){ local FAM=$1 MID=$2 REV=$3 TEMPLATE_=$4 SEQ_=$5 DATA=$6 DATA_SHA=$7 TOK=$8
  say "tokenise $FAM ($TEMPLATE_, seq $SEQ_) -> $(basename $TOK)"
  HF_HUB_OFFLINE=1 $PY_E4B $W/tp4_arm.py --prepare --fam $FAM --model "$MID" --revision $REV --data $DATA --data-sha $DATA_SHA --seq $SEQ_ --eval-n $EVAL_N --template $TEMPLATE_ --tokens $TOK > logs/prepare_${FAM}_$TEMPLATE_.log 2>&1 || return 1
  tail -1 logs/prepare_${FAM}_$TEMPLATE_.log; return 0; }
tok_sha(){ $PY_E4B -c "import json; print(json.load(open('$1'))['sha256'])"; }
# family FAM MID REV FETCH_AL FUSED_AL UNS_AL HF_AL REF_AL OFFLOAD MODE UNS_TARGETS
#   MODE normal: e4b fused (primary) -> unsloth -> hf -> e4b reference (control) -> the mb1 secondary pair when a primary arm OOMed
#   MODE gptoss: e4b fused REFUSED stub (bare experts; tp1/tp2 cited) + attn_only (secondary row, probes and refreshes the stubs) -> unsloth -> hf
family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 FUAL=$5 UAL=$6 HAL=$7 RAL=$8 OFF=$9 MODE=${10} UT=${11}
  if skip $FAM; then say "skip family $FAM (TP4_SKIP)"
    for t in "e4b fused_attn4 fused" "unsloth ckpt_unsloth unsloth" "hf hf_peft hf" "e4b reference_attn4 reference"; do set -- $t; stubw $FAM $1 $2 $3 not_run "family skipped by TP4_SKIP"; done
    [ "$MODE" = gptoss ] && stubw $FAM e4b attn_only attn_only not_run "family skipped by TP4_SKIP"; return 0; fi
  say "===== family $FAM ($MID @ $REV; mode=$MODE offload=$OFF unsloth_targets=$UT)"
  FETCH_REASON=""
  fetch $FAM $MID $REV $FAL; local frc=$?
  if [ $frc -ne 0 ]; then
    local st=not_run; [ $frc -eq 2 ] && st=load_fault
    for t in "e4b fused_attn4 fused" "unsloth ckpt_unsloth unsloth" "hf hf_peft hf" "e4b reference_attn4 reference"; do set -- $t; stubw $FAM $1 $2 $3 $st "$FETCH_REASON"; done
    [ "$MODE" = gptoss ] && stubw $FAM e4b attn_only attn_only $st "$FETCH_REASON"
    free_family $FAM ${MID//\//--}; return 0
  fi
  local TOK=$W/tokens_$FAM.json
  if ! tokenise $FAM "$MID" $REV alpaca $SEQ $W/data/ds_alpaca.json $DS_ALPACA_SHA $TOK; then
    tail -3 logs/prepare_${FAM}_alpaca.log; echo "$FAM: TOKENS FAIL" | tee -a summary.txt
    local why; why="tokenise failed (logs/prepare_${FAM}_alpaca.log): $(tail -1 logs/prepare_${FAM}_alpaca.log | cut -c1-200)"
    for t in "e4b fused_attn4 fused" "unsloth ckpt_unsloth unsloth" "hf hf_peft hf" "e4b reference_attn4 reference"; do set -- $t; stubw $FAM $1 $2 $3 harness_error "$why"; done
    [ "$MODE" = gptoss ] && stubw $FAM e4b attn_only attn_only harness_error "$why"
    free_family $FAM ${MID//\//--}; return 0
  fi
  local TS; TS=$(tok_sha $TOK); echo "TOKENS $FAM alpaca sha=$TS" | tee -a summary.txt
  if [ "$MODE" = gptoss ]; then
    stubw $FAM e4b fused_attn4 fused refused "SKIPPED as REFUSED: tp1 (P36) + tp2 (P40) rows cited -- enable_fast_train(dgrad=True) patched 0 modules on gpt-oss (experts built bare, no ExpertsLoRA: 'GPT-OSS-aware training LoRA is a separate change', loader.py); attn_only is the secondary row and refreshes this stub with its own probe" '{"cited": "tp1,tp2", "n_patched": 0}'
    can_run 600 $FAM/e4b/attn_only && arm $FAM e4b attn_only attn_only $FUAL "$MID" $REV $OFF field $TOK $TS --attn-4bit 0
  else
    can_run 600 $FAM/e4b/fused && arm $FAM e4b fused_attn4 fused $FUAL "$MID" $REV $OFF field $TOK $TS --attn-4bit 1
  fi
  can_run 600 $FAM/unsloth && arm $FAM unsloth ckpt_unsloth unsloth $UAL "$MID" $REV 0 field $TOK $TS --grad-ckpt unsloth --unsloth-targets "$UT"
  can_run 600 $FAM/hf && arm $FAM hf hf_peft hf $HAL "$MID" $REV 0 field $TOK $TS
  if [ "$MODE" != gptoss ]; then
    can_run 900 $FAM/e4b/reference && arm $FAM e4b reference_attn4 reference $RAL "$MID" $REV $OFF field $TOK $TS --attn-4bit 1
    # the secondary pair (TP4-PREREG "Arms"): micro-batch 1 x accum 8 -- same tokens per step -- for EVERY framework, only when a primary arm OOMed
    local se su sh; se=$(status_of $FAM e4b fused_attn4); su=$(status_of $FAM unsloth ckpt_unsloth); sh=$(status_of $FAM hf hf_peft)
    if [ "$se" = oom ] || [ "$su" = oom ] || [ "$sh" = oom ]; then
      echo "SECONDARY $FAM: a primary arm OOMed (e4b=$se unsloth=$su hf=$sh) -> mb1 pair" | tee -a summary.txt
      can_run 600 $FAM/e4b/fused_mb1 && arm $FAM e4b fused_attn4_mb1 fused $FUAL "$MID" $REV $OFF mb1 $TOK $TS --attn-4bit 1
      can_run 600 $FAM/unsloth/mb1 && arm $FAM unsloth ckpt_unsloth_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS --grad-ckpt unsloth --unsloth-targets "$UT"
      [ "$sh" = oom ] && can_run 600 $FAM/hf/mb1 && arm $FAM hf hf_peft_mb1 hf $HAL "$MID" $REV 0 mb1 $TOK $TS
    fi
  fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# the Qwen3 ANCHOR pair (box A): tp2's fixture on tp2's text, so this box's e4b/Unsloth ratio reads against tp2's 1.457 and P38's 1.413
anchor_pair(){ local FAM=qwen3 MID=Qwen/Qwen3-30B-A3B REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
  say "===== ANCHOR pair: $FAM at tp2/P38's fixture (clinical, seq 512, batch 1, r 8, lr 1e-4, torch AdamW)"
  FETCH_REASON=""; fetch $FAM $MID $REV 5400; local frc=$?
  if [ $frc -ne 0 ]; then local st=not_run; [ $frc -eq 2 ] && st=load_fault
    stubw $FAM e4b fused_attn4_p38 fused $st "$FETCH_REASON"; stubw $FAM unsloth ckpt_unsloth_p38 unsloth $st "$FETCH_REASON"; free_family $FAM ${MID//\//--}; return 0; fi
  local TOK=$W/tokens_${FAM}_p38.json CLIN_SHA; CLIN_SHA=$($PY_E4B -c "import json; print(json.load(open('$W/ds_manifest.json'))['clinical']['sha256'])")
  if ! tokenise $FAM $MID $REV clinical $A_SEQ $W/data/ds_clinical.json $CLIN_SHA $TOK; then
    local why="tokenise failed (logs/prepare_${FAM}_clinical.log)"; stubw $FAM e4b fused_attn4_p38 fused harness_error "$why"; stubw $FAM unsloth ckpt_unsloth_p38 unsloth harness_error "$why"
    free_family $FAM ${MID//\//--}; return 0; fi
  local TS; TS=$(tok_sha $TOK); echo "TOKENS ${FAM}_p38 clinical sha=$TS (tp2's tokens_qwen3 sha for cross-check: the tp2 receipt records it)" | tee -a summary.txt
  can_run 600 $FAM/e4b/fused_p38 && arm $FAM e4b fused_attn4_p38 fused 1800 "$MID" $REV 0 anchor $TOK $TS --attn-4bit 1
  can_run 600 $FAM/unsloth/p38 && arm $FAM unsloth ckpt_unsloth_p38 unsloth 1800 "$MID" $REV 0 anchor $TOK $TS --grad-ckpt unsloth --unsloth-targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  echo "ANCHOR DONE" | tee -a summary.txt; free_family $FAM ${MID//\//--}; }
notrun_rows(){  # the two registered families no single box of this class can hold (TP4-PREREG "Families"): a row each, never a silent omission
  local why_ds="NOT RUN by registration: deepseek-ai/DeepSeek-V4-Flash is 159.6 GB of safetensors (43 layers, per-expert MXFP4); its expert bytes exceed the registered class (32 GB VRAM) and the ordered host RAM (98 GB) even with e4b's expert offload; e4b's NVMe tier is not a registered training path; neither Unsloth nor plain HF can load it in 4-bit on one card. No framework was attempted."
  local why_k3="NOT RUN by registration: moonshotai/Kimi-K3 is 1,560.9 GB of safetensors; no single box of any rentable class holds it; no framework was attempted."
  for fam in deepseek_v4 kimi_k3; do
    local why="$why_ds"; [ "$fam" = kimi_k3 ] && why="$why_k3"
    for t in "e4b fused_attn4 fused" "unsloth ckpt_unsloth unsloth" "hf hf_peft hf" "e4b reference_attn4 reference"; do set -- $t; stubw $fam $1 $2 $3 not_run "$why" '{"registered": "TP4-PREREG.md Families"}'; done
  done; }
UT7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"     # the notebooks' seven targets
UT4="q_proj,k_proj,v_proj,o_proj"                                 # attention only: families with a SHARED dense expert (qwen3_5) so both frameworks adapt the same set
# ---------------------------------------------------------------- the plan (TP4-PREREG "Families"; revisions = the HF API on 2026-09-10, tp1/tp2's where they exist)
#      FAM      MID                                        REV                                         FETCH FUSED UNS  HF   REF  OFF MODE   UNS_TARGETS
for FAM in $FAMILIES; do case "$FAM" in
  granite)   family granite  ibm-granite/granite-3.1-3b-a800m-instruct a02780686e08a03fe0d2679a293b5c74a90efa89 1800 1800 1800 1800 2400 0 normal $UT7;;
  olmoe)     family olmoe    allenai/OLMoE-1B-7B-0924-Instruct         7f1c97f440f06ce36705e4f2b843edb5925f4498 2400 2400 2400 2400 3000 0 normal $UT7;;
  gptoss)    family gptoss   openai/gpt-oss-20b                        6cee5e81ee83917806bbde320786a8fb61efebee 3000 3600 2400 2400 3600 0 gptoss $UT7;;
  qwen3)     family qwen3    Qwen/Qwen3-30B-A3B                        ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 1800 5400 0 normal $UT7;;
  qwen3_5)   family qwen3_5  Qwen/Qwen3.6-35B-A3B                      995ad96eacd98c81ed38be0c5b274b04031597b0 6000 3600 3600 1800 5400 0 normal $UT4;;
  gemma4)    family gemma4   google/gemma-4-26B-A4B-it                 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 5400 3600 3600 1800 5400 0 normal $UT7;;
  mixtral)   family mixtral  mistralai/Mixtral-8x7B-Instruct-v0.1      eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 5400 2400 1800 6000 1 normal $UT7;;
  qwen3anchor) anchor_pair;;
  notrun)    notrun_rows;;
  *) say "unknown family token $FAM"; echo "UNKNOWN $FAM" >> summary.txt;;
esac; done
# ---------------------------------------------------------------- reduce, summarise, mark
say "reduce"
$PY_E4B $W/tp4_reduce.py $W --md $W/RESULTS-tp4-box$TP4_BOX.md --steps $STEPS > RESULTS.txt 2>&1; tail -30 RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
[ "$UNS_OK" = 1 ] || echo "NO UNSLOTH COMPARATOR on this box: the Unsloth side did not install/import (rows = install_failed)" | tee -a summary.txt
finish 0
