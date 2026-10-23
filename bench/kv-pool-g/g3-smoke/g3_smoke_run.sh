#!/bin/bash
# bench/kv-pool-g/g3-smoke/g3_smoke_run.sh -- lever G3's GPU smoke (bench/kv-pool-g/DESIGN.md, G3), BOX side. Started
# detached by g3_smoke_drive.sh with the run's nonce; G3S_RUN_NONCE first, then G3S_EXIT_CODE.<nonce> + TP_DONE.<nonce> on
# every exit and G3S_SUCCESS.<nonce> only when the smoke's verdict is PASS.
#
# One RTX 5090. Installs e4b at the commit under test (E4B_SHA) and grouped-nf4-gemm at GNF4_SHA on torch 2.8.0+cu128,
# fetches Granite-3.1-3b-a800m at its pin, bakes the NF4 arena (bench/p39/k8_bake.py), then runs g3_smoke.py six times,
# one engine per process, each on the same workload and schedule at 64 slots on the default bucket list (1,2,4,8,16:
# a 64-row step runs as chained 16-row replays), 2,048 tokens a slot, graphs and bulk KV on: claim-all twice (the
# floor) and shared, both lookahead settings, and a shared run whose pool is sized from a cut free-memory reading so it
# is smaller than its slots can reach. Then the verdict (g3_smoke.py verdict). Nothing here creates, destroys or
# approves compute.
set -uo pipefail
W=/root/g3s; mkdir -p $W/logs $W/runs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] g3s: $*"; }
NONCE=${G3S_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/G3S_RUN_NONCE.tmp && mv $W/G3S_RUN_NONCE.tmp $W/G3S_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > G3S_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > G3S_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in G3S_RUN_ID G3S_DEADLINE_EPOCH G3S_INSTANCE_ID E4B_SHA GNF4_SHA GR_REV; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA GR_REV; do val=${!v}; case "$val" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac
  [ ${#val} -eq 40 ] || { say "refusing: $v is not a 40-char sha"; finish 78; }; done
# the CUDA devel image ships python3 without venv / ensurepip, git, cmake or rsync (g1-smoke-2: VENV FAIL); bench/sc1/sc1_run.sh's
# ensure_tools list, plus rsync for the receipt fetch
need=0; for t in git cmake curl rsync; do command -v $t > /dev/null 2>&1 || need=1; done
python3 -c "import venv, ensurepip" 2> /dev/null || need=1
if [ "$need" = 1 ]; then
  say "apt: python3-venv / git / cmake / curl / rsync / build-essential (missing on this image)"
  perl -e 'alarm 900; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 python3-dev python3-venv python3-pip git cmake ninja-build curl rsync ca-certificates build-essential' > logs/apt.log 2>&1 \
    || { tail -3 logs/apt.log; say "APT FAIL"; finish 9; }
fi
python3 -c "import venv, ensurepip" || { say "python3 still lacks venv / ensurepip after apt"; finish 9; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
: > summary.txt; echo "$G3S_INSTANCE_ID" > INSTANCE_ID
line(){ echo "$*" | tee -a summary.txt; }
for f in g3_smoke_run.sh g3_smoke.py k8_bake.py calib.json staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 > /dev/null) || { say "STAGED FILES DIFFER FROM bench/kv-pool-g/g3-smoke/staged.sha256"; finish 9; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader > forensics.txt 2>&1 || { say "DUD BOX (nvidia-smi)"; finish 10; }
cat forensics.txt
GR=ibm-granite/granite-3.1-3b-a800m-instruct
PY=python3
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" "$PY" -m pip install -q --no-input "$@" > "$log" 2>&1 && return 0
  say "pip failed ($(tail -1 "$log" | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" "$PY" -m pip install -q --no-input "$@" >> "$log" 2>&1; }
say "install torch 2.8.0+cu128, e4b @$E4B_SHA, gnf4 @$GNF4_SHA (venv --system-site-packages; transformers 5.16.1 / bitsandbytes 0.50.1)"
python3 -m venv --system-site-packages $W/venv > logs/venv.log 2>&1 || { tail -3 logs/venv.log; say "VENV FAIL"; finish 9; }
PY=$W/venv/bin/python
pipx logs/pip_torch.log 1800 "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 || { tail -3 logs/pip_torch.log; say "PIP FAIL (torch)"; finish 9; }
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" accelerate sentencepiece safetensors "huggingface_hub>=0.23" || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA "$PY" - <<'PYT' 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the commit under test: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import torch
assert torch.cuda.is_available(), "no CUDA"
from experts4bit_qlora.serve_paged import PagedServeConfig
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
assert "kv_pool_rows" in PagedServeConfig.__dataclass_fields__, "e4b lacks the shared pool sizing (lever G3)"
assert hasattr(PagedModelRunner, "_claim_all_last") and "pool_rows" in Fp8PagedKV.__init__.__code__.co_varnames, "e4b lacks G3's pool_rows"
import experts4bit_qlora as e
print("G3S_TRIPWIRE ok: e4b", e.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
PYT
say "fetch $GR @ $GR_REV"
perl -e "alarm 1200; exec @ARGV" "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$GR', revision='$GR_REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 \
  || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
line "FETCH $GR@$GR_REV $(tail -1 logs/fetch.log)"
say "bake the NF4 arena"
mkdir -p $W/work_granite
K8_MODEL="$GR" K8_WORK="$W/work_granite" perl -e "alarm 1800; exec @ARGV" "$PY" $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work_granite/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
# SC1's Granite env (bench/sc1/sc1_run.sh GR_ENV) and route pins (ROUTEENV), unchanged; the engine as serve_paged builds it
GR_ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=auto E4B_FUSE_ROUTER_EPI=1"
ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
BASE="$GR_ENV $ROUTEENV E4B_PAGED_MODEL=$GR E4B_PAGED_REVISION=$GR_REV E4B_PAGED_ARENA=$W/work_granite/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_SEQS=64 E4B_PAGED_MAX_TOKENS_PER_SEQ=2048 E4B_PAGED_GRAPHS=1 E4B_PAGED_BUCKETS=1,2,4,8,16 E4B_PAGED_BULK_KV=1"
RC=0
# one NAME [ENV ...] [-- ARGS]: one engine, one process, the same workload and schedule
one(){ local NAME=$1 E="" A=""; shift
  while [ $# -gt 0 ] && [ "$1" != -- ]; do E="$E $1"; shift; done; [ "${1:-}" = -- ] && { shift; A="$*"; }
  [ $(( $(date +%s) + 900 )) -le "$G3S_DEADLINE_EPOCH" ] || { line "G3S $NAME SKIPPED: deadline"; RC=23; return; }
  say "run $NAME:$E $A"
  # shellcheck disable=SC2086  # assignment lists by design
  env -u E4B_PAGED_KV_POOL -u E4B_PAGED_DECODE_LOOKAHEAD $BASE $E perl -e "alarm 2400; exec @ARGV" "$PY" $W/g3_smoke.py run --out $W/runs/g3_$NAME.json $A \
      > logs/run_$NAME.log 2>&1; local r=$?
  [ "$r" = 0 ] || { tail -5 logs/run_$NAME.log; line "G3S $NAME FAILED rc=$r"; RC=23; return; }
  grep -a '^G3SMOKE run' logs/run_$NAME.log | sed "s/^/$NAME /" | tee -a summary.txt; }
one ca_a E4B_PAGED_KV_POOL=claim-all E4B_PAGED_DECODE_LOOKAHEAD=0
one sh E4B_PAGED_KV_POOL=shared E4B_PAGED_DECODE_LOOKAHEAD=0
one ca_b E4B_PAGED_KV_POOL=claim-all E4B_PAGED_DECODE_LOOKAHEAD=0
one ca_la1 E4B_PAGED_KV_POOL=claim-all E4B_PAGED_DECODE_LOOKAHEAD=1
one sh_la1 E4B_PAGED_KV_POOL=shared E4B_PAGED_DECODE_LOOKAHEAD=1
one sh_short E4B_PAGED_KV_POOL=shared E4B_PAGED_DECODE_LOOKAHEAD=0 -- --short
"$PY" $W/g3_smoke.py verdict --dir $W/runs --out $W/verdict_g3.json 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] || { [ "$RC" = 0 ] && RC=1; }
finish "$RC"
