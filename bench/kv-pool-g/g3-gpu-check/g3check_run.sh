#!/bin/bash
# bench/kv-pool-g/g3-gpu-check/g3check_run.sh -- the G3 GPU check (CHECK.md), BOX side. Started detached by
# g3check_drive.sh with the run's nonce; G3C_RUN_NONCE first, then G3C_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit
# and G3C_SUCCESS.<nonce> only when the verdict is PASS.
#
# One RTX 5090. Installs e4b at the commit under test (E4B_SHA, a merged main carrying e4b#2087) and grouped-nf4-gemm at
# GNF4_SHA on torch 2.8.0+cu128, records the card (nvidia-smi, enforced power limit, card_clock.py burn: recorded, not a
# gate -- this is a correctness check, not a reading), fetches Qwen3-30B-A3B at SC1's pin, bakes the NF4 arena
# (bench/p39/k8_bake.py), then starts ONE server as SC2j's box Q started its shared arm (bench/sc2/sc2j_box_q.sh
# q_server_start: E4B_PAGED_MAX_SEQS=auto, E4B_PAGED_KV_POOL=shared, 4,096 tokens a slot, SC1's SPEEDENV + ROUTEENV +
# E4B_PAGED_FUSE_QKV=1, the prefill graph at its default auto), sends the prompts (g3check.py drive), reads /health and
# gives the verdict from the server's own state (g3check.py verdict). Nothing here creates, destroys or approves compute.
set -uo pipefail
W=/root/g3c; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] g3c: $*"; }
NONCE=${G3C_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/G3C_RUN_NONCE.tmp && mv $W/G3C_RUN_NONCE.tmp $W/G3C_RUN_NONCE
SRV_PID=""
stop_srv(){ [ -n "$SRV_PID" ] || return 0; kill -TERM -- "-$SRV_PID" 2>/dev/null || kill -TERM "$SRV_PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$SRV_PID" 2>/dev/null || { SRV_PID=""; return 0; }; sleep 1; done
  kill -KILL -- "-$SRV_PID" 2>/dev/null || kill -KILL "$SRV_PID" 2>/dev/null; SRV_PID=""; }
finish(){ local rc=$1; stop_srv; printf '%s\n' "$rc" > G3C_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > G3C_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in G3C_RUN_ID G3C_DEADLINE_EPOCH G3C_INSTANCE_ID E4B_SHA GNF4_SHA QWEN_REV; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA QWEN_REV; do val=${!v}; case "$val" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac
  [ ${#val} -eq 40 ] || { say "refusing: $v is not a 40-char sha"; finish 78; }; done
left(){ echo $(( G3C_DEADLINE_EPOCH - $(date +%s) )); }
# an alarm no longer than the time left before the deadline, less a 5-minute fetch reserve
cap(){ local want=$1 l; l=$(( $(left) - 300 )); [ "$l" -lt "$want" ] && want=$l; [ "$want" -lt 60 ] && want=60; echo "$want"; }
need_left(){ [ "$(left)" -ge "$1" ] || { say "DEADLINE: $2 needs ${1}s, $(left)s left"; finish 23; }; }
# Guard fit (bench/common/README.md "Guard fit"), from sc2j-5090-1's walls on the same model and card: installs 372 s (+ the
# ~60 s card burn here), the bf16 fetch and the bake 1,184 s, the shared server's start with its capture about 140 s;
# the static side (boot 300 s + FIT) x 1.25 <= the 1.0 h guard is tests/test_g3_gpu_check.py's.
INSTALL_S=480; FETCH_BAKE_S=1380; INSTALL_TO_GATE_S=$((INSTALL_S + FETCH_BAKE_S)); NEED_SERVER=360; RESERVE=300
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_SERVER + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((G3C_DEADLINE_EPOCH - $(date +%s)))
# rc 41 (the cross-lane guard-fit code; 17 would read to the launcher as a power cap and exclude a healthy host): a
# refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; finish 41; }
say "guard fit: ${LEFT}s left >= ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%)"
# the CUDA devel image ships python3 without venv / ensurepip, git, cmake or rsync: bench/kv-pool-g/g3-smoke's list
need=0; for t in git cmake curl rsync; do command -v $t > /dev/null 2>&1 || need=1; done
python3 -c "import venv, ensurepip" 2> /dev/null || need=1
if [ "$need" = 1 ]; then
  say "apt: python3-venv / git / cmake / curl / rsync / build-essential (missing on this image)"
  perl -e 'alarm 900; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 python3-dev python3-venv python3-pip git cmake ninja-build curl rsync ca-certificates build-essential' > logs/apt.log 2>&1 \
    || { tail -3 logs/apt.log; say "APT FAIL"; finish 9; }
fi
python3 -c "import venv, ensurepip" || { say "python3 still lacks venv / ensurepip after apt"; finish 9; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
: > summary.txt; echo "$G3C_INSTANCE_ID" > INSTANCE_ID
line(){ echo "$*" | tee -a summary.txt; }
for f in g3check_run.sh g3check.py k8_bake.py calib.json card_clock.py staged.sha256; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 > /dev/null) || { say "STAGED FILES DIFFER FROM bench/kv-pool-g/g3-gpu-check/staged.sha256"; finish 9; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,power.limit,enforced.power.limit --format=csv,noheader > forensics.txt 2>&1 || { say "DUD BOX (nvidia-smi)"; finish 10; }
cat forensics.txt; line "CARD $(cut -d, -f1-3,5- forensics.txt)"   # the GPU UUID stays in forensics.txt (store-only)
QWEN=Qwen/Qwen3-30B-A3B
FASTAPI_PIN=0.141.1; UVICORN_PIN=0.54.0   # bench/sc2/sc2_box_e.sh install_sc2_client's pins
PY=python3
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" "$PY" -m pip install -q --no-input "$@" > "$log" 2>&1 && return 0
  say "pip failed ($(tail -1 "$log" | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" "$PY" -m pip install -q --no-input "$@" >> "$log" 2>&1; }
say "install torch 2.8.0+cu128, e4b @$E4B_SHA, gnf4 @$GNF4_SHA (venv --system-site-packages; transformers 5.16.1 / bitsandbytes 0.50.1)"
python3 -m venv --system-site-packages $W/venv > logs/venv.log 2>&1 || { tail -3 logs/venv.log; say "VENV FAIL"; finish 9; }
PY=$W/venv/bin/python
pipx logs/pip_torch.log "$(cap 1800)" "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 || { tail -3 logs/pip_torch.log; say "PIP FAIL (torch)"; finish 9; }
# the card, recorded before anything else loads it (bench/common/card_clock.py; RTX 5090 floor 2700 MHz, TC1 Am. 80)
perl -e 'alarm 180; exec @ARGV' "$PY" $W/card_clock.py burn --floor 2700 --out $W/card_clock.json > logs/card_clock.log 2>&1; CC=$?
line "CARD_CLOCK rc=$CC (recorded, not a gate) $(tail -1 logs/card_clock.log | cut -c1-200)"
pipx logs/pip_e4b.log "$(cap 1800)" --prefer-binary "experts4bit-qlora[serve] @ git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" accelerate sentencepiece safetensors "huggingface_hub>=0.23" "brotli>=1.2.0" \
  "fastapi==$FASTAPI_PIN" "uvicorn==$UVICORN_PIN" || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
# ^ e4b WITH its own serve extra (fastapi>=0.110, uvicorn>=0.29, pydantic>=2, ...): the shipped dependency set of the
#   server this box runs. The two exact pins sit inside the extra's ranges, at bench/sc2/sc2_box_e.sh
#   install_sc2_client's versions, for reproducibility. g3-gpu-check-1 died at the server's start with "No module named
#   'uvicorn'": its install came from the in-process g3-smoke kit, which starts no server.
pipx logs/pip_gnf4.log "$(cap 900)" --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA "$PY" - <<'PYT' 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the commit under test: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import torch
assert torch.cuda.is_available(), "no CUDA"
from experts4bit_qlora import serve_recipe as r, serve_paged as s
from experts4bit_qlora.serve_paged import create_app, main as serve_main     # noqa: F401 -- the entry point the box runs
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
assert hasattr(r, "prefill_graph_headroom_bytes") and hasattr(r, "bulk_flush_bytes"), "e4b lacks #2087's headroom rule"
assert "prefill_graph_headroom_bytes" in inspect.getsource(r.size_shared_pool), "size_shared_pool does not hold back the headroom (#2087)"
assert "prefill_graph_headroom_bytes" in inspect.getsource(PagedModelRunner.enable_prefill_graph), "the runtime gate is not #2087's rule"
assert "kv_pool_sizing" in inspect.getsource(s.create_app), "e4b lacks /health engine.kv_pool_sizing"
import experts4bit_qlora as e
print("G3C_TRIPWIRE ok: e4b", e.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
PYT
# the server's web stack, by NAME and at its pins, before the 61 GB fetch: serve_paged imports fastapi, its responses
# and pydantic inside create_app, and uvicorn inside main, so importing serve_paged proves none of them
# (g3-gpu-check-1). Exit 9 on any miss.
WANT_FASTAPI=$FASTAPI_PIN WANT_UVICORN=$UVICORN_PIN "$PY" - <<'PYW' 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "WEBSTACK TRIPWIRE FAIL"; finish 9; }
import os
import fastapi
import uvicorn
import pydantic
from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: F401 -- create_app's call-time imports
from fastapi.responses import StreamingResponse  # noqa: F401
from pydantic import BaseModel  # noqa: F401
assert fastapi.__version__ == os.environ["WANT_FASTAPI"], f"fastapi {fastapi.__version__}, not the pin"
assert uvicorn.__version__ == os.environ["WANT_UVICORN"], f"uvicorn {uvicorn.__version__}, not the pin"
print("G3C_WEBSTACK ok: fastapi", fastapi.__version__, "uvicorn", uvicorn.__version__, "pydantic", pydantic.VERSION)
PYW
need_left $((FETCH_BAKE_S + NEED_SERVER)) "the fetch, bake and server"
say "fetch $QWEN @ $QWEN_REV"
# e4b#2041's fix, as bench/sc1/sc1_run.sh's fetch(): huggingface_hub on httpx decodes brotli with an API that needs
# brotli >= 1.2.0 (pinned in the e4b install above); a fetch that still fails is retried once after 20 s (the snapshot
# resumes), its alarm re-armed from the time left, before it is a DL FAIL
SNAP="from huggingface_hub import snapshot_download as s; print(s('$QWEN', revision='$QWEN_REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))"
perl -e "alarm $(cap 2400); exec @ARGV" "$PY" -c "$SNAP" > logs/fetch.log 2>&1 \
  || { say "fetch failed ($(tail -1 logs/fetch.log | cut -c1-160)) -- one retry in 20 s"; sleep 20
       perl -e "alarm $(cap 2400); exec @ARGV" "$PY" -c "$SNAP" >> logs/fetch.log 2>&1; } \
  || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
line "FETCH $QWEN@$QWEN_REV $(tail -1 logs/fetch.log)"
say "bake the NF4 arena"
mkdir -p $W/work_qwen3
K8_MODEL="$QWEN" K8_WORK="$W/work_qwen3" perl -e "alarm $(cap 1800); exec @ARGV" "$PY" $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work_qwen3/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
# SC1's Qwen3 stack (bench/sc1/sc1_run.sh FOLDS / SPEEDENV / ROUTEENV) and box Q's int4 LEV, unchanged
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
PORT=18431; URL=http://127.0.0.1:$PORT
need_left $NEED_SERVER "the server"
say "server: shared pool, auto width, 4,096 tokens a slot"
# shellcheck disable=SC2086  # assignment lists by design
setsid env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN -u E4B_PAGED_PREFILL_GRAPH -u E4B_PAGED_BULK_KV \
    -u E4B_PAGED_GRAPHS -u E4B_PAGED_BUCKETS -u E4B_PAGED_MAX_SEQS -u E4B_PAGED_KV_POOL -u E4B_PAGED_CHUNK_TOKENS PYTHONPATH= \
    $ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1 \
    E4B_PAGED_MAX_SEQS=auto E4B_PAGED_KV_POOL=shared E4B_PAGED_MODEL=$QWEN E4B_PAGED_REVISION=$QWEN_REV \
    E4B_PAGED_ARENA=$W/work_qwen3/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_TOKENS_PER_SEQ=4096 \
    E4B_HOST=127.0.0.1 E4B_PORT=$PORT \
    "$PY" -m experts4bit_qlora.serve_paged > logs/server.log 2>&1 < /dev/null &
SRV_PID=$!
t0=$(date +%s); RDY=1
while :; do
  st=$(curl -s -m 5 "$URL/health" 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null)
  case "$st" in ready|busy) RDY=0; break;; error) say "engine error"; break;; esac
  kill -0 "$SRV_PID" 2>/dev/null || { say "server exited before ready: $(tail -2 logs/server.log | tr '\n' ' ' | cut -c1-240)"; break; }
  [ $(( $(date +%s) - t0 )) -ge "$(cap 1500)" ] && { say "server not ready after $(( $(date +%s) - t0 ))s (status=${st:-none})"; break; }
  sleep 3
done
# the start record is kept whatever happened: a refused graph or an engine error is read from it
curl -fsS -m 30 "$URL/health" -o $W/health_start.json 2>/dev/null || say "no /health at start"
line "SERVER ready_rc=$RDY after $(( $(date +%s) - t0 ))s"
DRV=0
if [ "$RDY" = 0 ]; then
  perl -e 'alarm 900; exec @ARGV' "$PY" $W/g3check.py drive --port $PORT --model "$QWEN" --n 6 --max-tokens 8 2>&1 | tee logs/drive.log | grep -a '^G3CHECK drive' | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || DRV=1
  curl -fsS -m 30 "$URL/health" -o $W/health_end.json || say "no /health after the prompts"
fi
HF=$W/health_end.json; [ -s "$HF" ] || HF=$W/health_start.json
"$PY" $W/g3check.py verdict --health "$HF" --out $W/verdict_g3check.json 2>&1 | tee -a summary.txt
VRC=${PIPESTATUS[0]}
# one exit code per cause (CHECK.md's exit table; none is the launcher's host-excluding 13/14/17/18):
#   0 PASS; 1 FAIL (a server record read, and it fails the check: an engine error at start or after the prompts, a
#   graph refused at start, or an end record that fails); 31 UNREAD (no server state; a server that never reached
#   ready; or ready but no end record after the prompts, so nothing was read after them: a harness fault, never a G3
#   FAIL); 32 the prompts failed while the server stayed up (its end record says so)
END=0; [ -s $W/health_end.json ] && END=1
ST=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$HF" 2>/dev/null)
RAS=$("$PY" -c 'import json,sys; print((json.load(open(sys.argv[1])).get("prefill_graph") or {}).get("status",""))' \
      $W/health_start.json 2>/dev/null)
if [ "$VRC" = 3 ]; then RC=31
elif [ "$ST" = error ]; then RC=1                       # an engine error is read, at start or after the prompts
elif [ "$RDY" != 0 ]; then RC=31                        # answered, never ready
elif [ "$RAS" = refused ]; then RC=1                    # the graph refused at start: decided there, with or without an end record
elif [ "$END" = 0 ]; then RC=31                         # ready, but nothing read after the prompts (replays unknown)
elif [ "$DRV" != 0 ]; then RC=32
elif [ "$VRC" = 0 ]; then RC=0
else RC=1; fi
finish "$RC"
