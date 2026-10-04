#!/bin/bash
# bench/p55/p55_run.sh -- lane P55, BOX side. Three load attempts of google/gemma-4-26B-A4B-it
# through load_moe_4bit_streaming: unarmed, armed (E4B_LOAD_SYNC_DEBUG=1 + CUDA_LAUNCH_BLOCKING=1),
# and armed with a host-memory trace. No timing, no arms, no bake. See bench/p55/P55-PREREG.md.
set -uo pipefail
: "${P55_RUN_ID:?}"; : "${P55_RUN_NONCE:?}"; : "${P55_DEADLINE_EPOCH:?}"; : "${E4B_SHA:?}"
W=$(cd "$(dirname "$0")" && pwd); cd "$W" || exit 20
mkdir -p logs
NONCE=$P55_RUN_NONCE
echo "$NONCE" > P55_RUN_NONCE                  # the controller's start handshake
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a summary.txt; }
# A terminal marker must encode SUCCESS, not just "the script stopped": TP_DONE on every exit,
# P55_SUCCESS only on a clean one, so the controller cannot read a crash as a completed lane.
finish(){ local rc=$1; printf '%s\n' "$rc" > "P55_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "P55_SUCCESS.$NONCE"; say "TP_DONE rc=$rc"; : > "TP_DONE.$NONCE"; exit "$rc"; }
trap 'finish 130' INT TERM
MODEL=${P55_MODEL:-google/gemma-4-26B-A4B-it}
REV=${P55_REVISION:-4d7ae4984b7db7de8f8457170b3f1a419ee76d52}

say "P55 $P55_RUN_ID: e4b $E4B_SHA, model $MODEL @ $REV"

# ---- forensics FIRST: STOP-1 is decided on these numbers, and a box that dies later must
# still have left behind the record of which class was drawn.
{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
  # `model name` is an x86 spelling; an ARM host has none and a bare grep leaves the field
  # blank, which reads as "not recorded" when it means "recorded as nothing". Fall back, and
  # say `unknown` rather than print an empty value -- this line is half the fingerprint.
  cpu=$(grep -m1 -E '^(model name|Model|Hardware)' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')
  echo "cpu: ${cpu:-unknown}"
  echo "kernel: $(uname -r)"
  grep -E '^(MemTotal|MemAvailable|SwapTotal)' /proc/meminfo 2>/dev/null
  cg=""
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [ -r "$f" ] && { echo "cgroup_limit($f): $(cat "$f")"; cg=1; break; }
  done
  # An ABSENT limit is a fact about the host, not a gap in the record: the 30 GiB box that
  # refused the map had a 31 GiB cgroup, so "no limit readable" is what distinguishes a host
  # with all its RAM from one that merely reports it.
  [ -z "$cg" ] && echo "cgroup_limit: none readable"
  echo "overcommit_memory: $(cat /proc/sys/vm/overcommit_memory 2>/dev/null)"
  echo "overcommit_ratio: $(cat /proc/sys/vm/overcommit_ratio 2>/dev/null)"
  echo "max_map_count: $(cat /proc/sys/vm/max_map_count 2>/dev/null)"
  # Amendment 1: the per-process limits a marginal mapping or a pinned copy can hit (RLIMIT_AS, RLIMIT_MEMLOCK).
  echo "ulimit:"; ulimit -a 2>/dev/null | sed 's/^/  /'
} > forensics.txt 2>&1
cat forensics.txt

# STOP-1 (Amendment 1): the class is the memory a process here can have -- min(MemTotal, the cgroup limit) -- not
# MemTotal, which inside a Vast container is the whole host's. A pass on a big-memory box says nothing about the
# failing class. Recorded, not refused: the arms still run (their logs are free evidence) but the verdict is withheld.
python3 p55_ram.py --class-max-gib "${P55_MAX_HOST_GIB:-72}" > ram.txt 2>&1 || { cat ram.txt; say "HARNESS: p55_ram.py could not read this host's memory"; finish 43; }
cat ram.txt >> forensics.txt
EFF_GIB=$(awk -F': ' '/^effective_ram_gib:/{print $2}' ram.txt)
CLASS_DRAWN=$(awk -F': ' '/^class_drawn:/{print $2}' ram.txt)
say "memory: MemTotal $(awk -F': ' '/^mem_total_gib:/{print $2}' ram.txt) GiB, cgroup limit $(awk -F': ' '/^cgroup_limit_bytes:/{print $2}' ram.txt) B -> effective ${EFF_GIB} GiB"
case "$CLASS_DRAWN" in
  1) ;;
  0) say "STOP-1: effective memory ${EFF_GIB} GiB > ${P55_MAX_HOST_GIB:-72} GiB -- CLASS NOT DRAWN; arms run, P1 gets no verdict" ;;
  *) say "HARNESS: p55_ram.py gave no class decision"; finish 43 ;;
esac
echo "$CLASS_DRAWN" > class_drawn.txt

# ---- install (Amendment 2): experts4bit-qlora's BASE dependencies are torch and bitsandbytes only. The loader's
# transformers / safetensors / huggingface_hub live in its extras, so the registered bare `pip install` could never
# load the model (p55-5090-1 died at the fetch on `No module named 'huggingface_hub'`). Installed as P113's runner
# does: the image's torch held by a constraint, the loader's stack pinned, bounded, one retry, exit 9 on failure.
# git is what `pip install git+https://...` runs. Vast's ssh runtime layer has supplied it so far (P113 relied on that
# without saying so); the lane's image itself does not ship it, which the $0 rehearsal found. Installed only when absent.
if ! command -v git >/dev/null 2>&1; then
  say "git absent -- installing it (apt, bounded)"
  perl -e "alarm 600; exec @ARGV" sh -c "DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git" > logs/apt_git.log 2>&1 \
    || { tail -3 logs/apt_git.log; say "HARNESS: cannot install git -- no verdict"; finish 9; }
fi
TORCH_PIN=$(python3 -c "import torch; print(torch.__version__.split('+')[0])") || { say "HARNESS: no torch in the image"; finish 9; }
echo "torch==$TORCH_PIN" > constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python3 -m pip install -q --no-input -c constraints.txt "$@" > "$log" 2>&1 && return 0
  say "pip failed ($(tail -1 "$log" | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python3 -m pip install -q --no-input -c constraints.txt "$@" >> "$log" 2>&1; }
say "install e4b @$E4B_SHA (transformers 5.17.0, bitsandbytes 0.50.2; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1200 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" accelerate safetensors "huggingface_hub>=0.23" \
  || { tail -4 logs/pip_e4b.log; say "PIP FAIL -- harness fault: no verdict"; finish 9; }
# The tripwire: the installed commit is the launch commit, and everything the probe and the fetch import imports.
WANT_E4B=$E4B_SHA python3 - <<'PYT' 2>&1 | tee -a summary.txt
import json, os, sys, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
if d.get("vcs_info", {}).get("commit_id") != os.environ["WANT_E4B"]:
    sys.exit(f"TRIPWIRE: installed e4b is not the launch commit: {d}")
import torch, transformers, huggingface_hub, safetensors, bitsandbytes  # noqa: F401
from experts4bit_qlora.loader import load_moe_4bit_streaming  # noqa: F401
import experts4bit_qlora as e
print(f"tripwire OK: e4b {e.__version__} @{os.environ['WANT_E4B'][:12]} torch {torch.__version__} transformers "
      f"{transformers.__version__} huggingface_hub {huggingface_hub.__version__} bitsandbytes {md.version('bitsandbytes')}")
PYT
grep -q "^tripwire OK:" summary.txt || { say "TRIPWIRE FAIL -- harness fault: no verdict"; finish 9; }

# ---- the fetch (Amendment 1, defect 5): BEFORE the arms, bounded, Xet disabled. Registered, the first arm downloaded the
# 51.6 GB checkpoint inside its own load -- unbounded, on the Xet backend that wedges at ~6.1 MB on this fleet, so a stall
# would have spent the whole guard with no reading, and A_baseline's failure window would have held a download. A fetch
# that fails is a harness fault (STOP-2 shape): no verdict, no redraw.
export HF_HUB_DISABLE_XET=1
FETCH_S=$(( P55_DEADLINE_EPOCH - $(date +%s) - 900 ))      # leave 15 min for the three loads
if [ "$FETCH_S" -lt 120 ]; then say "STOP-2: under 17 min of guard left before the fetch"; finish 11; fi
say "fetch: $MODEL @ $REV (51.6 GB, Xet disabled, alarm ${FETCH_S}s)"
perl -e "alarm $FETCH_S; exec @ARGV" python3 -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors', '*.json', 'tokenizer*', '*.model', '*.txt'], max_workers=8))" > logs/fetch.log 2>&1 \
  || { tail -3 logs/fetch.log; say "DL FAIL -- harness fault (STOP-2): no verdict"; finish 11; }
say "fetch: done, $(du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | cut -f1) in the HF cache"

# ---- the probe. One process per arm: CUDA_LAUNCH_BLOCKING is read at context creation, so
# the armed arm MUST be a fresh process -- an in-process second attempt would silently run
# unarmed and produce exactly the uninformative result #344 already has five of.
cat > p55_probe.py <<'PYEOF'
import json, os, sys, traceback
name = sys.argv[1]
out = {"arm": name, "env": {k: os.environ.get(k) for k in
                            ("E4B_LOAD_SYNC_DEBUG", "CUDA_LAUNCH_BLOCKING")}}
try:
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    model, cfg = load_moe_4bit_streaming(
        os.environ["P55_MODEL"], "cuda", __import__("torch").bfloat16,
        r=8, alpha=16, quant_type="nf4", revision=os.environ["P55_REVISION"])
    out["status"] = "OK"
    out["n_layers"] = int(getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", -1))
except BaseException as exc:                      # every outcome is data here
    out["status"] = "FAILED"
    out["exc_type"] = type(exc).__name__
    out["exc_module"] = type(exc).__module__
    out["exc_msg"] = str(exc)
    out["notes"] = list(getattr(exc, "__notes__", []))
    out["tb"] = traceback.format_exc()
json.dump(out, open(f"result_{name}.json", "w"), indent=1)
print(f"P55 {name}: {out['status']}" + ("" if out["status"] == "OK" else f" {out['exc_type']}: {out['exc_msg'][:200]}"))
PYEOF

export P55_MODEL="$MODEL" P55_REVISION="$REV"

run_arm(){  # name, then the env assignments for this arm
  local name=$1; shift
  if [ "$(date +%s)" -ge $(( P55_DEADLINE_EPOCH - 300 )) ]; then
    say "arm $name SKIPPED: under 5 min of guard left (STOP-2 shape: never shortened)"; return 0
  fi
  say "arm $name: starting ($*)"
  ( env "$@" python3 p55_probe.py "$name" ) > "logs/$name.log" 2>&1
  local rc=$?
  say "arm $name: rc=$rc  $(grep -m1 '^P55 ' "logs/$name.log" 2>/dev/null)"
  # The diagnosis is the deliverable; echo it into the summary so the fetched summary.txt
  # is readable without opening the logs.
  grep -E '^e4b: |^  \[sync\] |^  E4B_LOAD_SYNC_DEBUG' "logs/$name.log" 2>/dev/null | tee -a summary.txt
}

run_arm A_baseline  E4B_DUMMY=0
run_arm B_sync      E4B_LOAD_SYNC_DEBUG=1 CUDA_LAUNCH_BLOCKING=1

# arm C: the same armed load with MemAvailable sampled alongside it.
if [ "$(date +%s)" -lt $(( P55_DEADLINE_EPOCH - 300 )) ]; then
  say "arm C_headroom: starting (armed, with a 2 s MemAvailable trace)"
  # Amendment 1: the cgroup's usage and limit beside MemAvailable -- inside a container MemAvailable is the host's.
  ( echo "epoch,mem_available_kb,mem_free_kb,cgroup_usage_bytes,cgroup_limit_bytes"
    while :; do
      python3 p55_ram.py --sample
      sleep 2
    done ) > mem_trace.csv &
  TRACE_PID=$!
  run_arm C_headroom E4B_LOAD_SYNC_DEBUG=1 CUDA_LAUNCH_BLOCKING=1
  kill "$TRACE_PID" 2>/dev/null; wait "$TRACE_PID" 2>/dev/null
  say "C_headroom: MemAvailable min $(awk -F, 'NR>1 && $2!=""{if(m==""||$2<m)m=$2}END{printf "%.1f GiB", m/1048576}' mem_trace.csv 2>/dev/null); cgroup usage max $(awk -F, 'NR>1 && $4!=""{if($4>m)m=$4}END{printf "%.1f GiB", m/1073741824}' mem_trace.csv 2>/dev/null)"
fi

python3 - <<'PYEOF' | tee -a summary.txt
import json, os
rows = []
for arm in ("A_baseline", "B_sync", "C_headroom"):
    p = f"result_{arm}.json"
    if os.path.exists(p):
        d = json.load(open(p))
        rows.append((arm, d.get("status"), d.get("exc_type", ""), (d.get("exc_msg") or "")[:90]))
print("=== P55 arms ===")
for a, s, t, m in rows:
    print(f"  {a:12s} {s:7s} {t} {m}")
PYEOF

# ---- the lane's own exit code. Note what is NOT a lane failure: an arm whose LOAD failed is
# the result this lane was registered to obtain (P1), so `A_baseline` dying with a CUDA error
# exits 0. What fails the lane is the harness being unable to answer: an arm that produced no
# result at all, or the armed arm producing no [sync] bound (prereg P2 -- the instrumentation
# is then the defect, and a green lane would hide it).
rc_any=0
for arm in A_baseline B_sync; do
  [ -s "result_$arm.json" ] || { say "HARNESS: arm $arm produced no result file"; rc_any=40; }
done
if [ -s result_B_sync.json ] && ! grep -aq "E4B_LOAD_SYNC_DEBUG=1: staged synchronisation ON" logs/B_sync.log; then
  say "HARNESS: B_sync ran without the debug banner -- the flag did not arm (prereg P2 refuted by the harness, not by the box)"
  rc_any=41
fi
if [ -s result_B_sync.json ] && grep -aq '"status": "FAILED"' result_B_sync.json \
   && ! grep -aq '\[sync\]' logs/B_sync.log; then
  say "HARNESS: B_sync failed with no [sync] stage bound at all -- prereg P2 refuted; the instrumentation is the defect"
  rc_any=42
fi
say "class_drawn=$CLASS_DRAWN (P1 gets a verdict only when 1)"
say "----- summary -----"
finish "$rc_any"
