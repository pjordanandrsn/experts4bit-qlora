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
} > forensics.txt 2>&1
cat forensics.txt

MEM_KB=$(awk '/^MemTotal/{print $2}' /proc/meminfo 2>/dev/null)
MEM_GIB=$(( ${MEM_KB:-0} / 1048576 ))
say "host RAM: ${MEM_GIB} GiB"
# STOP-1: a pass on a big-RAM host says nothing about the failing class. Recorded, not refused
# -- the arms still run (their logs are free evidence) but the verdict is withheld.
CLASS_DRAWN=1
if [ "$MEM_GIB" -gt "${P55_MAX_HOST_GIB:-72}" ]; then
  CLASS_DRAWN=0
  say "STOP-1: host has ${MEM_GIB} GiB > ${P55_MAX_HOST_GIB:-72} GiB -- CLASS NOT DRAWN; arms run, P1 gets no verdict"
fi
echo "$CLASS_DRAWN" > class_drawn.txt

pip install -q "experts4bit-qlora @ git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" 2>&1 | tail -3
python3 -c "import experts4bit_qlora, torch; print('e4b', experts4bit_qlora.__version__, 'torch', torch.__version__)" | tee -a summary.txt

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
  ( echo "epoch,mem_available_kb,mem_free_kb"
    while :; do
      echo "$(date +%s),$(awk '/^MemAvailable/{print $2}' /proc/meminfo),$(awk '/^MemFree/{print $2}' /proc/meminfo)"
      sleep 2
    done ) > mem_trace.csv &
  TRACE_PID=$!
  run_arm C_headroom E4B_LOAD_SYNC_DEBUG=1 CUDA_LAUNCH_BLOCKING=1
  kill "$TRACE_PID" 2>/dev/null; wait "$TRACE_PID" 2>/dev/null
  say "C_headroom: MemAvailable min $(awk -F, 'NR>1 && $2!=""{if(m==""||$2<m)m=$2}END{printf "%.1f GiB", m/1048576}' mem_trace.csv 2>/dev/null)"
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
