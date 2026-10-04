# shellcheck shell=bash
# bench/sc2/sc2b_box_f.sh -- lane SC2b (#846), box F: e4b serve_paged's prefill graph, OFF against ON, on today's stack
# (bench/sc2/SC2b-PREREG.md). Sourced by sc1_run.sh when SC1_BOX=F, AFTER sc2_box_e.sh (whose helpers it uses:
# gpu_free, wait_e4b_ready, stop_pid, drive, install_sc2_client, sc2_prompts) and SC1's common setup.
#
# One e4b arm, int4 (SC1's levers + E4B_PAGED_FUSE_QKV=1, as SC2's e4b_int4), with the prefill graph knob OFF or ON. The
# knob is read at server start, so every (draw, arm) is its own server: draw 1 runs OFF then ON, draw 2 ON then OFF.
# Both arms of a draw use the SAME plan seeds, so the A/B is paired on identical arrivals (SC2's per-draw seeds made the
# two draws realise different rates; that remains between draws, never between arms). Draw 1's OFF server also runs the
# serial plan twice: the determinism control the identity gate needs.
#
# Every e4b server runs with NEITHER prefill route pin in its environment (`env -u`, belt and braces: box F's sc1_run.sh
# exports none), and the server's own /health `prefill_routes` must read k19 / k19 / flash with device grouping on, or
# the arm STOPs (rc 47) -- SC2's correction (#1061).

SC2B_KNOB=E4B_PAGED_PREFILL_GRAPH
SC2B_RATES="1 2 4 8"; SC2B_DRAWS=2; SC2B_N=120; SC2B_SERIAL_N=24

# routes_ok HEALTH.json -- the registered routes, read from the server's own record
routes_ok(){ "$PY" - "$1" <<'PYR'
import json, sys
h = json.load(open(sys.argv[1]))
r = h.get("prefill_routes") or {}
want = {"int4_prefill": "k19", "int4_prefill_above_256_rows": "k19", "prefill_attn": "flash", "device_grouping": True,
        "int4_prefill_env": None, "prefill_attn_env": None}
bad = {k: r.get(k, "<missing>") for k, v in want.items() if r.get(k, "<missing>") != v}
e = h.get("engine") or {}
for k in ("chunk_tokens", "max_prefill_tokens_per_step"):   # every 512-token prompt is ONE first chunk
    if e.get(k) != 512:
        bad[f"engine.{k}"] = e.get(k, "<missing>")
print("SC2B_ROUTES " + json.dumps({"ok": not bad, "bad": bad, "routes": r}), flush=True)
sys.exit(1 if bad else 0)
PYR
}

# f_early TAG ARM WARM_JSON -- right after the 4 warm requests, before any paid workload: the knob's own counters must
# already read as registered (OFF: status off; ON: status on, replays == 4, eager_chunks == 0) and every warm response
# must report 512 prompt tokens (a tokenization drift would give short or later chunks and VOID only at the end). STOP
# rc 48 otherwise.
f_early(){ curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${1}_warm.json || { line "SC2B STOP: $1 no /health after warm"; return 48; }
  "$PY" - "$2" $W/sc2/health_${1}_warm.json "$3" <<'PYE' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2B STOP: $1 early engagement"; return 48; }
import json, sys
arm, h, warm = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
g = h.get("prefill_graph") or {}
bad = []
if arm == "off" and (g.get("status") != "off" or g.get("requested") is not False):
    bad.append(f"prefill_graph {g}")
if arm == "on" and (g.get("status") != "on" or g.get("replays") != 4 or g.get("eager_chunks") != 0):
    bad.append(f"prefill_graph {g}")
pt = sorted({r.get("prompt_tokens") for r in warm["requests"]}, key=str)
if pt != [512]:
    bad.append(f"warm prompt_tokens {pt}")
print("SC2B_EARLY " + json.dumps({"arm": arm, "ok": not bad, "bad": bad}), flush=True)
sys.exit(1 if bad else 0)
PYE
}

# f_server_start TAG ARM(off|on) MODEL REV ARENA LEV -- one e4b server; returns 0 healthy on the registered routes
f_server_start(){ local TAG=$1 ARM=$2 MODEL=$3 R=$4 ARENA=$5 LEV=$6 LOG=$W/logs/sc2b_server_$1.log K=0
  [ "$ARM" = on ] && K=1
  # shellcheck disable=SC2086  # assignment lists by design
  setsid env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN PYTHONPATH= $ROUTEENV $LEV $SC2B_KNOB=$K \
      E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA E4B_PAGED_CALIB=$W/calib.json \
      E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2_MAXLEN E4B_PAGED_TRACE=$W/sc2/trace_$TAG.jsonl E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B \
      "$PY" -m experts4bit_qlora.serve_paged > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${TAG}_start.json || return 45
  routes_ok $W/sc2/health_${TAG}_start.json | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2B STOP: $TAG not on the registered routes"; return 47; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits > $W/sc2/vram_${TAG}.txt 2>/dev/null   # record only: the graph's pool costs VRAM
  line "SC2B $TAG healthy ($SC2B_KNOB=$K) vram_used_mib=$(cut -d, -f1 $W/sc2/vram_${TAG}.txt 2>/dev/null | tr -d ' ')"; }
# f_health_end TAG -- the server's record after its runs (the knob's engagement counters live here)
f_health_end(){ curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${1}_end.json || line "SC2B $1: no end-of-arm /health"; }

# f_arm DRAW ARM -- one server, warm, (draw 1 OFF: serial twice), serial, every rate; same seeds as the paired arm
f_arm(){ local D=$1 ARM=$2 TAG="e4b_${2}_d$1" r
  if can_run 900 "$TAG" && f_server_start "$TAG" "$ARM" "$MID" "$REV" "$QA" "$SPEEDENV E4B_PAGED_FUSE_QKV=1"; then
    sampler_start sc2b_$TAG
    drive ${TAG}_warm $PORT_E4B "$MID" e4b serial 0 4 999
    if f_early "$TAG" "$ARM" $W/sc2/${TAG}_warm.json; then
      drive ${TAG}_serial $PORT_E4B "$MID" e4b serial 0 $SC2B_SERIAL_N $D
      [ "$D" = 1 ] && [ "$ARM" = off ] && drive ${TAG}_serial_repeat $PORT_E4B "$MID" e4b serial 0 $SC2B_SERIAL_N $D
      for r in $SC2B_RATES; do
        can_run $(( SC2B_N / r + 240 )) "$TAG rate $r" && drive ${TAG}_r$r $PORT_E4B "$MID" e4b poisson $r $SC2B_N $(( D * 100 + r ))
      done
    fi
    f_health_end "$TAG"; sampler_stop sc2b_$TAG
  fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180; }

# ---- the real lane
box_f(){
  phase 0 "fetch (bf16 for the bake), bake, the SC2 prompt pool"
  fetch_common || finish 11; bake_qwen3 || finish 12; sc2_prompts || finish 19
  quiesce arms
  phase F1 "draw 1: OFF then ON (same seeds)"; f_arm 1 off; f_arm 1 on
  phase F2 "draw 2: ON then OFF (same seeds)"; f_arm 2 on; f_arm 2 off
  phase RD "the reading"
  "$PY" $W/sc2b_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2b.json 2>&1 | tail -30 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): both arms on the proof model, each on the registered routes, the identity gate on a
# serial smoke, and the knob's engagement as the reducer reads it
prove_f(){ local a
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_reduce.py sc2_identity.py sc2b_reduce.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$GR" --revision "$GR_REV" --out $W/sc2/prompts.json > logs/sc2_prompts.log 2>&1 \
    || { tail -3 logs/sc2_prompts.log; say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  for a in off on; do
    if f_server_start "prove_$a" "$a" "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$GR_ENV"; then
      { drive prove_${a}_warm $PORT_E4B "$GR" e4b serial 0 4 999 && f_early "prove_$a" "$a" $W/sc2/prove_${a}_warm.json \
        && drive prove_${a}_serial $PORT_E4B "$GR" e4b serial 0 6 1 && drive prove_${a}_poisson $PORT_E4B "$GR" e4b poisson 4 16 2; } \
        || { say "PROVE: $a smoke failed -- NOT PROVED"; rec 23; }
      f_health_end "prove_$a"
    else say "PROVE: $a server failed"; rec 23; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  done
  "$PY" $W/sc2_identity.py $W/sc2/prove_off_serial.json $W/sc2/prove_on_serial.json | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: OFF and ON differ on the serial smoke -- NOT PROVED"; rec 23; }
  "$PY" $W/sc2b_reduce.py --engagement $W/sc2/health_prove_off_end.json $W/sc2/health_prove_on_end.json | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: the knob did not engage as registered -- NOT PROVED"; rec 23; }; }
