# shellcheck shell=bash
# bench/sc2/sc2e_box_l.sh -- lane SC2e (#846), box L: e4b serve_paged at 16, 32 and 64 slots, with decode-graph buckets
# that end at max_seqs (E4B_PAGED_BUCKETS=auto) against the default list (bench/sc2/SC2e-PREREG.md). Sourced by
# sc1_run.sh when SC1_BOX=L, AFTER sc2_box_e.sh (drive, gpu_free, wait_e4b_ready, stop_pid, install_sc2_client,
# sc2_prompts) and sc2c_box_h.sh (routes_ok, h_early, h_health_end), and SC1's common setup.
#
# Four servers a draw, each SC2c's ON server (SC1's levers + E4B_PAGED_FUSE_QKV=1, bulk KV and the prefill graph at
# their defaults, both traces on), differing only in E4B_PAGED_MAX_SEQS and E4B_PAGED_BUCKETS:
#   s16  16 slots, the default list (control)     s32a 32 slots, auto (1..32)
#   s64c 64 slots, the default list (1..16)       s64a 64 slots, auto (1..64)
# Draw 1 runs s16 s32a s64c s64a, draw 2 the reverse; every arm of a draw uses the same plan seeds. Per server: 4 warm,
# the burst (64 requests at once: the arm's widest decode step must replay before any run that counts), the serial
# plan (twice on s16 draw 1: the determinism control), then Poisson at every rate.

SC2E_ARMS_D1="s16 s32a s64c s64a"; SC2E_ARMS_D2="s64a s64c s32a s16"
SC2E_RATES="1 2 4 8 12 16"; SC2E_N=120; SC2E_SERIAL_N=24
# The burst: every request asks >= 64 tokens (SC2_LO) and prefill admits one prompt a step, so when 64 arrive at once
# the 64th admission finds the first 63 still decoding: the arm's widest bucket runs.
SC2E_BURST_N=64; SC2E_BURST_RATE=1000; SC2E_BURST_SEED=998

l_slots(){ case "$1" in s16) echo 16;; s32a) echo 32;; s64c|s64a) echo 64;; *) return 1;; esac; }
l_buckets(){ case "$1" in s32a|s64a) echo auto;; s16|s64c) echo "";; *) return 1;; esac; }

# l_tripwire -- the installed e4b is the code under test: E4B_PAGED_BUCKETS=auto (serve_recipe.default_buckets) and
# /health's per-bucket graph_stats. Without them SLOTS and ENGAGED would read <missing> on every server. Refuse (rc 9).
l_tripwire(){ "$PY" -c "import inspect, experts4bit_qlora.serve_paged as s; from experts4bit_qlora.serve_recipe import default_buckets; assert default_buckets(64) == (1, 2, 4, 8, 16, 32, 64) and hasattr(s, '_graph_stats') and 'buckets_requested' in inspect.getsource(s.create_app), 'e4b predates E4B_PAGED_BUCKETS=auto'" \
    2>&1 | tail -1 | sed 's/^/SC2E_TRIPWIRE /' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SC2e: the installed e4b has no E4B_PAGED_BUCKETS=auto"; finish 9; }
  line "SC2E_TRIPWIRE ok: E4B_PAGED_BUCKETS=auto and /health graph_stats present"; }

# slots_ok HEALTH.json ARM -- the arm's slots and buckets, every bucket captured, the scratch slots sized by the largest
slots_ok(){ "$PY" - "$1" "$2" <<'PYS'
import json, sys
h, arm = json.load(open(sys.argv[1])), sys.argv[2]
WANT = {"s16": (16, [1, 2, 4, 8, 16], "default"), "s32a": (32, [1, 2, 4, 8, 16, 32], "auto"),
        "s64c": (64, [1, 2, 4, 8, 16], "default"), "s64a": (64, [1, 2, 4, 8, 16, 32, 64], "auto")}
m, b, req = WANT[arm]
e = h.get("engine") or {}
kv = (h.get("levers") or {}).get("kv") or {}
bad = {f"engine.{k}": e.get(k, "<missing>") for k, v in (("max_seqs", m), ("kv_slots", m), ("buckets", b),
                                                           ("buckets_requested", req)) if e.get(k, "<missing>") != v}
gs = e.get("graph_status")
if not isinstance(gs, dict) or set(gs) != {str(x) for x in b} or any(v != "graph" for v in gs.values()):
    bad["engine.graph_status"] = gs if gs is not None else "<missing>"
if kv.get("scratch_slots") != b[-1]:
    bad["levers.kv.scratch_slots"] = kv.get("scratch_slots", "<missing>")
print("SC2E_SLOTS " + json.dumps({"arm": arm, "ok": not bad, "bad": bad, "pool_mib": kv.get("pool_mib")}), flush=True)
sys.exit(1 if bad else 0)
PYS
}

# burst_ok HEALTH.json BURST.json ARM -- after the burst: every request VALID with 512 prompt tokens, the widest bucket
# replayed at least once, and no bucket ran an eager step
burst_ok(){ "$PY" - "$1" "$2" "$3" <<'PYB'
import json, sys
h, burst, arm = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
top = {"s16": 16, "s32a": 32, "s64c": 16, "s64a": 64}[arm]
bad = []
reqs = burst.get("requests") or []
if len(reqs) != 64 or not all(r.get("valid") for r in reqs):
    bad.append(f"burst: {sum(1 for r in reqs if r.get('valid'))} of {len(reqs)} VALID (want 64 of 64)")
pt = sorted({r.get("prompt_tokens") for r in reqs}, key=str)
if pt != [512]:
    bad.append(f"burst prompt_tokens {pt}")
gs = (h.get("engine") or {}).get("graph_stats") or {}
if (gs.get(str(top)) or {}).get("replays", 0) < 1:
    bad.append(f"bucket {top} never replayed: {gs.get(str(top))!r}")
eager = {k: v.get("eager_steps") for k, v in gs.items() if v.get("eager_steps")}
if eager or not gs:
    bad.append(f"eager decode steps {eager or gs!r}")
print("SC2E_BURST " + json.dumps({"arm": arm, "ok": not bad, "bad": bad, "graph_stats": gs}), flush=True)
sys.exit(1 if bad else 0)
PYB
}

# l_server_start TAG ARM MODEL REV ARENA LEV [SEEN_MOE_PREFIX] -- one e4b server at the arm's slots and buckets, on the
# registered routes (rc 47 otherwise) with the arm's slots and every bucket captured (rc 49 otherwise)
l_server_start(){ local TAG=$1 ARM=$2 MODEL=$3 R=$4 ARENA=$5 LEV=$6 SEEN=${7:-} LOG=$W/logs/sc2e_server_$1.log M B
  M=$(l_slots "$ARM") || { line "SC2E: unknown arm $ARM"; return 78; }; B=$(l_buckets "$ARM")
  # shellcheck disable=SC2086  # assignment lists by design; B empty leaves E4B_PAGED_BUCKETS unset (the default list)
  setsid env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN -u E4B_PAGED_PREFILL_GRAPH -u E4B_PAGED_BULK_KV \
      -u E4B_PAGED_GRAPHS -u E4B_PAGED_BUCKETS PYTHONPATH= $ROUTEENV $LEV E4B_PAGED_MAX_SEQS=$M ${B:+E4B_PAGED_BUCKETS=$B} \
      E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA E4B_PAGED_CALIB=$W/calib.json \
      E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2_MAXLEN E4B_PAGED_TRACE=$W/sc2/trace_$TAG.jsonl \
      E4B_PAGED_STEP_TRACE=$W/sc2/steps_$TAG.jsonl E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B \
      "$PY" -m experts4bit_qlora.serve_paged > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${TAG}_start.json || return 45
  routes_ok $W/sc2/health_${TAG}_start.json "$SEEN" | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2E STOP: $TAG not on the registered routes"; return 47; }
  slots_ok $W/sc2/health_${TAG}_start.json "$ARM" | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2E STOP: $TAG slots or buckets not as registered"; return 49; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits > $W/sc2/vram_${TAG}.txt 2>/dev/null   # record only
  line "SC2E $TAG healthy (max_seqs=$M buckets=${B:-default}) vram_used_mib=$(cut -d, -f1 $W/sc2/vram_${TAG}.txt 2>/dev/null | tr -d ' ')"; }

# l_burst TAG ARM MODEL -- the burst, then its engagement check (rc 48)
l_burst(){ local TAG=$1 ARM=$2 MODEL=$3
  drive ${TAG}_burst $PORT_E4B "$MODEL" e4b poisson $SC2E_BURST_RATE $SC2E_BURST_N $SC2E_BURST_SEED || { line "SC2E STOP: $TAG burst not VALID"; return 48; }
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${TAG}_burst.json || { line "SC2E STOP: $TAG no /health after the burst"; return 48; }
  burst_ok $W/sc2/health_${TAG}_burst.json $W/sc2/${TAG}_burst.json "$ARM" | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2E STOP: $TAG burst engagement"; return 48; }; }

# l_arm DRAW ARM -- one server: warm, early engagement (SC2c's h_early, bulk on), the burst, serial (+ the repeat on s16
# draw 1), every rate; the same seeds as the draw's other arms
l_arm(){ local D=$1 ARM=$2 TAG="e4b_${2}_d$1" r
  if can_run 1200 "$TAG" && l_server_start "$TAG" "$ARM" "$MID" "$REV" "$QA" "$SPEEDENV E4B_PAGED_FUSE_QKV=1" "int4_k19|"; then
    sampler_start sc2e_$TAG
    drive ${TAG}_warm $PORT_E4B "$MID" e4b serial 0 4 999
    if h_early "$TAG" on $W/sc2/${TAG}_warm.json && l_burst "$TAG" "$ARM" "$MID"; then
      drive ${TAG}_serial $PORT_E4B "$MID" e4b serial 0 $SC2E_SERIAL_N $D
      [ "$D" = 1 ] && [ "$ARM" = s16 ] && drive ${TAG}_serial_repeat $PORT_E4B "$MID" e4b serial 0 $SC2E_SERIAL_N $D
      for r in $SC2E_RATES; do
        can_run $(( SC2E_N / r + 240 )) "$TAG rate $r" && drive ${TAG}_r$r $PORT_E4B "$MID" e4b poisson $r $SC2E_N $(( D * 100 + r ))
      done
    fi
    h_health_end "$TAG"; sampler_stop sc2e_$TAG
  fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180; }

# ---- the real lane
box_l(){ local a
  l_tripwire
  phase 0 "fetch (bf16 for the bake), bake, the SC2 prompt pool"
  fetch_common || finish 11; bake_qwen3 || finish 12; sc2_prompts || finish 19
  quiesce arms
  phase L1 "draw 1: s16, s32a, s64c, s64a (same seeds)"; for a in $SC2E_ARMS_D1; do l_arm 1 "$a"; done
  phase L2 "draw 2: s64a, s64c, s32a, s16 (same seeds)"; for a in $SC2E_ARMS_D2; do l_arm 2 "$a"; done
  phase RD "the reading"
  "$PY" $W/sc2e_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2e.json 2>&1 | tail -40 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): every arm on the proof model, on the registered routes and slots, the burst replaying the
# widest bucket, the identity gate on a serial smoke, engagement as the reducer reads it, and step traces with wide steps
prove_l(){ local a s
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_reduce.py sc2_identity.py sc2c_census.py sc2e_census.py sc2e_reduce.py sc2e_basis.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  l_tripwire
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$GR" --revision "$GR_REV" --out $W/sc2/prompts.json > logs/sc2_prompts.log 2>&1 \
    || { tail -3 logs/sc2_prompts.log; say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  for a in s16 s32a s64c s64a; do
    if l_server_start "prove_$a" "$a" "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$GR_ENV" ""; then
      { drive prove_${a}_warm $PORT_E4B "$GR" e4b serial 0 4 999 && h_early "prove_$a" on $W/sc2/prove_${a}_warm.json \
        && l_burst "prove_$a" "$a" "$GR" && drive prove_${a}_serial $PORT_E4B "$GR" e4b serial 0 6 1 \
        && drive prove_${a}_poisson $PORT_E4B "$GR" e4b poisson 8 16 2; } || { say "PROVE: $a smoke failed -- NOT PROVED"; rec 23; }
      h_health_end "prove_$a"
    else say "PROVE: $a server failed"; rec 23; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  done
  for a in s32a s64c s64a; do
    "$PY" $W/sc2_identity.py $W/sc2/prove_s16_serial.json $W/sc2/prove_${a}_serial.json | tee -a summary.txt
    [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: s16 and $a differ on the serial smoke -- NOT PROVED"; rec 23; }
  done
  "$PY" $W/sc2e_reduce.py --engagement $W/sc2/health_prove_s16_end.json $W/sc2/health_prove_s32a_end.json \
      $W/sc2/health_prove_s64c_end.json $W/sc2/health_prove_s64a_end.json | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: a server did not engage as registered -- NOT PROVED"; rec 23; }
  for a in s16 s32a s64c s64a; do
    "$PY" $W/sc2e_census.py steps $W/sc2/steps_prove_$a.jsonl --arm "$a" --require-wide | tee -a summary.txt
    [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: the $a step trace does not decompose with its wide steps -- NOT PROVED"; rec 23; }
  done; }
