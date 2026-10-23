# shellcheck shell=bash
# bench/sc2/sc2j_box_q.sh -- lane SC2j (#846), box Q: e4b serve_paged at E4B_PAGED_MAX_SEQS=auto and 4,096 tokens a slot,
# the shared KV pool (E4B_PAGED_KV_POOL=shared, levers G1-G3) against claim-all (bench/sc2/SC2j-PREREG.md). Sourced by
# sc1_run.sh when SC1_BOX=Q, AFTER sc2_box_e.sh (drive, gpu_free, wait_e4b_ready, stop_pid, install_sc2_client,
# sc2_prompts), sc2c_box_h.sh (routes_ok, h_early, h_health_end), sc2e_box_l.sh (l_tripwire) and sc2f_box_n.sh
# (n_tripwire), and SC1's common setup.
#
# Phases: the contract arm (sc2j_contract.py, in process, explicit 32 slots: claim-all twice and shared, every token
# equal), the QUALITY arm (P117's teacher-forced width check at the launch stack, plus P117's split plans at its table
# width), then two draws of three servers, ca_a sh ca_b, each SC2f's `on` server at 4,096 tokens a slot, auto width and
# auto buckets, differing only in the pool. Per server: 4 warm, the burst (64 at once: the server's widest bucket must
# replay), serial 24, the SC2 ladder (Q2) and the long ladder (Q3, /stats around every rung), then the end /health and
# /stats.

SC2J_ORDER="ca_a sh ca_b"; SC2J_Q2="1 2 4 8 12 16"; SC2J_Q2_N=120; SC2J_SERIAL_N=24
SC2J_Q3="0.5 1 2 4"; SC2J_Q3_N=96; SC2J_Q3_LO=768; SC2J_Q3_HI=1536
SC2J_TOKENS=4096; SC2J_CONTRACT_SLOTS=32
SC2J_BURST_N=64; SC2J_BURST_RATE=1000; SC2J_BURST_SEED=998
unset E4B_PAGED_KV_POOL E4B_PAGED_MAX_SEQS E4B_PAGED_BUCKETS E4B_PAGED_DECODE_LOOKAHEAD   # only an arm names the pool

q_pool(){ case "$1" in sh) echo shared;; ca_a|ca_b|ca) echo claim-all;; *) return 1;; esac; }
q_rtag(){ echo "$1" | tr . p; }

# q_tripwire -- the installed e4b carries levers G1-G3: Fp8PagedKV(pool_rows=...), serve_recipe.size_shared_pool and
# /health's engine.kv_pool_sizing. Refuse (rc 9).
q_tripwire(){ "$PY" -c "import inspect
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
from experts4bit_qlora import serve_recipe as r, serve_paged as s
assert 'pool_rows' in inspect.signature(Fp8PagedKV.__init__).parameters, 'e4b lacks Fp8PagedKV(pool_rows=...) (G3)'
assert hasattr(r, 'size_shared_pool') and hasattr(s, 'resolve_kv_pool'), 'e4b lacks the shared pool sizing (G3)'
assert 'kv_pool_sizing' in inspect.getsource(s.create_app), 'e4b lacks /health engine.kv_pool_sizing'
import experts4bit_qlora as e; print('ok: levers G1-G3 present, experts4bit-qlora', e.__version__)" \
    2>&1 | tail -1 | sed 's/^/SC2J_TRIPWIRE /' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SC2j: the installed e4b lacks levers G1-G3"; finish 9; }; }

# q_server_start TAG ARM MODEL REV ARENA LEV [SEEN] -- one e4b server at auto width, 4,096 tokens a slot, auto buckets,
# the arm's pool; on the registered routes (rc 47) and reading its pool and auto's buckets for its own width (rc 49)
q_server_start(){ local TAG=$1 ARM=$2 MODEL=$3 R=$4 ARENA=$5 LEV=$6 SEEN=${7:-} LOG=$W/logs/sc2j_server_$1.log P
  P=$(q_pool "$ARM") || { line "SC2J: unknown arm $ARM"; return 78; }
  # shellcheck disable=SC2086  # assignment lists by design
  setsid env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN -u E4B_PAGED_PREFILL_GRAPH -u E4B_PAGED_BULK_KV \
      -u E4B_PAGED_GRAPHS -u E4B_PAGED_BUCKETS -u E4B_PAGED_MAX_SEQS -u E4B_PAGED_KV_POOL PYTHONPATH= $ROUTEENV $LEV \
      E4B_PAGED_MAX_SEQS=auto E4B_PAGED_KV_POOL=$P E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA \
      E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2J_TOKENS E4B_PAGED_TRACE=$W/sc2/trace_$TAG.jsonl \
      E4B_PAGED_STEP_TRACE=$W/sc2/steps_$TAG.jsonl E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B \
      "$PY" -m experts4bit_qlora.serve_paged > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${TAG}_start.json || return 45
  routes_ok $W/sc2/health_${TAG}_start.json "$SEEN" | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2J STOP: $TAG not on the registered routes"; return 47; }
  "$PY" $W/sc2j_reduce.py --pool $W/sc2/health_${TAG}_start.json "$ARM" | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2J STOP: $TAG pool or slots not as registered"; return 49; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits > $W/sc2/vram_${TAG}.txt 2>/dev/null   # record only
  line "SC2J $TAG healthy (pool $P) vram_used_mib=$(cut -d, -f1 $W/sc2/vram_${TAG}.txt 2>/dev/null | tr -d ' ')"; }

# q_burst TAG MODEL -- 64 requests at once, every one VALID with 512 prompt tokens; the server's widest bucket replayed
# and no eager decode step (rc 48)
q_burst(){ local TAG=$1 MODEL=$2
  drive ${TAG}_burst $PORT_E4B "$MODEL" e4b poisson $SC2J_BURST_RATE $SC2J_BURST_N $SC2J_BURST_SEED || { line "SC2J STOP: $TAG burst not VALID"; return 48; }
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${TAG}_burst.json || { line "SC2J STOP: $TAG no /health after the burst"; return 48; }
  "$PY" - $W/sc2/health_${TAG}_burst.json $W/sc2/${TAG}_burst.json <<'PYB' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2J STOP: $TAG burst engagement"; return 48; }
import json, sys
h, burst = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
e = h.get("engine") or {}
top = max(e.get("buckets") or [0])
bad = []
reqs = burst.get("requests") or []
if len(reqs) != 64 or not all(r.get("valid") for r in reqs):
    bad.append(f"burst: {sum(1 for r in reqs if r.get('valid'))} of {len(reqs)} VALID (want 64 of 64)")
if sorted({r.get("prompt_tokens") for r in reqs}, key=str) != [512]:
    bad.append("burst prompt_tokens not all 512")
gs = e.get("graph_stats") or {}
if (gs.get(str(top)) or {}).get("replays", 0) < 1:
    bad.append(f"bucket {top} never replayed: {gs.get(str(top))!r}")
if not gs or any(v.get("eager_steps") for v in gs.values()):
    bad.append(f"eager decode steps {gs!r}")
print("SC2J_BURST " + json.dumps({"ok": not bad, "bad": bad, "top": top}), flush=True)
sys.exit(1 if bad else 0)
PYB
}

# q_drive_long TAG MODEL RATE N SEED -- the long ladder's driver run: 2,048-token prompts, max_tokens U[768, 1536]
q_drive_long(){ local TAG=$1 MODEL=$2 RATE=$3 N=$4 SEED=$5 AL rc
  AL=$(arm_alarm 1800)
  perl -e "alarm $AL; exec @ARGV" "$PY" $W/sc2_driver.py run --base "http://127.0.0.1:$PORT_E4B" --model "$MODEL" \
      --prompts $W/sc2/prompts_2048.json --mode poisson --rate "$RATE" --n "$N" --seed "$SEED" \
      --max-tokens-lo $SC2J_Q3_LO --max-tokens-hi $SC2J_Q3_HI --profile e4b --out $W/sc2/$TAG.json > logs/sc2_$TAG.log 2>&1
  rc=$?; line "SC2J $TAG rc=$rc $(grep -a '^SC2_RUN' logs/sc2_$TAG.log | tail -1 | cut -c1-320)"; return $rc; }

q_stats(){ curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/stats" -o "$2" || line "SC2J $1: no /stats"; }

# q_long_prompts MODEL REV -- the long ladder's prompt pool (rc 19)
q_long_prompts(){ say "SC2j long prompt pool (2,048 tokens)"
  perl -e "alarm 1200; exec @ARGV" "$PY" $W/sc2j_prompts.py --model "$1" --revision "$2" --out $W/sc2/prompts_2048.json \
    > logs/sc2j_prompts.log 2>&1 || { tail -3 logs/sc2j_prompts.log; return 19; }
  grep -a "^SC2J_PROMPTS" logs/sc2j_prompts.log | tee -a summary.txt; }

# q_arm DRAW SLOT -- one server (slot ca_a, sh or ca_b): start, warm, early engagement, the burst, serial, Q2, Q3
q_arm(){ local D=$1 S=$2 TAG="e4b_${2}_d$1" r q
  if can_run 1500 "$TAG" && q_server_start "$TAG" "$S" "$MID" "$REV" "$QA" "$SPEEDENV E4B_PAGED_FUSE_QKV=1" "int4_k19|"; then
    sampler_start sc2j_$TAG
    drive ${TAG}_warm $PORT_E4B "$MID" e4b serial 0 4 999
    if h_early "$TAG" on $W/sc2/${TAG}_warm.json && q_burst "$TAG" "$MID"; then
      drive ${TAG}_serial $PORT_E4B "$MID" e4b serial 0 $SC2J_SERIAL_N $D
      for r in $SC2J_Q2; do
        can_run $(( SC2J_Q2_N / r + 240 )) "$TAG rate $r" && drive ${TAG}_r$r $PORT_E4B "$MID" e4b poisson $r $SC2J_Q2_N $(( D * 100 + r ))
      done
      for r in $SC2J_Q3; do
        q=$(q_rtag "$r")
        can_run $(( $(awk -v n=$SC2J_Q3_N -v r="$r" 'BEGIN{printf "%d", n / r}') + 300 )) "$TAG q3 rate $r" || continue
        q_stats "$TAG" $W/sc2/stats_${TAG}_q3r${q}_before.json
        q_drive_long ${TAG}_q3r$q "$MID" "$r" $SC2J_Q3_N $(( D * 1000 + $(awk -v r="$r" 'BEGIN{printf "%d", 10 * r}') ))
        q_stats "$TAG" $W/sc2/stats_${TAG}_q3r${q}_after.json
      done
    fi
    h_health_end "$TAG"; q_stats "$TAG" $W/sc2/stats_${TAG}_end.json; sampler_stop sc2j_$TAG
  fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180; }

# q_contract MODEL REV ARENA LEV -- the contract arm: claim-all twice and shared at 32 slots, in process (rc 41 on VOID)
q_contract(){ local MODEL=$1 R=$2 ARENA=$3 LEV=$4 k P
  mkdir -p $W/sc2/contract
  for k in ca_a sh ca_b; do
    P=$(q_pool "$k")
    # shellcheck disable=SC2086  # assignment lists by design
    env -u E4B_PAGED_BUCKETS -u E4B_PAGED_KV_POOL PYTHONPATH= $ROUTEENV $LEV E4B_PAGED_MAX_SEQS=$SC2J_CONTRACT_SLOTS \
        E4B_PAGED_KV_POOL=$P E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA E4B_PAGED_CALIB=$W/calib.json \
        E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2J_TOKENS E4B_PAGED_GRAPHS=1 \
        perl -e "alarm $(arm_alarm 1500); exec @ARGV" "$PY" $W/sc2j_contract.py run --out $W/sc2/contract/contract_$k.json \
        > logs/sc2j_contract_$k.log 2>&1 || { tail -5 logs/sc2j_contract_$k.log; line "SC2J CONTRACT $k failed"; }
    grep -a '^SC2J_CONTRACT run' logs/sc2j_contract_$k.log | sed "s/^/$k /" | tee -a summary.txt
    gpu_free 120
  done
  "$PY" $W/sc2j_contract.py verdict --dir $W/sc2/contract --out $W/sc2/contract.json | tee -a summary.txt; }

# q_quality MODEL REV ARENA LEV WINDOWS CONT -- P117's teacher-forced width check at the launch stack (its own box and
# reducer, unchanged, in $W/sc2/quality), and P117's split plans at its table width for the reducer's comparison
q_quality(){ local MODEL=$1 R=$2 ARENA=$3 LEV=$4 WIN=$5 CONT=$6 QD=$W/sc2/quality
  mkdir -p $QD
  (cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' "$PY" -m pytest test_decode_graph_buckets.py -q -rs -p no:cacheprovider) \
      > logs/sc2j_premise.log 2>&1
  if [ $? = 0 ] && tail -1 logs/sc2j_premise.log | grep -q "passed" && ! tail -1 logs/sc2j_premise.log | grep -q skipped; then
    echo "premise ok" > $QD/summary.txt
  else echo "premise failed" > $QD/summary.txt; fi
  tail -1 logs/sc2j_premise.log | sed 's/^/SC2J_QUALITY premise: /' | tee -a summary.txt
  # shellcheck disable=SC2086  # assignment lists by design
  env -u E4B_PAGED_KV_POOL -u E4B_PAGED_BUCKETS PYTHONPATH= GNF4_SHA=$GNF4_SHA E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA \
      E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 \
      E4B_PAGED_PREFILL_GRAPH=0 $ROUTEENV $LEV \
      perl -e "alarm $(arm_alarm 2400); exec @ARGV" "$PY" $W/p117_box.py --out $QD/box.json --windows "$WIN" --cont "$CONT" \
      > logs/sc2j_quality.log 2>&1 || { tail -5 logs/sc2j_quality.log; line "SC2J QUALITY box failed"; }
  # P117's reducer and bar at box Q's two stack constants (SC2j Amendment 1): box Q's grouped-nf4-gemm, SC2j's windows
  "$PY" $W/sc2j_quality.py --dir $QD --out $QD/verdict.json --e4b-sha "$E4B_SHA" --gnf4-sha "$GNF4_SHA" 2>&1 | tail -3 \
      | sed 's/^/SC2J_QUALITY /' | tee -a summary.txt
  "$PY" - $QD/plans.json "$CONT" <<'PYP' | tee -a summary.txt
import json, math, sys
import torch, fp8_paged_attn
cols = math.ceil((512 + int(sys.argv[2]) + 16) / 16)          # p117_box's pool: max_tokens_per_seq = P + C + 16, P 512
sms = torch.cuda.get_device_properties(0).multi_processor_count
plans = {str(B): fp8_paged_attn.decode_split_plan(B, 4, cols, block_tokens=16, ktile=64, compute="fp8",
                                                  pack_heads=False, sms=sms) for B in (16, 32, 64)}
json.dump(plans, open(sys.argv[1], "w"))
print("SC2J_QUALITY plans " + json.dumps({"table_cols": cols, "sms": sms, "plans": plans}))
PYP
}

# ---- the real lane
box_q(){ local s
  l_tripwire; n_tripwire; q_tripwire
  phase 0 "fetch (bf16 for the bake), bake, both prompt pools"
  fetch_common || finish 11; bake_qwen3 || finish 12; sc2_prompts || finish 19; q_long_prompts "$MID" "$REV" || finish 19
  phase Q0 "the contract arm (32 slots, in process)"; q_contract "$MID" "$REV" "$QA" "$SPEEDENV E4B_PAGED_FUSE_QKV=1"
  phase Q0b "the QUALITY arm (P117 at the launch stack)"; q_quality "$MID" "$REV" "$QA" "$SPEEDENV E4B_PAGED_FUSE_QKV=1" 64 128
  quiesce arms
  phase Q1 "draw 1: $SC2J_ORDER (same seeds)"; for s in $SC2J_ORDER; do q_arm 1 "$s"; done
  phase Q2 "draw 2: $SC2J_ORDER (same seeds)"; for s in $SC2J_ORDER; do q_arm 2 "$s"; done
  phase RD "the reading"
  "$PY" $W/sc2j_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2j.json 2>&1 | tail -40 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1), on Granite: the self-tests, the contract arm at 32 slots, both pools' servers through
# POOL, SLOTS and the burst with a serial smoke and a 16-request Poisson smoke on each ladder's top rate (every request
# VALID), and the QUALITY arm on 8 windows. WIDTH is not read here: Granite fits 64 slots under both pools.
prove_q(){ local a s
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_reduce.py sc2_identity.py sc2j_reduce.py sc2j_contract.py sc2j_prompts.py p117_reduce.py sc2j_quality.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  l_tripwire; n_tripwire; q_tripwire
  for s in sc2_prompts.py sc2j_prompts.py; do
    perl -e "alarm 900; exec @ARGV" "$PY" $W/$s --model "$GR" --revision "$GR_REV" \
        --out $W/sc2/$( [ $s = sc2_prompts.py ] && echo prompts.json || echo prompts_2048.json ) > logs/$s.log 2>&1 \
      || { tail -3 logs/$s.log; say "PROVE: $s failed -- NOT PROVED"; rec 23; return; }
  done
  q_contract "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$GR_ENV"
  "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$W/sc2/contract.json'))['verdict'] == 'OK' else 1)" \
    || { say "PROVE: the contract arm is not OK on Granite -- NOT PROVED"; rec 23; }
  q_quality "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$GR_ENV" 8 32
  "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$W/sc2/quality/verdict.json'))['verdict'] not in ('VOID', 'NO_READING') else 1)" \
    || { say "PROVE: the QUALITY arm did not read on Granite -- NOT PROVED"; rec 23; }
  for a in ca sh; do
    if q_server_start "prove_$a" "$a" "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$GR_ENV" ""; then
      { drive prove_${a}_warm $PORT_E4B "$GR" e4b serial 0 4 999 && h_early "prove_$a" on $W/sc2/prove_${a}_warm.json \
        && q_burst "prove_$a" "$GR" && drive prove_${a}_serial $PORT_E4B "$GR" e4b serial 0 6 1 \
        && drive prove_${a}_r16 $PORT_E4B "$GR" e4b poisson 16 16 116 \
        && q_drive_long prove_${a}_q3r4 "$GR" 4 16 1040; } || { say "PROVE: $a smoke failed -- NOT PROVED"; rec 23; }
      h_health_end "prove_$a"; q_stats "prove_$a" $W/sc2/stats_prove_${a}_end.json
    else say "PROVE: $a server failed"; rec 23; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  done
  "$PY" $W/sc2_identity.py $W/sc2/prove_ca_serial.json $W/sc2/prove_sh_serial.json | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: ca and sh differ on the serial smoke -- NOT PROVED"; rec 23; }; }
