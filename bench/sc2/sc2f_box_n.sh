# shellcheck shell=bash
# bench/sc2/sc2f_box_n.sh -- lane SC2f (#846), box N: e4b serve_paged at 64 slots (SC2e's s64a server) with P130's two
# prefill defaults on against both off (bench/sc2/SC2f-PREREG.md). Sourced by sc1_run.sh when SC1_BOX=N, AFTER
# sc2_box_e.sh (drive, gpu_free, wait_e4b_ready, stop_pid, install_sc2_client, sc2_prompts), sc2c_box_h.sh (routes_ok,
# h_early, h_health_end) and sc2e_box_l.sh (l_tripwire, l_server_start with slots_ok, l_burst with burst_ok), and SC1's
# common setup.
#
# Eight servers, each SC2e's s64a server (SPEEDENV + E4B_PAGED_FUSE_QKV=1, 64 slots, buckets auto, bulk KV and the
# prefill graph at their defaults, both traces on), differing only in P130's two knobs:
#   on   E4B_FUSE_PREFILL_GLUE and E4B_PREFILL_LEAN_DISPATCH unset: the release's defaults (on qwen3_moe the decode folds
#        take prefill rows, and K19 above 256 rows takes the lean dispatch)
#   off  E4B_FUSE_PREFILL_GLUE=0 E4B_PREFILL_LEAN_DISPATCH=0
# Draw 1 runs on_a off_a off_b on_b, draw 2 off_a on_a on_b off_b. Every server of a draw runs the same seeds, so the two
# servers of an arm in a draw are a same-seed replicate: the noise the rule reads. Per server: 4 warm, SC2e's burst, the
# FLIPS check, serial 24, the ladder (4, 8, 12, 13, 14 and 16 req/s, 120 requests each), then the knee block (12, 13 and
# 14 req/s, four more seeds each).

SC2F_ORDER_D1="on_a off_a off_b on_b"; SC2F_ORDER_D2="off_a on_a on_b off_b"
SC2F_RATES="4 8 12 13 14 16"; SC2F_KNEE="12 13 14"; SC2F_KNEE_J="1 2 3 4"; SC2F_N=120; SC2F_SERIAL_N=24
SC2F_OFF="E4B_FUSE_PREFILL_GLUE=0 E4B_PREFILL_LEAN_DISPATCH=0"
unset E4B_FUSE_PREFILL_GLUE E4B_PREFILL_LEAN_DISPATCH   # the on arm reads the defaults; only the off arm names the knobs

# n_tripwire -- the installed e4b is the release under test: both P130 defaults (#1608, #1615) are in it. Refuse (rc 9).
n_tripwire(){ "$PY" -c "import os; [os.environ.pop(k, None) for k in ('E4B_FUSE_PREFILL_GLUE', 'E4B_PREFILL_LEAN_DISPATCH')]
import experts4bit_qlora as e, experts4bit_qlora.serve_paged as s
from experts4bit_qlora.engines import hot_residency as hr
assert hasattr(s, 'resolve_prefill_glue') and s.resolve_prefill_glue(s.FUSION_UNSET, 'qwen3_moe') == ('1', 'default-allowlisted'), 'e4b lacks the P1 default (#1615)'
assert hr._prefill_lean_dispatch_env() == 'auto', 'e4b lacks the P2 default (#1608)'
print('ok: both P130 defaults present, experts4bit-qlora', e.__version__)" \
    2>&1 | tail -1 | sed 's/^/SC2F_TRIPWIRE /' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SC2f: the installed e4b lacks P130's defaults"; finish 9; }; }

# n_flips_ok TAG ARM [proof] -- the server's start /health reads as its arm of the two knobs (sc2f_reduce.flips_bad:
# the resolved glue mode and its source, both raw env values, the lean dispatch setting, and what the startup captures
# took: K19 above 256 rows lean or gather, the prefill folds). The proof's Granite engages neither. STOP rc 43.
n_flips_ok(){ "$PY" $W/sc2f_reduce.py --flips $W/sc2/health_${1}_start.json "$2" ${3:+--proof} | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2F STOP: $1 not on its arm's P130 knobs"; return 43; }; }

# n_arm DRAW SLOT(on_a|off_a|on_b|off_b) -- one server: SC2e's s64a start (routes rc 47, slots rc 49), the FLIPS check
# (rc 43), warm, early engagement (SC2c's h_early, bulk on) and the burst (rc 48), serial, the ladder, the knee block
n_arm(){ local D=$1 ARM=${2%_*} REP=${2#*_} TAG LEV r j
  TAG="e4b_${ARM}_d${D}${REP}"; LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1"; [ "$ARM" = off ] && LEV="$LEV $SC2F_OFF"
  if can_run 1500 "$TAG" && l_server_start "$TAG" s64a "$MID" "$REV" "$QA" "$LEV" "int4_k19|" && n_flips_ok "$TAG" "$ARM"; then
    sampler_start sc2f_$TAG
    drive ${TAG}_warm $PORT_E4B "$MID" e4b serial 0 4 999
    if h_early "$TAG" on $W/sc2/${TAG}_warm.json && l_burst "$TAG" s64a "$MID"; then
      drive ${TAG}_serial $PORT_E4B "$MID" e4b serial 0 $SC2F_SERIAL_N $D
      for r in $SC2F_RATES; do
        can_run $(( SC2F_N / r + 240 )) "$TAG rate $r" && drive ${TAG}_r$r $PORT_E4B "$MID" e4b poisson $r $SC2F_N $(( D * 100 + r ))
      done
      for r in $SC2F_KNEE; do for j in $SC2F_KNEE_J; do
        can_run $(( SC2F_N / r + 240 )) "$TAG knee $r/$j" && drive ${TAG}_k${r}_$j $PORT_E4B "$MID" e4b poisson $r $SC2F_N $(( 10000 * D + 100 * r + j ))
      done; done
    fi
    h_health_end "$TAG"; sampler_stop sc2f_$TAG
  fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180; }

# ---- the real lane
box_n(){ local s
  l_tripwire; n_tripwire
  phase 0 "fetch (bf16 for the bake), bake, the SC2 prompt pool"
  fetch_common || finish 11; bake_qwen3 || finish 12; sc2_prompts || finish 19
  quiesce arms
  phase N1 "draw 1: on_a, off_a, off_b, on_b (same seeds)"; for s in $SC2F_ORDER_D1; do n_arm 1 "$s"; done
  phase N2 "draw 2: off_a, on_a, on_b, off_b (same seeds)"; for s in $SC2F_ORDER_D2; do n_arm 2 "$s"; done
  phase RD "the reading"
  "$PY" $W/sc2f_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2f.json 2>&1 | tail -40 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): both arms on the proof model at 64 slots, the knobs resolved as each arm sets them, the
# burst replaying bucket 64, a knee-rate smoke at a ladder seed and a knee seed, serial IDENTICAL (on Granite neither knob
# engages), engagement as the reducer reads it, and step traces with 64-row steps
prove_n(){ local a s LEV
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_reduce.py sc2_identity.py sc2c_census.py sc2e_census.py sc2e_reduce.py sc2f_reduce.py sc2f_basis.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  l_tripwire; n_tripwire
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$GR" --revision "$GR_REV" --out $W/sc2/prompts.json > logs/sc2_prompts.log 2>&1 \
    || { tail -3 logs/sc2_prompts.log; say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  for a in on off; do
    LEV="$GR_ENV"; [ "$a" = off ] && LEV="$LEV $SC2F_OFF"
    if l_server_start "prove_$a" s64a "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$LEV" "" && n_flips_ok "prove_$a" "$a" proof; then
      { drive prove_${a}_warm $PORT_E4B "$GR" e4b serial 0 4 999 && h_early "prove_$a" on $W/sc2/prove_${a}_warm.json \
        && l_burst "prove_$a" s64a "$GR" && drive prove_${a}_serial $PORT_E4B "$GR" e4b serial 0 6 1 \
        && drive prove_${a}_r13 $PORT_E4B "$GR" e4b poisson 13 16 113 \
        && drive prove_${a}_k13_1 $PORT_E4B "$GR" e4b poisson 13 16 11301; } || { say "PROVE: $a smoke failed -- NOT PROVED"; rec 23; }
      h_health_end "prove_$a"
    else say "PROVE: $a server failed"; rec 23; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  done
  "$PY" $W/sc2_identity.py $W/sc2/prove_on_serial.json $W/sc2/prove_off_serial.json | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: on and off differ on Granite, where neither knob engages -- NOT PROVED"; rec 23; }
  "$PY" $W/sc2f_reduce.py --engagement $W/sc2/health_prove_on_end.json $W/sc2/health_prove_off_end.json --arms on off | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: a server did not engage as registered -- NOT PROVED"; rec 23; }
  for a in on off; do
    "$PY" $W/sc2e_census.py steps $W/sc2/steps_prove_$a.jsonl --arm s64a --require-wide | tee -a summary.txt
    [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: the $a step trace does not decompose with its 64-row steps -- NOT PROVED"; rec 23; }
  done; }
