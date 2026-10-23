# shellcheck shell=bash
# bench/sc2/sc2i_box_p.sh -- lane SC2i (#846), box P: e4b serve_paged at 64 slots on the release's defaults (SC2h's
# `base` server, SC2f's `on`) against e4b#2017's two prefill interleave knobs (bench/sc2/SC2i-PREREG.md). Sourced by
# sc1_run.sh when SC1_BOX=P, AFTER sc2_box_e.sh (drive, gpu_free, wait_e4b_ready, stop_pid, install_sc2_client,
# sc2_prompts), sc2c_box_h.sh (routes_ok, h_early, h_health_end), sc2e_box_l.sh (l_tripwire, l_server_start with
# slots_ok, l_burst) and sc2f_box_n.sh (n_tripwire: P130's two defaults in the installed e4b), and SC1's common setup.
#
# Twelve servers, each SC2e's s64a server at the release's defaults (SPEEDENV + E4B_PAGED_FUSE_QKV=1, 64 slots, buckets
# auto, bulk KV, the prefill graph, P130's two prefill defaults, the lookahead off, both traces on), differing only in
# the two knobs:
#   base   unset (prefill_every 1, loaded_chunk_tokens 0: the earlier plans)
#   pe2    E4B_PAGED_PREFILL_EVERY=2          pe4    E4B_PAGED_PREFILL_EVERY=4
#   lc256  E4B_PAGED_LOADED_CHUNK_TOKENS=256  lc128  E4B_PAGED_LOADED_CHUNK_TOKENS=128
# Draw 1 runs base_a pe2 pe4 lc256 lc128 base_b, draw 2 base_a lc128 lc256 pe4 pe2 base_b; every server of a draw runs
# the same seeds. Per server: 4 warm, SC2e's burst, the KNOBS check, serial 24, the ladder 4, 8, 12 req/s (120 each),
# then the end /health and /stats.

SC2I_ORDER_D1="base_a pe2 pe4 lc256 lc128 base_b"; SC2I_ORDER_D2="base_a lc128 lc256 pe4 pe2 base_b"
SC2I_LADDER="4 8 12"; SC2I_N=120; SC2I_SERIAL_N=24
unset E4B_PAGED_PREFILL_EVERY E4B_PAGED_LOADED_CHUNK_TOKENS E4B_PAGED_DECODE_LOOKAHEAD   # only a knob arm names a knob

# p_env ARM -- the knob assignment an arm adds to the server's environment (empty for base)
p_env(){ case "$1" in
  base) echo "" ;; pe2) echo "E4B_PAGED_PREFILL_EVERY=2" ;; pe4) echo "E4B_PAGED_PREFILL_EVERY=4" ;;
  lc256) echo "E4B_PAGED_LOADED_CHUNK_TOKENS=256" ;; lc128) echo "E4B_PAGED_LOADED_CHUNK_TOKENS=128" ;;
  *) return 1 ;; esac; }

# p_tripwire -- the installed e4b carries e4b#2017's knobs, both at the earlier plans by default. Refuse (rc 9).
p_tripwire(){ "$PY" -c "import os
for k in ('E4B_PAGED_PREFILL_EVERY', 'E4B_PAGED_LOADED_CHUNK_TOKENS'): os.environ.pop(k, None)
import inspect, experts4bit_qlora as e, experts4bit_qlora.serve_paged as s
from experts4bit_qlora.engines import scheduler as sc
f = s.PagedServeConfig.__dataclass_fields__
assert 'prefill_every' in f and 'loaded_chunk_tokens' in f, 'e4b lacks e4b#2017 (E4B_PAGED_PREFILL_EVERY / E4B_PAGED_LOADED_CHUNK_TOKENS)'
assert f['prefill_every'].default == 1 and f['loaded_chunk_tokens'].default == 0, 'the knobs are not at the earlier plans by default'
p = inspect.signature(sc.ContinuousScheduler).parameters
assert 'prefill_every' in p and 'loaded_chunk_tokens' in p, 'the scheduler takes neither knob'
print('ok: e4b#2017 knobs present at their defaults, experts4bit-qlora', e.__version__)" \
    2>&1 | tail -1 | sed 's/^/SC2I_TRIPWIRE /' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SC2i: the installed e4b lacks e4b#2017's knobs"; finish 9; }; }

# p_knobs_ok TAG ARM [proof] -- the server's start /health reads as its arm (sc2i_reduce.knobs_bad: the two knobs, SC2f's
# on defaults, the lookahead off; the proof's Granite engages neither P130 knob). STOP rc 44.
p_knobs_ok(){ "$PY" $W/sc2i_reduce.py --knobs $W/sc2/health_${1}_start.json "$2" ${3:+--proof} | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2I STOP: $1 not on its arm"; return 44; }; }

# p_stats_end TAG -- the end /stats (the scheduler's prefill_deferred), beside h_health_end's /health
p_stats_end(){ curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/stats" -o $W/sc2/stats_${1}_end.json || line "SC2I $1: no end /stats"; }

# p_arm DRAW SLOT -- one server: SC2e's s64a start (routes rc 47, slots rc 49), the KNOBS check (rc 44), warm, early
# engagement (SC2c's h_early, bulk on) and the burst (rc 48), serial, the ladder, then the end /health and /stats
p_arm(){ local D=$1 S=$2 ARM TAG LEV r
  ARM=${S%_[ab]}; TAG="e4b_${S}_d${D}"; LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1 $(p_env "$ARM")"
  if can_run 1200 "$TAG" && l_server_start "$TAG" s64a "$MID" "$REV" "$QA" "$LEV" "int4_k19|" && p_knobs_ok "$TAG" "$ARM"; then
    sampler_start sc2i_$TAG
    drive ${TAG}_warm $PORT_E4B "$MID" e4b serial 0 4 999
    if h_early "$TAG" on $W/sc2/${TAG}_warm.json && l_burst "$TAG" s64a "$MID"; then
      drive ${TAG}_serial $PORT_E4B "$MID" e4b serial 0 $SC2I_SERIAL_N $D
      for r in $SC2I_LADDER; do
        can_run $(( SC2I_N / r + 240 )) "$TAG rate $r" && drive ${TAG}_r$r $PORT_E4B "$MID" e4b poisson $r $SC2I_N $(( D * 100 + r ))
      done
    fi
    h_health_end "$TAG"; p_stats_end "$TAG"; sampler_stop sc2i_$TAG
  fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180; }

# ---- the real lane
box_p(){ local s
  l_tripwire; n_tripwire; p_tripwire
  phase 0 "fetch (bf16 for the bake), bake, the SC2 prompt pool"
  fetch_common || finish 11; bake_qwen3 || finish 12; sc2_prompts || finish 19
  quiesce arms
  phase P1 "draw 1: $SC2I_ORDER_D1 (same seeds)"; for s in $SC2I_ORDER_D1; do p_arm 1 "$s"; done
  phase P2 "draw 2: $SC2I_ORDER_D2 (same seeds)"; for s in $SC2I_ORDER_D2; do p_arm 2 "$s"; done
  phase RD "the reading"
  "$PY" $W/sc2i_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2i.json 2>&1 | tail -40 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): base, pe2 and lc128 on the proof model at 64 slots, each on its arm by its start /health,
# the burst replaying bucket 64, a serial smoke and a Poisson smoke at 12 req/s with every request VALID, engagement as
# the reducer reads it (lc128's end /health is the real record the adapted ENGAGED gate is checked on), and step traces
# with 64-row steps
prove_p(){ local a s LEV
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_reduce.py sc2_identity.py sc2c_census.py sc2e_census.py sc2e_reduce.py sc2f_reduce.py sc2i_reduce.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  l_tripwire; n_tripwire; p_tripwire
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$GR" --revision "$GR_REV" --out $W/sc2/prompts.json > logs/sc2_prompts.log 2>&1 \
    || { tail -3 logs/sc2_prompts.log; say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  for a in base pe2 lc128; do
    LEV="$GR_ENV $(p_env "$a")"
    if l_server_start "prove_$a" s64a "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$LEV" "" && p_knobs_ok "prove_$a" "$a" proof; then
      { drive prove_${a}_warm $PORT_E4B "$GR" e4b serial 0 4 999 && h_early "prove_$a" on $W/sc2/prove_${a}_warm.json \
        && l_burst "prove_$a" s64a "$GR" && drive prove_${a}_serial $PORT_E4B "$GR" e4b serial 0 6 1 \
        && drive prove_${a}_r12 $PORT_E4B "$GR" e4b poisson 12 16 112; } || { say "PROVE: $a smoke failed -- NOT PROVED"; rec 23; }
      h_health_end "prove_$a"; p_stats_end "prove_$a"
    else say "PROVE: $a server failed"; rec 23; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  done
  "$PY" $W/sc2i_reduce.py --engagement $W/sc2/health_prove_base_end.json $W/sc2/health_prove_pe2_end.json \
    $W/sc2/health_prove_lc128_end.json --arms base pe2 lc128 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: a server did not engage as registered -- NOT PROVED"; rec 23; }
  for a in base pe2 lc128; do
    "$PY" $W/sc2e_census.py steps $W/sc2/steps_prove_$a.jsonl --arm s64a | tee -a summary.txt     # the census, reported
    "$PY" $W/sc2i_reduce.py --census $W/sc2/steps_prove_$a.jsonl $a | tee -a summary.txt          # SC2e's wide check; lc: the 64-row step
    [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: the $a step trace does not decompose with its 64-row steps -- NOT PROVED"; rec 23; }
  done; }
