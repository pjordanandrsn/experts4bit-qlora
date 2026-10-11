# shellcheck shell=bash
# bench/sc2/sc2h_box_o.sh -- lane SC2h (#846), box O: e4b serve_paged at 64 slots on the release's defaults (SC2f's `on`
# server) with P118's decode lookahead against without it (bench/sc2/SC2h-PREREG.md). Sourced by sc1_run.sh when
# SC1_BOX=O, AFTER sc2_box_e.sh (drive, gpu_free, wait_e4b_ready, stop_pid, install_sc2_client, sc2_prompts),
# sc2c_box_h.sh (routes_ok, h_early, h_health_end), sc2e_box_l.sh (l_tripwire, l_server_start with slots_ok, l_burst)
# and sc2f_box_n.sh (n_tripwire: P130's two defaults in the installed e4b), and SC1's common setup.
#
# Eight servers, each SC2e's s64a server at the release's defaults (SPEEDENV + E4B_PAGED_FUSE_QKV=1, 64 slots, buckets
# auto, bulk KV, the prefill graph and P130's two prefill defaults, both traces on), differing only in the lookahead:
#   la    E4B_PAGED_DECODE_LOOKAHEAD=1 (P118: each decode step issued before the previous one is read back)
#   base  unset (the release's default: off)
# Draw 1 runs la_a base_a base_b la_b, draw 2 base_a la_a la_b base_b; every server of a draw runs the same seeds. Per
# server: 4 warm, SC2e's burst, the LOOKAHEAD check, serial 24, the ladder (SC2f's 4, 8, 12, 13, 14, 16 and the knee
# rates), then the knee block (the knee rates, four more seeds each). The knee rates come from SC2f's reading by the
# registered rule (sc2h_reduce.knee_rates), resolved here from the staged SC2f verdict, never typed in.

SC2H_ORDER_D1="la_a base_a base_b la_b"; SC2H_ORDER_D2="base_a la_a la_b base_b"
SC2H_LADDER0="4 8 12 13 14 16"; SC2H_KNEE_J="1 2 3 4"; SC2H_N=120; SC2H_SERIAL_N=24
SC2H_LA="E4B_PAGED_DECODE_LOOKAHEAD=1"
unset E4B_PAGED_DECODE_LOOKAHEAD   # the base arm reads the default; only the la arm names the knob

# o_tripwire -- the installed e4b carries P118's lookahead (PagedServeConfig.decode_lookahead, the scheduler's issue /
# collect path) and reads it off by default. Refuse (rc 9).
o_tripwire(){ "$PY" -c "import os; os.environ.pop('E4B_PAGED_DECODE_LOOKAHEAD', None)
import inspect, experts4bit_qlora as e, experts4bit_qlora.serve_paged as s
from experts4bit_qlora.engines import paged_runner as pr, scheduler as sc
assert 'decode_lookahead' in s.PagedServeConfig.__dataclass_fields__, 'e4b lacks E4B_PAGED_DECODE_LOOKAHEAD (#1339)'
assert s.PagedServeConfig.__dataclass_fields__['decode_lookahead'].default is False, 'the lookahead is not off by default'
assert hasattr(pr.PagedModelRunner, 'issue_decode') and hasattr(pr.PagedModelRunner, 'collect_decode'), 'no issue/collect'
assert 'lookahead' in inspect.signature(sc.ContinuousScheduler).parameters, 'the scheduler takes no lookahead'
print('ok: P118 lookahead present and off by default, experts4bit-qlora', e.__version__)" \
    2>&1 | tail -1 | sed 's/^/SC2H_TRIPWIRE /' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SC2h: the installed e4b lacks P118's lookahead"; finish 9; }; }

# o_knee -- the knee rates by the registered rule, on the SC2f verdict sc1_drive.sh staged as sc2f_verdict.json:
# writes sc2/knee_rates.json (with the verdict's sha256) and a copy of the verdict beside it; sets SC2H_KNEE and
# SC2H_LADDER. SC2f VOID or UNREAD, or a row not VALID: SC2h does not launch (rc 45).
o_knee(){ "$PY" $W/sc2h_reduce.py --knee-rule $W/sc2f_verdict.json --write $W/sc2 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "SC2h: the knee rule refuses SC2f's verdict -- SC2h does not launch"; finish 45; }
  SC2H_KNEE=$("$PY" -c "import json; print(' '.join(str(r) for r in json.load(open('$W/sc2/knee_rates.json'))['rates']))")
  SC2H_LADDER=$(for r in $SC2H_LADDER0 $SC2H_KNEE; do echo "$r"; done | sort -n | uniq | tr '\n' ' ')
  line "SC2H knee rates [$SC2H_KNEE], ladder [$SC2H_LADDER]"; }

# o_la_ok TAG ARM [proof] -- the server's start /health reads as its arm (sc2h_reduce.lookahead_bad: engine.decode_lookahead
# and SC2f's on defaults; the proof's Granite engages neither P130 knob). STOP rc 44.
o_la_ok(){ "$PY" $W/sc2h_reduce.py --lookahead $W/sc2/health_${1}_start.json "$2" ${3:+--proof} | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2H STOP: $1 not on its arm"; return 44; }; }

# o_arm DRAW SLOT(la_a|base_a|la_b|base_b) -- one server: SC2e's s64a start (routes rc 47, slots rc 49), the LOOKAHEAD
# check (rc 44), warm, early engagement (SC2c's h_early, bulk on) and the burst (rc 48), serial, the ladder, the knee block
o_arm(){ local D=$1 ARM=${2%_*} REP=${2#*_} TAG LEV r j
  TAG="e4b_${ARM}_d${D}${REP}"; LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1"; [ "$ARM" = la ] && LEV="$LEV $SC2H_LA"
  if can_run 1500 "$TAG" && l_server_start "$TAG" s64a "$MID" "$REV" "$QA" "$LEV" "int4_k19|" && o_la_ok "$TAG" "$ARM"; then
    sampler_start sc2h_$TAG
    drive ${TAG}_warm $PORT_E4B "$MID" e4b serial 0 4 999
    if h_early "$TAG" on $W/sc2/${TAG}_warm.json && l_burst "$TAG" s64a "$MID"; then
      drive ${TAG}_serial $PORT_E4B "$MID" e4b serial 0 $SC2H_SERIAL_N $D
      for r in $SC2H_LADDER; do
        can_run $(( SC2H_N / r + 240 )) "$TAG rate $r" && drive ${TAG}_r$r $PORT_E4B "$MID" e4b poisson $r $SC2H_N $(( D * 100 + r ))
      done
      for r in $SC2H_KNEE; do for j in $SC2H_KNEE_J; do
        can_run $(( SC2H_N / r + 240 )) "$TAG knee $r/$j" && drive ${TAG}_k${r}_$j $PORT_E4B "$MID" e4b poisson $r $SC2H_N $(( 10000 * D + 100 * r + j ))
      done; done
    fi
    h_health_end "$TAG"; sampler_stop sc2h_$TAG
  fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180; }

# ---- the real lane
box_o(){ local s
  l_tripwire; n_tripwire; o_tripwire; o_knee
  phase 0 "fetch (bf16 for the bake), bake, the SC2 prompt pool"
  fetch_common || finish 11; bake_qwen3 || finish 12; sc2_prompts || finish 19
  quiesce arms
  phase O1 "draw 1: la_a, base_a, base_b, la_b (same seeds)"; for s in $SC2H_ORDER_D1; do o_arm 1 "$s"; done
  phase O2 "draw 2: base_a, la_a, la_b, base_b (same seeds)"; for s in $SC2H_ORDER_D2; do o_arm 2 "$s"; done
  phase RD "the reading"
  "$PY" $W/sc2h_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2h.json 2>&1 | tail -40 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): both arms on the proof model at 64 slots, the lookahead engaged on la (Granite's decode
# graphs run it) and absent on base, the burst replaying bucket 64, a knee-rate smoke at a ladder seed and a knee seed,
# serial IDENTICAL la against base (the lookahead keeps the synchronous path's tokens), engagement as the reducer reads
# it, and step traces with 64-row steps and the lookahead notes where they belong
prove_o(){ local a s LEV
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_reduce.py sc2_identity.py sc2c_census.py sc2e_census.py sc2e_reduce.py sc2f_reduce.py sc2h_reduce.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  l_tripwire; n_tripwire; o_tripwire
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$GR" --revision "$GR_REV" --out $W/sc2/prompts.json > logs/sc2_prompts.log 2>&1 \
    || { tail -3 logs/sc2_prompts.log; say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  for a in la base; do
    LEV="$GR_ENV"; [ "$a" = la ] && LEV="$LEV $SC2H_LA"
    if l_server_start "prove_$a" s64a "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$LEV" "" && o_la_ok "prove_$a" "$a" proof; then
      { drive prove_${a}_warm $PORT_E4B "$GR" e4b serial 0 4 999 && h_early "prove_$a" on $W/sc2/prove_${a}_warm.json \
        && l_burst "prove_$a" s64a "$GR" && drive prove_${a}_serial $PORT_E4B "$GR" e4b serial 0 6 1 \
        && drive prove_${a}_r13 $PORT_E4B "$GR" e4b poisson 13 16 113 \
        && drive prove_${a}_k13_1 $PORT_E4B "$GR" e4b poisson 13 16 11301; } || { say "PROVE: $a smoke failed -- NOT PROVED"; rec 23; }
      h_health_end "prove_$a"
    else say "PROVE: $a server failed"; rec 23; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  done
  "$PY" $W/sc2_identity.py $W/sc2/prove_la_serial.json $W/sc2/prove_base_serial.json | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: la and base differ on Granite: the lookahead changed tokens -- NOT PROVED"; rec 23; }
  "$PY" $W/sc2h_reduce.py --engagement $W/sc2/health_prove_la_end.json $W/sc2/health_prove_base_end.json --arms la base | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: a server did not engage as registered -- NOT PROVED"; rec 23; }
  for a in la base; do
    "$PY" $W/sc2e_census.py steps $W/sc2/steps_prove_$a.jsonl --arm s64a --require-wide | tee -a summary.txt
    [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: the $a step trace does not decompose with its 64-row steps -- NOT PROVED"; rec 23; }
    "$PY" $W/sc2h_reduce.py --steps $W/sc2/steps_prove_$a.jsonl $a | tee -a summary.txt
    [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: the $a step trace's lookahead notes are not its arm's -- NOT PROVED"; rec 23; }
  done; }
