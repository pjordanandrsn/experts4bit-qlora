# PGX THE READING, box side (bench/pgx/PREREG-pgx.md, Amendment 3). SOURCED by pgx_run.sh on a read box (PGX_BOX=read),
# after the shared setup: the refusals and the guard fit (this box's pgx_schedule.py --need), STARTED, the installs and
# tripwires, the self-tests, the fetch, the bytes and K, the bake, calibrate and the placement preview, the clock burn. It
# uses pgx_run.sh's helpers (say, finish, can_run, step_alarm, img_guard, the server starts, drive, probe, the sampler,
# stop_pid) and its variables, and never runs alone. In order:
#   1. the tier counters' own cost on THIS image python (pgx_dram_bench.py): over 0.1 % of the shakedown's E-hyb decode
#      step, or a wrap that is not in place, is a premise failure (25), before any lifetime;
#   2. the measured K at every budget (box_bytes.json), beside the registered table; the lifetime records carry both;
#   3. registration.json for the part reduction (the fit profile's sha, the budgets, the prompt sets, the box's preview);
#   4. every lifetime of this box in pgx_schedule.py's order, each started only if its modelled need fits what is left
#      (a skipped lifetime leaves its cells to VOID, never a guess);
#   5. the box's own reduction (pgx_reduce.py --mode part): PART_OK -> 0, anything else -> 27.
# shellcheck shell=bash
# shellcheck disable=SC2154  # every variable here is pgx_run.sh's
# ---- 1. the tier counters' cost, on the interpreter the E-hyb server runs
img_guard dram-bench
PYTHONPATH=$W/hook_e4b PGX_DRAM_TIME_OUT=$W/logs/dram_bench_counters.jsonl PGX_DRAM_TIME_PERIOD_S=3600 \
  "$PY_IMG" $W/pgx_dram_bench.py > logs/dram_bench.json 2> logs/dram_bench.err \
  || { cat logs/dram_bench.json logs/dram_bench.err | tail -5; say "TIER COUNTER PREMISE FAILED (wrap, split or the 0.1 % bound)"; finish 25; }
echo "DRAM_BENCH $(cat logs/dram_bench.json)" | tee -a summary.txt
# ---- 2. the measured K at every budget, from the box's own checkpoint headers
KM_ALL=$(W=$W "$PY_IMG" - <<'PYK'
import json, os
k = json.load(open(os.path.join(os.environ["W"], "box_bytes.json")))["k_table"]
print(" ".join(str(k[a][b]) for a in ("S4", "S8", "V8") for b in ("4", "8", "12")))
PYK
) || { say "K TABLE UNREAD from box_bytes.json"; finish 9; }
read -r KM_S4_4 KM_S4_8 KM_S4_12 KM_S8_4 KM_S8_8 KM_S8_12 KM_V8_4 KM_V8_8 KM_V8_12 <<< "$KM_ALL"
echo "K_MEASURED_ALL S4=$KM_S4_4/$KM_S4_8/$KM_S4_12 S8=$KM_S8_4/$KM_S8_8/$KM_S8_12 V8=$KM_V8_4/$KM_V8_8/$KM_V8_12 (registered $K_S4_4/$K_S4_8/$K_S4_12, $K_S8_4/$K_S8_8/$K_S8_12, $K_V8_4/$K_V8_8/$K_V8_12)" | tee -a summary.txt
# ---- 3. the part reduction's registration, from the staged (sha-pinned) files and this box's preview
W=$W "$PY_IMG" - <<'PYG' > $W/registration.json || { say "REGISTRATION RECORD FAILED"; finish 9; }
import hashlib, json, os
w = os.environ["W"]
man = json.load(open(f"{w}/prompts/manifest.json"))
pv = json.load(open(f"{w}/place_preview.json"))
print(json.dumps({"profile_sha": hashlib.sha256(open(f"{w}/fit_profile.jsonl", "rb").read()).hexdigest(),
                  "budget_bytes": {str(b): b * 10**9 for b in (4, 8, 12)},
                  "sets": {k: v["prompts_sha"] for k, v in man["sets"].items()},
                  "preview": {k: v["vram_mass_share"] for k, v in pv.items() if k.startswith("E-hyb")}}, indent=1))
PYG
# ---- 4. this box's lifetimes
"$PY_IMG" $W/pgx_schedule.py --box "$BOX_SPEC" > $W/box_schedule.txt || { say "SCHEDULE FAILED for $BOX_SPEC"; finish 78; }
DRAWS=$(awk '{print $2}' $W/box_schedule.txt | sort -un | tr '\n' ' ')
echo "BOX $BOX_SPEC draws $DRAWS: $(wc -l < $W/box_schedule.txt | tr -d ' ') lifetimes" | tee -a summary.txt
pass_set(){ case "$1" in P1|P2) echo "warm_c1 1";; P3) echo "warm_c4 4";; P4) echo "shift 1";; *) return 1;; esac; }
read_lifetime(){ local kind=$1 arm=$2 b=$3 d=$4 passes=$5 cap=$6 lt log t0 rc port smp failed="" probes="" p set conc ps k="" km="" gib m rev xtra
  if [ "$kind" = instr ]; then lt=$W/instr/$arm/$b/d$d; log=$W/logs/server_${arm}_${b}_d${d}_instr.log
  else lt=$W/read/$arm/$b/d$d; log=$W/logs/server_${arm}_${b}_d${d}.log; fi
  mkdir -p "$lt"; : > "$lt/windows.txt"; t0=$(date +%s)
  say "$kind lifetime $arm ($b GB, draw $d): $passes"
  case "$arm" in S4|S8|V8) eval "k=\$K_${arm}_$b; km=\$KM_${arm}_$b";; esac
  case "$arm" in
    S4|S8)
      port=$PORT_SGL
      if [ "$arm" = S4 ]; then m=$INT4; rev=$INT4_REV; xtra="--dtype float16"; else m=$FP8; rev=$FP8_REV; xtra=""; fi
      if [ "$kind" = instr ]; then
        echo W0 > "$lt/PASS"          # the server's own warmup is tagged W0: replayed for the cache state, never scored
        SGL_HOOK_PATH=$W/hook SGL_DUMP_ENV="PGX_ROUTE_DUMP=$lt" sgl_start "$m" "$rev" "$km" "$log" \
          "--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled $xtra"
      else
        sgl_start "$m" "$rev" "$km" "$log" "$xtra"
      fi;;
    V8) port=$PORT_VLLM; vllm_start "$km" "$log";;
    E-*) port=$PORT_E4B; gib=$EHYB_GIB_8; [ "$b" != all ] && eval "gib=\$EHYB_GIB_$b"; e4b_start "$log" "$lt" "$arm" "$gib";;
  esac
  rc=$?
  if [ "$rc" != 0 ]; then failed="rc $rc: $(grep -aiE 'error|exception|refus' "$log" | tail -1 | cut -c1-200)"; say "$arm ($b GB, d$d, $kind) did not start ($failed)"
  else
    smp=$(sampler_start "$lt/samples.txt" "$SRV_PID")
    [ "$arm" = V8 ] && probes="$(probe $port "$log")"
    for p in $passes; do
      read -r set conc <<< "$(pass_set "$p")"
      [ "$kind" = instr ] && echo "$p" > "$lt/PASS"
      ps=$(date +%s)
      drive $port "$set" "$p" "$lt/$p.json" "${arm}_${b}_d${d}_${kind}" "$conc" "$cap"
      rc=$?
      echo "$p $ps $(date +%s)" >> "$lt/windows.txt"
      echo "$arm $b d$d $kind $p rc=$rc $("$PY_IMG" -c "import json; r=json.load(open('$lt/$p.json')); print('faults', r['faults'], 'tok/s', round(r['out_tok_s'] or 0, 1))" 2>/dev/null)" | tee -a summary.txt
      [ "$arm" = V8 ] && probes="$probes $(probe $port "$log")"
    done
    case "$arm" in E-*) curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o "$lt/health.json" 2>/dev/null;; esac
    sampler_stop "$smp"
  fi
  stop_pid "$SRV_PID"; SRV_PID=""
  [ "$kind" = instr ] && return 0               # the instrument lifetime's record is its dump; its timed lifetime reads it
  local wall=$(( $(date +%s) - t0 ))
  # shellcheck disable=SC2206  # passes is a space-separated pass list by design
  local args=(lifetime --arm "$arm" --lt "$lt" --wall-s "$wall" --samples "$lt/samples.txt" --budget "$b" --draw "$d"
              --passes $passes --windows "$lt/windows.txt")
  [ -n "$failed" ] && args+=(--startup-failed "$failed")
  case "$arm" in
    S4|S8) args+=(--k "$k" --k-measured "$km" --route-dump "$W/instr/$arm/$b/d$d" --instr-windows "$W/instr/$arm/$b/d$d/windows.txt"
                  --instr-log "$W/logs/server_${arm}_${b}_d${d}_instr.log" --timed-log "$log");;
    # one probe before the first pass and one after each; a failed probe leaves fewer offsets: counters_ok false, never a guess
    # shellcheck disable=SC2206  # probes is a list of byte offsets by design
    V8) args+=(--k "$k" --k-measured "$km" --pager-log "$log" --probes $probes);;
    E-*) args+=(--health "$lt/health.json" --placement "$lt/placement.json" --step-trace "$lt/step_trace.jsonl"
                --dram-time "$lt/dram_time.jsonl");;
  esac
  "$PY_IMG" $W/rt/pgx/pgx_box.py "${args[@]}" 2>&1 | tee -a summary.txt; }
mapfile -t ROWS < $W/box_schedule.txt
for row in "${ROWS[@]}"; do
  read -r _idx draw kind arm budget plist <<< "$row"
  need=$("$PY_IMG" $W/pgx_schedule.py --lifetime-need "$kind" "$arm" "$budget") || { say "no modelled need for $row"; finish 78; }
  can_run "$need" "${kind}_${arm}_${budget}_d${draw}" || continue
  read_lifetime "$kind" "$arm" "$budget" "$draw" "${plist//,/ }" "$need"
done
# ---- 5. the box's own reduction: its draws' faults only (P238's part form); the reading is the controller's merge
say "part reduction ($BOX_SPEC, draws $DRAWS)"
# shellcheck disable=SC2086  # DRAWS is a list of draw numbers by design
(cd $W/rt/pgx && "$PY_IMG" pgx_reduce.py --work $W/read --registration $W/registration.json --mode part --draws $DRAWS \
  --out $W/part_verdict.json) 2>&1 | tee -a summary.txt
case "${PIPESTATUS[0]}" in 0) finish 0;; *) finish 27;; esac
