#!/usr/bin/env bash
# Tests for tp4_drive.sh's stall detector (e4b #624).
#
# The bug this guards: on 2026-09-19 `tp4-b-p46cut-C2` printed STALL? for 26
# minutes while downloading a checkpoint normally. Both of the old conditions
# -- GPU idle, summary unchanged -- are TRUE BY DEFINITION during a fetch, so
# the detector could not distinguish a fetch from a hang. The rows marked
# "observed" below are copied from that run's heartbeat log.
set -u
# shellcheck disable=SC1090
TP4_MIN_PROGRESS_MB=16
eval "$(sed -n '/^tp4_progress_verdict()/,/^}/p' "$(dirname "$0")/tp4_drive.sh")"
eval "$(sed -n '/^tp4_lane_dead()/,/^}/p' "$(dirname "$0")/tp4_drive.sh")"

pass=0; fail=0
check() { # name expected idle stall util dfk_now dfk_prev du_now du_prev
  local name=$1 want=$2; shift 2
  local got; got=$(tp4_progress_verdict "$@")
  if [ "$got" = "$want" ]; then pass=$((pass+1));
  else fail=$((fail+1)); echo "FAIL $name: want '$want' got '$got' (args: $*)"; fi
}

# --- the regression: a real fetch must never be called a stall ---
# observed 16:38:39Z -> 16:40:47Z: du flat at 8366M, free disk 308G -> 303G.
# du is blind here because the HF cache lives outside the working dir.
check "observed model fetch, du flat, disk falling" "fetching:-5154M on disk" \
      1568 900 0 318767104 324044800 8366 8366
# the old detector's exact inputs, minus disk: still a stall, correctly
check "genuinely hung: idle, flat du, flat disk" "stall:1568" \
      1568 900 0 318767104 318767104 8366 8366

# --- in-tree growth (dataset prep, adapters) still counts as progress ---
check "in-tree growth, disk unchanged" "fetching:+1136M in tree" \
      1568 900 0 318767104 318767104 5412 4276
# observed 16:25:55Z -> 16:26:58Z, early fetch: both move, disk dominates
check "both move, disk dominates" "fetching:-5154M on disk" \
      1568 900 0 318767104 324044800 1622 1220

# --- healthy lanes are silent ---
check "GPU busy, nothing moving" "" 1568 900 97 1 1 1 1
check "idle but under the stall threshold" "" 300 900 0 1 1 1 1

# --- boundary and robustness ---
check "idle exactly at threshold" "stall:900" 900 900 0 1 1 1 1
check "first heartbeat, no previous samples" "" 1568 900 97 318767104 "" 8366 ""
check "first heartbeat, idle and past threshold" "stall:1568" 1568 900 0 318767104 "" 8366 ""
check "disk FREED (cleanup) is not progress" "stall:1568" \
      1568 900 0 324044800 318767104 8366 8366
check "non-numeric probe output is not progress" "stall:1568" \
      1568 900 0 "n/a" "n/a" "n/a" "n/a"
check "missing util defaults to idle" "stall:1568" 1568 900 "" 1 1 1 1

# --- e4b#634: "any byte" is not a threshold -----------------------------------
# tp4-c-parity-1's remote process died at ~20:36. The controller polled the dead
# box for 53 more minutes because the heartbeat logged `fetching (-0M on disk)`
# every tick: a sub-megabyte write (a log line) counted as a fetch and silenced
# a stall alarm that had already, correctly, fired six times.
# 200 KB of movement, idle GPU, summary unchanged -- this is a STALL:
check "sub-MB disk noise is not progress" "stall:1568" \
      1568 900 0 318767104 318767304 8366 8366
# 1 MB in the tree is likewise noise, not work
check "sub-threshold tree growth is not progress" "stall:1568" \
      1568 900 0 318767104 318767104 8367 8366
# a real shard still reads as a fetch
check "a real fetch still reports progress" "fetching:-5154M on disk" \
      1568 900 0 318767104 324044800 8366 8366
# exactly at the 16 MB threshold counts (boundary, in tree)
check "at the threshold, in tree" "fetching:+16M in tree" \
      1568 900 0 318767104 318767104 8382 8366
# one below it does not
check "one MB below the threshold, in tree" "stall:1568" \
      1568 900 0 318767104 318767104 8381 8366

# --- e4b#634: a lane that DIED is not a lane that is slow ----------------------
# The remote process was killed without writing TP4_EXIT_CODE or TP_DONE, so
# every marker the controller polls for was absent forever -- indistinguishable
# from "still working" unless you ask the box whether the process is still there.
dead() { local want=$1 got; got=$(tp4_lane_dead "$2" "$3"); if [ "$got" = "$want" ]; then pass=$((pass+1));
  else fail=$((fail+1)); echo "FAIL lane_dead($2,$3): want '$want' got '$got'"; fi; }
dead "dead" 0 0          # two consecutive definite absences
dead "" 0 1              # it was alive last tick -- not yet
dead "" 1 0              # it is alive now
dead "" 1 1              # running
dead "" 0 ""             # first poll has no previous: never call a lane dead on one sample
dead "" "" ""            # probe returned nothing (ssh hiccup) -- not evidence of death
dead "" "" 0             # ditto, in the other order

# --- proof this suite is not inert: the OLD rule must FAIL the observed case ---
# Verbatim reconstruction of the pre-#624 condition, which consulted only the
# idle timer and GPU utilisation. If this ever stops reporting a stall on a
# known-good fetch, the suite has stopped testing anything.
tp4_progress_verdict_v1() {
  if [ "$1" -ge "$2" ] && [ "${3:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$1"; fi
}
v1=$(tp4_progress_verdict_v1 1568 900 0)
if [ "$v1" = "stall:1568" ]; then
  pass=$((pass+1))
else
  fail=$((fail+1)); echo "FAIL suite is inert: old rule did not misfire on the observed fetch (got '$v1')"
fi

# ...and the PRE-#634 progress rule must have been fooled by the -0M case, or
# this suite has stopped testing the bug it was written for.
tp4_progress_verdict_v2() {  # verbatim pre-#634: any nonzero movement is progress
  local consumed=0 grew=0
  [ "$4" -lt "$5" ] 2>/dev/null && consumed=$(( $5 - $4 ))
  [ "$6" -gt "$7" ] 2>/dev/null && grew=$(( $6 - $7 ))
  if [ "$consumed" -gt 0 ] || [ "$grew" -gt 0 ]; then echo "fetching"; return 0; fi
  if [ "$1" -ge "$2" ] && [ "${3:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$1"; fi
}
v2=$(tp4_progress_verdict_v2 1568 900 0 318767104 318767304 8366 8366)
if [ "$v2" = "fetching" ]; then pass=$((pass+1));
else fail=$((fail+1)); echo "FAIL suite is inert: the pre-#634 rule did not misread 200 KB as a fetch (got '$v2')"; fi

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
