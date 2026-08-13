#!/bin/bash
# Descend the cap until the arm stops completing. The lowest cap at which an arm
# still finishes all four steps is that arm's host-RAM requirement -- a measured
# number, not a pass/fail anecdote. Stops one rung past the first failure so the
# threshold is bracketed on both sides rather than merely exceeded.
#
#   usage: ./ladder.sh <host|arena> <cap> [cap ...]      caps in descending order
#   e.g.   ./ladder.sh host 8192m 6144m 5888m 5632m 5120m
set -uo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ARM=${1:?arm: host|arena}; shift
FAILS=0
for CAP in "$@"; do
  OUT=$("$HERE/runarm.sh" "$ARM" "$CAP" 2>&1 | tail -1)
  echo "$OUT"
  case "$OUT" in
    *COMPLETED*) FAILS=0 ;;
    *) FAILS=$((FAILS+1))
       [ $FAILS -ge 2 ] && { echo "LADDER-STOP $ARM (2 consecutive failures)"; break; } ;;
  esac
done
echo "LADDER-DONE $ARM"
