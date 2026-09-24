#!/bin/bash
# bench/p67/p67_drive.sh -- lane P67, CONTROLLER side: the launcher's --command (bench/p67/P67-PREREG.md).
# NOT a second driver. It checks the staged pins against this tree, fixes the registered knobs, and execs
# bench/tp4/tp4_drive.sh -- tp4's stage / nonce / heartbeat / fetch, unchanged -- with the box told to start
# bench/p67/p67_run.sh (the registration's guard), which re-checks the pins and knobs ON the box and then execs
# tp4_run.sh. Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p67_drive] $*"; }
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# staged.sha256 names files as the BOX sees them (flat in /root/tp4); resolve each to its source here.
src_of(){ case "$1" in
  p67_run.sh|registered.knobs) echo "$HERE/$1";;
  tp4_run.sh|tp4_arm.py|tp4_reduce.py|tp4_alpaca.py) echo "$REPO/bench/tp4/$1";;
  n9_datasets.py) echo "$REPO/bench/flagship-matrix/drivers/$1";;
  ds_manifest.json) echo "$REPO/bench/flagship-matrix/$1";;
  *) echo "";; esac; }
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
n=0
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  src=$(src_of "$name"); [ -n "$src" ] || { say "refusing: staged.sha256 names $name, which this driver does not stage"; exit 78; }
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want -- re-pin, or the box refuses after it is paid for"; exit 78; }
  n=$((n + 1))
done < "$HERE/staged.sha256"
say "pins OK ($n files)"
# The registered knobs: registered.knobs, the SAME file p67_run.sh enforces on the box (staged and pinned above).
# A value already in the environment that differs is refused here too, before anything is staged.
while IFS='|' read -r k v; do
  case "$k" in \#*|"") continue;; esac
  if [ -n "${!k:-}" ] && [ "${!k}" != "$v" ]; then say "refusing: $k='${!k}' differs from the registered '$v'"; exit 78; fi
  export "$k=$v"
done < "$HERE/registered.knobs"
export TP4_RUNNER=p67_run.sh TP4_EXTRA_STAGE="$HERE/p67_run.sh $HERE/registered.knobs $HERE/staged.sha256"
say "registered: TP4_BOX=$TP4_BOX families=$TP4_FAMILIES steps=$TP4_STEPS P56=$TP4_P56 P67=$TP4_P67 perms='$TP4_P67_PERMS' gnf4=$GNF4_SHA prereg=$TP4_PREREG"
exec bash "$REPO/bench/tp4/tp4_drive.sh"
