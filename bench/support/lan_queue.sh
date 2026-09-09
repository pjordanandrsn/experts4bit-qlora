#!/usr/bin/env bash
# Work through NAS checkpoints one at a time: copy over LAN, probe, reclaim disk.
#
# Why this exists: the support lane was planned as download-bound, on the premise
# that checkpoints had to come over the WAN and that Vast bandwidth varies >10x
# between boxes -- so it wanted rented boxes chosen for measured bandwidth. That
# premise is wrong on this fleet. /share/models holds real checkpoints for 8 of
# the 9 claimed families, on the LAN. The binding constraint is this host's 64 GB
# of RAM, not download cost, and most of the matrix needs no rental at all.
#
# SEQUENTIAL on purpose. The pool is HDD and serves Plex and the arr stack; two
# concurrent tars already measured ~20 MB/s combined. One at a time is no slower
# in aggregate and leaves the media stack alone.
#
# The local copy is DELETED after a successful probe. The row is the deliverable,
# the NAS keeps the checkpoint, and 424 GB free does not survive Qwen3-30B (57 G)
# + Qwen3.6-35B (68 G) + a DeepSeek-V4-Flash (157 G) held simultaneously.
set -uo pipefail

NAS=${NAS:-admin@10.0.0.68}
SRC=${SRC:-/share/models}
DEST=${DEST:-$HOME/models}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$HOME/.venvs/e4b/bin/python}
ROWS=$REPO/bench/support/rows
KEEP=${KEEP:-0}          # KEEP=1 to leave copies on disk

# name : min_gb : row : dtype
QUEUE=${QUEUE:-"granite-3.0-1b-a400m-instruct:2:granitemoe_1b_a400m:bfloat16
Qwen3-30B-A3B:50:qwen3_moe:bfloat16
Qwen3.6-35B-A3B:60:qwen3_5_moe:bfloat16"}

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# Echo the queue we actually received, one line per entry. A run that processed
# Qwen3-30B and then said "queue done" -- skipping the Qwen3.6 entry passed in
# the same variable -- left no way to tell whether the entry was lost in the
# environment, filtered by a guard, or never there. A queue that does not state
# its own input cannot be debugged after the fact.
log "queue received $(echo "$QUEUE" | grep -c .) entr$([ "$(echo "$QUEUE" | grep -c .)" = 1 ] && echo y || echo ies):"
echo "$QUEUE" | grep . | while IFS= read -r q; do log "  | $q"; done

echo "$QUEUE" | while IFS=: read -r name min_gb row dtype; do
  [ -n "$name" ] || continue
  out=$ROWS/$row.json
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    log "SKIP $name -- $row.json exists (FORCE=1 to redo)"
    continue
  fi

  free_gb=$(df -g "$DEST" 2>/dev/null | awk 'NR==2{print $4}')
  if [ -n "$free_gb" ] && [ "$free_gb" -lt $((min_gb + 40)) ]; then
    log "STOP: ${free_gb}GB free, $name needs ~${min_gb}GB plus headroom for the load"
    exit 1
  fi

  log "copying $name ($(ssh "$NAS" "du -sh $SRC/$name 2>/dev/null | cut -f1"))"
  mkdir -p "$DEST/$name"
  # Build the member list on the REMOTE from what exists, rather than passing
  # globs. Passing globs made tar exit non-zero on any checkpoint lacking one of
  # them -- granite-3.0 has a single model.safetensors and no *.index.json, so a
  # perfectly good copy reported an error. That is not cosmetic: it made the exit
  # status carry no information, so a genuinely truncated copy looked identical
  # to a fine one and only the size guard below told them apart. `set --` from a
  # filtered `ls` keeps it to ONE tar invocation (xargs could split the list and
  # emit concatenated archives, of which an extractor reads only the first).
  #
  # Excluded: metal/, original/, .cache/ -- duplicate formats a load never
  # touches, and what made gpt-oss-20b 39 GB instead of 13 GB.
  if ! ssh "$NAS" "cd $SRC/$name && set -- \$(ls -1 | grep -vxE 'metal|original|\.cache|README\.md|\.gitattributes|\.filelist|\.api\.json|download\.log|fetch\.pid|\.eval_results') && tar -cf - \"\$@\"" \
      | tar -C "$DEST/$name" -xf -; then
    log "copy of $name FAILED (exit status is meaningful now); size guard decides next"
  fi

  got=$(du -sg "$DEST/$name" 2>/dev/null | cut -f1)
  if [ -z "$got" ] || [ "$got" -lt "$min_gb" ]; then
    log "REFUSING to probe $name: ${got:-0}GB < ${min_gb}GB expected. A load failure on a"
    log "  truncated checkpoint would land in the rows as a fact about the family."
    [ "$KEEP" = "1" ] || rm -rf "$DEST/$name"
    continue
  fi

  log "probing $name (${got}GB)"
  ( cd "$REPO" && PYTHONPATH=$REPO "$PY" bench/support/support_probe.py \
      --model "$DEST/$name" --device cpu --dtype "$dtype" --out "$out" )
  rc=$?
  log "$name probe exit $rc -> $row.json"

  # If the probe died without writing, record that. The OS can kill it outright
  # -- an OOM on the largest checkpoints is the likeliest outcome and leaves no
  # row at all, which the reducer then reports as `none`: indistinguishable from
  # "never attempted". Absence reading as untested is the failure this whole
  # harness exists to remove, so it must not reappear at the one point where the
  # answer matters most.
  if [ ! -f "$out" ]; then
    log "  probe wrote NO row (exit $rc) -- synthesising one so the attempt is not invisible"
    PYTHONPATH=$REPO "$PY" - "$DEST/$name" "$out" "$rc" <<'PYEOF'
import json, os, platform, sys, time
model_dir, out, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
mt = None
try:
    c = json.load(open(os.path.join(model_dir, "config.json")))
    mt = c.get("model_type")
except Exception:
    pass
row = {
    "model": model_dir, "revision": None, "source": "local-path",
    "provenance": "published", "device": "cpu", "dtype": "bfloat16",
    "probe": "bench/support/lan_queue.sh (stub: the probe produced no row)",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "versions": {"python": platform.python_version(), "platform": platform.platform()},
    "model_type": mt,
    "stages": {"load": {
        "status": "error",
        "error": f"the probe exited {rc} without writing a row -- most likely killed by "
                 f"the OS (OOM). Nothing here says the family is unsupported: it says "
                 f"this HOST could not complete the attempt.",
    }},
    "exit_code": rc if rc != 0 else 9,
}
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write(json.dumps(row, indent=1) + "\n")
print(f"stub row written to {out}")
PYEOF
  fi

  if [ "$KEEP" != "1" ]; then
    log "reclaiming $(du -sh "$DEST/$name" | cut -f1) from $name"
    rm -rf "$DEST/$name"
  fi
done

log "queue done"
