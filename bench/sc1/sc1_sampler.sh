#!/bin/bash
# bench/sc1/sc1_sampler.sh -- the per-arm VRAM / power / PCIe sampler and a host sampler (lane SC1, experts4bit-qlora#846;
# the notes/SC1-PREREG-draft-v3.md "Phase EN" sampler line, run around EVERY arm so the resource columns come from the
# same instrument as the energy windows).
#
#   sc1_sampler.sh start <tag>   nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw.instant,
#                                pcie.link.gen.current --format=csv,noheader,nounits -lms 50   ->  $SC1_SAMPLES_DIR/<tag>.csv
#                                + a 1 s host sampler (epoch, loadavg, the top-5 processes by RSS with %CPU) -> <tag>.host.txt
#   sc1_sampler.sh stop <tag>    stops both; writes <tag>.meta.json {tag, start_epoch, stop_epoch, fields, n_gpu_samples, ...}
#   sc1_sampler.sh stop-all      stops every sampler this directory knows (the runner's finish)
#
# J/token for an energy window = the reducer's integral of power.draw.instant over the window the RECEIPT names
# (e4b-sched: energy_window_epoch; vLLM: the LONG rep's timed_windows_epoch) divided by the tokens it decoded -- board
# power, stated. `power.draw.instant` needs a recent nvidia-smi; when the field list is refused the sampler falls back
# to `power.draw` (an averaged reading) and the meta says which fields it sampled.
set -uo pipefail
DIR=${SC1_SAMPLES_DIR:-$PWD/samples}; mkdir -p "$DIR"
FIELDS=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw.instant,pcie.link.gen.current
FALLBACK=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,pcie.link.gen.current
cmd=${1:-}; tag=${2:-}
_kill(){ local f=$1; [ -f "$f" ] || return 0; local p; p=$(cat "$f"); kill "$p" 2>/dev/null; rm -f "$f"; }
case "$cmd" in
  start)
    [ -n "$tag" ] || { echo "usage: sc1_sampler.sh start <tag>" >&2; exit 2; }
    f=$FIELDS
    nvidia-smi --query-gpu=$f --format=csv,noheader,nounits >/dev/null 2>&1 || f=$FALLBACK
    nohup nvidia-smi --query-gpu=$f --format=csv,noheader,nounits -lms 50 > "$DIR/$tag.csv" 2>/dev/null &
    echo $! > "$DIR/$tag.gpu.pid"
    nohup bash -c 'while :; do printf "%s load=%s |" "$(date +%s.%N)" "$(cut -d" " -f1-3 /proc/loadavg)"; ps -eo pid=,rss=,pcpu=,comm= --sort=-rss | head -5 | tr -s " " | tr "\n" "|"; echo; sleep 1; done' \
      > "$DIR/$tag.host.txt" 2>/dev/null &
    echo $! > "$DIR/$tag.host.pid"
    printf '%s %s\n' "$(date +%s.%N)" "$f" > "$DIR/$tag.meta"
    ;;
  stop)
    [ -n "$tag" ] || { echo "usage: sc1_sampler.sh stop <tag>" >&2; exit 2; }
    _kill "$DIR/$tag.gpu.pid"; _kill "$DIR/$tag.host.pid"
    start=""; fields=""
    [ -f "$DIR/$tag.meta" ] && read -r start fields < "$DIR/$tag.meta"
    stop=$(date +%s.%N)
    n_gpu=$(wc -l < "$DIR/$tag.csv" 2>/dev/null | tr -d ' '); n_host=$(wc -l < "$DIR/$tag.host.txt" 2>/dev/null | tr -d ' ')
    printf '{"tag": "%s", "start_epoch": %s, "stop_epoch": %s, "fields": "%s", "interval_ms": 50, "csv": "%s", "host": "%s", "n_gpu_samples": %s, "n_host_samples": %s}\n' \
      "$tag" "${start:-null}" "$stop" "$fields" "samples/$tag.csv" "samples/$tag.host.txt" "${n_gpu:-0}" "${n_host:-0}" > "$DIR/$tag.meta.json"
    ;;
  stop-all)
    for p in "$DIR"/*.pid; do [ -f "$p" ] && _kill "$p"; done
    ;;
  *) echo "usage: sc1_sampler.sh start <tag> | stop <tag> | stop-all" >&2; exit 2 ;;
esac
exit 0
