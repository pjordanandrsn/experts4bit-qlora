#!/bin/bash
# Controller-side helper; sent over SSH stdin, never added to a lane's staged kit.
# A known launch PID + /proc start time survives exec and cannot match the probe
# command, an SSH ancestor, or an unrelated process that reuses the PID.
lane_proc_snapshot() {  # pid [proc-root] -> live start-ticks boot-id uptime-seconds
  local pid=$1 root=${2:-/proc} uptime boot stat state ticks=- live=unknown
  local -a fields
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  read -r uptime _ < "$root/uptime" || return 1
  [[ "$uptime" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  boot=$(cat "$root/sys/kernel/random/boot_id" 2>/dev/null) || boot=-
  [[ "$boot" =~ ^(-|[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})$ ]] || boot=-
  # An existing entry with unreadable stat leaves live unknown.
  if stat=$(cat "$root/$pid/stat" 2>/dev/null); then
    # comm (field 2) can contain spaces and ')'; use the final ') ' separator.
    read -r -a fields <<< "${stat##*) }"
    state=${fields[0]:-}; ticks=${fields[19]:--}  # field 22, counting from field 3
    if [[ "$ticks" =~ ^[0-9]+$ ]] && [[ "$state" =~ ^[A-Z]$ ]]; then
      case "$state" in Z|X) live=0;; *) live=1;; esac
    else ticks=-; fi
  elif [ ! -d "$root/$pid" ]; then
    live=0
  fi
  printf '%s %s %s %s\n' "$live" "$ticks" "$boot" "$uptime"
}

lane_valid_snapshot() {
  local live ticks boot uptime extra
  read -r live ticks boot uptime extra <<< "$1"
  [ -z "$extra" ] && [[ "$live" =~ ^(0|1|unknown)$ ]] &&
    [[ "$ticks" =~ ^([0-9]+|-)$ ]] && [[ "$boot" =~ ^(-|[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})$ ]] &&
    [[ "$uptime" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

lane_snapshot_verdict() {  # current-snapshot initial-snapshot lane-age-seconds
  # Echo 1 (same live process), 0 (gone/reused/zombie), reboot, or nothing (unknown).
  local live ticks boot uptime extra old_live old_ticks old_boot old_uptime age=$3
  lane_valid_snapshot "$1" && lane_valid_snapshot "$2" || return 0
  read -r live ticks boot uptime extra <<< "$1"
  read -r old_live old_ticks old_boot old_uptime extra <<< "$2"
  [[ "$uptime" =~ ^[0-9]+([.][0-9]+)?$ && "$old_uptime" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 0
  [[ "$age" =~ ^[0-9]+$ ]] || return 0
  if { [ "$boot" != - ] && [ "$old_boot" != - ] && [ "$boot" != "$old_boot" ]; } ||
      awk -v up="$uptime" -v initial="$old_uptime" -v age="$age" \
          'BEGIN {exit !(up + 5 < age || up + 5 < initial)}'; then
    echo reboot; return 0  # 5s tolerance for sampling/clock rounding, not a stall rule
  fi
  case "$live" in
    0) echo 0;;
    1)
      [[ "$ticks" =~ ^[0-9]+$ && "$old_ticks" =~ ^[0-9]+$ ]] || return 0
      if [ "$ticks" = "$old_ticks" ]; then echo 1; else echo 0; fi;;
  esac
}

lane_two_missing() { [ "${1:-}" = 0 ] && [ "${2:-}" = 0 ] && echo dead; }

# bash -s -- --probe PID executes the same probe via SSH stdin; production uses /proc.
if [ "${1:-}" = --probe ]; then lane_proc_snapshot "${2:-}" "${3:-/proc}"; fi
