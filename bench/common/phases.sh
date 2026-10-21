# phases.sh -- the box-side phase stamps every new rental kit writes (bench/common/README.md, "Phase stamps").
# Source it, then bracket each phase:  phase_start NAME KIND ... phase_end NAME
#   KIND is one of setup | measure | reduce | other. The file is ${PHASES_FILE:-phases.jsonl} in the current directory.
# One JSON object per line, appended, never rewritten. A name is [A-Za-z0-9_.:-]+; anything else is refused (rc 2)
# rather than written, so a reader never has to guess. Safe under set -u (a missing argument is a refusal, rc 2).
phase_start() {
  case "${1-}" in ''|*[!A-Za-z0-9_.:-]*) echo "phase_start: bad phase name '${1-}'" >&2; return 2;; esac
  case "${2-}" in setup|measure|reduce|other) ;; *) echo "phase_start: bad kind '${2-}' for $1" >&2; return 2;; esac
  printf '{"phase": "%s", "event": "start", "kind": "%s", "t": %s}\n' "$1" "$2" "$(date +%s)" >> "${PHASES_FILE:-phases.jsonl}"
}
phase_end() {
  case "${1-}" in ''|*[!A-Za-z0-9_.:-]*) echo "phase_end: bad phase name '${1-}'" >&2; return 2;; esac
  printf '{"phase": "%s", "event": "end", "t": %s}\n' "$1" "$(date +%s)" >> "${PHASES_FILE:-phases.jsonl}"
}
