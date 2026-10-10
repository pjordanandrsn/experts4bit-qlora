#!/bin/bash
# Draft box side. A stub-complete marker cannot masquerade as TC1_SUCCESS.
set -uo pipefail
W=${DQ11_W:-/root/tc1}; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}
case "$NONCE" in *[!a-zA-Z0-9_-]*|"") exit 78;; esac
printf '%s\n' "$NONCE" > TC1_RUN_NONCE
finish(){ printf '%s\n' "$1" > "TC1_EXIT_CODE.$NONCE"; : > "TP_DONE.$NONCE"; exit "$1"; }
[ "${DQ11_DRY_RUN:-0}" = 1 ] || { echo 'DQ11 DRAFT_NOT_LAUNCHABLE'; finish 78; }
sha256sum -c instrument.sha256 || finish 9
python3 dq11_box.py --dry-run --nonce "$NONCE" --out "dq11-dry-run.$NONCE.json" || finish 11
: > "DQ11_DRY_RUN_COMPLETE.$NONCE"
finish 0
