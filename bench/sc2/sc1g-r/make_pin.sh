#!/bin/bash
# bench/sc2/sc1g-r/make_pin.sh -- regenerate staged-r.sha256: every file box R stages, named as the box sees it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
WIN="$REPO/bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g"
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
{
  for f in sc1g_r_run.sh window_shas.json; do echo "$(sha_of "$HERE/$f")  $f"; done
  for f in sc1g_ref.py sc1g_kl.py; do echo "$(sha_of "$REPO/bench/sc2/$f")  $f"; done
  echo "$(sha_of "$REPO/bench/kl_fidelity.py")  kl_fidelity.py"
  for s in conv1 conv2 conv3 conv4 wikitext; do echo "$(sha_of "$WIN/k8_window_$s.json")  windows/k8_window_$s.json"; done
} > "$HERE/staged-r.sha256"
echo "pinned $(wc -l < "$HERE/staged-r.sha256" | tr -d ' ') files -> $HERE/staged-r.sha256"
