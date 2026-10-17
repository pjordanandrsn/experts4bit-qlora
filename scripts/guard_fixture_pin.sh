#!/bin/bash
# Holds one private-marker-guard fixture exemption to its pin. The fixture's
# bytes arrive on stdin. Exit 0 = at most MAX_LINES lines and the pinned sha256,
# so the exemption stands. Exit 1 = the fixture grew or changed, and the guard
# must fail rather than skip it.
#
# Usage: scripts/guard_fixture_pin.sh PATH MAX_LINES SHA256 < bytes
# The pins live in .github/private-markers.txt as "#exempt-fixture:" lines.
set -euo pipefail
[ "$#" -eq 3 ] || { echo "usage: $0 PATH MAX_LINES SHA256 < bytes" >&2; exit 2; }
path=$1 cap=$2 pin=$3
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cat > "$tmp"
# grep -c '' also counts a last line with no newline; wc -l would not.
n=$(grep -c '' "$tmp" || true)
if [ "$n" -gt "$cap" ]; then
  echo "exempt fixture $path has $n lines; its cap is $cap" >&2
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha=$(sha256sum < "$tmp" | cut -d' ' -f1)
else
  sha=$(shasum -a 256 < "$tmp" | cut -d' ' -f1)
fi
if [ "$sha" != "$pin" ]; then
  echo "exempt fixture $path changed: sha256 $sha, pinned $pin" >&2
  exit 1
fi
