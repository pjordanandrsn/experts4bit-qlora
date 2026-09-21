#!/bin/bash
# bench/p55x/p55x_publish.sh -- lane P55x, AFTER the run: verify the fetched pack artifact and place it on an
# archive host, which is where the licensed BYTES live (the receipts tree is git and the pack is ~15.2 GiB).
#
#   P55X_NAS_HOST=user@host bench/p55x/p55x_publish.sh <local-artifact-dir> <sha256:...expected fingerprint>
#
# It refuses rather than publishing something it cannot identify. Three checks, in this order:
#   1. the library's own verify_artifact on the local copy -- re-hashes every payload, cross-checks the hashed
#      identity and assignment payloads against the manifest's copies, recomputes the root fingerprint, and
#      refuses a mismatch (#405). Run from THIS checkout, so the verifier is the code the pack was built by.
#   2. rsync to the NAS.
#   3. on the NAS, re-hash every payload and recompute the root fingerprint from the documented canonical rule
#      (sha256 over the ordered (path, size, sha256) tuples). The NAS has no e4b install, so that rule is
#      re-implemented inline (the archive host has no e4b install) -- and is only trusted because step 1
#      computed the same number with the library on the same bytes. Two implementations agreeing on one
#      artifact is the check; either alone is not. P55X_NAS_PY names that host's python if it is not python3.
# Nothing here writes to the register. A pack is quotable only once this exits 0.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p55x_publish] $*"; }
[ "$#" -eq 2 ] || { echo "usage: $(basename "$0") <local-artifact-dir> <sha256:...>" >&2; exit 95; }
ART=$1; FP=$2
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# The archive host is a PARAMETER, not a default. This tree is public; an operator's host name does not
# belong in it, and a default would quietly send someone else's bytes to a machine they do not own.
NAS_HOST=${P55X_NAS_HOST:?set P55X_NAS_HOST to the ssh destination of the archive host (user@host)}
NAS_POOL=${P55X_NAS_POOL:-/srv/e4b-packs}   # a path on that host with room for ~16 GiB per pack
NAS_LINK=${P55X_NAS_LINK:-}                 # optional: a stable symlink to $NAS_POOL that readers are told
PY=${P55X_PY:-python3}
case "$FP" in sha256:*) ;; *) say "refusing: '$FP' is not sha256:<64 hex>"; exit 95;; esac
[ -d "$ART" ] && [ -s "$ART/manifest.json" ] || { say "refusing: $ART has no manifest.json"; exit 95; }
SHORT=${FP#sha256:}; SHORT=${SHORT:0:16}
DEST="$NAS_POOL/qwen3-30b-a3b-calibexp-streamed-64k-$SHORT"

say "1/3 verifying the local copy with this checkout's pack_manifest"
PYTHONPATH="$REPO" "$PY" - "$ART" "$FP" <<'PY' || { say "REFUSING: the local artifact does not verify"; exit 1; }
import sys
from experts4bit_qlora.engines.pack_manifest import verify_artifact
art, fp = sys.argv[1], sys.argv[2]
man = verify_artifact(art, expected_fingerprint=fp)
print(f"   verify_artifact OK: {man['pack_fingerprint']}")
print(f"   model {man.get('model_id')} @ {man.get('model_revision')} layout {man.get('layout')}")
print(f"   payloads {len(man['payloads'])} min_rows {man.get('min_rows')} damping {man.get('damping')} "
      f"counts {man.get('calibrated_counts')}")
PY

BYTES=$(du -sm "$ART" | cut -f1)
say "2/3 placing ${BYTES}M on the archive host at $DEST"
ssh -o BatchMode=yes "$NAS_HOST" "mkdir -p '$DEST'${NAS_LINK:+ && { [ -e '$NAS_LINK' ] || ln -s '$NAS_POOL' '$NAS_LINK'; \}}" \
  || { say "REFUSING: cannot prepare $DEST"; exit 2; }
rsync -a --partial --inplace --no-compress --info=progress2 "$ART/" "$NAS_HOST:$DEST/" \
  || { say "REFUSING: rsync to the archive host failed -- the bytes are not retained"; exit 2; }

say "3/3 re-hashing on the archive host and recomputing the root fingerprint there"
ssh -o BatchMode=yes "$NAS_HOST" "${P55X_NAS_PY:-python3} - '$DEST' '$FP'" <<'PY' || { say "REFUSING: the archive copy does not verify -- treat it as absent"; exit 3; }
import hashlib, json, sys, os
root, want = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(root, "manifest.json")))
rows = []
for p in man["payloads"]:
    fp = os.path.join(root, p["path"])
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    size, digest = os.path.getsize(fp), h.hexdigest()
    if size != int(p["size"]) or digest != str(p["sha256"]).lower():
        sys.exit(f"   payload {p['path']} differs after transfer: {size} vs {p['size']}, {digest} vs {p['sha256']}")
    rows.append((p["path"], size, digest))
rows.sort(key=lambda r: r[0])
blob = json.dumps([list(r) for r in rows], separators=(",", ":"), ensure_ascii=True).encode()
got = "sha256:" + hashlib.sha256(blob).hexdigest()
if got != want:
    sys.exit(f"   archive copy fingerprints {got}, expected {want}")
print(f"   archive copy verified: {len(rows)} payloads, {got}")
PY

say "PUBLISHED $FP"
say "  bytes:    $NAS_HOST:$DEST${NAS_LINK:+  (reader-facing path $NAS_LINK/$(basename "$DEST"))}"
say "  manifest: keep manifest.json + payloads/identity.json + payloads/assignment.json in the receipts"
say "  a loader pinned to this fingerprint refuses anything else, so a reader can check any copy they are given"
