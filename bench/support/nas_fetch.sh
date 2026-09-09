#!/bin/sh
# Pull a Hugging Face model straight onto the NAS. Runs ON the QNAP (BusyBox +
# Entware), so: /bin/sh, no bashisms, no `nice` (there is none on this userland
# and adding it to a command line just makes the command fail -- that cost 15
# spurious FAILs during the K3 fetch).
#
# Modelled on the K3 fetch loop already in this fleet
# (/share/ZFS532_DATA/hf-models/moonshotai_Kimi-K3/fetch.sh), keeping the parts
# that were earned the hard way:
#
#   * a pid file that is re-checked with `kill -0` -- a stale pid from a killed
#     run is NOT a lock, and two concurrent loops silently corrupt each other
#   * `curl -C -` resume plus a per-file `.done` marker, so a relaunch skips
#     completed files instead of re-pulling terabytes
#   * HF's `x-linked-etag` (the LFS sha256) captured as an INDEPENDENT hash to
#     check against, and re-fetched with a HEAD when a completed file answers
#     the resume with 416 and omits the header -- "no etag" must not be read as
#     "no objection"
#
# Usage: nas_fetch.sh <hf-model-id> <dest-dir>
set -u

MODEL=$1
DEST=$2
LOG=$DEST/download.log
RESOLVE="https://huggingface.co/$MODEL/resolve/main"

mkdir -p "$DEST" || exit 1

if [ -f "$DEST/fetch.pid" ] && kill -0 "$(cat "$DEST/fetch.pid" 2>/dev/null)" 2>/dev/null; then
  echo "ALREADY-RUNNING pid=$(cat "$DEST/fetch.pid") -- refusing to start a second loop" >> "$LOG"
  exit 0
fi
echo $$ > "$DEST/fetch.pid"
trap 'rm -f "$DEST/fetch.pid"' EXIT INT TERM

echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL pid=$$ ===" >> "$LOG"

# File list from the API rather than a hardcoded list: a hardcoded one is how a
# fetch silently misses a shard that was added to the repo.
LIST=$DEST/.filelist
if [ ! -s "$LIST" ]; then
  curl -sSL "https://huggingface.co/api/models/$MODEL" -o "$DEST/.api.json" 2>>"$LOG"
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
keep = []
for s in d.get("siblings", []):
    f = s["rfilename"]
    # Skip formats a load never touches. The gpt-oss-20b directory on this NAS is
    # 39 GB where 13 GB is the model, entirely because of these.
    if f.startswith(("original/", "metal/", ".cache/")):
        continue
    if f.endswith((".gguf", ".md", ".png", ".jpg")):
        continue
    keep.append(f)
for f in keep:
    print(f)
' "$DEST/.api.json" > "$LIST" 2>>"$LOG"
fi
n=$(wc -l < "$LIST")
echo "file list: $n entries" >> "$LOG"
if [ "$n" -eq 0 ]; then
  echo "REFUSING: empty file list for $MODEL (gated, renamed, or the API call failed)" >> "$LOG"
  exit 2
fi

while read -r f; do
  [ -z "$f" ] && continue
  if [ -f "$DEST/$f.done" ]; then
    echo "SKIP $f (already complete)" >> "$LOG"
    continue
  fi
  mkdir -p "$DEST/$(dirname "$f")" 2>/dev/null
  t0=$(date +%s)
  hdr=$(curl -sSL -m 21600 -C - -D - -o "$DEST/$f" \
        -w '%{http_code} %{size_download}' "$RESOLVE/$f" 2>>"$LOG" | tr -d '\r')
  code=$(printf '%s' "$hdr" | tail -1 | awk '{print $1}')
  etag=$(printf '%s' "$hdr" | grep -i '^x-linked-etag:' | head -1 | sed 's/.*"\(.*\)".*/\1/')
  if [ -z "$etag" ]; then
    etag=$(curl -sIL "$RESOLVE/$f" 2>/dev/null | tr -d '\r' \
           | grep -i '^x-linked-etag:' | head -1 | sed 's/.*"\(.*\)".*/\1/')
  fi
  sz=$(wc -c < "$DEST/$f" 2>/dev/null || echo 0)
  t1=$(date +%s)

  case "$code" in
    200|206|416)
      # x-linked-etag is an LFS sha256 (64 hex) only for LFS-tracked files. For a
      # small non-LFS file HF returns the git blob SHA-1 (40 hex), which is not a
      # hash of the content in any comparable sense -- comparing it to a sha256
      # reported HASH-MISMATCH on config.json, .gitattributes and every other
      # small file, and left them without .done markers. That is not merely
      # noisy: the driver decides a model is complete by counting .done against
      # the file list, so those files would be re-fetched forever and the model
      # would never register as finished. Check the LENGTH before trusting it.
      if [ -n "$etag" ] && [ ${#etag} -eq 64 ]; then
        have=$(sha256sum "$DEST/$f" 2>/dev/null | awk '{print $1}')
        if [ "$have" = "$etag" ]; then
          : > "$DEST/$f.done"
          echo "OK $f $sz B in $((t1 - t0))s sha256=$have (matches x-linked-etag)" >> "$LOG"
        else
          echo "HASH-MISMATCH $f ours=$have hf=$etag -- leaving WITHOUT a .done marker" >> "$LOG"
        fi
      elif [ -n "$etag" ]; then
        : > "$DEST/$f.done"
        echo "OK $f $sz B in $((t1 - t0))s (etag ${#etag} hex = git blob SHA-1, not an LFS" \
             "sha256; size-only, not hash-verified)" >> "$LOG"
      else
        # No etag from either the GET or the HEAD: a non-LFS small file. Size is
        # all we have, so say so rather than marking it verified.
        : > "$DEST/$f.done"
        echo "OK $f $sz B in $((t1 - t0))s (no x-linked-etag; size-only, not hash-verified)" >> "$LOG"
      fi
      ;;
    401|403)
      echo "GATED $f http=$code -- $MODEL needs an HF token; stopping" >> "$LOG"
      exit 3
      ;;
    *)
      echo "FAIL $f http=$code size=$sz -- will retry on the next run" >> "$LOG"
      ;;
  esac
done < "$LIST"

done_n=$(find "$DEST" -name '*.done' | wc -l)
echo "=== end $(date -u +%Y-%m-%dT%H:%M:%SZ) $done_n/$n complete ===" >> "$LOG"
