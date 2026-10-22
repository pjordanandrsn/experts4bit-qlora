#!/bin/bash
# T1 step 3's box runner (bench/first_use/PREREG-first_use.md): the release candidate's own first install, once per cell
# this box was rented for. Each cell is scripts/first_run.sh, from the release candidate's commit, run in a new directory
# with the RC wheel as PACKAGE_SPEC and the cell's TORCH_INDEX; fu_reduce.py grades every cell by the driver this box
# reads. HO2's runner (bench/ho2/ho2_run.sh in experts4bit-qlora) for the nonce, the markers and the staged-hash check.
# No instance id and no host name are written here. This script never creates compute.
set -uo pipefail
W=${FU_WORK_DIR:-/root/first_use}; mkdir -p $W/cells $W/logs; cd $W || exit 78   # the override is for the kit's own tests
say(){ echo "[$(date -u +%FT%TZ)] first_use: $*"; }
NONCE=${FU_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/FU_RUN_NONCE.tmp && mv $W/FU_RUN_NONCE.tmp $W/FU_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > FU_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > FU_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
sha256(){ if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
for v in FU_RUN_ID FU_DEADLINE_EPOCH FU_CELLS FU_WHEEL FU_WHEEL_SHA256 FU_FIRST_RUN_SHA256; do
  [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }
done
: > summary.txt; echo "$FU_RUN_ID" > RUN_ID
# ---- staged pieces, byte for byte: the kit by its manifest, the release candidate's two files by the hashes the
# controller computed at the RC commit (and records in the receipt)
for f in fu_reduce.py staged-first_use.sha256 first_run.sh "$FU_WHEEL"; do [ -s "$W/$f" ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256 -c staged-first_use.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM staged-first_use.sha256"; finish 9; }
[ "$(sha256 "$W/$FU_WHEEL" | cut -d' ' -f1)" = "$FU_WHEEL_SHA256" ] || { say "RC WHEEL HASH MISMATCH"; finish 9; }
[ "$(sha256 "$W/first_run.sh" | cut -d' ' -f1)" = "$FU_FIRST_RUN_SHA256" ] || { say "first_run.sh HASH MISMATCH"; finish 9; }
python3 $W/fu_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the cells: each registered name, run in order, each in its own new directory (first_run.sh refuses an existing one)
NEED_CELL=1500; RESERVE=300
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader > forensics.txt 2>&1 || true
echo "BOX $(head -1 forensics.txt)" | tee -a summary.txt
ran=0
for cell in $FU_CELLS; do
  case "$cell" in
    below-default|at-default|current-default) tix=pypi;;
    below-matched|at-matched|current-matched) tix="";;
    cpu-torch) tix=cpu;;
    no-gpu) say "refusing: the no-gpu cell runs on a host with no NVIDIA GPU, never on a rented GPU box"; finish 78;;
    *) say "refusing: unregistered cell $cell"; finish 78;;
  esac
  # the cell's directory exists BEFORE the deadline check: a skipped cell reads MISSING (the run INCOMPLETE), never absent
  mkdir -p "$W/cells/$cell"
  left=$((FU_DEADLINE_EPOCH - $(date +%s)))
  if [ $((NEED_CELL + RESERVE)) -gt "$left" ]; then
    say "STOP: $cell needs $((NEED_CELL + RESERVE))s, ${left}s left -- skipped"; echo "SKIPPED $cell" >> summary.txt
    printf '{"cell": "%s", "torch_index_env": "%s", "listed_driver": "%s", "skipped": "deadline: %ss left, %ss needed"}\n' \
      "$cell" "$tix" "${FU_LISTED_DRIVER:-}" "$left" "$((NEED_CELL + RESERVE))" > "$W/cells/$cell/cell.json"
    continue
  fi
  printf '{"cell": "%s", "torch_index_env": "%s", "listed_driver": "%s"}\n' "$cell" "$tix" "${FU_LISTED_DRIVER:-}" > "$W/cells/$cell/cell.json"
  say "cell $cell (TORCH_INDEX=${tix:-<unset>})"
  # first_run.sh makes the run directory itself and refuses an existing one: hand it a fresh path inside the cell dir
  # (the whole cell directory comes back in the fetch; the venv is excluded there)
  PACKAGE_SPEC="$W/$FU_WHEEL" TORCH_INDEX="$tix" PYTHON=python3 \
    perl -e "alarm $NEED_CELL; exec @ARGV" bash "$W/first_run.sh" "$W/cells/$cell/run" > "logs/$cell.out" 2> "logs/$cell.err"
  rc=$?
  [ -f "$W/cells/$cell/run/first_run.json" ] && cp "$W/cells/$cell/run/first_run.json" "$W/cells/$cell/first_run.json"
  echo "cell $cell rc=$rc" | tee -a summary.txt
  ran=$((ran + 1))
done
# the reduce runs even when no cell ran, so every registered cell this box was given is on record (MISSING if skipped)
python3 $W/fu_reduce.py --dir $W/cells --run-id "$FU_RUN_ID" --out $W/verdict_first_use.json 2>&1 | tee -a summary.txt
[ -s $W/verdict_first_use.json ] || { say "REDUCER FAILED"; finish 22; }
[ "$ran" -gt 0 ] || { say "no cell ran"; finish 40; }
finish 0
