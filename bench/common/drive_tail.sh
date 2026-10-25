# bench/common/drive_tail.sh -- the driver's last step: the box's own exit code, handed to the launcher.
#
#   . "$REPO/bench/common/drive_tail.sh"
#   drive_finish "$OUT" "$NONCE" PREFIX "9 10 15 16 27 28 41 78 130"     # the lane's registered box codes
#
# Reads PREFIX_EXIT_CODE.<nonce> and PREFIX_SUCCESS.<nonce> in the fetched directory and EXITS the driver with:
#   0   the box exited 0 AND left this nonce's success marker (another nonce's marker does not count);
#   24  the box exited 0 without this nonce's marker: a success the box never recorded;
#   23  no exit file, or one that is not a number;
#   rc  the box's own code, when it is one of the launcher's host codes (13 disk, 14 egress, 17 power cap, 18 host floor:
#       the launcher excludes the machine only on `command exited 13|14|17|18`, the DRIVER's exit) or one the lane registers;
#   26  any other code: unregistered, so it is never passed on as something the launcher would act on.
# tests/test_drive_tail_codes.py drives this function directly, and every driver's tail against the same rule.

drive_finish() {
  local dir=$1 nonce=$2 prefix=$3 codes=" 13 14 17 18 ${4:-} " rc
  rc=$(cat "$dir/${prefix}_EXIT_CODE.$nonce" 2>/dev/null) || { echo "[drive_tail] no ${prefix}_EXIT_CODE.$nonce: rc 23" >&2; exit 23; }
  case "$rc" in ''|*[!0-9]*) echo "[drive_tail] unreadable exit code '$rc': rc 23" >&2; exit 23;; esac
  if [ "$rc" = 0 ]; then
    [ -f "$dir/${prefix}_SUCCESS.$nonce" ] || { echo "[drive_tail] rc 0 without ${prefix}_SUCCESS.$nonce: rc 24" >&2; exit 24; }
    exit 0
  fi
  case "$codes" in *" $rc "*) echo "[drive_tail] box rc $rc passed through" >&2; exit "$rc";; esac
  echo "[drive_tail] box rc $rc is not registered: rc 26" >&2
  exit 26
}
