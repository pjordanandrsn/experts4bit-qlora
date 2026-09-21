"""The P55x lane's controller-side scripts must run on the controller we actually have.

`p55x-prove-1` (2026-09-21) reported `upload 0.00 MB/s` and `0 GB free on /root` on a box whose
pre-flight had just measured 320 GB free and 99.3 MB/s of Hugging Face CDN egress. Two defects, both
on the controller side and both invisible to `bash -n`:

1. **The controller is macOS, whose `rsync` is openrsync** ("rsync version 2.6.9 compatible"). It
   rejects `--no-compress` and `--info=progress2` with a usage error (rc 1), so the probe's transfer
   never ran and the lane read the failure as a throughput of zero. `-a` is already uncompressed, so
   the flag bought nothing even where it is supported.
2. **The Vast image prints a two-line login banner before the command's own output**, so anything
   parsed by line number reads "Welcome to vast.ai..." as its first value. Same family as the ssh
   banner fused to a curl HTTP status.

These tests are cheap and would have caught both before a box was rented.
"""
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p55x"
# Long options macOS openrsync accepts, verified on the controller with `rsync <flag> --version`.
# Anything outside this set must be justified, because the controller is where these scripts run.
OPENRSYNC_OK = {"--partial", "--inplace", "--timeout", "--exclude", "--delete", "--archive",
                "--verbose", "--quiet", "--compress", "--recursive", "--times", "--links",
                "--perms", "--owner", "--group", "--devices", "--specials", "--rsh", "--dry-run"}
KNOWN_BAD = {"--no-compress", "--info", "--outbuf", "--mkpath", "--old-args"}

SCRIPTS = sorted(LANE.glob("*.sh"))

pytestmark = pytest.mark.skipif(not SCRIPTS, reason="p55x lane not present")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_rsync_invocations_use_only_flags_the_controller_supports(script):
    text = script.read_text()
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#") or "rsync " not in line:
            continue
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", line):
            assert flag not in KNOWN_BAD, (
                f"{script.name}:{line_no} passes {flag} to rsync. The controller is macOS, whose rsync "
                f"is openrsync, and it rejects that with a usage error (rc 1) -- p55x-prove-1 read the "
                f"resulting failure as 0.00 MB/s of upload. Drop it (-a is already uncompressed)."
            )
            assert flag in OPENRSYNC_OK, (
                f"{script.name}:{line_no} passes {flag} to rsync, which is not in the set verified "
                f"against the controller's openrsync. Verify it there first, then add it."
            )


def test_the_prove_script_reads_box_values_by_marker_not_by_line_number():
    """A login banner shifts every line, so a line-indexed read returns the banner."""
    text = (LANE / "p55x_prove.sh").read_text()
    assert "P55X_GPU=" in text and "P55X_AVAIL_GB=" in text, "box facts must be emitted as KEY= markers"
    assert re.search(r'grep -m1 "\^\$1=" ', text), "the values must be read back by their marker"
    for banned in (r"head -1 \"\$OUT/box.txt\"", r"sed -n '2p' \"\$OUT/box.txt\""):
        assert not re.search(banned, text), f"{banned} reads the Vast login banner, not the value"


def test_the_prove_script_fails_when_it_measured_nothing():
    """A low reading is a result; no reading is not, and the first run reported OK over a dead transfer."""
    text = (LANE / "p55x_prove.sh").read_text()
    assert 'if [ "$PRC" != 0 ] || [ "$GOT" -lt $((PROBE_MB * 1048576)) ]; then' in text
    assert "exit 24" in text, "a failed probe transfer must not exit 0"


def test_the_drive_script_still_refuses_a_low_but_real_upload():
    """The fix above must not have removed the lane's actual STOP-0 floor."""
    text = (LANE / "p55x_drive.sh").read_text()
    assert "P55X_MIN_UP_MBS" in text and "exit 13" in text
    assert "the box cannot push its own product" in text
