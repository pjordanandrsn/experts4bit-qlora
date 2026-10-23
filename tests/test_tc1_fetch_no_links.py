"""tc1_drive.sh never imports a link from the box (bench/common rule 8).

The receipt store's writer hashes a linked file's target, and a rented box is a third-party host, so every fetch from
it passes --no-links. Checked two ways: statically, every rsync INVOCATION in the driver (command position: quoted
text and comments stripped, so `say "fetch failed: rsync"` is not one and `if ! rsync` is) carries --no-links; and by
running the driver's own fetch code (the function and the final-fetch block, cut from the script) against a fake rsync
on PATH that records its argv.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVE = (REPO / "bench" / "tc1" / "tc1_drive.sh").read_text(encoding="utf-8")

FAKE_RSYNC = r"""#!/usr/bin/env bash
printf '%s\n' "$@" >> "$FAKE_ARGV"; echo --- >> "$FAKE_ARGV"
dest="${@: -1}"
mkdir -p "$dest" && cp -R "$FAKE_BOX/." "$dest"
"""


def invocations(script: str) -> list[str]:
    """The lines that run rsync: double-quoted, then single-quoted text and then comments are removed before the match,
    so a message or a comment naming rsync is not an invocation, and `if ! rsync`, `$(rsync` and `&& rsync` are."""
    out = []
    for line in script.splitlines():
        code = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        code = re.sub(r"'[^']*'", "''", code)
        code = re.sub(r"(^|\s)#.*$", "", code)
        if re.search(r"(?<![\w./-])rsync(?![\w.-])", code):
            out.append(line)
    return out


def missing_no_links(script: str) -> list[str]:
    return [ln for ln in invocations(script) if not re.search(r"(?<![\w-])--no-links(?![\w-])", ln)]


def test_every_rsync_in_the_driver_passes_no_links():
    found = invocations(DRIVE)
    assert len(found) == 2, found            # the incremental fetch and the final fetch; none found fails here
    assert not missing_no_links(DRIVE), missing_no_links(DRIVE)


def test_the_check_catches_every_form_and_skips_messages():
    for form in ("if ! rsync -az a b; then", "x=$(rsync -az a b)", "true && rsync -az a b", "  rsync -az a b",
                 "while ! rsync -az a b; do"):
        assert missing_no_links(form) == [form], form
        assert not missing_no_links(form.replace("-az", "-az --no-links")), form
    for text in ('say "fetch failed: rsync (no incremental copy to keep)"', "# the final rsync", "echo 'rsync -az'",
                 'tc1_partial_fetch() {  # dest_dir -> rc of rsync; never exits the caller'):
        assert invocations(text) == [], text
    old = DRIVE.replace(" --no-links", "")
    assert len(missing_no_links(old)) == 2                       # the driver before this change fails on both fetches


def _cut(pattern: str) -> str:
    m = re.search(pattern, DRIVE, re.DOTALL | re.MULTILINE)
    assert m, f"pattern no longer matches tc1_drive.sh: {pattern}"
    return m.group(0)


def test_both_fetches_pass_no_links_to_rsync(tmp_path: Path):
    """Run the driver's own code: the incremental fetch, then the final fetch, against a fake rsync that records argv."""
    (tmp_path / "box").mkdir()
    (tmp_path / "box" / "summary.txt").write_text("ok\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "rsync").write_text(FAKE_RSYNC)
    (bindir / "rsync").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "FAKE_BOX": str(tmp_path / "box"),
           "FAKE_ARGV": str(tmp_path / "argv")}
    prelude = ('say(){ echo "SAY $*"; }\nHOST=box.example; PORT=2222; W=/root/tc1; E4B_RENT_SSH_OPTS="-o UserKnownHostsFile=/x"\n'
               f'RUN_DIR="{tmp_path}/run"; PARTIAL_AT=2026-10-23T17:00:00Z\n')
    script = (prelude + _cut(r"^TC1_RSYNC_EXCLUDES=\(.*?\)\n") + _cut(r"^tc1_partial_fetch\(\) \{.*?^\}\n")
              + 'tc1_partial_fetch "$RUN_DIR/tc1.partial"\n' + _cut(r"^if ! rsync -az .*?^rm -rf \"\$RUN_DIR/tc1\.partial\"\n"))
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, check=False, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    calls = [c.split("\n") for c in (tmp_path / "argv").read_text().split("---\n") if c.strip()]
    assert len(calls) == 2, calls
    assert all("--no-links" in c for c in calls), calls
    assert (tmp_path / "run" / "tc1" / "summary.txt").exists()
