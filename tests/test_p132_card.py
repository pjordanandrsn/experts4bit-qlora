# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P132 Amendment 2 (#1313; bench/p132/PREREG-p132.md): the card-identity refusal, tested on real records first.

P132's rule reads absolute milliseconds against sd2-5090-3. That run's card was GPU-38c66e15. P116's matched pair
showed another RTX 5090 running 13-15 % faster at B = 1, so the box refuses any other card before it installs or
fetches anything (rc 18). The records the check reads are:
- sd2-5090-3's forensics.txt, committed in bench/sd2/receipts/sd2-5090-3 with SHA256SUMS: it must pass;
- p132-5090-1's forensics.txt (tests/fixtures/p132_canary/, byte-identical to that run's receipt), from a different
  card: it must refuse."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN = REPO / "bench" / "p132" / "p132_run.sh"
SD2 = REPO / "bench" / "sd2" / "receipts" / "sd2-5090-3"
FIX = REPO / "tests" / "fixtures" / "p132_canary"
RECORDS = {   # name -> (path, sha256, the card in its first line)
    "sd2-5090-3": (SD2 / "forensics.txt", "49acadba5416519b2d841955f555c67368f7172721aeccc950f537026ac4996a",
                   "GPU-38c66e15-7fac-e3bc-9760-a49deb41e4d6"),
    "p132-5090-1": (FIX / "forensics.txt", "8937620ac7926dec900fa58d1fe2c6978fcadcc953b2c45bc51c54733dc450eb",
                    "GPU-280958b0-9ceb-4341-a905-2398d664f1c3"),
}


def _record(name):
    path, want, _ = RECORDS[name]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == want, f"{path} is not {name}'s record"
    return raw


def _block():
    """The Amendment 2 check as p132_run.sh runs it: from REF_UUID= through the rc 18 refusal."""
    lines = RUN.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("REF_UUID="))
    end = next(i for i, ln in enumerate(lines) if "finish 18" in ln)
    return "\n".join(lines[start:end + 1])


def test_the_records_are_the_runs():
    sums = (SD2 / "SHA256SUMS").read_text(encoding="ascii")
    assert f"{RECORDS['sd2-5090-3'][1]} *forensics.txt" in sums          # the committed receipt vouches for it
    for name in RECORDS:
        _record(name)


def test_the_reference_is_sd2_5090_3s_card_and_the_runner_reads_field_4():
    run = RUN.read_text(encoding="utf-8")
    assert f"REF_UUID={RECORDS['sd2-5090-3'][2]}" in run
    assert "card_uuid(){ awk -F', ' 'NR == 1 {print $4}' \"$1\"; }" in run
    for name, (_, _, card) in RECORDS.items():
        assert _record(name).decode("ascii").splitlines()[0].split(", ")[3] == card     # field 4 is the uuid
    assert RECORDS["p132-5090-1"][2] != RECORDS["sd2-5090-3"][2]


def test_the_refusal_comes_before_anything_is_installed_or_fetched():
    run = RUN.read_text(encoding="utf-8")
    i_class, i_card = run.index("finish 15"), run.index("finish 18")
    assert i_class < i_card < run.index("pipx(){") < run.index("--mode canary")
    assert run.count("finish 18") == 1


@pytest.mark.parametrize("name,rc,match", [("sd2-5090-3", 0, True), ("p132-5090-1", 18, False)])
def test_the_runners_check_on_the_real_forensics(name, rc, match, tmp_path):
    if sys.platform == "win32" or shutil.which("bash") is None or shutil.which("awk") is None:
        pytest.skip("the runner's check needs a POSIX bash and awk")
    (tmp_path / "forensics.txt").write_bytes(_record(name))
    script = ('say(){ echo "SAY $*"; }\nfinish(){ echo "FINISH $1"; exit "$1"; }\n: > summary.txt\n'
              + _block() + "\nfinish 0\n")
    out = subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert out.returncode == rc, out.stdout + out.stderr
    card = json.loads((tmp_path / "card.json").read_text())
    assert card == {"uuid": RECORDS[name][2], "reference": RECORDS["sd2-5090-3"][2], "match": match}
    assert f"P132_CARD uuid={RECORDS[name][2]}" in (tmp_path / "summary.txt").read_text()
    if rc:
        assert (tmp_path / "REFUSAL").read_text().strip() == f"refused: card {RECORDS[name][2]}"
