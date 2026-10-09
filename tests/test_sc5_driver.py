# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC5's closed-loop driver (bench/sc5/sc5_driver.py): its self-test, and SC2's driver left byte-identical.

``bench/sc2/sc2_driver.py`` is pinned by ``bench/sc1/staged.sha256`` and ``bench/ra/source-pins.json``; SC5 imports it
rather than editing it, and this test fails if the two pins and the file ever disagree.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_closed_loop_driver_self_tests():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc5" / "sc5_driver.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "sc5_driver self-test OK" in out.stdout, out.stdout + out.stderr


def test_sc2_driver_is_left_at_its_pinned_bytes():
    got = hashlib.sha256((REPO / "bench" / "sc2" / "sc2_driver.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    ra = json.loads((REPO / "bench" / "ra" / "source-pins.json").read_text(encoding="utf-8"))
    assert got in json.dumps(ra), "bench/sc2/sc2_driver.py no longer matches RA's source pin"
    sc1 = (REPO / "bench" / "sc1" / "staged.sha256").read_text(encoding="utf-8")
    assert f"{got}  sc2_driver.py" in sc1, "bench/sc2/sc2_driver.py no longer matches bench/sc1/staged.sha256"
