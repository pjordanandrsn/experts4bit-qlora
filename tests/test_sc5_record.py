# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC5's record assembler (bench/sc5/sc5_record.py): the box's block and quality files round-trip to the very record
the reducer's own fixture reduces to; a missing cell or block.json is a VOID, never filled in; a wrong reference hash
VOIDs every quality row -- through the module's self-test."""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_record_assembler_self_tests():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc5" / "sc5_record.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "sc5_record self-test OK" in out.stdout, out.stdout + out.stderr
