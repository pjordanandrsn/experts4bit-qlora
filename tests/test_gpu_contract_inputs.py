# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/gpu-contract's controller refuses a malformed GC_E4B_TESTS / GC_NEED_E4B_S before any connection.

Each case runs `gc_drive.sh` with `GC_DRIVE_DRYRUN=1` (it prints the box's start command instead of connecting). A refusal
here costs nothing; the same input reaching a rented box would be refused only after the rental was paid, or worse:
- a glob (`tests/test_*.py`) used to expand in the launch tree into real paths, pass, and reach the box as the literal;
- a value of blanks only would reach the box as no file at all, and pytest would collect every test in the repository;
- a leading zero (`0900`) is octal in the box's `$(( ))`, which exits without the TP_DONE marker the driver polls for.
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVE = os.path.join(REPO, "bench", "gpu-contract", "gc_drive.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _drive(tmp_path, **env):
    full = {**os.environ,
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q",
            "E4B_RENT_RUN_DIR": str(tmp_path / "run"), "E4B_RENT_RUN_ID": "gc-inputs-1",
            "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "GC_DRIVE_DRYRUN": "1",
            "GC_GNF4_SHA": "a" * 40, "GC_E4B_SHA": "b" * 40, "GC_CARD": "8.9/128", **env}
    r = subprocess.run(["bash", DRIVE], cwd=REPO, env=full, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("env, why", [
    ({"GC_E4B_TESTS": "tests/test_gpu_contract_*.py"}, "is not tests/test_<name>.py"),
    ({"GC_E4B_TESTS": "tests/../evil.py"}, "is not tests/test_<name>.py"),
    ({"GC_E4B_TESTS": "   "}, "names no file"),
    ({"GC_E4B_TESTS": "tests/test_epilogue_contract.py\ntests/test_fast_v4.py"}, "holds a newline"),
    ({"GC_NEED_E4B_S": "0900"}, "without a leading zero"),
    ({"GC_NEED_E4B_S": "1801"}, "without a leading zero"),
    ({"GC_NEED_E4B_S": "6O0"}, "without a leading zero"),
])
def test_the_controller_refuses(tmp_path, env, why):
    rc, out = _drive(tmp_path, **env)
    assert rc == 78 and why in out, out


def test_a_valid_list_reaches_the_box_escaped(tmp_path):
    rc, out = _drive(tmp_path, GC_E4B_TESTS="tests/test_gpu_contract_inputs.py tests/test_epilogue_contract.py",
                     GC_NEED_E4B_S="1200")
    if "launch tree" in out and "dirty" in out:
        pytest.skip("the working tree is dirty; the driver refuses before printing the start command")
    assert rc == 0, out
    assert r"GC_E4B_TESTS=tests/test_gpu_contract_inputs.py\ tests/test_epilogue_contract.py" in out
    assert "GC_NEED_E4B_S=1200" in out
