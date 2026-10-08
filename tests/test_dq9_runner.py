"""Exercise nonce failure markers and stop-before-reading semantics without CUDA or compute creation."""
import os
from pathlib import Path
import subprocess

import pytest

RUNNER = Path(__file__).parents[1]/"bench/dq9/dq9_run.sh"


@pytest.mark.parametrize("stop,expected", [("vram", 18), ("proof", 11), ("paired", 11), ("admission", 11)])
def test_preflight_failure_stops_before_capacity_process_and_preserves_nonce(tmp_path, stop, expected):
    commands = tmp_path/"bin"
    commands.mkdir()
    scripts = {
        "nvidia-smi": "#!/bin/sh\necho 'NVIDIA GeForce RTX 5090, 32768'\n",
        "df": "#!/bin/sh\necho 'Filesystem 1024-blocks Used Available Capacity Mounted'\necho 'fixture 320000000 0 320000000 0% /'\n",
        "sha256sum": "#!/bin/sh\nexit 0\n",
        "python": """#!/usr/bin/env python3
import os,sys
from pathlib import Path
args=sys.argv[1:]
with open('calls.txt','a') as f: f.write(' '.join(args)+'\\n')
stop=os.environ['FIXTURE_STOP']
if args[0]=='dq3_vram_probe.py' and stop=='vram': sys.exit(3)
if args[0]=='-': sys.stdin.read()
if args[0]=='-c': print('4ca5a3494e746b9aaa812270e03f4d0c1ec050a9')
if args[0]=='dq7_subject.py':
    assert args[1]=='tiny', 'full checkpoint built after preflight failure'
    Path(args[2]).mkdir(parents=True)
    (Path(args[2])/'dq7-subject.json').write_text('{}')
if args[0]=='dq9_proof.py' and stop=='proof': sys.exit(11)
if args[0]=='dq9_reduce.py' and stop=='paired': sys.exit(11)
if args[0]=='dq9_admission.py' and stop=='admission': sys.exit(11)
if args[0]=='dq9_arm.py': raise AssertionError('reading after preflight failure')
""",
    }
    for name, text in scripts.items():
        p = commands/name
        p.write_text(text)
        p.chmod(0o755)
    for subject in ("llama31_8b", "qwen3_14b", "qwen3_32b"):
        (tmp_path/(subject+".json")).write_text("{}")
    env = dict(os.environ, PATH=str(commands)+os.pathsep+os.environ["PATH"], DQ9_W=str(tmp_path),
               TC1_RUN_NONCE="fixture-dq9", E4B_SHA="4ca5a3494e746b9aaa812270e03f4d0c1ec050a9", FIXTURE_STOP=stop)
    result = subprocess.run(["bash", str(RUNNER)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, result.stdout+result.stderr
    assert (tmp_path/"TC1_EXIT_CODE.fixture-dq9").read_text().strip() == str(expected)
    assert (tmp_path/"TP_DONE.fixture-dq9").exists()
    assert not (tmp_path/"TC1_SUCCESS.fixture-dq9").exists()
    assert "dq9_arm.py" not in (tmp_path/"calls.txt").read_text()
