"""Every lane driver trusts the rented box's ssh host key from the RUN's file, which the launcher hands it.

Vast reuses proxy endpoints (`sshN.vast.ai:PORT`) across instances. With `StrictHostKeyChecking=accept-new` against
the controller's shared ~/.ssh/known_hosts, a key an earlier box pinned there for the same endpoint refuses the next
box to draw it -- sc1a-prove-4 and p68-5090-1 each paid for a box and ended NOT_RUN on `Host key ... has changed`.
The launcher (adertha-agents `rent.py`) now pins the box's key in `<run dir>/known_hosts` during the pre-flight and
hands `--command` the options that trust that file and nothing else, as `E4B_RENT_SSH_OPTS`. A driver that kept its
own options would stage, poll and fetch against the shared file again -- the pre-flight would pass and the lane would
then be refused, later and dearer -- so the options are the launcher's to give and the driver's to splice in, never
its own to choose.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVERS = sorted(p for p in (REPO / "bench").rglob("*_drive.sh") if "E4B_RENT_SSH_HOST" in p.read_text())
OPTS = "$E4B_RENT_SSH_OPTS"
# an ssh / scp invocation with options -- `ssh -o ...`, `scp -q ...`, `-e "ssh -o ..."` -- that is not text being echoed
CONNECTION = re.compile(r"(?<![\w./$-])(ssh|scp) +-")


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO)) for p in paths]


def test_the_drivers_that_read_the_launcher_endpoint_are_all_here():
    assert len(DRIVERS) >= 60, _ids(DRIVERS)   # a glob that matched nothing would make every test below vacuous


@pytest.mark.parametrize("driver", DRIVERS, ids=_ids(DRIVERS))
def test_the_driver_requires_the_launchers_host_key_options(driver: Path):
    text = driver.read_text()
    loop = re.search(r"^for v in ([^;\n]*\bE4B_RENT_SSH_HOST\b[^;\n]*); do", text, re.M)
    assert loop and "E4B_RENT_SSH_OPTS" in loop.group(1).split(), "the required-variable loop must name it"
    # the launcher owns host-key policy; a driver that names its own would override or contradict the run's file
    assert "StrictHostKeyChecking" not in text and "KnownHostsFile" not in text


@pytest.mark.parametrize("driver", DRIVERS, ids=_ids(DRIVERS))
def test_every_connection_to_the_box_carries_the_options(driver: Path):
    found = 0
    for n, line in enumerate(driver.read_text().splitlines(), 1):
        code = "" if line.lstrip().startswith("#") else line
        for m in CONNECTION.finditer(code):
            if "echo " in code[:m.start()]:
                continue                     # a DRYRUN line describing a command, not running one
            found += 1
            assert OPTS in code, f"{driver.relative_to(REPO)}:{n}: {line.strip()}"
    assert found, "no ssh/scp invocation recognised -- the pattern no longer fits this driver, so it checks nothing"


@pytest.mark.parametrize("driver", DRIVERS, ids=_ids(DRIVERS))
def test_without_the_options_the_driver_refuses_before_any_connection(driver: Path, tmp_path: Path):
    """An older launcher that does not export them gets a refusal (78) before the first ssh, scp or rsync."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    touched = tmp_path / "connected"
    for tool in ("ssh", "scp", "rsync"):
        (fake_bin / tool).write_text(f"#!/bin/sh\necho {tool} >> '{touched}'\nexit 0\n")
        (fake_bin / tool).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("E4B_RENT_")}
    env.update({"PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}", "HOME": str(tmp_path),
                "E4B_RENT_SSH_HOST": "ssh8.vast.ai", "E4B_RENT_SSH_PORT": "27278", "E4B_RENT_SSH": "ssh8.vast.ai:27278",
                "E4B_RENT_RUN_DIR": str(tmp_path / "run"), "E4B_RENT_RUN_ID": "known-hosts-refusal",
                "E4B_RENT_DEADLINE_EPOCH": "1", "E4B_RENT_INSTANCE_ID": "7000123", "E4B_RENT_WALLCLOCK_S": "60",
                "E4B_RENT_USD_PER_HOUR": "0.66", "E4B_RENT_EST_USD": "0.66", "E4B_RENT_PROVIDER": "vast:verified-secure"})
    r = subprocess.run(["bash", str(driver)], capture_output=True, text=True, env=env, timeout=60, cwd=tmp_path)
    assert r.returncode == 78 and "E4B_RENT_SSH_OPTS" in r.stdout, r.stdout + r.stderr
    assert not touched.exists(), touched.read_text()
