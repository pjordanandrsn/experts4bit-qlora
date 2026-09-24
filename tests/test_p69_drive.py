"""Lane P69's controller script (bench/p69/p69_drive.sh; bench/p69/P69-PREREG.md): it refuses to run outside the
launcher's environment, its refusals carry the registered exit codes, and it reads the probe fields the registration
names -- checked without a box, by grepping the script and running it with the environment cleared."""
import os
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bench" / "p69" / "p69_drive.sh"
PREREG = REPO / "bench" / "p69" / "P69-PREREG.md"


def test_it_refuses_without_the_launcher_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("E4B_RENT_") and k != "GNF4_SHA"}
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env)
    assert r.returncode == 78 and "refusing" in r.stdout


def test_a_missing_gnf4_pin_is_refused_too():
    env = {k: v for k, v in os.environ.items() if k != "GNF4_SHA"}
    env.update({"E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_RUN_DIR": "/nonexistent/x",
                "E4B_RENT_RUN_ID": "t", "E4B_RENT_INSTANCE_ID": "0"})
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env)
    assert r.returncode == 78 and "GNF4_SHA" in r.stdout


def test_the_registered_exit_codes_and_probe_fields_are_the_scripts():
    s = SCRIPT.read_text()
    for code in ("exit 78", "exit 15", "exit 20", "exit 21", "exit 22"):
        assert code in s, code
    for field in ("h2d_64mb_single", "h2d_64mb", "--skip-cpu", "pcie.link.gen.current"):
        assert field in s, field
    assert "RTX 5090" in s                       # the card-class refusal the registration names
    assert re.search(r"for r in 1 2", s)         # two runs, as registered
    p = PREREG.read_text()
    for field in ("h2d_64mb_single", "h2d_64mb", "[0.55, 0.75]", "exit 78", "15 the card"):
        assert field in p, field
