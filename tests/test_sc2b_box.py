"""Lane SC2b's box F (#846): the serving stack it runs is today's, and that is checked in the ENVIRONMENT a process
inherits, not in the text of a launch line.

SC2's read needed a correction (#1061): ``sc1_run.sh`` exported SC1's prefill route pins (``E4B_INT4_PREFILL=loop``,
``E4B_PAGED_PREFILL_ATTN=math``) to every box, and box E's e4b server inherited them while ``tests/test_sc2_box.py``
only checked that ``e4b_server_start``'s own text did not set them. These tests run the box script's scrub-and-export
block under bash and read what a child process actually sees.
"""
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()


def _child_env(box):
    """Run sc1_run.sh's scrub + route-pin block for BOX under bash, with both pins pre-set by the parent (as a stale
    shell would have them), and return the environment a child process inherits."""
    start = RUN.index("# every lever is unset; each arm names its own switches")
    end = RUN.index(": > summary.txt")
    block = RUN[start:end]
    script = f'BOX={box}\nexport E4B_INT4_PREFILL=batched E4B_PAGED_PREFILL_ATTN=bogus\n{block}\nenv\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)


def test_box_f_children_inherit_neither_prefill_route_pin():
    env = _child_env("F")
    assert "E4B_INT4_PREFILL" not in env and "E4B_PAGED_PREFILL_ATTN" not in env, \
        {k: v for k, v in env.items() if "PREFILL" in k}


def test_boxes_a_to_e_keep_sc1s_pins_as_they_ran():
    for box in "ABCDE":
        env = _child_env(box)
        assert env.get("E4B_INT4_PREFILL") == "loop" and env.get("E4B_PAGED_PREFILL_ATTN") == "math", (box, env)


def test_box_f_pins_todays_kernel_package_and_asserts_the_default_routes():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M) ;;' in RUN
    assert 'case "$BOX" in F) GNF4_SHA=5a887c48acc90207fdd30f2e9b23d21d62102b14;;' in RUN   # v0.38.0's commit
    assert 'hr._int4_prefill_mode_env() == "k19"' in RUN and 'pa._prefill_attn_mode_env() == "flash"' in RUN
    assert RUN.count('os.environ["TRIP_BOX"] in ("F", "G", "H", "I", "J", "K", "L")') == 2
    assert 'TRIP_BOX=$BOX "$PY" -' in RUN
