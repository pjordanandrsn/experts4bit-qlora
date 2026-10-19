"""The SMT fuse measurement's staged-file pin must match the repo (the e4b#642 check, mirrored from test_p146_staged_pin.py).

`bench/smt-fuse/smt_fuse_drive.sh` refuses to run when a staged file's sha256 differs from `bench/smt-fuse/staged.sha256`;
this test runs the same comparison in CI, runs the probe's self-test, and pins the measurement's shape: the pins, the
declared run ids, the refusals before the install, the interleaved pools, the byte-identity rule and the on-box cap.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "smt-fuse"
PIN = LANE / "staged.sha256"
SOURCES = {n: LANE / n for n in ("smt_fuse_run.sh", "smt_fuse_box.py", "smt_sib_box.py", "q235_bootstrap.py", "wheels.json",
                                  "dq11_require_git.sh")}
LOCK_SHA = "84f361ed35fff9a857f340b65488ad582dc4270aef7e67cbeb6b46b8370021f3"   # Q235's wheels.json at d65793e4 (bench/q235/q235.sha256)
RUN = (LANE / "smt_fuse_run.sh").read_text()
DRIVE = (LANE / "smt_fuse_drive.sh").read_text()
BOX = (LANE / "smt_fuse_box.py").read_text()
SIB = (LANE / "smt_sib_box.py").read_text()
PREREG = (LANE / "PREREG-smt-fuse.md").read_text(encoding="utf-8")
E4B_PIN = "d65793e4a106c7db6ecfc219668a2d6881d207e5"
GNF4_PIN = "724ccc454f006c1a46836e434e997f31f293747f"


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _drive(tmp_path, run_id, **extra):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "SMT_FUSE_DRIVE_DRYRUN": "1", **extra}
    return subprocess.run(["bash", str(LANE / "smt_fuse_drive.sh")], capture_output=True, text=True, env=env)


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next launch refuses ON A RENTED BOX"
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    assert "smt_fuse_box.py staged.sha256" in RUN and "sha256sum -c staged.sha256" in RUN


def test_the_self_test_passes():
    out = subprocess.run([sys.executable, str(LANE / "smt_fuse_box.py"), "--self-test"], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": ""})
    assert out.returncode == 0 and "smt_fuse_box self-test OK (6/6 cases)" in out.stdout, out.stdout + out.stderr


def test_the_pins_are_exact_and_agree():
    assert f"E4B_PIN={E4B_PIN}" in RUN and f"E4B_PIN={E4B_PIN}" in DRIVE and E4B_PIN in PREREG
    assert f"GNF4_SHA={GNF4_PIN}" in RUN and GNF4_PIN in PREREG
    assert '[ "$E4B_SHA" = "$E4B_PIN" ] ||' in RUN and "TORCH_WANT=2.11.0+cu128" in RUN and f"LOCK_SHA={LOCK_SHA}" in RUN
    assert hashlib.sha256((LANE / "wheels.json").read_bytes()).hexdigest() == LOCK_SHA          # Q235's lock, byte for byte
    assert "python $W/q235_bootstrap.py --directory $W" in RUN and "torch.__version__ == os.environ[\"WANT_TORCH\"]" in RUN
    assert 'md.version("transformers") == "5.19.0" and md.version("bitsandbytes") == "0.50.2"' in RUN


def test_the_declared_run_ids_and_nothing_else(tmp_path):
    for run_id, extra in (("smt-fuse-1", {}), ("smt-fuse-2", {}), ("smt-fuse-3", {}),
                          ("smt-sib-1", {"E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 9000)}),
                          ("smt-sib-2", {"E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 9000)}),
                          ("smt-sib-3", {"E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 7200)}),
                          ("smt-sib-4", {"E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 9000)}),       # Amendment 1a
                          ("smt-sib-8", {"E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 9000)}),
                          ("smt-fuse-dry-1", {"SMT_FUSE_DRY": "1", "SMT_FUSE_BOX_EXEC": "/bin/false"}),
                          ("smt-sib-dry-1", {"SMT_FUSE_DRY": "1", "SMT_FUSE_BOX_EXEC": "/bin/false"})):
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> "), (run_id, out.stdout + out.stderr)
        assert f"E4B_SHA={E4B_PIN}" in out.stdout, out.stdout
    for run_id, extra in (("smt-fuse-4", {}), ("smt-fuse-10", {}), ("p146-4090-6", {}), ("smt-fuse-dry-1", {}),
                          ("smt-sib-9", {}), ("smt-sib-dry-1", {}), ("smt-sib", {}),
                          ("smt-sib-1", {"E4B_RENT_DEADLINE_EPOCH": str(int(time.time()) + 3600)}),   # a 1.0 h guard: refused
                          ("smt-sib-1", {"SMT_FUSE_DRY": "1", "SMT_FUSE_BOX_EXEC": "/bin/false"}),
                          ("smt-fuse-1", {"SMT_FUSE_DRY": "1", "SMT_FUSE_BOX_EXEC": "/bin/false"}),
                          ("smt-fuse-1", {"E4B_SHA": "0" * 40})):
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 78 and "refusing" in out.stdout, (run_id, extra, out.stdout + out.stderr)
    a1a = PREREG[PREREG.index("### Amendment 1a"):]                          # new text since the 10-19 citation rule
    assert not re.search(r"receipt store `[0-9a-f]{7,40}`", a1a) and "$12.24" in a1a and "$1.53 per created box" in a1a
    short = _drive(tmp_path, "smt-sib-1", E4B_RENT_DEADLINE_EPOCH=str(int(time.time()) + 3600))
    assert short.returncode == 78 and "< the registered fit 7110s" in short.stdout, short.stdout   # the guard fit, not another refusal
    for s in ("smt-fuse-1", "smt-fuse-2", "smt-fuse-3", "smt-fuse-dry-", "smt-sib-1", "smt-sib-2", "smt-sib-3", "smt-sib-4", "smt-sib-8", "smt-sib-dry-"):
        assert s in PREREG, s


def test_the_refusals_come_before_the_install_and_the_probe():
    order = ["CUDA_PROBE=$(python -", 'say "REFUSED: not an SMT host', 'say "REFUSED: $AFF CPUs in the affinity',
             'say "REFUSED: quota $QUOTA gives a high arm', 'say "REFUSED: the host cannot carry the sibling arms',
             'say "REFUSED: ${RAM_GB', 'say "REFUSED: ${FREE_GB', "say \"install: Q235's wheel lock",
             "smt_fuse_box.py --self-test", "smt_sib_box.py --self-test", "smt_sib_box.py --out $W/result.json",
             "smt_fuse_box.py --out $W/result.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    codes = {int(c) for c in re.findall(r"finish (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18} and {9, 10, 16, 21, 22, 40} <= codes, codes
    assert "MIN_LOGICAL=32; MIN_P=24" in RUN and "EXPERTS=128" in RUN and "MIN_HIGH = 24" in BOX


def test_the_design_is_the_registered_design():
    assert "POOLS = (16, 32, 32, 16)" in BOX and "CYCLES = 4" in BOX
    assert "return min(want, affinity, math.ceil(quota) if quota else want)" in BOX    # the high arm under the quota cap
    assert "THROTTLE_FLAG = 0.05" in BOX and 'if k in p and not p["flagged"]' in BOX     # flagged pairs are not read
    assert "E_FULL, INTER, HIDDEN = 128, 1536, 4096" in BOX
    assert "fuse_experts(gate, up, down)" in BOX and 'quant_type="nf4"' in BOX and "torch.set_num_threads(pool)" in BOX
    assert 'return 0 if res["summary"]["bytes_identical"] else 3' in BOX                 # the pool must be speed-only
    assert "return min(time_left_s, max(CAP_FLOOR_S, 1.5 * sum(warm_by_pool[pool] for pool in plan)))" in BOX
    assert "CAP_FLOOR_S = 600" in BOX                                                      # the on-box phase cap
    for s in ("16, HIGH, HIGH, 16", "4 cycles", "8", "back-to-back", "min(32, ceil(quota), affinity)", "range", "on_box",
              "never a ruling", "Worst $0.64"):
        assert s in PREREG, s


def test_amendment_1_the_sibling_probe_is_the_registered_design():
    out = subprocess.run([sys.executable, str(LANE / "smt_sib_box.py"), "--self-test"], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": ""})
    assert out.returncode == 0 and "smt_sib_box self-test OK (7/7 cases)" in out.stdout, out.stdout + out.stderr
    assert 'ORDER = ("PHYS", "SIB", "WIDE", "WIDE", "SIB", "PHYS")' in SIB and "CYCLES = 4" in SIB
    assert "K_MAX, MIN_K = 16, 8" in SIB and "QUOTA_PER_K = 2.5" in SIB and 'READ = ("fuse_s",)' in SIB and 'REPORTED = ("block_s",)' in SIB
    assert "process_start_s = time.time() - a.spawned" in SIB and '"--spawned", repr(time.time())' in SIB
    assert "REPEATS = 15" in SIB and "GUARD_S = 9000" in SIB and "def time_fit(" in SIB
    assert "THROTTLE_FLAG = 0.05" in SIB and "COMPANION_FLAG = 0.25" in SIB and "MIN_TRIPLES = 6" in SIB
    # pinned before torch is imported, in a fresh process per rep, around the registered block unchanged
    child = SIB[SIB.index("def child(a):"):SIB.index("def flagged(r):")]
    assert child.index("os.sched_setaffinity(0, cpus)") < child.index("import smt_fuse_box as B")
    heavy = ("import torch", "import smt_fuse_box as B")                    # --plan and the child's pin stay torch-free
    assert not [x for x in SIB[:SIB.index("def child(a):")].splitlines() if x.strip() in heavy]
    assert all(child.index(f"\n    {h}\n") > child.index("os.sched_setaffinity(0, cpus)") for h in heavy)
    assert child.count("B.one_rep(a.pool, a.slab, a.experts)") == 2 and "for _ in range(REPEATS)" in child
    assert 'return 0 if res["summary"]["bytes_identical"] else 3' in SIB
    assert 'case "$SMT_FUSE_RUN_ID" in smt-sib-*) MODE=sib;; *) MODE=fuse;; esac' in RUN
    assert "--exclude '/venv*'" in DRIVE and "--exclude='./venv*'" in DRIVE          # the box venv never comes back
    for s in ("Amendment 1", "SIB/PHYS", "fresh process", "before torch is", "K = min(16, floor(quota / 2.5))",
              "PHYS, SIB, WIDE, WIDE, SIB, PHYS", "8 triples", "25 %", "VOID_BY_HOST", "15 times timed", "SIB/PHYS not resolved below about 0.80", "7110 s", "2.5 h", "about 3x its timed `block_s`", "The fit's terms are recorded on the box", "never a ruling",
              "Worst $1.53"):
        assert s in PREREG, s
