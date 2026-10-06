"""Lane DQ6 (bench/dq6/DQ6-PREREG.md): DQ4's capacity read on a 24 GB RTX 4090. The runner is DQ4's with the registered
differences only -- a card gate that refuses anything but a 24 GB RTX 4090 at rc 19 (not an admitted machine-evidence code)
before any install, DQ3's VRAM probe at a 24 GB floor, a 512-token ladder, and DQ4's reducer re-registered to the 4090.
CPU-only."""
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DQ6 = REPO / "bench" / "dq6"
DQ4 = REPO / "bench" / "dq4"
RUN = (DQ6 / "dq6_run.sh").read_text()
DQ4_RUN = (DQ4 / "dq4_run.sh").read_text()

REGISTERED = [
    ('W=${DQ4_W:-/root/tc1}; cd "$W" || exit 9   # DQ4_W: tests only', 'W=${DQ6_W:-/root/tc1}; cd "$W" || exit 9   # DQ6_W: tests only'),
    ('dq4: $*"', 'dq6: $*"'),
    ('STEP=1024\n', 'START=512; STEP=512   # a 24 GB card: R is predicted near 1.5-2k tokens, so the ladder starts low and steps finely\n'),
    ('--query-gpu=name,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits 2>&1 | head -1)',
     '--query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits 2>&1 | head -1)'),
    ('python dq3_vram_probe.py > logs/vram_probe.log', 'python dq6_vram_probe.py > logs/vram_probe.log'),
    ('print("dq4 tripwire OK"', 'print("dq6 tripwire OK"'),
    ('python dq4_reduce.py --self-test', 'python dq6_reduce.py --self-test'),
    ('--step "$STEP" --seq "$seq"', '--start "$START" --step "$STEP" --seq "$seq"'),
    ('python dq4_reduce.py receipts > receipts/dq4_read.json', 'python dq6_reduce.py receipts > receipts/dq6_read.json'),
    ("d=json.load(open('receipts/dq4_read.json')); print('DQ4 read:'", "d=json.load(open('receipts/dq6_read.json')); print('DQ6 read:'"),
]


def _body(text):
    return text[text.index("set -uo pipefail"):]


def test_the_runner_is_dq4s_but_for_the_registered_differences():
    expected = _body(DQ4_RUN)
    for a, b in REGISTERED:
        assert expected.count(a) == 1, a
        expected = expected.replace(a, b)
    old_gate = next(ln for ln in expected.splitlines() if ln.startswith('case "$link" in'))
    got = _body(RUN)
    gate = got[got.index("# The CARD, not the link"):]
    gate = gate[:gate.index("\n", gate.index('case "$link" in')) + 1]
    assert got.replace(gate, old_gate + "\n") == expected, "dq6_run.sh drifted from dq4_run.sh beyond the registered differences"
    assert "finish 19" in gate and "finish 13" not in gate
    for s in ("dq4_cap.py", "for spec in", "c_def chunked default 1", "Qwen3ForCausalLM"):
        assert s in RUN, s
    assert subprocess.run(["bash", "-n", str(DQ6 / "dq6_run.sh")]).returncode == 0


def _run_box(tmp_path, link, vram_rc=0):
    b = tmp_path / "bin"
    b.mkdir(parents=True)
    (b / "nvidia-smi").write_text(f"#!/bin/sh\necho '{link}'\n")
    (b / "python").write_text("#!/bin/sh\n"
                              f'case "$*" in *dq6_vram_probe.py*) exit {vram_rc};; *dq3_vram_probe.py*) exit 99;; '
                              '*dq3_egress_probe.py*) exit 0;; esac\n'
                              'echo "$*" >> "$DQ6_W/reached"; exit 1\n')
    for f in b.iterdir():
        f.chmod(0o755)
    w = tmp_path / "w"
    w.mkdir()
    env = {"PATH": f"{b}:/usr/bin:/bin", "DQ6_W": str(w), "TC1_RUN_NONCE": "n", "E4B_SHA": "0" * 40}
    rc = subprocess.run(["bash", str(DQ6 / "dq6_run.sh")], env=env, capture_output=True, text=True).returncode
    return rc, w


def test_anything_but_a_24gb_4090_is_out_of_band_at_19_before_any_install(tmp_path):
    for i, link in enumerate(("NVIDIA GeForce RTX 5090, 32607, 5, 16", "NVIDIA GeForce RTX 4090, 49140, 4, 16",
                              "NVIDIA GeForce RTX 4090 D, 24564, 4, 16", "NVIDIA RTX A2000 12GB, 12282, 4, 16",
                              "NVIDIA GeForce RTX 4090, 2456, 4, 16")):
        rc, w = _run_box(tmp_path / str(i), link)
        assert rc == 19, (link, rc)
        assert not (w / "reached").exists() and not (w / "REFUSAL").exists(), link
        assert (w / "TC1_EXIT_CODE.n").read_text().strip() == "19" and (w / "TP_DONE.n").exists()


def test_a_24gb_4090_proceeds_on_any_link_and_the_vram_floor_still_refuses(tmp_path):
    for i, link in enumerate(("NVIDIA GeForce RTX 4090, 24564, 4, 16", "NVIDIA GeForce RTX 4090, 24564, 3, 8")):
        rc, w = _run_box(tmp_path / str(i), link)
        assert rc == 9 and (w / "reached").exists(), (link, rc)          # on to the install, which the fake fails
    rc, w = _run_box(tmp_path / "v", "NVIDIA GeForce RTX 4090, 24564, 4, 16", vram_rc=3)
    assert rc == 18 and (w / "REFUSAL").exists(), "DQ3's host floor still names the host"


def test_the_vram_probe_is_dq3s_at_a_24gb_floor():
    sys.path[:0] = [str(DQ6), str(REPO / "bench" / "dq3")]
    try:
        import dq3_vram_probe
        import dq6_vram_probe
    finally:
        del sys.path[:2]
    assert dq6_vram_probe.P is dq3_vram_probe
    assert dq3_vram_probe.FIRST_GIB < dq6_vram_probe.TOTAL_GIB < 23.5 and dq3_vram_probe.TOTAL_GIB == 28.0
    assert "P.TOTAL_GIB = TOTAL_GIB" in (DQ6 / "dq6_vram_probe.py").read_text()


def test_the_reducer_self_test_passes():
    r = subprocess.run([sys.executable, str(DQ6 / "dq6_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0 and "dq6 self-test OK" in r.stdout, r.stdout + r.stderr


def test_dq4s_own_5090_receipts_are_void_under_dq6():
    r = subprocess.run([sys.executable, str(DQ6 / "dq6_reduce.py"), str(DQ4 / "receipts" / "dq4-5090-2")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["schema"] == "dq6-read/1" and d["verdicts"]["lane"] == "VOID"
    assert any("NVIDIA GeForce RTX 5090" in v for v in d["configs"]["c_def"]["void"])


def test_dq4s_reducer_is_untouched_by_dq6():
    """DQ6 re-registers DQ4's rule by patching an imported module in its own process; DQ4's committed read must still
    re-derive byte for byte from DQ4's own reducer."""
    d = DQ4 / "receipts" / "dq4-5090-2"
    r = subprocess.run([sys.executable, str(DQ4 / "dq4_reduce.py"), str(d)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == (d / "dq4_read.json").read_text()
