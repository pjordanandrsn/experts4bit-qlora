"""Lane DQ5 (bench/dq5/DQ5-PREREG.md): DQ3's lane on a gen-4 x16 link. The runner is DQ3's with a gen-4 gate that refuses
anything else at rc 19 (not an admitted machine-evidence code) BEFORE any install, plus a descriptive H2D probe that never
refuses; the arms, the palindrome and the rule are DQ3's files. CPU-only."""
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
DQ5 = REPO / "bench" / "dq5"
DQ3 = REPO / "bench" / "dq3"
RUN = (DQ5 / "dq5_run.sh").read_text()
DQ3_RUN = (DQ3 / "dq3_run.sh").read_text()


def _body(text):
    return text[text.index("set -uo pipefail"):]


def test_the_runner_is_dq3s_but_for_the_link_gate_and_the_h2d_probe():
    expected = _body(DQ3_RUN)
    expected = expected.replace('W=${DQ3_W:-/root/tc1}; cd "$W" || exit 9   # DQ3_W: tests only',
                                'W=${DQ5_W:-/root/tc1}; cd "$W" || exit 9   # DQ5_W: tests only')
    expected = expected.replace('dq3: $*"', 'dq5: $*"')
    expected = expected.replace(
        'case "$link" in *"RTX 5090, 5, 16") ;; *) echo "HOST REFUSED: not an RTX 5090 on PCIe gen 5 x16 ($link)" | tee -a summary.txt; finish 13;; esac',
        'case "$link" in *"RTX 5090, 4, 16") ;; *) echo "OUT OF BAND: not an RTX 5090 on PCIe gen 4 x16 ($link)" | tee -a summary.txt; finish 19;; esac')
    got = _body(RUN)
    probe = got[got.index("# Descriptive only:"):got.index("\ncommand -v git")]
    gate = got[got.index("# width.max is what the slot CAN negotiate"):got.index("# Host floor")]
    assert got.replace(probe, "").replace(gate, "") == expected, \
        "dq5_run.sh drifted from dq3_run.sh beyond the registered differences"
    assert "python dq5_link_gate.py" in gate and 'finish 19' in gate and 'finish 9' in gate
    assert "dq5_h2d_probe.py receipts/h2d.json" in probe and "not a refusal" in probe and "finish" not in probe
    for s in ("for arm in R S S0 S0 S R; do", "dq3_arm.py", "dq3_reduce.py", "pcie.link.gen.max,pcie.link.width.max"):
        assert s in RUN, s
    assert subprocess.run(["bash", "-n", str(DQ5 / "dq5_run.sh")]).returncode == 0


def _run_box(tmp_path, link, vram_rc=0, gate_rc=0):
    b = tmp_path / "bin"
    b.mkdir(parents=True)
    (b / "nvidia-smi").write_text(f"#!/bin/sh\necho '{link}'\n")
    (b / "python").write_text("#!/bin/sh\n"
                              f'case "$*" in *dq5_link_gate.py*) exit {gate_rc};; *dq3_vram_probe.py*) exit {vram_rc};; '
                              '*dq3_egress_probe.py*) exit 0;; '
                              '*dq5_h2d_probe.py*) exit 2;; esac\n'
                              'echo "$*" >> "$DQ5_W/reached"; exit 1\n')
    for f in b.iterdir():
        f.chmod(0o755)
    w = tmp_path / "w"
    w.mkdir()
    env = {"PATH": f"{b}:/usr/bin:/bin", "DQ5_W": str(w), "TC1_RUN_NONCE": "n", "E4B_SHA": "0" * 40}
    rc = subprocess.run(["bash", str(DQ5 / "dq5_run.sh")], env=env, capture_output=True, text=True).returncode
    return rc, w


def test_a_host_outside_gen4_x16_is_out_of_band_at_19_before_any_install(tmp_path):
    for i, link in enumerate(("NVIDIA GeForce RTX 5090, 5, 16", "NVIDIA GeForce RTX 5090, 4, 8",
                              "NVIDIA GeForce RTX 5090, 3, 16", "NVIDIA RTX A2000 12GB, 4, 16")):
        rc, w = _run_box(tmp_path / str(i), link)
        assert rc == 19, (link, rc)
        assert not (w / "reached").exists() and not (w / "REFUSAL").exists(), link
        assert (w / "TC1_EXIT_CODE.n").read_text().strip() == "19" and (w / "TP_DONE.n").exists()


def test_19_is_not_a_code_adertha_admits_as_machine_evidence():
    assert "finish 13" not in RUN.split("# Host floor")[0], "the link gate must not use an admitted code"


def test_a_gen4_x16_host_proceeds_and_a_failing_h2d_probe_never_refuses(tmp_path):
    rc, w = _run_box(tmp_path, "NVIDIA GeForce RTX 5090, 4, 16")
    assert rc == 9 and (w / "reached").exists(), "a sound gen-4 host goes on to the install (which the fake fails)"
    assert "H2D PROBE DID NOT COMPLETE" in (w / "summary.txt").read_text()
    rc, w = _run_box(tmp_path / "v", "NVIDIA GeForce RTX 5090, 4, 16", vram_rc=3)
    assert rc == 18, "DQ3's host floor still applies"


def test_a_link_running_narrower_than_x16_under_load_is_out_of_band(tmp_path):
    """width.max read 16 on the A2000 while it ran x8: the gate reads the negotiated width under load."""
    rc, w = _run_box(tmp_path, "NVIDIA GeForce RTX 5090, 4, 16", gate_rc=3)
    assert rc == 19 and not (w / "reached").exists() and not (w / "REFUSAL").exists()
    rc, w = _run_box(tmp_path / "e", "NVIDIA GeForce RTX 5090, 4, 16", gate_rc=2)
    assert rc == 9 and not (w / "REFUSAL").exists(), "an unreadable link is a harness error, not a host refusal"
