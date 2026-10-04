"""Lane P55 (e4b#344): the staged-file pin, Amendment 1's memory arithmetic, and the reducer on synthetic receipts.

`bench/p55/p55_drive.sh` refuses on the controller, after a box is rented, when a staged file's sha256 differs from
`bench/p55/staged.sha256`. This test runs that comparison in CI, where it costs nothing.

It also tests Amendment 1's three fixes, which the registration could not have caught on a rented box:
- STOP-1 reads the memory a process in the container can have, min(MemTotal, the cgroup limit), not MemTotal. Inside
  a Vast container MemTotal is the whole host's.
- P4's headroom is min(MemAvailable, limit - usage) for the same reason.
- The largest shard is 49,907,246,508 bytes, 46.48 GiB. The registration's "49.9 GiB" was GB.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p55"
PIN = LANE / "staged.sha256"
RUN = (LANE / "p55_run.sh").read_text()
DRIVE = (LANE / "p55_drive.sh").read_text()
sys.path.insert(0, str(LANE))
import p55_ram  # noqa: E402


def _entries():
    for line in PIN.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256((LANE / name).read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the launch refuses ON A RENTED BOX"


def test_every_pinned_file_is_staged_by_the_driver():
    staged = {n for _w, n in _entries()}
    assert staged == {"p55_run.sh", "p55_ram.py"}
    for name in staged:
        assert f"$HERE/{name}" in DRIVE, f"{name} is pinned but p55_drive.sh does not stage it"


def test_stop1_reads_the_effective_memory_not_memtotal():
    assert "python3 p55_ram.py --class-max-gib" in RUN
    assert "MEM_GIB=$(( ${MEM_KB" not in RUN, "the MemTotal-only STOP-1 is gone"
    assert "python3 p55_ram.py --sample" in RUN, "the C_headroom trace samples the cgroup beside MemAvailable"
    assert "ulimit -a" in RUN


def test_the_install_brings_the_loaders_stack_and_a_tripwire_proves_it():
    """Amendment 2: e4b's BASE dependencies are torch and bitsandbytes only. The registered bare install could never
    load the model, and p55-5090-1 died at the fetch on `No module named 'huggingface_hub'`."""
    assert 'pip install -q "experts4bit-qlora @' not in RUN, "the bare install is gone"
    assert RUN.index("command -v git") < RUN.index("pipx logs/pip_e4b.log"), "git is ensured before pip needs it"
    for pin in ('"transformers==5.17.0"', '"bitsandbytes==0.50.2"', "safetensors", '"huggingface_hub>=0.23"', "accelerate"):
        assert pin in RUN, pin
    assert "from experts4bit_qlora.loader import load_moe_4bit_streaming" in RUN
    assert 'grep -q "^tripwire OK:" summary.txt' in RUN and "TRIPWIRE FAIL" in RUN
    assert RUN.index("tripwire OK") < RUN.index("snapshot_download") < RUN.index("run_arm A_baseline"), \
        "install and tripwire, then the fetch, then the arms"


def test_the_checkpoint_is_fetched_before_the_arms_bounded_and_without_xet():
    """Amendment 1, defect 5: registered, the first arm downloaded 51.6 GB inside its own load -- unbounded, on the Xet
    backend that wedges on this fleet. The fetch now precedes every arm, under an alarm, and failing it is exit 11."""
    fetch = RUN.index("snapshot_download")
    assert RUN.index("export HF_HUB_DISABLE_XET=1") < fetch < RUN.index("run_arm A_baseline")
    assert 'perl -e "alarm $FETCH_S; exec @ARGV"' in RUN and "finish 11" in RUN[fetch:fetch + 600]


def _world(tmp_path, *, total_kib, avail_kib=None, v2=None, v1=None, usage=None):
    mi = tmp_path / "meminfo"
    mi.write_text(f"MemTotal: {total_kib} kB\nMemFree: 1 kB\n" + (f"MemAvailable: {avail_kib} kB\n" if avail_kib else ""))
    cg = tmp_path / "cg"
    cg.mkdir()
    if v2 is not None:
        (cg / "memory.max").write_text(f"{v2}\n")
        if usage is not None:
            (cg / "memory.current").write_text(f"{usage}\n")
    if v1 is not None:
        (cg / "memory").mkdir()
        (cg / "memory" / "memory.limit_in_bytes").write_text(f"{v1}\n")
        if usage is not None:
            (cg / "memory" / "memory.usage_in_bytes").write_text(f"{usage}\n")
    return str(mi), str(cg)


def _report(capsys, *args):
    assert p55_ram.main(list(args)) == 0
    return dict(line.split(": ", 1) for line in capsys.readouterr().out.strip().splitlines())


GIB = 1 << 30


@pytest.mark.parametrize("v2,v1,want_eff,want_class", [
    (63 * GIB, None, 63.0, "1"),        # a Vast 63 GB allotment on a 512 GB host: in the class
    ("max", None, 512.0, "0"),          # no v2 limit: the host's RAM, not the class
    (None, 63 * GIB, 63.0, "1"),        # cgroup v1 limit
    (None, (1 << 63) - 4096, 512.0, "0"),  # v1's "no limit"
    (None, None, 512.0, "0"),           # nothing readable: MemTotal stands
])
def test_effective_memory_is_the_smaller_of_memtotal_and_the_cgroup_limit(tmp_path, capsys, v2, v1, want_eff, want_class):
    mi, cg = _world(tmp_path, total_kib=512 * 1048576, v2=v2, v1=v1)
    r = _report(capsys, "--meminfo", mi, "--cgroup-dir", cg, "--class-max-gib", "72")
    assert float(r["effective_ram_gib"]) == want_eff and r["class_drawn"] == want_class, r


def test_a_small_host_without_a_cgroup_is_still_in_the_class(tmp_path, capsys):
    mi, cg = _world(tmp_path, total_kib=64 * 1048576)
    r = _report(capsys, "--meminfo", mi, "--cgroup-dir", cg)
    assert r["class_drawn"] == "1" and float(r["effective_ram_gib"]) == 64.0


def test_unreadable_memtotal_is_a_harness_failure_not_a_class(tmp_path, capsys):
    cg = tmp_path / "cg"
    cg.mkdir()
    assert p55_ram.main(["--meminfo", str(tmp_path / "absent"), "--cgroup-dir", str(cg)]) == 2


def test_the_shard_is_measured_in_gib():
    assert p55_ram.SHARD_BYTES == 49_907_246_508
    assert round(p55_ram.SHARD_GIB, 2) == 46.48


def test_headroom_takes_the_cgroup_term_when_both_halves_are_known():
    assert p55_ram.headroom_bytes(400 * 1048576, 60 * GIB, 63 * GIB) == 3 * GIB          # host has plenty; cgroup does not
    assert p55_ram.headroom_bytes(10 * 1048576, None, 63 * GIB) == 10 * GIB              # no usage: MemAvailable only
    assert p55_ram.headroom_bytes(None, None, None) is None


def test_sample_row_has_the_trace_columns(tmp_path, capsys):
    mi, cg = _world(tmp_path, total_kib=512 * 1048576, avail_kib=7, v2=63 * GIB, usage=5 * GIB)
    assert p55_ram.main(["--sample", "--meminfo", mi, "--cgroup-dir", cg]) == 0
    cells = capsys.readouterr().out.strip().split(",")
    assert len(cells) == 5 and cells[1] == "7" and cells[3] == str(5 * GIB) and cells[4] == str(63 * GIB)


def _receipts(tmp_path, *, class_drawn, c_status, trace_rows):
    d = tmp_path / "p55"
    (d / "logs").mkdir(parents=True)
    (d / "forensics.txt").write_text("NVIDIA GeForce RTX 5090, 32607 MiB, 580.95.05\nMemTotal: 536870912 kB\n"
                                     f"cgroup_limit(/sys/fs/cgroup/memory.max): {63 * GIB}\n"
                                     "cgroup_limit_source: /sys/fs/cgroup/memory.max\n"
                                     f"effective_ram_gib: 63.0\ncgroup_limit_bytes: {63 * GIB}\n")
    (d / "class_drawn.txt").write_text(f"{class_drawn}\n")
    fail = {"status": "FAILED", "exc_type": "RuntimeError", "exc_msg": "CUDA error: invalid argument", "tb": "safe_open"}
    for arm in ("A_baseline", "B_sync"):
        (d / f"result_{arm}.json").write_text(json.dumps(fail))
    (d / "result_C_headroom.json").write_text(json.dumps(dict(fail, status=c_status)))
    (d / "mem_trace.csv").write_text("epoch,mem_available_kb,mem_free_kb,cgroup_usage_bytes,cgroup_limit_bytes\n"
                                     + "".join(f"{i},{a},1,{u},{63 * GIB}\n" for i, (a, u) in enumerate(trace_rows)))
    return d


def _reduce(d):
    p = subprocess.run([sys.executable, str(LANE / "p55_reduce.py"), str(d)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_reducer_reads_p4_from_the_cgroup_headroom_not_the_hosts_memavailable(tmp_path):
    # The host's MemAvailable stays ~400 GiB all the way; the cgroup's usage climbs to 60 of 63 GiB.
    out = _reduce(_receipts(tmp_path, class_drawn=1, c_status="FAILED",
                            trace_rows=[(400 * 1048576, 10 * GIB), (400 * 1048576, 60 * GIB)]))
    assert "**P4 — headroom falls.** HOLDS — headroom bottomed at 3.0 GiB, below the 46.48 GiB shard." in out
    assert "| 63.0 GiB |" in out, "the host table shows the effective memory"
    assert f"| {63 * GIB} |" in out, "the cgroup column keeps the box's own limit line, not p55_ram.py's provenance line"


def test_reducer_refutes_p4_when_headroom_stays_above_the_shard(tmp_path):
    out = _reduce(_receipts(tmp_path, class_drawn=1, c_status="FAILED",
                            trace_rows=[(400 * 1048576, 1 * GIB), (400 * 1048576, 5 * GIB)]))
    assert "**P4 — headroom falls.** REFUTED" in out


def test_reducer_withholds_p1_when_the_class_was_not_drawn(tmp_path):
    out = _reduce(_receipts(tmp_path, class_drawn=0, c_status="FAILED", trace_rows=[(1, 1)]))
    assert "STOP-1 fired" in out and "**P1 — reproduction.** UNDECIDED" in out
