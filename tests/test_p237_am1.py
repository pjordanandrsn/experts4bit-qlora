"""P237 amendment 1 (pre-data): the host floors as the market and a container see them.

- rc 16 reads the RAM the process can use: min(MemTotal, every cgroup memory limit on its path). A 1 TB host held to 64 GiB by its
  cgroup refuses 16; the registered MemTotal-only line would have passed it.
- The pin screen maps Vast's ``cpu_name`` onto the registered CPU class (``vast_cpu_class``); the runner's lscpu regexes are unchanged.
  The table is real Vast names: the 2026-10-26 07:51Z RTX 4090 census and the host registry's ``offer_host.cpu_name``.
- The screen requires Vast's listed ``cpu_ram`` (the offer's share) >= the runner's floor.
- The shake's launch wall is the fit plus 1,500 s of launch slack: 18,000 s (5.0 h), worst USD 4.10, under the 6 h cap.
"""
from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
P = REPO / "bench" / "p237"
sys.path.insert(0, str(P))
host = importlib.import_module("p237_host")
RUN = (P / "p237_run.sh").read_text(encoding="utf-8")
PREREG = (P / "PREREG-p237.md").read_text(encoding="utf-8")
AM1 = re.sub(r"\s+", " ", PREREG[PREREG.index("### Amendment 1"):])
GIB = 2 ** 30

E5, RY, C14, CU = "epyc-zen5", "ryzen-7000-9000-desktop", "core-13-14-desktop", "core-ultra-desktop"


# -- the screen's CPU class: real Vast names, as Vast writes them (trailing spaces kept) --------------------------------------------
@pytest.mark.parametrize("name,cls", [
    # fast classes
    ("AMD EPYC 9115 16-Core Processor", E5), ("AMD EPYC 9135 16-Core Processor", E5), ("AMD EPYC 9655 96-Core Processor", E5),
    ("AMD EPYC 9755 128-Core Processor", E5),
    ("AMD Ryzen 5 7600 6-Core Processor", RY), ("AMD Ryzen 7 7800X3D 8-Core Processor", RY), ("AMD Ryzen 7 9700X 8-Core Processor", RY),
    ("AMD Ryzen 7 9800X3D 8-Core Processor", RY), ("AMD Ryzen 7 9850X3D 8-Core Processor", RY), ("AMD Ryzen 9 7900 12-Core Processor", RY),
    ("AMD Ryzen 9 7900X 12-Core Processor", RY), ("AMD Ryzen 9 7950X 16-Core Processor", RY), ("AMD Ryzen 9 7950X3D 16-Core Processor", RY),
    ("AMD Ryzen 9 9900X 12-Core Processor", RY), ("AMD Ryzen 9 9900X3D 12-Core Processor", RY), ("AMD Ryzen 9 9950X 16-Core Processor", RY),
    ("AMD Ryzen 9 9950X3D 16-Core Processor", RY),
    ("13th Gen Core™ i5-13600K", C14), ("13th Gen Core™ i9-13900F", C14), ("Core™ i5-14400F", C14), ("Core™ i5-14500", C14),
    ("Core™ i9-14900K", C14),
    ("Core™ Ultra 9 285K", CU), ("Core™ Ultra 7 270K Plus", CU),
    # EPYC Zen 4 (9xx4, the F and X parts included), Zen 2/3, and the cloud-named EPYCs: refused
    ("AMD EPYC 9174F 16-Core Processor", None), ("AMD EPYC 9354P 32-Core Emb Processor", None), ("AMD EPYC 9374F 32-Core Processor", None),
    ("AMD EPYC 9554 64-Core Processor", None), ("AMD EPYC 9654 96-Core Processor", None), ("AMD EPYC 9654 96-Core Emb Processor", None),
    ("AMD EPYC 9684X 96-Core Processor", None), ("AMD EPYC 9T24 96-Core Processor", None), ("AMD EPYC 9V74 80-Core Processor", None),
    ("AMD EPYC 9V84 96-Core Processor", None), ("AMD EPYC 7742 64-Core Processor", None), ("AMD EPYC 7B13 64-Core Processor", None),
    ("AMD EPYC 7K62 48-Core Processor", None), ("AMD EPYC 7R13 48-Core Processor", None), ("AMD EPYC 7V12 64-Core Processor", None),
    # older or workstation Ryzen, a mobile part, Threadripper: refused
    ("AMD Ryzen 5 3600 6-Core Processor", None), ("AMD Ryzen 5 5600 6-Core Processor", None), ("AMD Ryzen 7 5700X3D 8-Core Processor", None),
    ("AMD Ryzen 7 5800X3D 8-Core Processor", None), ("AMD Ryzen 9 5900XT 16-Core Processor", None),
    ("AMD Ryzen 9 5950X 16-Core Processor", None), ("AMD Ryzen 9 7945HX with Radeon Graphics", None),
    ("AMD Ryzen Threadripper 9980X 64-Cores", None), ("AMD Ryzen Threadripper PRO 7975WX 32-Cores", None),
    ("AMD Ryzen Threadripper PRO 5995WX 64-Cores", None),
    # older Core and every Xeon: refused
    ("Core™ i7-5820K ", None), ("Core™ i7-7700 ", None), ("Core™ i9-9900K ", None), ("Core™ i9-10940X ", None),
    ("Core™ i9-10980XE ", None), ("Xeon® Gold 6338 ", None), ("Xeon® Platinum 8481C", None), ("XEON® GOLD 6548Y+", None),
    ("Xeon® 6521P", None), ("Xeon® E5-2695 v4 ", None), ("", None), (None, None),
])
def test_the_screen_maps_vast_names_onto_the_registered_class(name, cls):
    assert host.vast_cpu_class(name) == cls


def test_the_screen_rule_is_the_runners_and_the_runners_is_unchanged():
    """No copy of the rule: vast_cpu_class re-spells the marks and calls cpu_class; CPU_CLASSES are the registered patterns."""
    import ast
    import inspect
    fn = ast.parse(inspect.getsource(host.vast_cpu_class))
    called = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "cpu_class" in called and "re" not in {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert [(c, r.pattern) for c, r in host.CPU_CLASSES] == [
        (E5, r"\bAMD EPYC 9\d\d5[A-Z]?\b"),
        (RY, r"\bAMD Ryzen [3579] [79]\d\d\d(X3D|X|F)?(\s|$)"),
        (C14, r"\bCore\(TM\) i[3579]-1[34]\d\d\d(K|KF|KS|F)?(\s|$)"),
        (CU, r"\bCore\(TM\) Ultra [579] 2\d\d(K|KF|F)?(\s|$)"),
    ]
    assert host.cpu_class("Core™ i9-14900K") is None          # the lscpu reading never saw Vast's mark: why the mapping exists
    assert host.cpu_class("13th Gen Intel(R) Core(TM) i9-13900K") == C14


def test_the_screens_listed_ram_floor_is_the_runners():
    floor = int(re.search(r"\bMIN_RAM_GIB=(\d+);", RUN).group(1))
    assert host.SCREEN_MIN_LISTED_RAM_GIB == floor == 128
    assert host.vast_ram_gib(146_432) == 143 and host.vast_ram_gib(131_071) == 127 and host.vast_ram_gib(None) == 0


# -- rc 16: the RAM the process can use ---------------------------------------------------------------------------------------------
def _box(tmp, mem_kb, v2=None, v1=None, path="/docker/abc", parents=None):
    """A fake /proc/meminfo, /proc/self/cgroup and /sys/fs/cgroup. v2: memory.max text at the leaf; v1: memory.limit_in_bytes text;
    parents: {relative dir: memory.max text} above the leaf."""
    (tmp / "meminfo").write_text("" if mem_kb is None else f"MemTotal:       {mem_kb} kB\nMemFree:  1 kB\n", encoding="utf-8")
    root = tmp / "cg"
    lines = []
    if v2 is not None or parents:
        lines.append(f"0::{path}")
        leaf = root.joinpath(*[p for p in path.split("/") if p])
        leaf.mkdir(parents=True, exist_ok=True)
        if v2 is not None:
            (leaf / "memory.max").write_text(v2 + "\n", encoding="utf-8")
        for rel, txt in (parents or {}).items():
            d = root.joinpath(*[p for p in rel.split("/") if p])
            d.mkdir(parents=True, exist_ok=True)
            (d / "memory.max").write_text(txt + "\n", encoding="utf-8")
    if v1 is not None:
        lines.append(f"9:memory:{path}")
        leaf = root.joinpath("memory", *[p for p in path.split("/") if p])
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "memory.limit_in_bytes").write_text(v1 + "\n", encoding="utf-8")
    (tmp / "self_cgroup").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(tmp / "meminfo"), str(tmp / "self_cgroup"), str(root)]


TB_KB = 1024 ** 3                                                           # 1 TiB in kB
CASES = [
    ("a 1 TB host held to 64 GiB (cgroup v2)", dict(mem_kb=TB_KB, v2=str(64 * GIB)), 64, 16),
    ("a 1 TB host, v2 max (no limit)", dict(mem_kb=TB_KB, v2="max"), 1024, 0),
    ("a 1 TB host held to 64 GiB (cgroup v1)", dict(mem_kb=TB_KB, v1=str(64 * GIB)), 64, 16),
    ("v1's no-limit sentinel", dict(mem_kb=TB_KB, v1="9223372036854771712"), 1024, 0),
    ("a parent's limit binds under a leaf max", dict(mem_kb=TB_KB, v2="max", parents={"/docker": str(100 * GIB)}), 100, 16),
    ("a 256 GiB limit on a 1 TB host passes", dict(mem_kb=TB_KB, v2=str(256 * GIB)), 256, 0),
    ("a 192 GiB desktop, no cgroup line", dict(mem_kb=192 * 1024 ** 2), 192, 0),
    ("a limit above MemTotal: MemTotal binds", dict(mem_kb=96 * 1024 ** 2, v2=str(512 * GIB)), 96, 16),
]


@pytest.mark.parametrize("label,box,gib,_rc", CASES)
def test_usable_ram(tmp_path, capsys, label, box, gib, _rc):
    assert host.main(["ram_gib", *_box(tmp_path, **box)]) == 0, label
    out = capsys.readouterr().out.split()
    assert int(out[0]) == gib and out[1].startswith("memtotal="), (label, out)


@pytest.mark.parametrize("box", [dict(mem_kb=None), dict(mem_kb=TB_KB, v2="garbage"), dict(mem_kb=TB_KB, v1="")])
def test_an_unread_value_is_never_no_limit(tmp_path, capsys, box):
    assert host.main(["ram_gib", *_box(tmp_path, **box)]) == 1
    assert capsys.readouterr().out == ""


def _ram_block(run):
    start = run.index("read -r RAM_GIB RAM_FROM")
    return run[start:run.index("finish 16; }", start) + len("finish 16; }")]


def _runner_rc(tmp_path, args, block):
    """The runner's own RAM lines, run by bash with $W/p237_host.py a shim that hands the real main this box's fake paths."""
    w = tmp_path / "w"
    w.mkdir(exist_ok=True)
    (w / "p237_host.py").write_text(
        f"import sys\nsys.path.insert(0, {str(P)!r})\nimport p237_host\nsys.exit(p237_host.main(['ram_gib', *{args!r}]))\n",
        encoding="utf-8")
    script = f'W={w}\nMIN_RAM_GIB=128\nsay(){{ echo "$*"; }}\nfinish(){{ exit "$1"; }}\n{block}\nexit 0\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path).returncode


@pytest.mark.parametrize("label,box,_gib,rc", CASES)
def test_the_runner_refuses_16_on_usable_ram(tmp_path, label, box, _gib, rc):
    assert _runner_rc(tmp_path, _box(tmp_path, **box), _ram_block(RUN)) == rc, label


@pytest.mark.parametrize("box", [dict(mem_kb=None), dict(mem_kb=TB_KB, v2="garbage")])
def test_the_runner_stops_9_on_an_unread_value(tmp_path, box):
    assert _runner_rc(tmp_path, _box(tmp_path, **box), _ram_block(RUN)) == 9


def test_mutant_the_registered_memtotal_line_passes_the_slice(tmp_path):
    """The bug amendment 1 fixes: the registered line read MemTotal only, so the 64 GiB slice of a 1 TB host passed."""
    old = ("RAM_GIB=$(awk '/^MemTotal:/{print int($2/1048576)}' " + str(tmp_path / "meminfo") + ")\n"
           + _ram_block(RUN).split("\n", 1)[1])
    args = _box(tmp_path, mem_kb=TB_KB, v2=str(64 * GIB))
    assert _runner_rc(tmp_path, args, old) == 0
    assert _runner_rc(tmp_path, args, _ram_block(RUN)) == 16


def test_the_host_line_records_the_ram_source():
    assert "ram=${RAM_GIB}GiB($RAM_FROM)" in RUN and 'echo "refused: ram ${RAM_GIB:-?} GiB ($RAM_FROM)" > REFUSAL' in RUN


# -- the shake's launch wall --------------------------------------------------------------------------------------------------------
def test_the_shake_wall_is_the_fit_plus_the_launch_slack():
    assert host.LAUNCH_SLACK_S == 1500
    assert host.fit_s(host.n_proc("shake"), 1500) == 16500 and host.worst_usd(16500) == 3.85
    assert host.wall_s("shake", 1500) == 18000 == 5.0 * 3600 and host.worst_usd(18000) == 4.10
    assert host.wall_s("shake", 1500) <= host.POLICY_MAX_WALLCLOCK_H * 3600                 # no override


def test_the_amendment_states_the_ruled_numbers():
    for s in ("min(MemTotal, every cgroup memory limit", "1 TB host held to 64 GiB", "`p237_host.vast_cpu_class`", "listed `cpu_ram` >= 128 GiB",
              "18,000 s (5.0 h)", "worst USD 4.10", "USD 3.85", "no override", "111 offers", "0 offers",
              "2026-10-28T07:51Z", "The CPU class is unchanged"):
        assert s in AM1, s
