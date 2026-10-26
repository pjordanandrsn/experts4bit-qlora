"""P237's kit (bench/p237): the staged pin, the schedule, the exit codes against the launcher, the host decisions, the egress probe,
the runner's argv against tc1_arm's own parser, the runner's knobs against the reducer's, and the reducer's self-test.

The hook (p237_arm.py) has its own tests (tests/test_p237_arm.py); the shared driver-tail rule covers p237_drive.sh with every other
driver (tests/test_drive_tail_codes.py).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
P = REPO / "bench" / "p237"
RUN, DRIVE, PIN = P / "p237_run.sh", P / "p237_drive.sh", P / "staged-p237.sha256"
sys.path.insert(0, str(P))
host = importlib.import_module("p237_host")
sched = importlib.import_module("p237_schedule")
egress = importlib.import_module("p237_egress_probe")
HOST_CODES = {13, 14, 17, 18}


def _run_text():
    return RUN.read_text(encoding="utf-8")


def _drive_staged():
    m = re.search(r"for f in (.*?); do", DRIVE.read_text(encoding="utf-8"), re.S)
    return [w for w in m.group(1).replace("\\\n", " ").split()]


# -- the staged pin -----------------------------------------------------------------------------------------------------------------
def test_the_pin_names_every_staged_file_at_its_bytes():
    pins = dict(reversed(ln.split()) for ln in PIN.read_text(encoding="utf-8").splitlines() if ln.strip())
    staged = [p for p in _drive_staged() if not p.endswith("staged-p237.sha256")]
    assert sorted(Path(p).name for p in staged) == sorted(pins), (staged, sorted(pins))
    for p in staged:
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == pins[Path(p).name], f"{p} changed: regenerate {PIN.name}"


def test_the_runner_checks_the_files_the_driver_stages():
    m = re.search(r'STAGED="(.*?)"', _run_text(), re.S)
    assert sorted(m.group(1).split()) == sorted(Path(p).name for p in _drive_staged())


def test_the_hook_pin_is_the_merged_harness():
    pins = dict(reversed(ln.split()) for ln in PIN.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert pins["tc1_arm.py"].startswith("434c77c1"), pins["tc1_arm.py"]       # e4b#2146's merged tc1_arm.py
    assert "awk '$2 == \"p237_arm.py\"" in _run_text()                       # the runner reads the hook's sha from the pin by name


# -- the schedule -------------------------------------------------------------------------------------------------------------------
def test_the_reading_order_is_primers_then_the_palindrome_per_budget():
    rows = sched.schedule()
    got = [(r["budget"], r["kind"], r["arm"], r["draw"]) for r in rows]
    per = [("primer", "E", 0), ("primer", "UA", 0), ("draw", "E", 1), ("draw", "UA", 1), ("probes", "UT", 0), ("draw", "UT", 1),
           ("draw", "UT", 2), ("draw", "UA", 2), ("draw", "E", 2)]
    assert got == [("12g", *x) for x in per] + [("full", *x) for x in per]
    assert [r["index"] for r in rows] == list(range(1, len(rows) + 1))
    draws = [r["arm"] for r in rows if r["budget"] == "12g" and r["kind"] == "draw"]
    assert draws == draws[::-1]                                                 # E UA UT UT UA E


def test_the_shakedown_order():
    got = [(r["budget"], r["kind"], r["arm"]) for r in sched.schedule(shake=True)]
    assert got == [("12g", "draw", "E"), ("12g", "draw", "UA"), ("12g", "probes", "UT"), ("full", "draw", "UA")]


def test_the_schedule_cli(capsys):
    assert sched.main([]) == 0
    assert capsys.readouterr().out.splitlines()[2] == "3 draw 12g E 1"
    assert sched.main(["--bogus"]) == 2


# -- the exit codes -----------------------------------------------------------------------------------------------------------------
def _finishes(text):
    out = []
    for ln in text.splitlines():
        for m in re.finditer(r"\bfinish (\d+)\b", ln):
            out.append((int(m.group(1)), ln))
    return out


def test_host_codes_appear_only_on_a_host_floor_line():
    for code, ln in _finishes(_run_text()):
        if code in HOST_CODES:
            assert re.search(r"BOX_REFUSED|host floor", ln), f"host code {code} on a line with no host-floor reason: {ln.strip()[:160]}"


def test_thirteen_is_the_disk_floor_only_and_the_dataset_is_21():
    thirteen = [ln for c, ln in _finishes(_run_text()) if c == 13]
    assert len(thirteen) == 1 and "disk" in thirteen[0], thirteen
    for ln in _run_text().splitlines():
        if "DATASET" in ln and "finish" in ln:
            assert "finish 21" in ln, ln                                           # TC1 amendment 82: a dataset refusal is never 13


def test_the_driver_passes_exactly_the_runners_lane_codes():
    lane = {c for c, _ in _finishes(_run_text())} - HOST_CODES - {0}
    m = re.search(r'BOX_CODES="([\d ]+)"', DRIVE.read_text(encoding="utf-8"))
    assert {int(x) for x in m.group(1).split()} == lane, (sorted(lane), m.group(1))
    header = re.search(r"# Exit codes.*?\nset -uo", _run_text(), re.S).group(0)
    for c in lane | (HOST_CODES - {17}):
        assert re.search(rf"\b{c}\b", header), f"code {c} is not in the runner's exit-code table"


@pytest.mark.parametrize("rc,success,want", [("0", True, 0), ("0", False, 24), ("21", False, 21), ("13", False, 13),
                                             ("41", False, 41), ("17", False, 17), ("22", False, 26), ("x", False, 23)])
def test_the_driver_tail(tmp_path, rc, success, want):
    (tmp_path / "P237_EXIT_CODE.n1").write_text(rc + "\n")
    if success:
        (tmp_path / "P237_SUCCESS.n1").write_text("")
    p = subprocess.run(["bash", str(DRIVE)], env={"PATH": "/usr/bin:/bin", "P237_DRIVE_TAIL_DIR": str(tmp_path),
                                                  "P237_DRIVE_TAIL_NONCE": "n1"}, capture_output=True, text=True, timeout=60)
    assert p.returncode == want, (p.returncode, p.stderr)


def test_the_run_ids_agree_between_the_driver_and_the_runner():
    def ids(text):
        return {m.group(1): m.group(2) for m in re.finditer(r"^\s*(p237-[^)]+)\)\s*(?:WANT_)?MODE=(\w+);;", text, re.M)}
    d, r = ids(DRIVE.read_text(encoding="utf-8")), ids(_run_text())
    assert d == r and set(d.values()) == {"shake", "read"}, (d, r)


# -- the host decisions -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,cls", [
    ("AMD EPYC 9655 96-Core Processor", "epyc-zen5"), ("AMD EPYC 9575F 64-Core Processor", "epyc-zen5"),
    ("AMD EPYC 9654 96-Core Processor", None), ("AMD EPYC 7763 64-Core Processor", None), ("AMD EPYC 7B13", None),
    ("AMD Ryzen 9 7950X 16-Core Processor", "ryzen-7000-9000-desktop"), ("AMD Ryzen 7 9800X3D 8-Core Processor", "ryzen-7000-9000-desktop"),
    ("AMD Ryzen 9 9950X 16-Core Processor", "ryzen-7000-9000-desktop"), ("AMD Ryzen Threadripper 7970X 32-Cores", None),
    ("AMD Ryzen 9 7945HX with Radeon Graphics", None), ("AMD Ryzen 9 5950X 16-Core Processor", None),
    ("13th Gen Intel(R) Core(TM) i9-13900K", "core-13-14-desktop"), ("Intel(R) Core(TM) i9-14900KF", "core-13-14-desktop"),
    ("13th Gen Intel(R) Core(TM) i9-13980HX", None), ("Intel(R) Core(TM) i9-12900K", None),
    ("Intel(R) Core(TM) Ultra 9 285K", "core-ultra-desktop"), ("Intel(R) Core(TM) Ultra 7 155H", None),
    ("Intel(R) Xeon(R) Platinum 8480+", None), ("", None),
])
def test_the_cpu_class(name, cls):
    assert host.cpu_class(name) == cls
    assert host.main(["cpu", name or "x"]) == (0 if cls else 18)


@pytest.mark.parametrize("total", [25_757_220_864, 25_386_352_640, 24_146_608_128, 1 << 34])
def test_the_full_card_cap_reads_back_as_exactly_one(total):
    cap = float(host.cap_gib(total))
    assert cap * 2 ** 30 / total == 1.0                                        # tc1_arm.apply_vram_cap's own arithmetic


def test_n_auto():
    ok = {"status": "ok", "unsloth_offload_layers": {"resolved": {"n_layers": 22}}}
    assert host.n_auto(ok, "") == 22
    line = "Unsloth: offload_layers = 'auto' loads every layer onto the GPU (fits)"
    ref = {"status": "refused", "reason": "--unsloth-offload-layers auto: the block swap holds no layer (the offload did not engage)"}
    assert host.n_auto(ref, f"x\n{line}\n") == 0
    assert host.n_auto(ref, "no such line") is None                           # a refusal without Unsloth's own line is not N_auto = 0
    assert host.n_auto(dict(ref, reason="something else"), line) is None
    assert host.n_auto(dict(ref, status="oom"), line) is None
    assert host.n_auto({"status": "ok", "unsloth_offload_layers": {"resolved": None}}, "") is None
    assert host.n_auto(None, line) is None


def test_probe_outcome_and_tune():
    cap = 12
    under, over = 12 * 2 ** 30 - 512 * 2 ** 20, 12 * 2 ** 30 - 512 * 2 ** 20 + 1
    cell = {"status": "ok", "steps": 8, "step_ms": [1.0] * 8, "step_mem": [{"max_reserved": 5}] * 7 + [{"max_reserved": under}]}
    assert host.probe_outcome(cell, cap) == f"ok {under}"
    cell["step_mem"][-1]["max_reserved"] = over
    assert host.probe_outcome(cell, cap) == f"over {over}"
    assert host.probe_outcome({"status": "oom"}, cap) == "oom"
    assert host.probe_outcome(dict(cell, steps=7), cap).startswith("fault")
    assert host.probe_outcome(dict(cell, step_mem=[]), cap) == "fault no-step-mem"
    assert host.probe_outcome(None, cap) == "fault none"
    assert host.tune(24, ["ok", "ok", "ok", "ok"]) == 20
    assert host.tune(24, ["ok", "over", "oom"]) == 23
    assert host.tune(24, ["oom"]) == 24 and host.tune(24, []) == 24 and host.tune(24, ["over", "over"]) == 24
    assert host.tune(24, ["ok", "fault"]) is None


# -- the egress probe ---------------------------------------------------------------------------------------------------------------
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_egress_first_shard_reads_the_listing():
    listing = json.dumps({"siblings": [{"rfilename": "config.json"}, {"rfilename": "model-00002-of-00016.safetensors"},
                                       {"rfilename": "model-00001-of-00016.safetensors"}]}).encode()
    assert egress.first_shard("m", "r", opener=lambda url, timeout: _Resp(listing)) == "model-00001-of-00016.safetensors"
    assert egress.first_shard("m", "r", opener=lambda url, timeout: _Resp(b'{"siblings": []}')) is None


def test_egress_measure_and_floor():
    t = iter([0.0, 0.5, 1.0, 1.5, 2.0, 40.0, 41.0])
    got, dt = egress.measure("u", opener=lambda url, timeout: _Resp(b"x" * (3 << 20)), clock=lambda: next(t), window=30)
    assert got == 3 << 20 and dt == 2.0                                        # the closing read of the clock
    t2 = iter([0.0] + [31.0] * 4)
    got, dt = egress.measure("u", opener=lambda url, timeout: _Resp(b"x" * (3 << 20)), clock=lambda: next(t2), window=30)
    assert got == 1 << 20 and dt == 31.0                                       # the window closes after the read that crossed it
    assert egress.FLOOR_BPS == 40_000_000 and egress.SLOW == 4 and egress.NO_TRANSFER == 2

    def refused(url, timeout):
        raise OSError("refused")
    with pytest.raises(OSError):
        egress.measure("u", opener=refused)
    assert egress.main(["only-one"]) == egress.NO_TRANSFER


# -- the runner's argv against tc1_arm's parser, and its knobs against the reducer's -------------------------------------------------
def _tc1():
    spec = importlib.util.spec_from_file_location("tc1_arm_p237_parse", REPO / "bench" / "tc1" / "tc1_arm.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["tc1_arm_p237_parse"] = m
    spec.loader.exec_module(m)
    return m


def _shell_value(name):
    m = re.search(rf'^{name}="(.*?)"$', _run_text(), re.M | re.S)
    return m.group(1)


def _arm_flags(arm):
    m = re.search(rf"^  {re.escape(arm)}\) echo \"(.*?)\";;", _run_text(), re.M)
    return m.group(1)


@pytest.mark.parametrize("arm,budget,off", [("E", "12g", None), ("E", "full", None), ("UA", "12g", "auto"), ("UT", "12g", "20"),
                                            ("UA", "full", "auto")])
def test_every_runner_argv_parses_with_tc1_arms_own_parser(arm, budget, off):
    tc1 = _tc1()
    field = _shell_value("FIELD").replace("$MODEL", "Qwen/Qwen3-30B-A3B").replace("$REVISION", "a" * 40).replace("$TOK_SHA", "b" * 64)
    field = field.replace("$TOK", "/root/p237/data/tokens_qwen3.json")
    flags = _arm_flags("E" if arm == "E" else "UA|UT").replace("$UT7", "q_proj,k_proj").replace("${EXPECT:+ --expect-trainable $EXPECT}",
                                                                                                    " --expect-trainable 123")
    cap = {"12g": "--vram-cap-gib 12", "full": "" if arm == "E" else "--vram-cap-gib 23.98828125"}[budget]
    argv = shlex.split(f"{field} {flags} {cap} {'--unsloth-offload-layers ' + off if off else ''} --steps 60 --eval-every 20 "
                       f"--profile-steps 3 --profile-warm 20 --tag t --out /tmp/o --adapter-dir /tmp/o/a")
    a = tc1.build_parser().parse_args(argv)
    assert a.accum == 4 and a.micro_batch == 2 and a.seq == 2048 and a.lr == 2e-4 and a.r == 16 and a.alpha == 16
    assert a.optim == "adamw_8bit" and a.lr_schedule == "linear" and a.warmup_steps == 5 and a.eval_n == 8
    assert a.adapter_dtype == "fp32" and a.lora_init == "matched:3407" and a.profile_warm == 20 and a.profile_steps == 3
    if arm == "E":
        assert a.framework == "e4b" and a.layer_major == 1 and int(a.offload) == 1 and a.unsloth_offload_layers is None
    else:
        assert a.framework == "unsloth" and a.unsloth_offload_layers == off and a.grad_ckpt == "unsloth"
        assert a.unsloth_moe_backend == "grouped_mm" and a.expect_trainable == 123
    assert (a.vram_cap_gib is None) == (budget == "full" and arm == "E")


def test_the_runner_pins_the_reducers_knobs():
    red = importlib.import_module("p237_reduce")
    text = _run_text()
    exported = dict(re.findall(r"\b(UNSLOTH_[A-Z0-9_]+)=(\S+)", re.search(r"^export UNSLOTH_.*$", text, re.M).group(0)))
    assert exported == red.RUNNER_UNSLOTH
    allowed = re.search(r'case "\$k" in (\S+)\) ;;', text).group(1).split("|")
    assert sorted(allowed) == sorted(red.RUNNER_UNSLOTH)
    for dist, var in (("experts4bit-qlora", "E4B_VERSION"), ("grouped-nf4-gemm", "GNF4_VERSION"), ("unsloth", "UNSLOTH_VERSION"),
                      ("unsloth_zoo", "ZOO_VERSION")):
        assert re.search(rf"\b{var}={re.escape(red.PINS[dist])}\b", text), dist
    assert f"CLOCK_FLOOR={red.CLOCK_FLOOR_MHZ}" in text and f"CAP_12G={red.CAP_12G_GIB}" in text and f"STEPS={red.STEPS}" in text
    assert "export GNF4_TRAIN_GEMM=auto" in text and 'env_e="E4B_OFFLOAD_STATS=1"' in text and "TC1_STEP_MEM=1" in text


def test_the_reducer_selftest():
    red = importlib.import_module("p237_reduce")
    assert red.selftest() >= 70
