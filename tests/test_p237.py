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
import os
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


def test_thirteen_is_the_disk_floor_only_and_the_dataset_is_29():
    thirteen = [ln for c, ln in _finishes(_run_text()) if c == 13]
    assert len(thirteen) == 1 and "disk" in thirteen[0], thirteen
    data = [ln for ln in _run_text().splitlines() if "DATASET" in ln and "finish" in ln]
    assert len(data) == 2 and all("finish 29" in ln for ln in data), data   # TC1 amendment 82: never 13, and never the driver's 21


def driver_codes(drive_text, tail_text):
    """Every literal ``exit N`` the driver can return on its own: p237_drive.sh's lines and the shared tail's."""
    return {int(c) for t in (drive_text, tail_text) for c in re.findall(r"\bexit (\d+)\b", t)}


def code_clashes(run_text, drive_text, tail_text):
    """Box codes that are also the driver's own exits (amendment 82's rule): only 0 and 78 may be both."""
    lane = {c for c, _ in _finishes(run_text)} - HOST_CODES
    listed = {int(x) for x in re.search(r'BOX_CODES="([\d ]+)"', drive_text).group(1).split()}
    return sorted((lane | listed) & driver_codes(drive_text, tail_text) - {0, 78})


def test_no_box_code_is_one_the_driver_exits_on_its_own():
    tail = (REPO / "bench" / "common" / "drive_tail.sh").read_text(encoding="utf-8")
    drive = DRIVE.read_text(encoding="utf-8")
    assert {20, 21, 22, 23, 24, 26, 78} <= driver_codes(drive, tail)               # the reader sees the driver's start-failure 21
    assert code_clashes(_run_text(), drive, tail) == []
    run21 = _run_text().replace("finish 29; }", "finish 21; }")                     # MUTANT: 21 put back
    drive21 = drive.replace('BOX_CODES="9 10 15 16 27 29 41 78 130"', 'BOX_CODES="9 10 15 16 21 27 41 78 130"')
    assert run21 != _run_text() and drive21 != drive, "the mutant's anchors are stale"
    assert code_clashes(run21, drive21, tail) == [21]


def test_the_driver_passes_exactly_the_runners_lane_codes():
    lane = {c for c, _ in _finishes(_run_text())} - HOST_CODES - {0}
    m = re.search(r'BOX_CODES="([\d ]+)"', DRIVE.read_text(encoding="utf-8"))
    assert {int(x) for x in m.group(1).split()} == lane, (sorted(lane), m.group(1))
    header = re.search(r"# Exit codes.*?\nset -uo", _run_text(), re.S).group(0)
    for c in lane | (HOST_CODES - {17}):
        assert re.search(rf"\b{c}\b", header), f"code {c} is not in the runner's exit-code table"


@pytest.mark.parametrize("rc,success,want", [("0", True, 0), ("0", False, 24), ("29", False, 29), ("21", False, 26), ("13", False, 13),
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


def _profile_flags():
    return re.search(r'pp="(--profile-steps 3 --profile-warm \d+)"', _run_text()).group(1)


def test_the_profile_window_is_past_the_post_eval_step_and_the_reducer_reads_it():
    """--profile-warm 21 (the maintainer's call): loop steps 21..23, so the profile never holds step 20, the cold step after the eval at
    20 (E stages 96, Unsloth re-arms after an eval-mode forward). The reducer's window edges are the same steps."""
    red = importlib.import_module("p237_reduce")
    warm = int(re.search(r"--profile-warm (\d+)", _profile_flags()).group(1))
    assert warm == 21 and red.PROFILED == (warm, warm + 3)
    assert re.search(r'\[ "\$s" -ge (\d+) \] && pp=', _run_text()).group(1) == str(warm + 3)   # the shakedown's 24 steps hold it


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
                       f"{_profile_flags()} --tag t --out /tmp/o --adapter-dir /tmp/o/a")
    a = tc1.build_parser().parse_args(argv)
    assert a.accum == 4 and a.micro_batch == 2 and a.seq == 2048 and a.lr == 2e-4 and a.r == 16 and a.alpha == 16
    assert a.optim == "adamw_8bit" and a.lr_schedule == "linear" and a.warmup_steps == 5 and a.eval_n == 8
    assert a.adapter_dtype == "fp32" and a.lora_init == "matched:3407" and a.profile_warm == 21 and a.profile_steps == 3
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
    assert red.selftest() >= 100


# -- the box environment: one list for the runner's refusal, its dump and the reducer --------------------------------------------
def _ambient_block(run=None):
    run = run or _run_text()
    return run.split("# ---- ambient knobs", 1)[1].split("\n", 1)[1].split("# ---- end ambient knobs", 1)[0]


@pytest.mark.parametrize("key,val,want", [
    ("E4B_TRAIN_DETERMINISTIC", "1", 78), ("E4B_OFFLOAD_STATS", "1", 78), ("GNF4_ANYTHING", "1", 78), ("NF4_QLORA_ANYTHING", "1", 78),
    ("UNSLOTH_DISABLE_MOE_TRITON", "1", 78), ("TC1_STEP_MEM", "1", 78), ("TC1_ARM_HOOK", "x:make", 78),
    ("E4B_SHA", "a" * 40, 0), ("GNF4_TRAIN_GEMM", "auto", 0), ("UNSLOTH_MOE_NF4_FUSED", "0", 0), ("PYTORCH_VERSION", "2.12.1", 0),
    (None, None, 0),
])
def test_the_runners_ambient_block_refuses_every_unregistered_key(key, val, want):
    """The runner's own marked block, run as written: any UNSLOTH_* / E4B_* / GNF4_* / NF4_QLORA_* / TC1_* key outside the registered
    set refuses 78 before STARTED (the per-process TC1_* and E4B_OFFLOAD_STATS keys included: they are set on each process's own
    line, never inherited); E4B_SHA (the driver's) and the image's keys pass."""
    red = importlib.import_module("p237_reduce")
    env = {k: v for k, v in os.environ.items() if not re.match(red.ENV_REFUSE_GREP, k)}
    if key:
        env[key] = val
    script = 'say(){ echo "$*"; }\nfinish(){ exit "$1"; }\n' + _ambient_block() + "exit 0\n"
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == want, (key, r.stdout, r.stderr)
    if want:
        assert key in r.stdout


def _env_disagreements(run, red):
    bad = []
    block = _ambient_block(run)
    refuse = re.search(r"env \| grep -oE '([^']+)' \| sort -u", block)
    if not refuse or refuse.group(1) != red.ENV_REFUSE_GREP:
        bad.append(("refusal grep", refuse and refuse.group(1)))
    allowed = set()
    for labels in re.findall(r"^\s*(?:case \"\$k\" in )?([A-Z0-9_|]+)\) ;;", block, re.M):
        allowed |= set(labels.split("|"))
    if allowed != set(red.ENV_BOX) - set(red.ENV_EXACT):
        bad.append(("refusal allowlist", sorted(allowed ^ (set(red.ENV_BOX) - set(red.ENV_EXACT)))))
    dump = re.search(r"^env \| grep -E '([^']+)' \| sort > \$W/env_box\.txt$", run, re.M)
    if not dump or dump.group(1) != red.ENV_BOX_GREP:
        bad.append(("env_box grep", dump and dump.group(1)))
    exported = {}
    for line in re.findall(r"^export (.+)$", run, re.M):
        for kv in line.split("#", 1)[0].split():
            k, _, v = kv.partition("=")
            if re.match(red.ENV_BOX_GREP, k + "="):
                exported[k] = v
    if exported != {k: v for k, v in red.ENV_BOX.items() if v is not None}:
        bad.append(("exports", exported))
    if not all(re.match(red.ENV_BOX_GREP, k + "=") for k in red.ENV_BOX):
        bad.append(("a registered key the dump would not record", sorted(red.ENV_BOX)))
    if any(k.startswith("TC1_") for k in red.ENV_BOX):
        bad.append(("a TC1_* key registered at the dump: they are set per process, after it", sorted(red.ENV_BOX)))
    return bad


def test_one_env_list_binds_the_refusal_the_dump_and_the_reducer(monkeypatch):
    red = importlib.import_module("p237_reduce")
    run = _run_text()
    assert _env_disagreements(run, red) == []
    mutants = [
        ("    E4B_SHA) ;;", "    E4B_SHA|E4B_TRAIN_DETERMINISTIC) ;;"),                                # a wider box allowlist
        ("grep -oE '^(UNSLOTH|E4B|GNF4|NF4_QLORA|TC1)_", "grep -oE '^(UNSLOTH|E4B|GNF4|NF4_QLORA)_"),        # a narrower refusal
        ("|TC1_)|^PYTORCH_CUDA_ALLOC_CONF=' | sort > $W/env_box.txt", "|TC1_|PYTORCH_)' | sort > $W/env_box.txt"),   # a wider dump
        ("export GNF4_TRAIN_GEMM=auto\n", "export GNF4_TRAIN_GEMM=auto E4B_TRAIN_DETERMINISTIC=1\n"),          # an extra export
    ]
    for old, new in mutants:
        assert run.count(old) == 1, old
        assert _env_disagreements(run.replace(old, new), red), f"mutant survives: {new}"
    monkeypatch.setattr(red, "ENV_BOX", {**red.ENV_BOX, "E4B_TRAIN_DETERMINISTIC": "1"})                 # a wider reducer set
    assert _env_disagreements(run, red)
    monkeypatch.setattr(red, "ENV_BOX", {**red.ENV_BOX, "TC1_STEP_MEM": "1"})
    assert _env_disagreements(run, red)


# -- the guard fit: the PREREG's numbers, the helper's constants and the runner's agree ---------------------------------------------
PREREG = P / "PREREG-p237.md"


def _fit_line(text):
    m = re.search(r"^`FIT: install (\d+) s, fetch (\d+) s, reserve (\d+) s, margin (\d+) %; processes (\d+) \(reading\), (\d+) \(shakedown\); "
                  r"price ([\d.]+) USD/h, downloads ([\d.]+) USD; run cap ([\d.]+) USD; policy max_wallclock_h (\d+) h`$", text, re.M)
    assert m, "the PREREG has no FIT line"
    g = m.groups()
    return {"install": int(g[0]), "fetch": int(g[1]), "reserve": int(g[2]), "margin": int(g[3]), "n_read": int(g[4]),
            "n_shake": int(g[5]), "usd_h": float(g[6]), "dl": float(g[7]), "cap_usd": float(g[8]), "policy_h": int(g[9])}


def fit_disagreements(prereg, run, h):
    f, bad = _fit_line(prereg), []
    want = {"install": h.INSTALL_S, "fetch": h.FETCH_S, "reserve": h.RESERVE_S, "margin": h.FIT_MARGIN_PCT, "n_read": h.n_proc("read"),
            "n_shake": h.n_proc("shake"), "usd_h": h.USD_PER_H, "dl": h.DOWNLOAD_USD, "cap_usd": h.RUN_CAP_USD, "policy_h": h.POLICY_MAX_WALLCLOCK_H}
    bad += [(k, f[k], v) for k, v in want.items() if f[k] != v]
    m = re.search(r"^INSTALL_S=(\d+); FETCH_S=(\d+); RESERVE=(\d+)$", run, re.M)
    if not m or tuple(map(int, m.groups())) != (h.INSTALL_S, h.FETCH_S, h.RESERVE_S):
        bad.append(("runner constants", m and m.groups()))
    for name, n in (("Reading", h.n_proc("read")), ("Shakedown", h.n_proc("shake"))):
        g = h.fit_s(n, 1500)
        lines = prereg.splitlines()
        at = next((i for i, ln in enumerate(lines) if ln.startswith(f"- **{name}:**")), None)
        block = [] if at is None else [lines[at]] + [ln for ln in __import__("itertools").takewhile(lambda x: x.startswith("  "), lines[at + 1:])]
        line = " ".join(ln.strip() for ln in block)                               # the bullet, its continuation lines joined
        stated = f"a guard of {g} s ({g / 3600:.{1 if name == 'Reading' else 2}f} h), worst USD {h.worst_usd(g):.2f}"
        if stated not in line:
            bad.append((name, stated, line[:120]))
        if (g > h.cap_s()) != ("HOLD" in line):
            bad.append((name, "HOLD stated iff the fit is over the cap", g, h.cap_s()))
    return bad


def test_the_registered_fit_is_the_runners_and_the_prereg_states_it():
    import p237_host as h
    assert h.POLICY_MAX_WALLCLOCK_H == 6                                         # adertha/compute/policy.json teardown.max_wallclock_h
    assert h.cap_s() == 6 * 3600 and h.PROBES_MAX == {"read": 4, "shake": 1}
    assert h.n_proc("read") == 24 and h.n_proc("shake") == 4
    assert h.fit_s(24, 1500) == 54000 and h.fit_s(4, 1500) == 16500 and h.worst_usd(16500) == 3.85
    prereg = PREREG.read_text(encoding="utf-8")
    assert fit_disagreements(prereg, _run_text(), h) == []
    assert 'read -r N_PROC FIT_NEED CAP_S PMAX < <(python3 $W/p237_host.py fit "$P237_MODE" "$LOAD_S")' in _run_text()
    assert '[ "$FIT_NEED" -le "$CAP_S" ]' in _run_text() and 'while [ $k -le "$PMAX" ]' in _run_text()


def test_mutants_of_the_fit_are_caught(monkeypatch):
    import p237_host as h
    prereg, run = PREREG.read_text(encoding="utf-8"), _run_text()
    for old, new in [("fetch 3600 s", "fetch 3000 s"), ("policy max_wallclock_h 6 h", "policy max_wallclock_h 8 h"),
                     ("worst USD 3.85", "worst USD 1.50"), ("**over today's 6 h policy cap, so the reading is on\n  HOLD**", "within the cap")]:
        assert prereg.count(old) == 1, old
        assert fit_disagreements(prereg.replace(old, new), run, h), f"PREREG mutant survives: {new}"
    assert fit_disagreements(prereg, run.replace("FETCH_S=3600", "FETCH_S=3000"), h)
    monkeypatch.setattr(h, "POLICY_MAX_WALLCLOCK_H", 24)                        # a wider policy alone: the PREREG no longer agrees
    assert fit_disagreements(prereg, run, h)


# -- the writer and the reader of each process's exit -------------------------------------------------------------------------------
@pytest.mark.parametrize("idx,kind,arm,off,rc", [("3", "draw", "E", "", "5"), ("5.2", "probe", "UT", "22", "0"), ("12", "draw", "UA", "auto", "137")])
def test_the_runners_proc_line_is_what_the_reducer_reads(idx, kind, arm, off, rc):
    """The OOM outcome needs tc1_arm's exit 5 from the runner's OWN PROC line: render that echo line with bash and parse it with
    p237_reduce.proc_rcs, so a change on either side fails here."""
    red = importlib.import_module("p237_reduce")
    line = next(ln for ln in _run_text().splitlines() if ln.lstrip().startswith('echo "PROC $idx $kind $budget $arm draw=$draw steps='))
    echo = line.strip().split(" | tee", 1)[0]
    script = (f'idx={idx}; kind={kind}; budget=12g; arm={arm}; draw=1; s=60; off={off}; rc={rc}; c0=10; c1=10; '
              f'RECEIPT=/root/p237/loads/{idx}/x.json\n{echo}\n')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "summary.txt"
        p.write_text("KNOBS x\n" + out + "PROC 9 draw 12g UT draw=1 skipped: N_t unread\n", encoding="utf-8")
        assert red.proc_rcs(str(p)) == {idx: int(rc)}, (out, red.proc_rcs(str(p)))


# -- the guard fit's registered wall ------------------------------------------------------------------------------------------------
def _load_s_block():
    run = _run_text()
    return run[run.index("LOAD_S=${P237_LOAD_S_REGISTERED"):run.index("INSTALL_S=")]


@pytest.mark.parametrize("val,want", [(None, 0), ("1500", 0), ("300", 0), ("299", 78), ("0300", 78), ("12.5", 78), ("abc", 78), ("", 0)])
def test_load_s_is_whole_seconds_over_a_floor(val, want):
    """P237_LOAD_S_REGISTERED: whole seconds, no leading 0 (bash arithmetic reads it as octal), at least 300; else 78. An empty value
    takes the default, as ${VAR:-1500} does."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if val is not None:
        env["P237_LOAD_S_REGISTERED"] = val
    script = 'say(){ echo "$*"; }\nfinish(){ exit "$1"; }\n' + _load_s_block() + "exit 0\n"
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == want, (val, r.stdout)


def test_mutant_load_s_without_its_floor_is_caught():
    block = _load_s_block()
    floor = next(ln for ln in block.splitlines() if "-ge 300" in ln)
    script = 'say(){ echo "$*"; }\nfinish(){ exit "$1"; }\n' + block.replace(floor, "") + "exit 0\n"
    r = subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin", "P237_LOAD_S_REGISTERED": "5"}, capture_output=True, text=True)
    assert r.returncode == 0                                                    # without the floor a 5 s wall passes: the test above binds it


# -- the fetch: receipts and logs only ----------------------------------------------------------------------------------------------
def _rsync_excludes():
    m = re.search(r"^P237_RSYNC_EXCLUDES=\((.*?)\)$", DRIVE.read_text(encoding="utf-8"), re.M | re.S)
    return shlex.split(m.group(1))


def _stub_box(root: Path):
    big = b"x" * (11 * 2 ** 20)
    files = {
        "summary.txt": b"ok", "box.json": b"{}", "env_box.txt": b"A=1", "logs/load_3.log": b"log", "verdict_p237.json": b"{}",
        "loads/3/qwen3_e4b_p237_3_draw_12g_E_d1.json": b"{}", "loads/3/qwen3_e4b_p237_3_draw_12g_E_d1_profile.json": b"{}",
        "loads/gpuclk_3.txt": b"1 2", "loads/dmon_3.txt": b"#", "loads/vram_3.txt": b"t, 1",
        "venv-unsloth/lib/python3.12/site-packages/torch/lib/libtorch.so": b"so",
        "data/alpaca_data_cleaned.json": big, "data/ds_alpaca.json": b"[]", "data/tokens_qwen3.json": b"{}",
        "loads/3/adapters/adapter_model.safetensors": b"t", "loads/4/adapters/x.json": b"{}",
        "unsloth_compiled_cache/UnslothSFTTrainer.py": b"py", "__pycache__/p237_host.cpython-312.pyc": b"pyc",
        ".cache/huggingface/hub/x": b"h", "loads/3/stray.pt": b"t", "loads/3/stray.bin": b"t", "logs/huge.log": big,
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)


def test_the_fetch_carries_receipts_and_logs_only(tmp_path):
    """p237_drive.sh's own exclude array over a stub box tree (rsync as the driver runs it, minus ssh): nothing over 10 MiB, no tensor
    file, no venv or cache directory, no adapters, and every receipt, log and record the reducer reads comes back."""
    src, dst = tmp_path / "box", tmp_path / "fetched"
    src.mkdir()
    dst.mkdir()
    _stub_box(src)
    r = subprocess.run(["rsync", "-a", "--no-links", *_rsync_excludes(), f"{src}/", f"{dst}/"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = {p.relative_to(dst).as_posix(): p.stat().st_size for p in dst.rglob("*") if p.is_file()}
    assert all(size <= 10 * 2 ** 20 for size in got.values()), got
    assert not [f for f in got if f.endswith((".safetensors", ".pt", ".bin", ".pth"))], got
    assert not [f for f in got if re.match(r"(venv|\.cache|unsloth_compiled_cache|__pycache__)|.*/adapters/", f)], got
    for keep in ("summary.txt", "box.json", "env_box.txt", "logs/load_3.log", "verdict_p237.json", "loads/gpuclk_3.txt",
                 "loads/3/qwen3_e4b_p237_3_draw_12g_E_d1.json", "loads/3/qwen3_e4b_p237_3_draw_12g_E_d1_profile.json",
                 "loads/dmon_3.txt", "loads/vram_3.txt", "data/ds_alpaca.json"):
        assert keep in got, keep


def test_mutant_a_narrower_fetch_list_is_caught(tmp_path):
    src, dst = tmp_path / "box", tmp_path / "fetched"
    src.mkdir()
    dst.mkdir()
    _stub_box(src)
    ex = [a for a in _rsync_excludes() if not a.startswith("--max-size")]          # MUTANT: no size cap
    subprocess.run(["rsync", "-a", "--no-links", *ex, f"{src}/", f"{dst}/"], check=True)
    assert (dst / "logs" / "huge.log").stat().st_size > 10 * 2 ** 20                 # the cap is what keeps it on the box
