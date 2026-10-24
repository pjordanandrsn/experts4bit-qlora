"""TC1 amendment 79 (qwen3pos79): amendment 69's box at the 0.58.0 release against Unsloth 2026.10.8.

The pins are constants of the reviewed kit, held equal across tc1_run.sh, tc1_reduce.py, the launch template and the registration; the
box refuses any other value before staging (rc 78); the staged release proof is byte-identical to loggetta's frozen helper and runs after
every install and before any arm, under the fresh bytecode prefix every arm then reads under; the family is amendment 69's four arms in
their ABBA order; the kit's screen and template carry one set of caps, inside the row's $6 cap. The box-side pieces are cut from
tc1_run.sh and run under bash with stubs, so no GPU, network or model is needed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TC1 = REPO / "bench" / "tc1"
RUN = (TC1 / "tc1_run.sh").read_text()
DRIVE = (TC1 / "tc1_drive.sh").read_text()
PREREG = (TC1 / "TC1-PREREG.md").read_text()
A79 = PREREG[PREREG.index("### Amendment 79 "):]
TEMPLATE = json.loads((TC1 / "pos79_manifest_template.json").read_text())
#: loggetta's bench/release_proof.py as frozen at loggetta#126's merge, the version this kit stages byte-identical
HELPER_SHA256 = "aff3fc80db8d35bbf94a9aaeed125c9d424cfe93bf16eb0e6868b6f9d03f13e0"
STAND_IN = "ab" * 20                                         # a stand-in commit for the gate's tests; never the kit's pin


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TC1 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RED = _load("tc1_reduce")
KIT = _load("pos79_kit")


def _run_const(name: str) -> str:
    m = re.search(rf"^{name}=([^\s;]*)", RUN, re.MULTILINE) or re.search(rf"[; ]{name}=([^\s;]*)", RUN)
    assert m, name
    return m.group(1).strip('"')


def _cmd(var: str) -> str:
    m = re.search(rf"(?:^| ){var}=(\S*)", TEMPLATE["command"])
    assert m, var
    return m.group(1)


def _cut(pattern: str) -> str:
    m = re.search(pattern, RUN, re.DOTALL | re.MULTILINE)
    assert m, f"pattern no longer matches tc1_run.sh: {pattern}"
    return m.group(0)


def test_the_pins_agree_across_the_kit_the_reducer_the_template_and_the_registration():
    e4b, gnf4 = _run_const("POS79_E4B_SHA"), _run_const("POS79_GNF4_SHA")
    uns, zoo = _run_const("POS79_UNSLOTH"), _run_const("POS79_UNSLOTH_ZOO")
    assert (e4b, gnf4) == (RED.POS79_E4B_SHA, RED.POS79_GNF4_SHA) == (_cmd("E4B_SHA"), _cmd("GNF4_SHA"))
    assert gnf4 == "30adb1bfa76f95f9ce4a4c29734b8f36de907eca"          # v0.51.0's peeled tag commit
    assert (uns, zoo) == (RED.POS79_PINS["unsloth"], RED.POS79_PINS["unsloth_zoo"]) == (_cmd("TC1_UNSLOTH_VERSION"), _cmd("TC1_UNSLOTH_ZOO_VERSION"))
    assert (uns, zoo) == ("2026.10.8", "2026.10.8")
    assert RED.POS79_PINS["experts4bit-qlora"] == "0.58.0" and RED.POS79_PINS["grouped-nf4-gemm"] == "0.51.0"
    for want in (gnf4, f"unsloth {uns}", f"unsloth_zoo {zoo}", HELPER_SHA256):
        assert want in A79, want
    if e4b:
        assert e4b in A79


@pytest.mark.xfail(strict=True, reason="unfilled until the v0.58.0 tag line: the commit that fills the pin removes this marker")
def test_the_e4b_pin_is_filled_before_merge():
    """v0.58.0's peeled tag commit must be written into the kit, the reducer, the template and the registration before this registration
    merges (the maintainer's ruling: a constant the review saw, never a launch-time lookup). Until then it is a strict xfail; filling the
    pin without removing the marker turns it into a failure, so the fill commit carries both. The box's pin gate refuses an unfilled pin."""
    pin = _run_const("POS79_E4B_SHA")
    assert re.fullmatch(r"[0-9a-f]{40}", pin), "fill POS79_E4B_SHA with v0.58.0's peeled tag commit when the tag line posts"


def test_the_staged_helper_is_loggettas_frozen_release_proof():
    got = hashlib.sha256((TC1 / "release_proof.py").read_bytes()).hexdigest()
    assert got == HELPER_SHA256 == RED.POS79_RELEASE_PROOF_SHA256
    assert re.search(rf"`bench/tc1/release_proof\.py`[^`]*`{HELPER_SHA256}`", A79), "the registration names the staged helper's sha256"
    assert 'case " ${TC1_FAMILIES:-} " in *" qwen3pos79 "*) STAGE="$STAGE $HERE/release_proof.py";; esac' in DRIVE


GATE = _cut(r"^POS79_E4B_SHA=.*?\n") + _cut(r"^POS79_GNF4_SHA=.*?\n") + _cut(r"^POS79_UNSLOTH=.*?\n") \
    + _cut(r"^case \" \$FAMILIES \" in \*\" qwen3pos79 \"\*\)\n.*?^esac\n")


def _gate(tmp: Path, families="qwen3pos79", e4b=STAND_IN, gnf4="30adb1bfa76f95f9ce4a4c29734b8f36de907eca", uns="2026.10.8",
          zoo="2026.10.8", staged=True, pin=STAND_IN, extra=""):
    w = tmp / "w"
    w.mkdir(exist_ok=True)
    if staged:
        (w / "release_proof.py").write_text("x")
    gate = GATE.replace('POS79_E4B_SHA=""', f"POS79_E4B_SHA={pin}", 1)
    script = ("say(){ echo \"SAY $*\"; }\nfinish(){ echo \"FINISH $1\"; exit $1; }\n" + f"W={w}\nFAMILIES='{families}'\nE4B_SHA={e4b}\n"
              f"GNF4_SHA={gnf4}\nTC1_UNSLOTH_VERSION={uns}\nTC1_UNSLOTH_ZOO_VERSION={zoo}\n{extra}\ncd {w}\n" + gate + "echo PASSED\n")
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    summary = (w / "summary.txt").read_text() if (w / "summary.txt").exists() else ""
    return p, summary


def test_the_gate_passes_at_the_kits_pins(tmp_path):
    p, summary = _gate(tmp_path)
    assert p.returncode == 0 and "PASSED" in p.stdout, p.stdout + p.stderr
    assert f"POS79_PINS e4b {STAND_IN} gnf4 30adb1bfa76f95f9ce4a4c29734b8f36de907eca unsloth 2026.10.8 unsloth_zoo 2026.10.8 release_proof " in summary
    assert "POS79_ARM_ENV e4b=[] extra=[]" in summary.splitlines()


@pytest.mark.parametrize("kw, why", [
    ({"e4b": "cd" * 20}, "E4B_SHA"),
    ({"gnf4": "ef" * 20}, "GNF4_SHA"),
    ({"uns": "2026.10.7"}, "TC1_UNSLOTH_VERSION"),
    ({"zoo": "2026.10.9"}, "TC1_UNSLOTH_ZOO_VERSION"),
    ({"families": "qwen3pos79 qwen3pos69"}, "runs alone"),
    ({"staged": False}, "release_proof.py is not staged"),
    ({"pin": "", "e4b": ""}, "unfilled"),
    ({"extra": "TC1_E4B_ENV='NF4_QLORA_COMPACT_DELTA=1'"}, "TC1_E4B_ENV is set"),
    ({"extra": "TC1_ARM_EXTRA_ENV='E4B_CHUNKED_LM_LOSS=256'"}, "TC1_ARM_EXTRA_ENV is set"),
])
def test_the_gate_refuses_any_other_pin_before_staging(tmp_path, kw, why):
    p, summary = _gate(tmp_path, **kw)
    assert p.returncode == 78 and "FINISH 78" in p.stdout and "PASSED" not in p.stdout, p.stdout
    assert why in summary and "BOX_REFUSED pos79_pins" in summary, summary


FAMILY = _cut(r"^tc1_pos79_family\(\)\{.*?free_family \$FAM \$\{MID//\\//--\}; \}\n")


def test_the_family_is_amendment_69s_four_arms_in_abba_order(tmp_path):
    stubs = ("say(){ :; }\ntc1_prepare(){ TOK=tok; TS=ts; return 0; }\ncan_run(){ return 0; }\nfree_family(){ echo \"FREE $1\"; }\n"
             "arm(){ echo \"ARM ${E4B_VENV:-} $2 $3 $*\"; }\ndraw2(){ echo \"ARM ${E4B_VENV:-} $2 ${3}_d2 $*\"; }\n"
             "MATCHED_SEED=3407 UT7=ut7 PROFILE_STEPS=3 PROFILE_WARM=3 POS79_UNSLOTH=2026.10.8\n")
    p = subprocess.run(["bash", "-c", stubs + FAMILY + "tc1_pos79_family qwen3pos79 M R 1 2 3\n"], capture_output=True, text=True, cwd=tmp_path)
    tags = [m.group(2) for m in re.finditer(r"^ARM (?:t212 )? *(e4b|unsloth) (\S+)", p.stdout, re.MULTILINE)]
    assert tags == ["fused_attn4_m_fd", "ckpt_unsloth_m_fv", "ckpt_unsloth_m_fv_d2", "fused_attn4_m_fd_d2"], p.stdout
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("ARM")]
    assert all(ln.startswith("ARM t212 e4b") for ln in lines if "fused_attn4" in ln)       # e4b runs in venv-unsloth, as in amendment 69
    assert all("--profile-steps 3 --profile-warm 3" in ln and "--adapter-dtype fp32 --lora-init matched:3407" in ln for ln in lines)
    assert all("--unsloth-moe-backend grouped_mm" in ln for ln in lines if "ckpt_unsloth" in ln)
    assert "FREE qwen3pos79" in p.stdout
    assert "TC1_ARM_EXTRA_ENV" not in FAMILY and "TC1_E4B_ENV" not in FAMILY      # the family hands its arms no extra environment


KNOB_PREFIXES = ("E4B_", "NF4_QLORA_", "GNF4_")


def _drive_pass_names():
    """Every variable the drive starts the box with: its PASS line's names and its forwarded knob list."""
    line = re.search(r'^PASS="(.*)"$', DRIVE, re.MULTILINE).group(1)
    names = set(re.findall(r"([A-Z][A-Z0-9_]*)=", line))
    fwd = re.search(r"^for v in (TC1_FAMILIES .*?); do", DRIVE, re.MULTILINE | re.DOTALL).group(1)   # the knob forward list
    return names | set(re.findall(r"[A-Z][A-Z0-9_]*", fwd))


def test_every_harness_variable_with_a_knob_prefix_is_exempt():
    """The exempt names are derived from what the harness actually exports, so a new harness variable cannot silently void every row:
    each name the drive starts the box with, or the box script exports, that carries a knob prefix must be in POS79_HARNESS_ENV."""
    names = _drive_pass_names() | set(re.findall(r"^export ((?:[A-Z][A-Z0-9_]*=\S* ?)+)", RUN, re.MULTILINE) and
                                      re.findall(r"^export ([A-Z][A-Z0-9_]*)", RUN, re.MULTILINE))
    names |= {"E4B_VENV"}                                    # arm_once's own call-prefix selector (E4B_VENV=t212 arm ...)
    harness = sorted(n for n in names if n.startswith(KNOB_PREFIXES))
    assert harness and set(harness) <= set(RED.POS79_HARNESS_ENV), (harness, RED.POS79_HARNESS_ENV)
    assert {"E4B_SHA", "GNF4_SHA", "GNF4_SHA_B"} <= set(harness)


def test_the_box_records_from_the_drives_real_environment_pass_the_reducers_box_facts(tmp_path, monkeypatch):
    """Not a hand-written fixture: the gate and the proof stage, cut from tc1_run.sh, run under the environment the drive's PASS line
    starts the box with (every name set), and the summary they write goes through pos79_box."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(KNOB_PREFIXES) and not k.startswith("TC1_")}
    for n in _drive_pass_names():
        env.setdefault(n, "1")
    env.update(E4B_SHA=STAND_IN, GNF4_SHA="30adb1bfa76f95f9ce4a4c29734b8f36de907eca", GNF4_SHA_B="cd" * 20,
               TC1_UNSLOTH_VERSION="2026.10.8", TC1_UNSLOTH_ZOO_VERSION="2026.10.8", TC1_FAMILIES="qwen3pos79")
    for n in ("TC1_E4B_ENV", "TC1_ARM_EXTRA_ENV"):
        env.pop(n, None)                                     # forwarded only when set; the launch template sets neither
    w = tmp_path / "w"
    (w / "logs").mkdir(parents=True)
    (w / "release_proof.py").write_bytes((TC1 / "release_proof.py").read_bytes())      # the staged helper itself: its sha goes on the pins line
    py = tmp_path / "py"
    rec = ('{"dist": "%s", "version": "2026.10.8", "wheel": "%s-2026.10.8-py3-none-any.whl", "record": "RECORD equal", '
           '"record_equal": true, "first_diff": null, "imports": {"%s": "/v"}, "pycache_prefix": "%s", "pycache_fresh": true}\\n')
    py.write_text("#!/usr/bin/env bash\nfor d in unsloth unsloth_zoo; do\n  printf '" + rec + "' \"$d\" \"$d\" \"$d\" \"$PYTHONPYCACHEPREFIX\"\ndone\n")
    py.chmod(0o755)
    gate = GATE.replace('POS79_E4B_SHA=""', f"POS79_E4B_SHA={STAND_IN}", 1)
    script = "\n".join(("say(){ :; }", "finish(){ echo FINISH $1; exit $1; }", f"W={w}", "FAMILIES=qwen3pos79", f"PY_UNS={py}", "UNS_OK=1",
                        "T212_OK=1", f"cd {w}")) + "\n" + gate + PROOF + "pos79_release_proof || exit 9\n"
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    (w / "versions.txt").write_text(f"e4b(t212) 0.58.0 @{STAND_IN}\ngnf4(t212) 0.51.0 @30adb1bfa76f95f9ce4a4c29734b8f36de907eca\n")
    (w / "box.json").write_text(json.dumps({"cpu": "AMD EPYC 7713 64-Core Processor"}))
    monkeypatch.setattr(RED, "POS79_E4B_SHA", STAND_IN)
    fx, bad = RED.pos79_box(str(w))
    assert bad == [] and fx["prefix"].startswith(RED.POS79_PYCACHE_PREFIX), (bad, (w / "summary.txt").read_text())


PROOF = _cut(r"^pos79_release_proof\(\)\{.*?POS79_KNOB_ENV .*?\| tee -a summary\.txt; \}\n")
FAKE_PY = r"""#!/usr/bin/env bash
echo "CWD=$PWD PREFIX=$PYTHONPYCACHEPREFIX MARK=$RELEASE_PROOF_FRESH_PYCACHE ARGS=$*" >> "$CALLS"
[ -d "$PYTHONPYCACHEPREFIX" ] && [ -z "$(ls -A "$PYTHONPYCACHEPREFIX")" ] && echo "PREFIX_FRESH" >> "$CALLS"
echo '{"dist": "unsloth"}'; echo '{"dist": "unsloth_zoo"}'
[ -n "${FAKE_FAIL:-}" ] && { echo "refused: unsloth: unsloth/x.py differs from the published wheel's" >&2; exit 1; }
exit 0
"""


def _proof(tmp: Path, uns_ok=1, t212_ok=1, fail=False, knob=""):
    w = tmp / "w"
    (w / "logs").mkdir(parents=True, exist_ok=True)
    py = tmp / "py"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    calls = tmp / "calls"
    script = (f"say(){{ echo \"SAY $*\"; }}\nW={w}\nPY_UNS={py}\nUNS_OK={uns_ok}\nT212_OK={t212_ok}\nPOS79_UNSLOTH=2026.10.8\n"
              f"POS79_UNSLOTH_ZOO=2026.10.8\nexport CALLS={calls}\n{'export FAKE_FAIL=1' if fail else ''}\n"
              f"export E4B_SHA={STAND_IN} GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca {knob}\ncd {w}\n" + PROOF
              + "pos79_release_proof; rc=$?; echo \"RC=$rc PREFIX_AFTER=${PYTHONPYCACHEPREFIX:-} MARK_AFTER=${RELEASE_PROOF_FRESH_PYCACHE:-}\"\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_", "NF4_QLORA_", "GNF4_"))}   # the box's own, not this runner's
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=clean)
    summary = (w / "summary.txt").read_text() if (w / "summary.txt").exists() else ""
    return p, summary, (calls.read_text() if calls.exists() else ""), w


def test_the_proof_runs_under_one_fresh_prefix_that_the_arms_inherit(tmp_path):
    p, summary, calls, w = _proof(tmp_path)
    assert "RC=0" in p.stdout, p.stdout + p.stderr
    m = re.search(r"CWD=(\S+) PREFIX=(\S+) MARK=(\S+) ARGS=(.*)", calls)
    assert m and m.group(1) == str(w) and m.group(2) == m.group(3) and m.group(2).startswith(RED.POS79_PYCACHE_PREFIX), calls
    assert m.group(4).split()[1:] == ["unsloth==2026.10.8", "unsloth_zoo==2026.10.8"] and "PREFIX_FRESH" in calls
    assert f"PREFIX_AFTER={m.group(2)} MARK_AFTER={m.group(2)}" in p.stdout            # exported: every later arm reads under it
    assert f"RELEASE_PROOF OK unsloth==2026.10.8 unsloth_zoo==2026.10.8 pycache_prefix {m.group(2)}" in summary
    assert (w / "release_proof_unsloth.json").read_text().count('"dist"') == 2
    kn = [ln for ln in summary.splitlines() if ln.startswith("POS79_KNOB_ENV")]
    assert len(kn) == 1 and kn[0].split()[1:] == ["E4B_SHA", "GNF4_SHA"], summary


def test_the_knob_line_names_every_e4b_and_gnf4_variable_the_arms_inherit(tmp_path):
    _, summary, _, _ = _proof(tmp_path, knob="NF4_QLORA_COMPACT_DELTA=1 E4B_CHUNKED_LM_LOSS=256 GNF4_X=1 TC1_STEPS=60")
    kn = [ln for ln in summary.splitlines() if ln.startswith("POS79_KNOB_ENV")][0].split()[1:]
    assert kn == ["E4B_CHUNKED_LM_LOSS", "E4B_SHA", "GNF4_SHA", "GNF4_X", "NF4_QLORA_COMPACT_DELTA"], kn


def test_the_arm_driver_is_not_changed_by_this_amendment():
    """bench/q235's kit pins tc1_arm.py by sha256 (q235.sha256): this amendment records its box facts in tc1_run.sh and leaves the arm
    driver byte-identical."""
    want = [ln.split()[0] for ln in (REPO / "bench" / "q235" / "q235.sha256").read_text().splitlines() if ln.endswith(" tc1_arm.py")]
    assert want == [hashlib.sha256((TC1 / "tc1_arm.py").read_bytes()).hexdigest()]


def test_a_refused_or_unrunnable_proof_ends_the_box_before_any_arm(tmp_path):
    p, summary, _, _ = _proof(tmp_path / "a", fail=True)
    assert "RC=1" in p.stdout and "RELEASE_PROOF FAILED rc=1" in summary and "differs from the published wheel" in summary
    p, summary, calls, _ = _proof(tmp_path / "b", uns_ok=0)
    assert "RC=1" in p.stdout and "RELEASE_PROOF NOT_RUN uns_ok=0" in summary and calls == ""
    assert '[ "$FAMILIES" = qwen3pos79 ] && { pos79_release_proof || finish 9; }' in RUN


def test_the_proof_follows_every_install_and_precedes_the_families():
    at = RUN.index('[ "$FAMILIES" = qwen3pos79 ] && { pos79_release_proof || finish 9; }')
    assert at > RUN.index('say "venv-unsloth + e4b @$E4B_SHA + gnf4 @$GNF4_SHA (the t212 row)"')     # the last install into venv-unsloth
    assert at < RUN.index('for FAM in $FAMILIES; do case "$FAM" in')
    assert "  qwen3pos79) tc1_pos79_family qwen3pos79 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600;;" in RUN


def test_the_screen_and_the_template_carry_one_set_of_caps_inside_the_rows_cap():
    assert KIT.offer_why(32607, 0.90, 0.01) == ""
    assert "32 GB" in KIT.offer_why(24564, 0.90, 0.01) and KIT.offer_why(None, 0.90, 0.01) != ""
    assert KIT.offer_why(32607, 1.01, 0.01) != "" and KIT.offer_why(32607, None, 0.01) != ""
    assert "download 0.021 USD/GB is over 0.02" in KIT.offer_why(32607, 0.90, 0.021) and KIT.offer_why(32607, 0.90, None) != ""
    assert KIT.offer_why(32607, 0.90, 0.01, num_gpus=2) != ""
    assert TEMPLATE["max_download_usd_per_gb"] == KIT.DL_USD_GB and TEMPLATE["usd_per_hour"] == KIT.RATE_USD_H
    assert TEMPLATE["wallclock_h"] == KIT.GUARD_H and TEMPLATE["download_gb"] == KIT.DOWNLOAD_GB
    assert TEMPLATE["storage"] == TEMPLATE["vast_disk_gb"] == KIT.DISK_GB and TEMPLATE["ssh_ready_s"] == KIT.SSH_READY_S == 420
    assert TEMPLATE["gpu"] == "RTX 5090" and TEMPLATE["est_usd"] <= TEMPLATE["usd_per_hour"] * TEMPLATE["wallclock_h"]
    for want in ("TC1_GPU_CLASS=5090", "TC1_FAMILIES=qwen3pos79", "TC1_STEPS=60", "TC1_LOAD_GATE=6.0", "bash bench/tc1/tc1_drive.sh"):
        assert want in TEMPLATE["command"], want
    rec = TEMPLATE["lane_record"]
    assert rec["worst_usd"] == KIT.box_worst() == 5.50 and rec["lane_worst_usd"] == KIT.lane_worst() == 5.93
    assert KIT.lane_worst() <= KIT.WORST_CAP_USD == 6.00 and KIT.nonstart_worst() == 0.43


def test_every_template_guard_names_text_that_exists():
    for g in TEMPLATE["guards"]:
        assert g["repo"] == "e4b" and g["pattern"] in (REPO / g["path"]).read_text(), g
