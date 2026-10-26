"""TC1 amendment 80 (qwen3pos80): e4b#2129's confirmation read. It is amendment 79's box at the 0.58.1 release, with every distribution
PyPI's proven wheel, F15 v2 (bench/tc1/f15v2.py) in place of F15 v1, kernels-only device time, and a card floor (bench/common/card_clock.py).

Every engagement and clock verdict here is read from REAL committed records:
- amendment 69's box (tc1-5090-137: unsloth_zoo 2026.9.9, the torch._grouped_mm path, a clean ~2900 MHz card);
- amendment 79's box (tc1-5090-pos79-r1: zoo 2026.10.8, the Triton path, a 2400 MHz card with NVML's Reliability reason).
Mutants are edits of those records, never hand-written receipts. The box-side pieces are cut from tc1_run.sh and run under bash with stubs,
so no GPU, network or model is needed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TC1 = REPO / "bench" / "tc1"
RUN = (TC1 / "tc1_run.sh").read_text()
DRIVE = (TC1 / "tc1_drive.sh").read_text()
PREREG = (TC1 / "TC1-PREREG.md").read_text()
A80 = PREREG[PREREG.index("### Amendment 80 "):]
A83 = PREREG[PREREG.index("### Amendment 83 "):]   # supersedes amendment 80's card-floor lines
TEMPLATE = json.loads((TC1 / "pos80_manifest_template.json").read_text())
RECEIPTS = REPO / "bench" / "h2h-2026-10-02" / "tc1" / "receipts"
P69 = RECEIPTS / "tc1-5090-137"
P79 = RECEIPTS / "tc1-5090-pos79-r1"
RELEASE_PROOF_SHA256 = "aff3fc80db8d35bbf94a9aaeed125c9d424cfe93bf16eb0e6868b6f9d03f13e0"
GNF4 = "30adb1bfa76f95f9ce4a4c29734b8f36de907eca"
E4B = "605d21e67958fa94d5e73fc3a95ffcb5c8eae02d"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RED = _load(TC1 / "tc1_reduce.py", "tc1_reduce")
F2 = _load(TC1 / "f15v2.py", "f15v2")
CC = _load(REPO / "bench" / "common" / "card_clock.py", "card_clock")
KIT = _load(TC1 / "pos80_kit.py", "pos80_kit")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run_const(name: str) -> str:
    m = re.search(rf"^{name}=([^\s;]*)", RUN, re.MULTILINE) or re.search(rf"[; ]{name}=([^\s;]*)", RUN)
    assert m, name
    return m.group(1).strip('"')


def _cut(pattern: str) -> str:
    m = re.search(pattern, RUN, re.DOTALL | re.MULTILINE)
    assert m, f"pattern no longer matches tc1_run.sh: {pattern}"
    return m.group(0)


def _real(box: Path, arm: str):
    """A real Unsloth arm: (its cell, its full profile). pos79's cells are read from the public copy (identifier-free)."""
    cell = box / "public" / f"{arm}.json" if (box / "public" / f"{arm}.json").exists() else box / f"{arm}.json"
    prof = box / "tc1" / f"{arm}_profile.json" if (box / "tc1").exists() else box / f"{arm}_profile.json"
    return json.loads(cell.read_text()), json.loads(prof.read_text())


REAL_UNSLOTH = [(P69, "qwen3pos69_unsloth_ckpt_unsloth_m_fv"), (P69, "qwen3pos69_unsloth_ckpt_unsloth_m_fv_d2"),
                (P79, "qwen3pos79_unsloth_ckpt_unsloth_m_fv"), (P79, "qwen3pos79_unsloth_ckpt_unsloth_m_fv_d2")]


# ------------------------------------------------------------------ pins, helpers, registration
def test_the_pins_agree_across_the_kit_the_reducer_the_template_and_the_registration():
    assert (_run_const("POS80_E4B_SHA"), _run_const("POS80_GNF4_SHA")) == (RED.POS80_E4B_SHA, RED.POS80_GNF4_SHA) == (E4B, GNF4)
    assert {"experts4bit-qlora": _run_const("POS80_E4B"), "grouped-nf4-gemm": _run_const("POS80_GNF4"), "unsloth": _run_const("POS80_UNSLOTH"),
            "unsloth_zoo": _run_const("POS80_UNSLOTH_ZOO")} == RED.POS80_PINS == {"experts4bit-qlora": "0.58.1", "grouped-nf4-gemm": "0.51.0",
                                                                                "unsloth": "2026.10.8", "unsloth_zoo": "2026.10.8"}
    cmd = TEMPLATE["command"]
    for want in (f"E4B_SHA={E4B}", f"GNF4_SHA={GNF4}", "TC1_UNSLOTH_VERSION=2026.10.8", "TC1_UNSLOTH_ZOO_VERSION=2026.10.8", "TC1_FAMILIES=qwen3pos80"):
        assert want in cmd, want
    for want in (E4B, GNF4, "**0.58.1**", "**0.51.0**", "**2026.10.8**", RELEASE_PROOF_SHA256, "**>= 2700 MHz**"):
        assert want in A80, want
    assert int(_run_const("POS80_CLOCK_FLOOR")) == RED.POS80_CLOCK_FLOOR == 2700


def test_the_registrations_burn_and_draw_numbers_are_the_codes():
    """The PREREG is the contract: its box-fact burn line, its host-floor text and its per-draw rule carry the reducer's and the box's own
    numbers, so text and code cannot drift (a stale min_clean 1.0 survived one review)."""
    m = re.search(r"one passing CARD_CLOCK burn line at (\d+) MHz, mode (\w+), persistent_share ([0-9.]+), with "
                  r"any_sample_bits_every_sample true", A83)
    assert m, "amendment 83's box-fact burn line"
    assert (int(m.group(1)), m.group(2), float(m.group(3))) == (RED.POS80_CLOCK_FLOOR, RED.POS80_CARD_MODE, RED.POS80_PERSISTENT_SHARE)
    assert RED.POS80_PERSISTENT_SHARE == CC.PERSISTENT_SHARE and RED.POS80_CARD_MODE in CC.MODES
    assert re.search(r"--mode (\w+)", _cut(r"card_clock\.py burn .*?\n")).group(1) == RED.POS80_CARD_MODE
    assert f"Reliability 0x400 on **>= {round(RED.POS80_PERSISTENT_SHARE * 100)} %** of busy samples" in A83     # the host-floor text
    assert "SwPowerCap 0x4 is **recorded and never refuses**" in A83
    assert "4. **The card floor on every draw:**" in A83                                               # the per-draw rule moved too
    for stale in ("min_clean 1.0", "20..22", "steps 3..5", "profile_warm 3"):
        assert stale not in A80, stale


def test_the_three_staged_helpers_are_pinned_by_digest_in_the_reducer_and_the_registration():
    got = {"release_proof": _sha(TC1 / "release_proof.py"), "card_clock": _sha(REPO / "bench" / "common" / "card_clock.py"),
           "f15v2": _sha(TC1 / "f15v2.py")}
    assert got == RED.POS80_HELPER_SHA256 and got["release_proof"] == RELEASE_PROOF_SHA256
    for v in got.values():
        assert v in A80, v
    assert 'STAGE="$STAGE $HERE/release_proof.py $HERE/f15v2.py $REPO/bench/common/card_clock.py"' in DRIVE


#: TC1 amendment 80's merge (e4b#2130): the arm driver this amendment left byte-identical. Read there, not in the live tree, since TC1
#: amendment 81 edits tc1_arm.py on main (bench/common rule 6, as ruled for amendment 81 on 2026-10-25).
POS80_FROZEN = "b8f618b10f9f82bff2f0267d06fc0112f47d3f06"


def test_the_arm_driver_is_not_changed_by_this_amendment():
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{POS80_FROZEN}:bench/tc1/tc1_arm.py"], capture_output=True)
    if r.returncode != 0:
        if os.environ.get("CI") == "true":
            pytest.fail(f"amendment 80's merge {POS80_FROZEN[:12]} is not readable: {r.stderr.decode()[:200]}")
        pytest.skip(f"amendment 80's merge {POS80_FROZEN[:12]} is not in this clone (shallow)")
    want = [ln.split()[0] for ln in (REPO / "bench" / "q235" / "q235.sha256").read_text().splitlines() if ln.endswith(" tc1_arm.py")]
    assert want == [hashlib.sha256(r.stdout).hexdigest()]


# ------------------------------------------------------------------ the box: gate, floors, installs, proofs, family
GATE = _cut(r"^POS80_E4B_SHA=.*?\n") + _cut(r"^POS80_GNF4_SHA=.*?\n") + _cut(r"^POS80_E4B=.*?\n") + _cut(r"^POS80_CLOCK_FLOOR=.*?\n") \
    + _cut(r"^case \" \$FAMILIES \" in \*\" qwen3pos80 \"\*\)\n  why=\"\".*?^esac\n")


def _gate(tmp: Path, families="qwen3pos80", e4b=E4B, gnf4=GNF4, uns="2026.10.8", zoo="2026.10.8", staged=("release_proof.py", "card_clock.py", "f15v2.py"), extra=""):
    w = tmp / "w"
    w.mkdir(exist_ok=True)
    for f in staged:
        (w / f).write_text("x")
    script = ("say(){ echo \"SAY $*\"; }\nfinish(){ echo \"FINISH $1\"; exit $1; }\n" + f"W={w}\nFAMILIES='{families}'\nE4B_SHA={e4b}\n"
              f"GNF4_SHA={gnf4}\nTC1_UNSLOTH_VERSION={uns}\nTC1_UNSLOTH_ZOO_VERSION={zoo}\n{extra}\ncd {w}\n" + GATE + "echo PASSED\n")
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return p, ((w / "summary.txt").read_text() if (w / "summary.txt").exists() else "")


def test_the_gate_passes_at_the_kits_pins(tmp_path):
    p, summary = _gate(tmp_path)
    assert p.returncode == 0 and "PASSED" in p.stdout, p.stdout + p.stderr
    assert f"POS80_PINS e4b 0.58.1 @{E4B} gnf4 0.51.0 @{GNF4} unsloth 2026.10.8 unsloth_zoo 2026.10.8 release_proof " in summary
    assert "POS80_ARM_ENV e4b=[] extra=[]" in summary.splitlines()


@pytest.mark.parametrize("kw, why", [
    ({"e4b": "cd" * 20}, "E4B_SHA"), ({"gnf4": "ef" * 20}, "GNF4_SHA"), ({"uns": "2026.10.7"}, "TC1_UNSLOTH_VERSION"),
    ({"zoo": "2026.10.9"}, "TC1_UNSLOTH_ZOO_VERSION"), ({"families": "qwen3pos80 qwen3pos79"}, "runs alone"),
    ({"staged": ("release_proof.py", "f15v2.py")}, "card_clock.py is not staged"), ({"staged": ("release_proof.py", "card_clock.py")}, "f15v2.py is not staged"),
    ({"extra": "TC1_E4B_ENV='NF4_QLORA_COMPACT_DELTA=1'"}, "TC1_E4B_ENV is set"), ({"extra": "TC1_ARM_EXTRA_ENV='E4B_CHUNKED_LM_LOSS=256'"}, "TC1_ARM_EXTRA_ENV is set"),
])
def test_the_gate_refuses_any_other_pin_before_staging(tmp_path, kw, why):
    p, summary = _gate(tmp_path, **kw)
    assert p.returncode == 78 and "PASSED" not in p.stdout and why in summary and "BOX_REFUSED pos80_pins" in summary, (p.stdout, summary)


FLOORS = _cut(r"^# TC1 amendment 80's host floors.*?^case \" \$FAMILIES \" in \*\" qwen3pos80 \"\*\)\n  MOPS=.*?^esac\n")


def _floors(tmp: Path, mops="20.00", burn_rc=0):
    w = tmp / "w"
    (w / "logs").mkdir(parents=True, exist_ok=True)
    bin_ = tmp / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "python3").write_text("#!/usr/bin/env bash\n"
                                  "case \"$*\" in *card_clock.py*burn*) echo 'CARD_CLOCK {\"pass\": " + ("true" if burn_rc == 0 else "false") + "}'; echo \"ARGS $*\" >> \"$ARGS_LOG\"; exit "
                                  f"{burn_rc};; *) exec /usr/bin/env -i PATH=/usr/bin:/bin python3 \"$@\";; esac\n")
    (bin_ / "python3").chmod(0o755)
    script = (f"export PATH={bin_}:$PATH ARGS_LOG={tmp / 'args'}\nsay(){{ echo \"SAY $*\"; }}\nfinish(){{ echo \"FINISH $1\"; exit $1; }}\n"
              f"single_thread_mops(){{ echo {mops}; }}\nW={w}\nFAMILIES=qwen3pos80\nPOS77_MIN_MOPS=15\nPOS80_CLOCK_FLOOR=2700\ncd {w}\n" + FLOORS + "echo PASSED\n")
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return p, ((w / "summary.txt").read_text() if (w / "summary.txt").exists() else ""), ((tmp / "args").read_text() if (tmp / "args").exists() else "")


def test_the_host_floors_pass_a_fast_host_and_a_clean_card(tmp_path):
    p, summary, args = _floors(tmp_path)
    assert "PASSED" in p.stdout and p.returncode == 0, p.stdout + p.stderr
    assert "--floor 2700 --seconds 20 --mode clock --min-clean 0.95" in args and "CARD_CLOCK" in summary and "SINGLE_THREAD 20.00" in summary


def test_a_capped_card_refuses_with_the_host_floor_code_before_any_install(tmp_path):
    p, summary, _ = _floors(tmp_path, burn_rc=18)
    assert p.returncode == 18 and "PASSED" not in p.stdout and "BOX_REFUSED card_clock rc=18 floor=2700" in summary, p.stdout + summary
    p, summary, args = _floors(tmp_path / "slow", mops="12.0")
    assert p.returncode == 18 and "single_thread_mops=12.0" in summary and args == ""     # the slow host refuses before the burn


def test_the_floors_run_after_the_probe_is_defined_and_before_any_install():
    at = RUN.index("# TC1 amendment 80's host floors")
    assert RUN.index("single_thread_mops(){") < at < RUN.index('pip(e4b) rc=')


PROOF = _cut(r"^pos80_release_proof\(\)\{.*?\n  return 0; \}\n")
FAKE_PY = r"""#!/usr/bin/env bash
echo "CWD=$PWD PREFIX=$PYTHONPYCACHEPREFIX MARK=$RELEASE_PROOF_FRESH_PYCACHE ARGS=$*" >> "$CALLS"
for d in experts4bit-qlora grouped-nf4-gemm unsloth unsloth_zoo; do echo "{\"dist\": \"$d\"}"; done
[ -n "${FAKE_FAIL:-}" ] && { echo "refused: unsloth: unsloth/x.py differs from the published wheel's" >&2; exit 1; }
exit 0
"""


def _proof(tmp: Path, kinds=("fastcheck", "binding"), fail=False):
    w = tmp / "w"
    (w / "logs").mkdir(parents=True, exist_ok=True)
    py = tmp / "py"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    calls = tmp / "calls"
    runs = "".join(f"pos80_release_proof {k}; echo \"RC_{k}=$? EXPORTED=${{PYTHONPYCACHEPREFIX:-none}}\"\n" for k in kinds)
    script = (f"say(){{ echo \"SAY $*\"; }}\nW={w}\nPY_UNS={py}\nPOS80_E4B=0.58.1\nPOS80_GNF4=0.51.0\nPOS80_UNSLOTH=2026.10.8\n"
              f"POS80_UNSLOTH_ZOO=2026.10.8\nexport CALLS={calls}\n{'export FAKE_FAIL=1' if fail else ''}\nexport E4B_SHA={E4B} GNF4_SHA={GNF4}\n"
              f"cd {w}\n" + PROOF + runs)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_", "NF4_QLORA_", "GNF4_", "PYTHONPYCACHEPREFIX", "RELEASE_PROOF"))}
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=clean)
    return p, ((w / "summary.txt").read_text() if (w / "summary.txt").exists() else ""), (calls.read_text() if calls.exists() else ""), w


def test_the_fastcheck_uses_a_throwaway_prefix_and_the_binding_proof_exports_its_own_fresh_one(tmp_path):
    p, summary, calls, w = _proof(tmp_path)
    c = re.findall(r"CWD=(\S+) PREFIX=(\S+) MARK=(\S+) ARGS=\S+ (.*)", calls)
    assert len(c) == 2 and all(x[0] == str(w) and x[1] == x[2] and x[1].startswith(RED.POS80_PYCACHE_PREFIX) for x in c), calls
    assert c[0][1] != c[1][1]                                                                   # two prefixes, not one reused
    assert all(x[3].split() == ["experts4bit-qlora==0.58.1", "grouped-nf4-gemm==0.51.0", "unsloth==2026.10.8", "unsloth_zoo==2026.10.8"] for x in c)
    assert "RC_fastcheck=0 EXPORTED=none" in p.stdout                                          # the fastcheck's prefix is never exported
    assert f"RC_binding=0 EXPORTED={c[1][1]}" in p.stdout                                       # the binding prefix is: every arm reads under it
    assert f"RELEASE_PROOF binding OK experts4bit-qlora==0.58.1 grouped-nf4-gemm==0.51.0 unsloth==2026.10.8 unsloth_zoo==2026.10.8 pycache_prefix {c[1][1]}" in summary
    kn = [ln for ln in summary.splitlines() if ln.startswith("POS80_KNOB_ENV")]
    assert len(kn) == 1 and kn[0].split()[1:] == ["E4B_SHA", "GNF4_SHA"]                       # once, with the binding proof only
    assert (w / "release_proof_pos80_binding.json").read_text().count('"dist"') == 4


def test_a_refused_proof_ends_the_box_before_any_arm(tmp_path):
    p, summary, _, _ = _proof(tmp_path, kinds=("binding",), fail=True)
    assert "RC_binding=1" in p.stdout and "RELEASE_PROOF binding FAILED rc=1" in summary and "differs from the published wheel" in summary
    line = '[ "$FAMILIES" = qwen3pos80 ] && { { [ "$UNS_OK" = 1 ] && [ "$T212_OK" = 1 ] && pos80_release_proof binding; } || '
    assert line in RUN and "finish 9; }; }" in RUN[RUN.index(line):RUN.index(line) + 300]


def test_the_proofs_are_defined_before_the_install_that_calls_them_and_bracket_it():
    """The fastcheck runs inside the t212 install, so the function is defined above it. The binding proof follows every install and
    precedes the family loop."""
    define = RUN.index("pos80_release_proof(){")
    fast = RUN.index("pos80_release_proof fastcheck")
    binding = RUN.index('[ "$FAMILIES" = qwen3pos80 ] && { { [ "$UNS_OK" = 1 ]')
    assert define < fast < binding < RUN.index('for FAM in $FAMILIES; do case "$FAM" in')
    assert RUN.index('say "venv-unsloth + experts4bit-qlora==$POS80_E4B') < fast


def test_the_arms_get_pypis_wheels_and_the_tripwire_refuses_a_vcs_install():
    blk = RUN[RUN.index('if [ "$UNS_OK" = 1 ] && [ "$FAMILIES" = qwen3pos80 ]; then'):RUN.index('elif [ "$UNS_OK" = 1 ]; then')]
    assert "--index-url https://pypi.org/simple" in blk and '"experts4bit-qlora==$POS80_E4B" "grouped-nf4-gemm==$POS80_GNF4"' in blk
    assert "git+https" not in blk and 'd.read_text("direct_url.json") is None' in blk
    assert 'e4b(t212) {e.__version__} (PyPI)' in blk


FAMILY = _cut(r"^tc1_pos80_family\(\)\{.*?free_family \$FAM \$\{MID//\\//--\}; \}\n")


def test_the_family_is_amendment_79s_four_arms_in_abba_order_every_one_profiled(tmp_path):
    stubs = ("say(){ :; }\ntc1_prepare(){ TOK=tok; TS=ts; return 0; }\ncan_run(){ return 0; }\nfree_family(){ echo \"FREE $1\"; }\n"
             "arm(){ echo \"ARM ${E4B_VENV:-} $2 $3 $*\"; }\ndraw2(){ echo \"ARM ${E4B_VENV:-} $2 ${3}_d2 $*\"; }\n"
             "MATCHED_SEED=3407 UT7=ut7 PROFILE_STEPS=3 PROFILE_WARM=3 POS80_PROFILE_WARM=20 POS80_E4B=0.58.1 POS80_UNSLOTH=2026.10.8\n")
    p = subprocess.run(["bash", "-c", stubs + FAMILY + "tc1_pos80_family qwen3pos80 M R 1 2 3\n"], capture_output=True, text=True, cwd=tmp_path)
    tags = [m.group(2) for m in re.finditer(r"^ARM (?:t212 )? *(e4b|unsloth) (\S+)", p.stdout, re.MULTILINE)]
    assert tags == ["fused_attn4_m_fd", "ckpt_unsloth_m_fv", "ckpt_unsloth_m_fv_d2", "fused_attn4_m_fd_d2"], p.stdout
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("ARM")]
    assert all("--profile-steps 3 --profile-warm 20" in ln for ln in lines) and all(ln.startswith("ARM t212 e4b") for ln in lines if "fused_attn4" in ln)
    assert int(_run_const("POS80_PROFILE_WARM")) == 20                    # past Unsloth's compile transient, inside the 11..60 median
    assert "  qwen3pos80) tc1_pos80_family qwen3pos80 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600;;" in RUN


# ------------------------------------------------------------------ F15 v2 on the real records, and its mutants
@pytest.mark.parametrize("box, arm", REAL_UNSLOTH, ids=lambda x: getattr(x, "name", x))
def test_f15v2_reads_every_real_unsloth_draw_engaged_on_its_own_path(box, arm):
    cell, prof = _real(box, arm)
    v, why, d = F2.engagement(cell, prof)
    assert v == "ENGAGED", why
    if cell["env"]["unsloth_zoo"] == "2026.9.9":
        assert d["path"] == "torch" and d["torch_grouped"] == 11 * d["E"] == 4224
    else:
        assert d["path"] == "triton" and (d["triton_gemm"], d["triton_wgrad"]) == (9 * d["E"], 2 * d["E"]) == (3456, 768)
    assert "reported, not binding" in F2.v1_line(cell)


def _scale(prof, names, k):
    p = copy.deepcopy(prof)
    for key in ("top_cpu", "top_device"):
        for r in p[key]:
            if r["name"] in names:
                r["count"] = r["count"] * k
    return p


MUTANTS = {
    "a 2x double dispatch": lambda c, p: (c, _scale(p, (F2.CPU_GROUPED, F2.TRITON_GEMM, F2.TRITON_WGRAD), 2)),
    "a 10 % bypass": lambda c, p: (c, _scale(p, (F2.CPU_GROUPED, F2.TRITON_GEMM), 0.9)),
    "a 50 % bypass": lambda c, p: (c, _scale(p, (F2.CPU_GROUPED, F2.TRITON_GEMM), 0.5)),
    "a full bypass": lambda c, p: (c, {**p, "top_cpu": [r for r in p["top_cpu"] if r["name"] != F2.CPU_GROUPED],
                                       "top_device": [r for r in p["top_device"] if r["name"] not in (F2.TRITON_GEMM, F2.TRITON_WGRAD)]}),
    "no expert forwards": lambda c, p: ({**c, "unsloth_backend_calls_per_step_min": {k: 0 for k in c["unsloth_backend_calls_per_step_min"]}}, p),
    "the manual fallback": lambda c, p: ({**c, "unsloth_manual_grouped_mm_calls_per_step_max": 1}, p),
    "the counted rows below the kept 80": lambda c, p: (c, {**p, "top_cpu": [r for r in p["top_cpu"] if r["name"] != F2.CPU_GROUPED][:80],
                                                             "top_device": [r for r in p["top_device"] if r["name"] not in (F2.TRITON_GEMM, F2.TRITON_WGRAD)][:80]}),
}


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
@pytest.mark.parametrize("box, arm", REAL_UNSLOTH, ids=lambda x: getattr(x, "name", x))
def test_every_mutant_of_a_real_record_voids(box, arm, mutant):
    cell, prof = _real(box, arm)
    c, p = MUTANTS[mutant](cell, prof)
    assert F2.engagement(c, p)[0] == "VOID", mutant


@pytest.mark.parametrize("arm", ["qwen3pos79_unsloth_ckpt_unsloth_m_fv", "qwen3pos79_unsloth_ckpt_unsloth_m_fv_d2"])
def test_a_triton_dw_short_by_ten_percent_voids(arm):
    cell, prof = _real(P79, arm)
    assert F2.engagement(cell, _scale(prof, (F2.TRITON_WGRAD,), 0.9))[0] == "VOID"


def test_the_torch_path_on_a_zoo_with_no_registered_bar_voids_and_a_mixed_dispatch_voids():
    cell, prof = _real(P69, "qwen3pos69_unsloth_ckpt_unsloth_m_fv")
    relabel = {**cell, "env": {**cell["env"], "unsloth_zoo": "2026.10.8"}}
    v, why, _ = F2.engagement(relabel, prof)
    assert v == "VOID" and "no bar is registered for that version" in why
    cell, prof = _real(P79, "qwen3pos79_unsloth_ckpt_unsloth_m_fv")
    mixed = copy.deepcopy(prof)
    mixed["top_cpu"].append({"name": F2.CPU_GROUPED, "count": 30, "self_cpu_ms": 1.0, "self_device_ms": 0.0, "device": False, "family": "matmul"})
    v, why, _ = F2.engagement(cell, mixed)
    assert v == "VOID" and "mixed dispatch" in why


def test_kernels_only_device_time_on_a_real_profile_drops_the_annotation_span_and_an_injected_one():
    prof = json.loads((P79 / "tc1" / "qwen3pos79_e4b_fused_attn4_m_fd_profile.json").read_text())
    ms, ann, names = F2.kernels_only_device_ms(prof)
    assert names == ["Optimizer.step#AdamW8bit.step"] and round(ann, 1) == 85.3 and round(ms, 1) == 1979.5
    inj = copy.deepcopy(prof)
    inj["top_device"].append({"name": "ProfilerStep#4", "count": 3, "self_cpu_ms": 0.0, "self_device_ms": 30.0, "device": True, "family": "other"})
    inj["summary"]["device_ms"] += 30.0
    assert round(F2.kernels_only_device_ms(inj)[0], 1) == 1979.5                         # an injected annotation never moves the kernel total


@pytest.mark.parametrize("edit", ["the row removed", "a wrong count", "two rows"])
def test_kernels_only_device_time_fails_closed_without_exactly_one_annotation_row(edit):
    """Review finding 1: without the Optimizer.step# row the recorded total would pass as kernels-only (2064.8, not 1979.5, on this real
    profile). Any profile without exactly one such row, counted once a profiled step, raises."""
    prof = json.loads((P79 / "tc1" / "qwen3pos79_e4b_fused_attn4_m_fd_profile.json").read_text())
    rows = prof["top_device"]
    i = next(k for k, r in enumerate(rows) if r["name"].startswith("Optimizer.step#"))
    if edit == "the row removed":
        del rows[i]
    elif edit == "a wrong count":
        rows[i]["count"] = 2
    else:
        rows.append(dict(rows[i]))
    with pytest.raises(F2.AnnotationError):
        F2.kernels_only_device_ms(prof)


@pytest.mark.parametrize("method", ["kernels_only_at_source", "something_else"])
def test_kernels_only_device_time_refuses_a_profile_already_corrected_at_source(method):
    """The maintainer's 02:51:56Z ruling: amendment 81's profiles carry device_ms_method == kernels_only_at_source; subtracting the
    annotation row from one again would correct it twice, so this method refuses any value but the legacy one."""
    prof = json.loads((P79 / "tc1" / "qwen3pos79_e4b_fused_attn4_m_fd_profile.json").read_text())
    assert round(F2.kernels_only_device_ms(prof)[0], 1) == 1979.5                       # legacy: no field
    prof["summary"]["device_ms_method"] = "all_cuda"
    assert round(F2.kernels_only_device_ms(prof)[0], 1) == 1979.5
    prof["summary"]["device_ms_method"] = method
    with pytest.raises(F2.AnnotationError, match="corrected twice"):
        F2.kernels_only_device_ms(prof)


def test_a_draw_whose_profile_lacks_the_annotation_row_is_void(tmp_path):
    d = _run_dir(tmp_path)
    pp = d / "qwen3pos80_e4b_fused_attn4_m_fd_profile.json"
    prof = json.loads(pp.read_text())
    prof["top_device"] = [r for r in prof["top_device"] if not r["name"].startswith("Optimizer.step#")]
    pp.write_text(json.dumps(prof))
    rows = {(x["fw"], x["tag"]): x for x in RED.reduce_dir(str(d))["qwen3pos80"]["rows"]}
    x = rows[("e4b", "fused_attn4_m_fd")]
    assert x["verdict"] == "VOID" and "kernels-only device time" in x["why"], x.get("why")


def test_a_helper_off_its_digest_is_never_executed(tmp_path, monkeypatch):
    """Review finding 2: the reducer executes f15v2.py / card_clock.py only at the kit's sha256."""
    monkeypatch.setitem(RED.POS80_HELPER_SHA256, "f15v2", "0" * 64)
    assert RED._pos80_helper("f15v2") is None
    monkeypatch.setitem(RED.POS80_HELPER_SHA256, "f15v2", _sha(TC1 / "f15v2.py"))
    assert RED._pos80_helper("f15v2") is not None


# ------------------------------------------------------------------ the reducer, end to end on a run built from the real receipts
def _run_dir(tmp: Path, clocks: str = "clean", pins: bool = True, burn: bool = True, mutate=None):
    """A qwen3pos80 run directory derived from amendment 79's committed receipt copy:
    - the four arms' cells (public/) and full profiles, renamed and re-pinned to amendment 80's distributions;
    - each arm's clock log: its own (capped) or amendment 69's same arm (clean);
    - the summary and proof records the box writes."""
    d = tmp / "tc1"
    d.mkdir(parents=True)
    for f in (P79 / "tc1").iterdir():
        if f.is_file() and not f.name.startswith(("gpuclk_", "qwen3pos79_")):
            shutil.copy(f, d / f.name)
    shutil.copytree(P79 / "tc1" / "logs", d / "logs")
    for fw, tag in (("e4b", "fused_attn4_m_fd"), ("unsloth", "ckpt_unsloth_m_fv"), ("unsloth", "ckpt_unsloth_m_fv_d2"), ("e4b", "fused_attn4_m_fd_d2")):
        src = f"qwen3pos79_{fw}_{tag}"
        dst = f"qwen3pos80_{fw}_{tag}"
        cell = json.loads((P79 / "public" / f"{src}.json").read_text())
        cell["fam"] = "qwen3pos80"
        if pins:
            cell["env"].update({"experts4bit-qlora": "0.58.1", "grouped-nf4-gemm": "0.51.0"})
        if mutate:
            cell = mutate(fw, tag, cell)
        (d / f"{dst}.json").write_text(json.dumps(cell))
        shutil.copy(P79 / "tc1" / f"{src}_profile.json", d / f"{dst}_profile.json")
        clock = P79 / "tc1" / f"gpuclk_{src}.txt" if clocks == "capped" else P69 / f"gpuclk_qwen3pos69_{fw}_{tag}.txt"
        shutil.copy(clock, d / f"gpuclk_{dst}.txt")
        lg = P79 / "tc1" / "logs" / f"run_{src}.log"
        if lg.exists():
            shutil.copy(lg, d / "logs" / f"run_{dst}.log")
    shutil.copy(P79 / "public" / "box.json", d / "box.json")
    summary = (P79 / "public" / "summary.txt").read_text().replace("qwen3pos79", "qwen3pos80")
    summary = "\n".join(ln for ln in summary.splitlines() if not ln.startswith(("POS79_", "RELEASE_PROOF")))
    pfx = "/tmp/tc1-pycache-abc123"
    hs = RED.POS80_HELPER_SHA256
    summary += (f"\nPOS80_PINS e4b 0.58.1 @{E4B} gnf4 0.51.0 @{GNF4} unsloth 2026.10.8 unsloth_zoo 2026.10.8 release_proof {hs['release_proof']} "
                f"card_clock {hs['card_clock']} f15v2 {hs['f15v2']}\nPOS80_ARM_ENV e4b=[] extra=[]\nSINGLE_THREAD 18.20 Mops (median of 5; floor 15; amendment 80)\n"
                + ("CARD_CLOCK " + json.dumps({"pass": True, "floor_mhz": 2700, "mode": "clock", "persistent_share": 0.95,
                                              "any_sample_bits_every_sample": True, "sm_median": 2895}) + "\n" if burn else "")
                + "RELEASE_PROOF fastcheck OK experts4bit-qlora==0.58.1 grouped-nf4-gemm==0.51.0 unsloth==2026.10.8 unsloth_zoo==2026.10.8 pycache_prefix /tmp/tc1-pycache-zzz999\n"
                f"RELEASE_PROOF binding OK experts4bit-qlora==0.58.1 grouped-nf4-gemm==0.51.0 unsloth==2026.10.8 unsloth_zoo==2026.10.8 pycache_prefix {pfx}\n"
                "POS80_KNOB_ENV E4B_SHA GNF4_SHA\n")
    (d / "summary.txt").write_text(summary)
    own = RED.POS80_OWN_IMPORT
    recs = [{"dist": k, "version": v, "wheel": f"{k.replace('-', '_')}-{v}-py3-none-any.whl", "record": "RECORD equal", "record_equal": True,
             "first_diff": None, "imports": {own[k]: "/v"}, "pycache_prefix": pfx, "pycache_fresh": True} for k, v in RED.POS80_PINS.items()]
    (d / "release_proof_pos80_binding.json").write_text("".join(json.dumps(r) + "\n" for r in recs))
    vt = (d / "versions.txt").read_text()
    (d / "versions.txt").write_text(vt + "e4b(t212) 0.58.1 (PyPI)\ngnf4(t212) 0.51.0 (PyPI)\n")
    return d


def test_the_real_capped_card_voids_every_draw_so_the_rule_reads_not_read(tmp_path):
    d = _run_dir(tmp_path, clocks="capped")
    F = RED.reduce_dir(str(d))
    rows = F["qwen3pos80"]["rows"]
    assert [x["verdict"] for x in rows] == ["VOID"] * 4
    assert all(re.search(r"card floor: busy SM median 2400(\.0)? MHz < floor 2700", x.get("why") or "") and "Reliability" in x["why"]
               for x in rows), [x.get("why") for x in rows]
    assert RED.score_pos80(F, str(d))[0] == "NOT_READ"


def test_on_a_clean_card_the_real_draws_read_valid_and_the_rule_runs_to_a_verdict(tmp_path):
    d = _run_dir(tmp_path, clocks="clean")
    F = RED.reduce_dir(str(d))
    rows = {(x["fw"], x["tag"]): x for x in F["qwen3pos80"]["rows"]}
    assert all(x["verdict"] == "VALID" for x in rows.values()), {k: x.get("why") for k, x in rows.items()}
    v, ev, q = RED.score_pos80(F, str(d))
    assert v in ("QUOTED", "UNSTABLE", "QUALITY_DIFFERS"), (v, ev)
    if v == "QUOTED":
        assert "kernels only" in ev[-1] and q["e4b_dev_ms"] < 2064.8 and q["unsloth_dev_ms"] < 1725.9        # the annotation rows are out
    preds = {p: (s, e) for p, _, s, e in RED.score_pos80_predictions(F, str(d))}
    assert preds["P225"][0] == "HELD"
    md = RED.render(F, str(d))
    assert "## Amendment 80" in md and "F15 v1 (reported, not binding)" in md and "**Amendment 80 rule:" in md
    assert "Mega-cache no trusted bundle" in md and "Mega-cache loaded bundle" in md        # the real logs' cache states, per draw


@pytest.mark.parametrize("drop, why", [
    ("POS80_PINS", "no POS80_PINS line"), ("RELEASE_PROOF binding", "RELEASE_PROOF binding OK lines"),
    ("RELEASE_PROOF fastcheck", "RELEASE_PROOF fastcheck OK lines"), ("CARD_CLOCK", "no single passing CARD_CLOCK"),
    ("SINGLE_THREAD", "single-thread speed None"), ("POS80_KNOB_ENV", "POS80_KNOB_ENV lines, not one"),
])
def test_every_box_fact_fails_closed(tmp_path, drop, why):
    d = _run_dir(tmp_path)
    sp = d / "summary.txt"
    sp.write_text("\n".join(ln for ln in sp.read_text().splitlines() if not ln.startswith(drop)) + "\n")
    fx, bad = RED.pos80_box(str(d))
    assert any(why in b for b in bad), bad


def test_a_proof_record_off_the_pins_and_a_knob_in_the_arms_environment_fail_closed(tmp_path):
    d = _run_dir(tmp_path)
    jp = d / "release_proof_pos80_binding.json"
    recs = [json.loads(x) for x in jp.read_text().splitlines()]
    recs[1]["record_equal"] = False
    jp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    sp = d / "summary.txt"
    sp.write_text(sp.read_text().replace("POS80_KNOB_ENV E4B_SHA GNF4_SHA", "POS80_KNOB_ENV E4B_SHA GNF4_SHA NF4_QLORA_COMPACT_DELTA"))
    _, bad = RED.pos80_box(str(d))
    assert any("release proof of grouped-nf4-gemm: not RECORD equal" in b for b in bad) and any("NF4_QLORA_COMPACT_DELTA" in b for b in bad), bad


def test_an_arm_off_the_pins_voids(tmp_path):
    d = _run_dir(tmp_path, pins=False)                                  # the real cells as recorded: e4b 0.58.0, amendment 79's pin
    F = RED.reduce_dir(str(d))
    assert all(x["verdict"] == "VOID" and "experts4bit-qlora '0.58.0' != 0.58.1" in x["why"] for x in F["qwen3pos80"]["rows"])


# ------------------------------------------------------------------ the kit, the drive
def test_the_screen_and_the_template_carry_one_set_of_caps_inside_the_rows_cap():
    assert KIT.offer_why(32607, 0.90, 0.01) == "" and KIT.offer_why(24564, 0.90, 0.01) != ""
    assert TEMPLATE["usd_per_hour"] == KIT.RATE_USD_H and TEMPLATE["wallclock_h"] == KIT.GUARD_H and TEMPLATE["download_gb"] == KIT.DOWNLOAD_GB
    rec = TEMPLATE["lane_record"]
    assert rec["worst_usd"] == KIT.box_worst() == 5.50 and rec["lane_worst_usd"] == KIT.lane_worst() == 5.94 and KIT.nonstart_worst() == 0.44
    assert KIT.lane_worst() <= KIT.WORST_CAP_USD == 6.00
    for want in ("**$5.50**", "**$0.44**", "**$5.94**"):
        assert want in A80, want


def test_every_template_guard_names_text_that_exists():
    for g in TEMPLATE["guards"]:
        assert g["repo"] == "e4b" and g["pattern"] in (REPO / g["path"]).read_text(), g


def _drive(tmp: Path, **override):
    import shlex
    import time
    words = shlex.split(TEMPLATE["command"])
    assigns = dict(w.split("=", 1) for w in words[:words.index("bash")])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_", "TC1_", "GNF4_"))}
    env.update(assigns, TC1_DRIVE_DRYRUN="1", E4B_RENT_SSH_HOST="fixture.invalid", E4B_RENT_SSH_PORT="22", E4B_RENT_SSH_OPTS="-o BatchMode=yes",
               E4B_RENT_RUN_DIR=str(tmp), E4B_RENT_RUN_ID="tc1-5090-pos80", E4B_RENT_DEADLINE_EPOCH=str(int(time.time()) + 7200),
               E4B_RENT_INSTANCE_ID="fixture-no-instance")
    for k, v in override.items():
        env[k] = v
    return subprocess.run(["bash", str(TC1 / "tc1_drive.sh")], cwd=REPO, env=env, text=True, capture_output=True, timeout=60)


def test_pos80_drive_dry_run_stages_the_three_helpers_at_the_kits_pins(tmp_path):
    r = _drive(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    staged = re.search(r"DRYRUN stage \[(.*?)\]", r.stdout).group(1).split()
    assert {"release_proof.py", "f15v2.py", "card_clock.py", "tc1_run.sh", "tc1_reduce.py"} <= set(staged), staged
    start = r.stdout[r.stdout.index("DRYRUN stage"):]
    for want in (f"E4B_SHA={E4B}", f"GNF4_SHA={GNF4}", "TC1_FAMILIES=qwen3pos80", "TC1_UNSLOTH_VERSION=2026.10.8", "TC1_UNSLOTH_ZOO_VERSION=2026.10.8"):
        assert want in start, want


@pytest.mark.parametrize("override, why", [({"E4B_SHA": "cd" * 20}, "E4B_SHA cdcd"), ({"GNF4_SHA": "ef" * 20}, "GNF4_SHA efef"),
                                           ({"TC1_UNSLOTH_VERSION": "2026.10.7"}, "TC1_UNSLOTH_VERSION 2026.10.7"),
                                           ({"TC1_E4B_ENV": "NF4_QLORA_COMPACT_DELTA=1"}, "must be empty")])
def test_the_drive_refuses_any_other_pin_before_anything_is_created(tmp_path, override, why):
    r = _drive(tmp_path, **override)
    assert r.returncode == 78 and "refusing (amendment 80 pins)" in r.stdout and why in r.stdout and "DRYRUN stage" not in r.stdout, r.stdout + r.stderr
