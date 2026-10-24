# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P235 (bench/p235/PREREG-p235.md): P58's same-box vLLM head-to-head re-read at the release pins (e4b 0.58.0,
grouped-nf4-gemm 0.51.0). The staged pins, resolved the way test_p58_staged_pin.py resolves P58's and held to P58's
digests; the runner as P58's by named edits (every changed line one of them: the release-pin asserts in the tripwire,
transformers 5.17.0, the 0.30.0 secondary, no GPU UUID); the reducer as P58's with its labels renamed (its output on
P58's receipts unchanged but for the title); and the drive's pod-launch contract, refusals and dry run (P233's form)."""
import ast
import difflib
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types

import pytest
from guard_outcomes import expect_outcome

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p235"
P58 = B / "p58"
PIN = LANE / "staged.sha256"
#: staged path (as the box sees it) -> the file in this checkout
SOURCES = {"p235_run.sh": LANE / "p235_run.sh", "p235_reduce.py": LANE / "p235_reduce.py",
           "p37_vllm.py": B / "h2h-20260905" / "p37" / "p37_vllm.py", "step_decomp.py": B / "p39" / "step_decomp.py",
           "k8_bake.py": B / "p39" / "k8_bake.py", "calib.json": B / "p39" / "calib.json",
           "hook/usercustomize.py": B / "p42" / "hook" / "usercustomize.py"}
RUN = (LANE / "p235_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p235_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p235.md").read_text(encoding="utf-8")
FLAT = " ".join(PREREG.split())     # the PREREG's wrapped lines joined: a phrase reads across a line break
NEW_FILES = [LANE / f for f in ("p235_run.sh", "p235_reduce.py", "p235_drive.sh", "staged.sha256", "PREREG-p235.md")] + [
    REPO / "tests" / "test_p235.py", REPO / "changelog.d" / "p235-registered.md"]
#: P58's names for what P235 renamed, so P58's text and this lane's compare line for line
TO_P58 = (("/root/p235", "/root/p58"), ("P235_", "P58_"), ("p235_reduce.py", "p58_reduce.py"),
          ("RESULTS-p235-generated.md", "RESULTS-p58-generated.md"), ("] p235: ", "] p58: "), ('"P235 arm=', '"P58 arm='),
          ("bench/p235/staged.sha256", "bench/p58/staged.sha256"))


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _resolve(name):
    """test_p58_staged_pin.py's map, for this lane's names: the runner and the reducer from bench/p235; p37_vllm.py
    from bench/h2h-20260905/p37 (byte-identical to P37's); the hook from bench/p42; everything else (step_decomp.py,
    k8_bake.py, calib.json) from bench/p39."""
    if name in ("p235_run.sh", "p235_reduce.py"):
        return LANE / name
    if name == "p37_vllm.py":
        return B / "h2h-20260905" / "p37" / name
    if name.startswith("hook/"):
        return B / "p42" / name
    return B / "p39" / name


@pytest.mark.parametrize("name,want", list(_pins(PIN).items()))
def test_staged_file_matches_its_pin(name, want):
    src = _resolve(name)
    assert src == SOURCES[name] and src.is_file(), (name, src)
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}"


def test_every_staged_piece_is_pinned_and_the_harness_is_p58s():
    """The pins name exactly the staged pieces, and every piece but the runner and the reducer is at P58's digest."""
    mine, p58 = _pins(PIN), _pins(P58 / "staged.sha256")
    assert list(mine) == list(SOURCES)
    shared = ("p37_vllm.py", "step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py")
    assert all(mine[n] == p58[n] for n in shared), {n: (mine[n][:8], p58[n][:8]) for n in shared}
    assert set(p58) - set(shared) == {"p58_run.sh", "p58_reduce.py"}


def _as_p58(text):
    for new, old in TO_P58:
        text = text.replace(new, old)
    return text


def _body(text):
    """the script below its header comment (from `set -uo pipefail` on)"""
    return text[text.index("set -uo pipefail\n"):]


#: one needle per line P235 adds to P58's runner body: each named edit, and nothing else
RUN_EDITS = ["VLLM_SECONDARY=${P58_VLLM_SECONDARY:-0.30.0}",
             "# ---- install: e4b at the launch commit (0.58.0) + P235's toolchain pins (image python); gnf4 at GNF4_SHA (0.51.0)",
             'say "install e4b @$E4B_SHA (image python; P235 pins)"', '"transformers==5.17.0" "bitsandbytes==0.50.1"',
             "import experts4bit_qlora", 'assert experts4bit_qlora.__version__ == "0.58.0"',
             'assert md.version("grouped-nf4-gemm") == "0.51.0"', "(P235 runs the two-launch default, as P58)",
             "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt",
             "SECONDARY = 0.30.0 pinned"]


def test_the_runner_is_p58s_by_named_edits():
    """With the lane's names mapped back, every line P235 adds to P58's runner is one of the named edits and every edit
    is there: the arms, their order, the alarms, the STOP rules, the engagement greps and the reducer call are P58's."""
    diff = list(difflib.unified_diff(_body((P58 / "p58_run.sh").read_text(encoding="utf-8")).splitlines(),
                                     _body(_as_p58(RUN)).splitlines(), lineterm="", n=0))
    added = [ln[1:] for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln[1:] for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    hits = {ln: [n for n in RUN_EDITS if n in ln] for ln in added}
    assert all(len(h) == 1 for h in hits.values()), hits
    assert sorted(n for h in hits.values() for n in h) == sorted(RUN_EDITS)
    assert len(removed) == len(RUN_EDITS) - 3          # the three tripwire lines are insertions
    assert any("driver_version,uuid" in ln for ln in removed) and any("transformers==5.16.1" in ln for ln in removed)


def test_the_pins_the_prereg_states_are_the_runners():
    assert 'MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;' in RUN
    assert "GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4; GPTQ_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441" in RUN
    assert '"transformers==5.17.0" "bitsandbytes==0.50.1"' in RUN and "VLLM_SECONDARY=${P235_VLLM_SECONDARY:-0.30.0}" in RUN
    assert "https://pypi.org/pypi/vllm/json" in RUN and "P235_VLLM_PRIMARY" not in DRIVE   # the drive passes no pin
    assert '"git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA"' in RUN


def _tripwire():
    head = "PYTHONPATH=$W/hook python - <<'PYT' || { say \"TRIPWIRE FAIL (e4b)\"; finish 9; }\n"
    assert RUN.count(head) == 1
    body = RUN[RUN.index(head) + len(head):]
    return body[:body.index("\nPYT\n")]


def test_the_tripwire_asserts_the_release_versions_first():
    """The version asserts sit in the e4b tripwire, whose failure finishes 9, ahead of every API assert."""
    tw = _tripwire()
    ast.parse(tw)
    lines = tw.splitlines()
    i = lines.index("import experts4bit_qlora")
    assert lines[i + 1].startswith('assert experts4bit_qlora.__version__ == "0.58.0", ')
    assert lines[i + 2].startswith('assert md.version("grouped-nf4-gemm") == "0.51.0", ')
    assert i < next(k for k, ln in enumerate(lines) if "_smallm_kernels" in ln)
    assert lines[0].startswith("import site, os, inspect, importlib.metadata as md")


@pytest.mark.parametrize("e4b,gnf4,ok", [("0.58.0", "0.51.0", True), ("0.57.0", "0.51.0", False),
                                         ("0.58.1", "0.51.0", False), ("0.58.0", "0.50.0", False),
                                         ("0.58.0", "0.51.1", False)])
def test_the_tripwire_refuses_any_other_release(monkeypatch, e4b, gnf4, ok):
    lines = _tripwire().splitlines()
    i = lines.index("import experts4bit_qlora")
    pin = "\n".join(lines[i:i + 3])
    monkeypatch.setitem(sys.modules, "experts4bit_qlora", types.SimpleNamespace(__version__=e4b))
    md = types.SimpleNamespace(version=lambda name: {"grouped-nf4-gemm": gnf4}[name])
    if ok:
        exec(pin, {"md": md})
    else:
        with pytest.raises(AssertionError, match="is not the 0.5"):
            exec(pin, {"md": md})


def test_the_runner_never_collects_the_gpu_uuid():
    """The runner never asks for the card's GPU UUID: no nvidia-smi query names uuid, and no nvidia-smi -L, -q or
    --list-gpus (which print it) runs. P58's query line is refused."""
    def bad(src):
        return [ln for ln in src.splitlines() if not ln.lstrip().startswith("#") and "nvidia-smi" in ln
                and ("uuid" in ln.lower() or re.search(r"nvidia-smi\s+(-L|-q|--list-gpus)(\s|$)", ln))]
    assert sum("nvidia-smi --query-gpu=" in ln for ln in RUN.splitlines()) >= 2   # the check has its input
    assert not bad(RUN), bad(RUN)
    assert bad(RUN.replace("driver_version --format", "driver_version,uuid --format"))
    assert bad((P58 / "p58_run.sh").read_text(encoding="utf-8"))
    assert bad("nvidia-smi -L") and bad("nvidia-smi -q | tee f") and bad("nvidia-smi --list-gpus")


def _docstring_split(text):
    a = text.index('"""') + 3
    b = text.index('"""', a)
    return text[a:b], text[b:]


def test_the_reducer_is_p58s_with_its_labels_renamed(tmp_path):
    """Below the docstring the reducer is P58's byte for byte but the title, and on P58's committed receipts it prints
    P58's table but for the title line."""
    mine_doc, mine = _docstring_split((LANE / "p235_reduce.py").read_text(encoding="utf-8"))
    p58_doc, p58 = _docstring_split((P58 / "p58_reduce.py").read_text(encoding="utf-8"))
    assert mine.count('"# P235 -- ') == 1 and mine.replace('"# P235 -- ', '"# P58 -- ') == p58
    changed = [k for k, (x, y) in enumerate(zip(mine_doc.splitlines(), p58_doc.splitlines())) if x != y]
    assert len(mine_doc.splitlines()) == len(p58_doc.splitlines()) and changed == [0, 1, 3, 7], changed
    rec = P58 / "receipts"
    outs = [subprocess.run([sys.executable, str(s), str(rec)], capture_output=True, text=True, timeout=120)
            for s in (LANE / "p235_reduce.py", P58 / "p58_reduce.py")]
    assert all(o.returncode == 0 for o in outs), [o.stderr for o in outs]
    (m, *m_rest), (p, *p_rest) = (o.stdout.splitlines() for o in outs)
    assert m.startswith("# P235 -- ") and p.startswith("# P58 -- ") and m_rest == p_rest and len(m_rest) > 40


# ----------------------------------------------------------------------------------------------------- drive --

def test_the_drive_follows_pod_launchs_contract():
    """pod-launch runs a reading's drive without E4B_SHA: the drive derives the launch commit from the clean HEAD,
    refuses an override, takes grouped-nf4-gemm's commit as a checked input and passes both to the runner, and fetches
    the box tree into $RUN_DIR/p235/, refusing a non-empty target."""
    assert 'if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi' in DRIVE
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)' in DRIVE and "is dirty" in DRIVE
    assert 'OUT="$RUN_DIR/p235"' in DRIVE and '"$RUN_DIR/"' not in DRIVE
    assert 'mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }' in DRIVE
    assert ('case "$RUN_ID" in\n  p235-5090-1|p235-5090-2|p235-5090-3|p235-5090-r1|p235-5090-r2|p235-5090-r3|p235-5090-r4|'
            'p235-5090-r5|p235-5090-r6) :;;\n  *) say "refusing unregistered run ID') in DRIVE
    assert 'E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA"' in DRIVE and DRIVE.index("GNF4_SHA=${GNF4_SHA:-}") < DRIVE.index("PASS=")
    for name, path in SOURCES.items():
        assert path.relative_to(REPO).as_posix() in DRIVE, name
    assert "bench/p235/staged.sha256" in DRIVE and 'cp "$REPO/$f" "$STAGE/hook/"' in DRIVE
    assert "--include 'work_qwen3/' --include 'work_qwen3/bake.json' --exclude 'work_qwen3/*'" in DRIVE
    for word in ("vast.py", "destroy", "/api/", "rent-watch"):
        assert word not in DRIVE.split("set -uo pipefail", 1)[1], word


def _fetches_without_no_links(drive):
    """(rsync lines, the ones without --no-links): the drive's live rsync invocations, comments skipped."""
    lines = [ln for ln in drive.splitlines() if re.search(r"\brsync\b", ln) and not ln.lstrip().startswith("#")]
    return lines, [ln for ln in lines if not re.search(r"\brsync\s(?:[^|;&]*\s)?--no-links\b", ln)]


def test_the_fetch_never_imports_a_link():
    lines, bad = _fetches_without_no_links(DRIVE)
    assert len(lines) == 1 and not bad, (lines, bad)
    assert _fetches_without_no_links(DRIVE.replace(" --no-links", ""))[1]          # the old line is refused
    assert not _fetches_without_no_links("echo fetch\n")[0]                          # no rsync -> the count above fails


GNF4 = "b" * 40


def _dry_env(tmp_path, run_id, **extra):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P235_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0",
            "GNF4_SHA": GNF4, **extra}


def _drive(tmp_path, run_id, **extra):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    return subprocess.run(["bash", str(LANE / "p235_drive.sh")], capture_output=True, text=True,
                          env=_dry_env(tmp_path, run_id, **extra), timeout=120)


@pytest.mark.parametrize("run_id", ["p235-5090-4", "p58-5090-1", "p235-shake-1", "p235-5090-r7", "p235-5090-r0"])
def test_the_driver_refuses_another_run_id(tmp_path, run_id):
    out = _drive(tmp_path, run_id)
    assert out.returncode == 78 and "is not p235-5090-N or p235-5090-rN" in out.stdout + out.stderr


def test_the_driver_refuses_an_e4b_sha_override(tmp_path):
    out = _drive(tmp_path, "p235-5090-1", E4B_SHA="a" * 40)
    assert out.returncode == 78 and "E4B_SHA override" in out.stdout + out.stderr


@pytest.mark.parametrize("gnf4", ["", "a" * 39, "a" * 41, "B" * 40, "g" * 40, "v0.51.0"])
def test_the_driver_refuses_a_gnf4_sha_that_is_not_one(tmp_path, gnf4):
    out = _drive(tmp_path, "p235-5090-1", GNF4_SHA=gnf4)
    assert out.returncode == 78 and "GNF4_SHA is not a 40-char hex sha" in out.stdout + out.stderr


def _clean_or_skip(dirty):
    """The driver refuses a dirty tree by design, so a dirty local checkout skips the dry run, named. Under CI the
    checkout is fresh: a dirty one is broken and fails, never a silent skip that drops the dry-run asserts."""
    if dirty.strip():
        why = f"the checkout is dirty ({dirty.splitlines()[0][:80]}): the driver refuses a dirty tree by design"
        if os.environ.get("CI"):
            pytest.fail(why + " (CI must run on a clean checkout)")
        pytest.skip(why)


@pytest.mark.parametrize("ci,outcome", [("1", pytest.fail.Exception), ("", pytest.skip.Exception)],
                         ids=["ci-fails", "local-skips"])
def test_a_dirty_checkout_fails_under_ci_and_skips_locally(monkeypatch, ci, outcome):
    if ci:
        monkeypatch.setenv("CI", ci)
    else:
        monkeypatch.delenv("CI", raising=False)
    with expect_outcome(outcome):
        _clean_or_skip(" M tests/test_p235.py\n")
    _clean_or_skip("")                                                   # a clean tree neither fails nor skips


@pytest.mark.parametrize("run_id", ["p235-5090-1", "p235-5090-3", "p235-5090-r6"])
def test_the_drivers_dry_run_names_the_launch(tmp_path, run_id):
    """The driver dry-runs on a clean checkout: it stages the seven pieces and the pin file, checks them there, and
    starts the runner with both commits."""
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    _clean_or_skip(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                  text=True).stdout)
    out = _drive(tmp_path, run_id)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"DRYRUN stage {len(SOURCES) + 1} files -> root@h:/root/p235" in out.stdout and "bash p235_run.sh" in out.stdout
    assert f"P235_RUN_ID={run_id} " in out.stdout and f"GNF4_SHA={GNF4} " in out.stdout
    assert re.search(r" E4B_SHA=[0-9a-f]{40} ", out.stdout)
    assert f"fetch -> {tmp_path}/p235" in out.stdout


# ------------------------------------------------------------------------------------------ prereg, files --

def test_the_prereg_states_the_pins_bars_and_budget():
    for s in ('`experts4bit_qlora.__version__ == "0.58.0"`', '`md.version("grouped-nf4-gemm") == "0.51.0"`',
              "| transformers | 5.16.1 | **5.17.0** |", "| bitsandbytes | 0.50.1 | 0.50.1 (e4b's floor is `>=0.43`) |",
              "latest PyPI at launch (**0.32.0** today)", "| vLLM SECONDARY | 0.29.0 | **0.30.0** (P58's primary) |",
              "**The one placeholder: `GNF4_SHA`.**", "`Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`",
              "`Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441`", "`a8e6ea1d…` (B=1) and `f67e7e4d…` (B=16)",
              "must be inside **1.03×**", "only with an equal `prompts_sha` on both sides", "**VOID arms never enter a ratio**",
              "**1.087** at B=1 and **1.396** at B=16", "one RTX 5090 at ≤ $1.05/h, guard 3.0 h, download ≤ $0.02/GB × 80 GB",
              "$3.15 + $1.60 = **$4.75**, inside a **$6** lane ceiling", "`p235-5090-1..-3`", "`p235-5090-r1..-r6`",
              "**No shakedown is registered.**", "**not collected**"):
        assert s in FLAT, s


def test_the_new_files_are_lf_only_and_the_shell_scripts_parse():
    for f in NEW_FILES:
        assert b"\r" not in f.read_bytes(), f
    if shutil.which("bash") is None:
        return
    for f in ("p235_run.sh", "p235_drive.sh"):
        out = subprocess.run(["bash", "-n", str(LANE / f)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
