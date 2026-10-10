# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC5 box M: the WHOLE proof path of bench/sc1/sc1_run.sh (SC1_BOX=M SC1_PROVE=1), run under bash with the heavy steps
stubbed, must reach PROVED with no command-not-found and no unexpected exit.

Three SC5 proofs failed on harness defects a dry run catches (2026-10-10): sc5-prove-3 called fetch_common before
sc1_run.sh defines it (rc 11), SC1's Granite smokes were refused by e4b 0.52.0, and the quiescence gate could not fail.
This test runs the script's OWN text, relocated from /root/sc1 to a temporary directory, with three stub hooks inserted
(after box M's sourcing, before the competitor installs, before the proof block):

* external commands on PATH: nvidia-smi, python (a venv, pip, every script: each ``--out`` file is written, each
  self-test passes), curl, perl's alarm wrapper, setsid, df, free, lscpu, nproc, sha256sum, nvcc, pgrep, git, cmake,
  apt-get;
* shell functions: fetch, bake, the four installs, the three server starts, gpu_free, quiesce (which writes the record
  the gate reads) -- the leaves only. sc1_run.sh's proof block, prove_m, m_block, m_cell, m_capacity, the quality
  commands and every SC1 / SC2 helper run as written.

It runs nothing on a GPU and fetches nothing; what it proves is that the control flow reaches PROVED."""
import importlib.util
import os
import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SC1 = REPO / "bench" / "sc1"

STUBS_BIN = {
    "nvidia-smi": r'''case "$*" in
  *"-L"*) echo "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-dryrun)";;
  *"query-gpu=name,memory.total"*) echo "NVIDIA GeForce RTX 5090, 32607 MiB, 595.84, GPU-dryrun, 12.0, 5, 16, 5, 16";;
  *"query-gpu=power.limit"*) echo "575.00 W, 3090 MHz";;
  *"query-gpu=name"*) echo "NVIDIA GeForce RTX 5090";;
  *"query-gpu=driver_version"*) echo "595.84";;
  *"query-gpu=memory.used"*) echo 1000;;
  *"query-compute-apps"*) :;;
  *) echo "nvidia-smi dryrun";;
esac''',
    "python": r'''case "${1:-}" in
  -m) case "${2:-}" in
        venv) mkdir -p "$3/bin" && cp "$0" "$3/bin/python" && chmod +x "$3/bin/python"; exit 0;;
        *) exit 0;; esac;;
  -c) exit 0;;
  -) cat > /dev/null; exit 0;;
  --version) echo "Python 3.12.3"; exit 0;;
esac
script=$(basename "${1:-x}")
prev=""
for a in "$@"; do
  case "$prev" in --out|--prefill-out|--decode-out) mkdir -p "$(dirname "$a")"; printf '{"summary": {}, "dryrun": true}\n' > "$a";; esac
  prev=$a
done
case " $* " in *" --self-test "*) echo "${script%.py} self-test OK (dryrun)"; exit 0;; esac
case "$script" in
  sc2_prompts.py) echo "SC2_PROMPTS dryrun";;
  sc5_driver.py) echo "SC5_RUN {}";;
  sc5_quality.py) echo "SC5_QUALITY dryrun";;
  sc5_e4b_quality.py) echo "SC5_E4B_QUALITY dryrun";;
esac
exit 0''',
    "curl": r'''out=""; w=""; prev=""
for a in "$@"; do case "$prev" in -o) out=$a;; -w) w=$a;; esac; prev=$a; done
body='{"status": "ready", "engine": {"max_seqs": 64, "max_tokens_per_seq": 1024}}'
[ -n "$out" ] && printf '%s\n' "$body" > "$out"
[ -n "$w" ] && { printf '200'; exit 0; }
[ -z "$out" ] && printf '%s\n' "$body"
exit 0''',
    "perl": r'''shift 2; exec "$@"''',
    "setsid": r'''exec "$@"''',
    "df": r'''case "$*" in *--output=avail*) printf 'Avail\n500G\n';; *) printf 'Filesystem Size Used Avail Use%% Mounted\noverlay 500G 1G 499G 1%% /\n';; esac''',
    "free": r'''printf '              total        used        free\nMem:            123           4         100\n' ''',
    "lscpu": r'''printf 'Vendor ID:  AuthenticAMD\nModel name:  AMD Ryzen 9 9950X (dryrun)\n' ''',
    "nproc": "echo 32",
    "sha256sum": r'''case "${1:-}" in -c) exit 0;; esac; for f; do echo "0000000000000000000000000000000000000000000000000000000000000000  $f"; done''',
    "nvcc": 'echo "Cuda compilation tools, release 13.0, V13.0.88"',
    "pgrep": "exit 1",
    "git": "exit 0",
    "cmake": "exit 0",
    "apt-get": "exit 0",
}

FUNCTION_STUBS = r'''
dry(){ echo "DRYRUN $*" >> "$W/dryrun.log"; }
m_install_e4b(){ dry m_install_e4b; mkdir -p "$W/venv-e4b/bin"; cp "$DRY_PY" "$W/venv-e4b/bin/python"; PY=$W/venv-e4b/bin/python
  echo "e4b 0.52.0 (wheel, dryrun)" >> versions.txt; }
install_vllm(){ dry install_vllm; mkdir -p "$W/venv-vllm/bin"; cp "$DRY_PY" "$W/venv-vllm/bin/python"; OK[vllm]=1; }
install_sglang(){ dry install_sglang; OK[sglang]=1; . "$W/sglang/server.sh"; export SGLANG_WORK=$W; }
install_sc2_client(){ dry install_sc2_client; OK[sc2client]=1; }
fetch(){ dry fetch "$@"; echo "$W/hf/$1" > "fetch_$1.path"; return 0; }
bake(){ dry bake "$@"; mkdir -p "$W/work_$1"; : > "$W/work_$1/nf4.arena"; return 0; }
m_e4b_start(){ dry m_e4b_start "$@"; SRV_PID=""; return 0; }
m_vllm_start(){ dry m_vllm_start "$@"; SRV_PID=""; return 0; }
m_sgl_start(){ dry m_sgl_start "$@"; return 0; }
sglang_server_stop(){ :; }
gpu_free(){ return 0; }
quiesce(){ dry quiesce "$1"; printf '{"tag": "%s", "quiesced": true}\n' "$1" > "quiesce_$1.json"; }
'''

HOOKS = [  # (anchor in sc1_run.sh, what replaces it) -- each must occur exactly once
    ("  . $W/sc2_box_e.sh; . $W/sc5_box_m.sh; m_install_e4b || finish 9",
     "  . $W/sc2_box_e.sh; . $W/sc5_box_m.sh; . $W/dryrun_stubs.sh; m_install_e4b || finish 9"),
    ('case "$BOX" in\n  A) install_vllm ;;', '. $W/dryrun_stubs.sh\ncase "$BOX" in\n  A) install_vllm ;;'),
    ('if [ "$PROVE" = 1 ]; then', '. $W/dryrun_stubs.sh\nif [ "$PROVE" = 1 ]; then'),
]


def _staged_names_and_sources():
    spec = importlib.util.spec_from_file_location("sc1_pin", REPO / "tests" / "test_sc1_staged_pin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for _want, name in mod._entries():
        yield name, mod.resolve(name)


def _bash4():
    """A bash >= 4 (the box's is 5; sc1_run.sh uses associative arrays): the PATH's, else Homebrew's, else none."""
    for cand in (shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        if cand and os.path.exists(cand):
            v = subprocess.run([cand, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout.strip()
            if v.isdigit() and int(v) >= 4:
                return cand
    return None


BASH = _bash4()



def test_ci_runs_the_dry_run_rather_than_skipping_it():
    """The two dry-run tests skip without a bash >= 4 (a developer's macOS /bin/bash is 3.2). In CI that skip would be a
    silent pass, so there it is a failure: Linux CI's bash is 5 and must be found."""
    if not (os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"):
        pytest.skip("checked only in CI")
    assert sys.platform != "win32" and BASH is not None, "CI found no bash >= 4: the box M dry run would be skipped"

def _stage(w: pathlib.Path, run_text: str) -> None:
    for name, src in _staged_names_and_sources():
        dst = w / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.copyfile(SC1 / "staged.sha256", w / "staged.sha256")     # stages itself; not one of its own entries
    (w / "sc1_run.sh").write_text(run_text)
    (w / "dryrun_stubs.sh").write_text(FUNCTION_STUBS)


def _stub_bin(d: pathlib.Path) -> pathlib.Path:
    d.mkdir()
    for name, body in STUBS_BIN.items():
        p = d / name
        p.write_text("#!/bin/bash\n" + body + "\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copyfile(d / "python", d / "python3")
    (d / "python3").chmod(0o755)
    return d


def _run(tmp_path: pathlib.Path, run_text: str):
    w = tmp_path / "w"
    w.mkdir()
    text = run_text.replace("/root/sc1", str(w))
    for anchor, repl in HOOKS:
        assert text.count(anchor) == 1, f"hook anchor moved: {anchor[:60]!r}"
        text = text.replace(anchor, repl)
    _stage(w, text)
    stubs = _stub_bin(tmp_path / "bin")
    env = {"PATH": f"{stubs}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp_path), "LANG": "C",
           "SC1_RUN_NONCE": "a" * 64, "SC1_RUN_ID": "sc5-dryrun", "SC1_DEADLINE_EPOCH": "4102444800",
           "SC1_INSTANCE_ID": "0", "SC1_BOX": "M", "E4B_SHA": "0" * 40, "SC1_PROVE": "1", "SC1_QUIESCE_S": "1",
           "DRY_PY": str(stubs / "python")}
    out = subprocess.run([BASH, str(w / "sc1_run.sh")], capture_output=True, text=True, env=env, timeout=300)
    return w, out


@pytest.mark.skipif(sys.platform == "win32" or BASH is None, reason="the box script needs a POSIX bash >= 4 with executable stubs")
def test_box_m_proof_path_reaches_proved_with_every_heavy_step_stubbed(tmp_path):
    w, out = _run(tmp_path, (SC1 / "sc1_run.sh").read_text())
    log = out.stdout + out.stderr
    assert "command not found" not in log, log[-3000:]
    assert out.returncode == 0, log[-3000:]
    assert (w / "PROVED").exists() and (w / ("SC1_SUCCESS." + "a" * 64)).exists(), log[-3000:]
    summary = (w / "summary.txt").read_text()
    assert "PROVED box=M" in summary and "SC5_PROVE ok" in summary, summary[-2000:]
    assert "PROVE box M: SC1's Granite smokes skipped" in summary
    dry = (w / "dryrun.log").read_text()
    for step in ("m_install_e4b", "install_vllm", "install_sglang", "install_sc2_client", "fetch qwen3", "fetch gptq",
                 "bake qwen3", "m_e4b_start default", "m_vllm_start default", "m_sgl_start default", "quiesce sc5_1"):
        assert step in dry, (step, dry)
    for fw, nn in (("e4b", 1), ("vllm", 2), ("sglang", 3)):
        for c in (1, 16):
            assert (w / "sc5" / "blocks" / f"{nn:02d}_d1_default_{fw}_b1" / f"c{c}.json").is_file(), (fw, c)
    for q in ("q_vllm", "q_sglang", "q_e4b", "q_e4b_decode"):
        assert (w / "sc5" / "quality" / f"{q}.json").is_file(), q


@pytest.mark.skipif(sys.platform == "win32" or BASH is None, reason="the box script needs a POSIX bash >= 4 with executable stubs")
def test_the_dry_run_catches_a_proof_that_calls_a_function_defined_after_the_proof_block(tmp_path):
    """The mutant is sc5-prove-3's own defect: prove_m calling fetch_common, which sc1_run.sh defines after its proof block."""
    box = (REPO / "bench" / "sc5" / "sc5_box_m.sh").read_text()
    anchor = '  fetch qwen3 "$MID" "$REV" 4800 || finish 11; fetch gptq "$GPTQ_MID" "$GPTQ_REV" 1800 || finish 11\n'
    assert box.count(anchor) == 1
    w, out = _run(tmp_path, (SC1 / "sc1_run.sh").read_text())
    (w / "sc5_box_m.sh").write_text(box.replace(anchor, "  fetch_common || finish 11\n"))
    # re-run the staged script against the mutated box file (the hooks are already in place in the staged copy)
    stubs = tmp_path / "bin"
    env = {"PATH": f"{stubs}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp_path), "LANG": "C",
           "SC1_RUN_NONCE": "b" * 64, "SC1_RUN_ID": "sc5-dryrun-mutant", "SC1_DEADLINE_EPOCH": "4102444800",
           "SC1_INSTANCE_ID": "0", "SC1_BOX": "M", "E4B_SHA": "0" * 40, "SC1_PROVE": "1", "SC1_QUIESCE_S": "1",
           "DRY_PY": str(stubs / "python")}
    for f in ("PROVED", "summary.txt"):
        (w / f).unlink(missing_ok=True)
    mut = subprocess.run([BASH, str(w / "sc1_run.sh")], capture_output=True, text=True, env=env, timeout=300)
    assert mut.returncode != 0 and not (w / "PROVED").exists()
    assert "fetch_common: command not found" in mut.stdout + mut.stderr
