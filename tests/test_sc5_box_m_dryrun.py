# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC1 boxes A-M: run each whole SC1_PROVE=1 path under bash with heavy leaves stubbed.

The script's own proof dispatch, prove_* functions and per-cell helpers must reach PROVED without command-not-found.
Three exact hooks install stubs after M's initial sourcing, before comparator installs and before proof dispatch;
staged box helpers also reinstall them after defining their heavy leaves. Python, downloads, installs, server starts,
profilers and sampling are fixtures; reducers and model execution are not validated by this control-flow test.

Box M retains its detailed SC5 cells and quality receipts, Granite skip and fetch_common failure mutant. Boxes A-L
exercise their actual Granite smoke commands and each box's additional proof. CI fails if bash >= 4 is unavailable.
No GPU or network work runs."""
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
        venv) d=${@: -1}; mkdir -p "$d/bin" && cp "$0" "$d/bin/python" && chmod +x "$d/bin/python"; exit 0;;
        *) exit 0;; esac;;
  -c) case "${2:-}" in *'print(json.load(open(sys.argv[1]))[sys.argv[2]])'*) printf '%064d\n' 0;; esac; exit 0;;
  -) cat > /dev/null; exit 0;;
  --version) echo "Python 3.12.3"; exit 0;;
esac
script=$(basename "${1:-x}")
printf '%s\n' "$*" >> "$DRY_CALLS"
if [ -n "${SC1_OUT:-}" ]; then mkdir -p "$(dirname "$SC1_OUT")"; printf '{"dryrun": true}\n' > "$SC1_OUT"; fi
prev=""
for a in "$@"; do
  case "$prev" in --out|--prefill-out|--decode-out|--write-prompts)
    if [ ! -d "$a" ]; then mkdir -p "$(dirname "$a")"; printf '{"summary": {}, "dryrun": true}\n' > "$a"; fi;; esac
  prev=$a
done
case " $* " in *" --self-test "*) echo "${script%.py} self-test OK (dryrun)"; exit 0;; esac
case "$script" in
  sc2_prompts.py) echo "SC2_PROMPTS dryrun";;
  sc1_prompts.py) echo "PROMPTS dryrun";;
  sc1g_k8.py) echo "WINDOW dryrun";;
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
    "nsys": "exit 0",
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
m_capacity(){ dry m_capacity "$1"; [ "$1" = "${DRY_NO_CAPACITY:-none}" ] && return 1; echo "65536 0"; }
install_llamacpp(){ dry install_llamacpp; OK[llamacpp]=1; LLAMACPP_BIN=$W/llamacpp/bin; mkdir -p "$LLAMACPP_BIN"; cp "$DRY_PY" "$LLAMACPP_BIN/nll_teacher_forced"; }
install_exl3(){ dry install_exl3; OK[exl3]=1; }
install_nsys(){ dry install_nsys; OK[nsys]=1; NSYS=nsys; }
llamacpp_fetch(){ mkdir -p "$3"; printf 'dryrun GGUF\n' > "$3/$2"; }
fetch_q4km(){ dry fetch_q4km; mkdir -p "$W/gguf"; printf 'dryrun GGUF\n' > "$W/gguf/$GGUF_Q4KM"; }
fetch_gptoss(){ fetch gptoss "$@"; }
bake_gptoss(){ bake gptoss; GA_GPTOSS=$W/work_gptoss/nf4.arena; }
fetch_gptoss_gguf(){ dry fetch_gptoss_gguf; mkdir -p "$W/gguf"; printf 'dryrun GGUF\n' > "$W/gguf/$SC2G_GGUF"; }
sampler_start(){ dry sampler_start "$@"; }
sampler_stop(){ dry sampler_stop "$@"; }
nsys_export(){ dry nsys_export "$@"; : > "$1.sqlite"; }
e4b_census(){ dry e4b_census "$@"; }
vllm_census(){ dry vllm_census "$@"; }
sglang_census(){ dry sglang_census "$@"; }
llamacpp_census(){ dry llamacpp_census "$@"; }
llamacpp_down(){ dry llamacpp_down; }
llamacpp_server_stop(){ dry llamacpp_server_stop; }

'''

SERVER_STARTS = (
    "llamacpp_up", "sglang_server_start", "e4b_server_start", "vllm_server_start", "sgl_server_start", "ll_server_start",
    "f_server_start", "h_server_start", "g_e4b_start", "g_vllm_start", "g_sgl_start", "g_ll_start", "k_server_start", "l_server_start",
)
FUNCTION_STUBS += "\n".join(f'{name}(){{ dry {name} "$@"; SRV_PID=""; return 0; }}' for name in SERVER_STARTS) + "\n"

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
    """The shell dry-run tests skip without a bash >= 4 (a developer's macOS /bin/bash is 3.2). In CI that skip would be a
    silent pass, so there it is a failure: Linux CI's bash is 5 and must be found."""
    if not (os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"):
        pytest.skip("checked only in CI")
    assert sys.platform != "win32" and BASH is not None, "CI found no bash >= 4: the SC1 proof dry runs would be skipped"

def _stage(w: pathlib.Path, run_text: str) -> None:
    for name, src in _staged_names_and_sources():
        dst = w / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if name.endswith(".sh") and ("_box_" in name or name == "llamacpp/llamacpp_box.sh"):
            with dst.open("a") as f:
                f.write('\n. "' + str(w / 'dryrun_stubs.sh') + '"\n')
    shutil.copyfile(SC1 / "staged.sha256", w / "staged.sha256")     # stages itself; not one of its own entries
    (w / "sc1_run.sh").write_text(run_text)
    (w / "dryrun_stubs.sh").write_text(FUNCTION_STUBS)
    # Box I's registered full rows are large; tiny fixture files exercise its actual existence/hash gates.
    ref = w / "sc1g_ref_full"
    ref.mkdir(exist_ok=True)
    (ref / "ref_full_conv1.npy").write_bytes(b"dryrun full rows")
    (w / "sc1g_ref" / "ref_full_shas.json").write_text('{"conv1": "' + "0" * 64 + '"}')


def _stub_bin(d: pathlib.Path) -> pathlib.Path:
    d.mkdir()
    for name, body in STUBS_BIN.items():
        p = d / name
        p.write_text("#!/bin/bash\n" + body + "\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (d / "bash").write_text(f'#!/bin/sh\nexec "{BASH}" "$@"\n')
    (d / "bash").chmod(0o755)
    shutil.copyfile(d / "python", d / "python3")
    (d / "python3").chmod(0o755)
    return d


def _run(tmp_path: pathlib.Path, run_text: str, box="M"):
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
           "SC1_INSTANCE_ID": "0", "SC1_BOX": box, "E4B_SHA": "0" * 40, "SC1_PROVE": "1", "SC1_QUIESCE_S": "1",
           "SC1_PROVE_SGLANG_MODEL": "dryrun/model@" + "0" * 40,
           "DRY_PY": str(stubs / "python"), "DRY_CALLS": str(w / "python_calls.log")}
    out = subprocess.run([BASH, str(w / "sc1_run.sh")], capture_output=True, text=True, env=env, timeout=300)
    return w, out


@pytest.mark.parametrize("box", list("ABCDEFGHIJKL"))
@pytest.mark.skipif(sys.platform == "win32" or BASH is None, reason="the box script needs a POSIX bash >= 4 with executable stubs")
def test_each_sc1_proof_path_reaches_proved(tmp_path, box):
    w, out = _run(tmp_path, (SC1 / "sc1_run.sh").read_text(), box)
    log = out.stdout + out.stderr
    assert "command not found" not in log, log[-6000:]
    assert out.returncode == 0, log[-6000:]
    assert (w / "PROVED").is_file() and (w / ("SC1_SUCCESS." + "a" * 64)).is_file(), log[-6000:]
    summary = (w / "summary.txt").read_text()
    assert f"PROVED box={box}" in summary, summary[-3000:]
    dry = (w / "dryrun.log").read_text()
    required_starts = {
        "B": ("llamacpp_up",), "C": ("sglang_server_start",),
        "D": ("e4b_census", "vllm_census", "sglang_census", "llamacpp_census"),
        "E": ("e4b_server_start", "vllm_server_start", "sgl_server_start", "ll_server_start"),
        "F": ("f_server_start",), "G": ("g_e4b_start", "g_vllm_start", "g_sgl_start", "g_ll_start"),
        "H": ("h_server_start",), "K": ("k_server_start granite_off", "k_server_start granite_on"),
        "L": tuple(f"l_server_start prove_{arm}" for arm in ("s16", "s32a", "s64c", "s64a")),
    }
    for step in required_starts.get(box, ()):
        assert step in dry, (step, dry)
    if box == "I":
        calls = (w / "python_calls.log").read_text()
        for step in ("sc1g_k8.py", "sc1_vllm_nll.py", "--mode decode", "--prove-a5"):
            assert step in calls, (step, calls)
    for b in (1, 16):
        assert (w / f"smoke_granite_b{b}.json").is_file(), log[-6000:]
        assert "E4B_FUSE_T1_GLUE_R2=auto" in (w / "logs" / f"run_smoke_granite_b{b}.log").read_text()


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
           "DRY_PY": str(stubs / "python"), "DRY_CALLS": str(w / "python_calls.log")}
    for f in ("PROVED", "summary.txt"):
        (w / f).unlink(missing_ok=True)
    mut = subprocess.run([BASH, str(w / "sc1_run.sh")], capture_output=True, text=True, env=env, timeout=300)
    assert mut.returncode != 0 and not (w / "PROVED").exists()
    assert "fetch_common: command not found" in mut.stdout + mut.stderr


@pytest.mark.skipif(sys.platform == "win32" or BASH is None, reason="the box script needs a POSIX bash >= 4 with executable stubs")
def test_the_dry_run_catches_a_proof_that_passes_with_a_failed_capacity_readout(tmp_path):
    """The mutant is sc5-prove-5's readout: SGLang's capacity read as "?", and the proof still proved. The parsers
    themselves run on each framework's response shape in test_sc5_box_m_static.py; here the readout is stubbed."""
    w, out = _run(tmp_path, (SC1 / "sc1_run.sh").read_text())
    assert (w / "PROVED").exists(), (out.stdout + out.stderr)[-3000:]
    assert "m_capacity sglang" in (w / "dryrun.log").read_text()
    stubs = tmp_path / "bin"
    env = {"PATH": f"{stubs}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp_path), "LANG": "C",
           "SC1_RUN_NONCE": "c" * 64, "SC1_RUN_ID": "sc5-dryrun-capacity", "SC1_DEADLINE_EPOCH": "4102444800",
           "SC1_INSTANCE_ID": "0", "SC1_BOX": "M", "E4B_SHA": "0" * 40, "SC1_PROVE": "1", "SC1_QUIESCE_S": "1",
           "DRY_PY": str(stubs / "python"), "DRY_CALLS": str(w / "python_calls.log"), "DRY_NO_CAPACITY": "sglang"}
    for f in ("PROVED", "summary.txt"):
        (w / f).unlink(missing_ok=True)
    mut = subprocess.run([BASH, str(w / "sc1_run.sh")], capture_output=True, text=True, env=env, timeout=300)
    assert mut.returncode != 0 and not (w / "PROVED").exists()
    summary = (w / "summary.txt").read_text()
    assert "PROVE: sglang capacity readout failed" in summary and "SC5_PROVE ok" not in summary, summary[-2000:]
