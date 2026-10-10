# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC5 box M: static checks for the defect classes the stubbed dry run cannot see (it checks control flow, not values).

sc5-prove-4 (2026-10-10, $0.905) failed on three of them:
* **versions:** SC1's SGLang engagement check compared the server against a hard-coded 0.5.20 while box M installs 0.5.21
  from its lock. Every version box M passes must equal its lock's pin, and the SC1 checks must take the version from
  the installer's knob, not from a literal.
* **imports:** p117_box.py imports p108_box, which imports p97_box; neither was staged. Every module a staged script
  imports, where some bench/ directory provides it, must itself be staged.
* **arguments:** box M drove e4b with the model name "sc5", which serve_paged refuses ("unknown model"). Every
  ``--flag`` box M passes to an SC5 script must be one that script's argparse defines, and the name box M sends e4b
  must be one serve_paged's own config serves."""
import ast
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SC5 = REPO / "bench" / "sc5"
BOX = (SC5 / "sc5_box_m.sh").read_text(encoding="utf-8")


def _pin_module():
    spec = importlib.util.spec_from_file_location("sc1_pin", REPO / "tests" / "test_sc1_staged_pin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _const(name: str) -> str:
    m = re.search(rf"\b{name}=([^\s;#]+)", BOX)
    assert m, f"{name} is not set in sc5_box_m.sh"
    return m.group(1).strip('"')


def _lock_pin(lock: str, package: str) -> str:
    m = re.search(rf"^{re.escape(package)}==(\S+) ", (SC5 / "locks" / lock).read_text(encoding="utf-8"), re.M)
    assert m, f"{lock} pins no {package}"
    return m.group(1)


# ---- (a) versions --------------------------------------------------------------------------------------------------
def test_every_version_box_m_passes_equals_its_lock():
    assert _const("SC5_VLLM_VERSION") == _lock_pin("vllm.lock.txt", "vllm")
    assert _const("SC5_SGLANG_VERSION") == _lock_pin("sglang.lock.txt", "sglang")
    # box M hands each installer, and SGLang's engagement check, the version through their knobs
    assert "SC1_VLLM_VERSION=$SC5_VLLM_VERSION" in BOX and "SGLANG_PIN=$SC5_SGLANG_VERSION" in BOX
    # e4b and gnf4: the box-M tripwire reads the versions from the wheel lock itself, and no e4b / gnf4 version is a literal
    wheels = {r.split()[0]: r.split()[1] for r in (SC5 / "locks" / "e4b-wheels.lock").read_text(encoding="utf-8").splitlines()
              if r.strip() and not r.startswith("#")}
    assert set(wheels) == {"grouped-nf4-gemm", "experts4bit-qlora"}
    assert "LOCK=$W/e4b-wheels.lock" in BOX and 'want = {r.split()[0]: r.split()[1] for r in open(os.environ["LOCK"])' in BOX
    install = BOX[BOX.index("m_install_e4b(){"):BOX.index("m_install_competitors(){")]
    for ver in wheels.values():
        assert ver not in install, f"m_install_e4b names {ver} literally; it must come from e4b-wheels.lock"


def test_sc1_version_checks_take_the_version_from_the_installer_knob():
    """A literal version in a check is how 0.5.21 met "!= 0.5.20". Defaults of a knob (``${SGLANG_PIN:-0.5.20}``) and
    comments are fine; a literal compared in code is not."""
    server = (REPO / "bench" / "sc1" / "sglang" / "server.sh").read_text(encoding="utf-8")
    assert 'SC1_SGLANG_PIN=${SGLANG_PIN:-' in server and 'want_v = os.environ.get("SC1_SGLANG_PIN"' in server
    for path in ("sglang/server.sh", "sglang/install.sh", "vllm/install.sh"):
        for n, line in enumerate((REPO / "bench" / "sc1" / path).read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            code = re.sub(r"\$\{[A-Z0-9_]+:-[0-9.]+\}", "", code)          # a knob's default
            code = re.sub(r'os\.environ\.get\("[A-Z0-9_]+", "[0-9.]+"\)', "", code)
            assert not re.search(r'(==|!=)\s*"?0\.\d+\.\d+"?', code), f"{path}:{n} compares a literal version: {line.strip()}"


# ---- (b) imports ---------------------------------------------------------------------------------------------------
def test_every_bench_module_a_staged_script_imports_is_staged_too():
    pin = _pin_module()
    entries = [(name, pin.resolve(name)) for _want, name in pin._entries()]
    staged = {name for name, _src in entries}
    # a module staged under a comparator directory (vllm/sc1_vllm_common.py) counts: its scripts put that directory on sys.path
    staged_modules = {pathlib.PurePosixPath(n).stem for n in staged if n.endswith(".py")}
    provided = {p.stem for p in (REPO / "bench").rglob("*.py")}
    missing = []
    for name, src in entries:
        if not name.endswith(".py") or "/" in name:
            continue
        for node in ast.walk(ast.parse(pathlib.Path(src).read_text(encoding="utf-8"))):
            mods = ([a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import)
                    else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                    else [])
            missing += [f"{name} imports {m}" for m in mods if m in provided and m not in staged_modules]
    assert not missing, missing


# ---- (c) arguments -------------------------------------------------------------------------------------------------
def _defined_flags(script: pathlib.Path) -> tuple:
    """(the --flags the script's argparse defines, the choices of its positional ``cmd``)."""
    flags, choices = set(), set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if first.value.startswith("--"):
                    flags.add(first.value)
                elif first.value == "cmd":
                    for kw in node.keywords:
                        if kw.arg == "choices" and isinstance(kw.value, (ast.Tuple, ast.List)):
                            choices |= {e.value for e in kw.value.elts if isinstance(e, ast.Constant)}
    return flags, choices


def _invocations(script_name: str) -> list:
    """Every logical line of sc5_box_m.sh that runs ``$W/<script_name>``: (the subcommand, the --flags it passes)."""
    logical = BOX.replace("\\\n", " ")
    out = []
    for line in logical.splitlines():
        m = re.search(rf"\$W/{re.escape(script_name)}\s+(.*)", line)
        if not m:
            continue
        rest = m.group(1).split("|")[0].split(">")[0]
        words = rest.split()
        cmd = words[0] if words and not words[0].startswith("-") else None
        out.append((cmd, {w for w in words if re.fullmatch(r"--[a-z][a-z0-9-]*", w)}))
    return out


@pytest.mark.parametrize("script", ["sc5_driver.py", "sc5_quality.py", "sc5_e4b_quality.py", "sc5_record.py",
                                    "sc5_reduce.py", "sc5_ref.py"])
def test_every_flag_box_m_passes_an_sc5_script_is_one_its_argparse_defines(script):
    flags, choices = _defined_flags(SC5 / script)
    calls = _invocations(script)
    assert calls, f"sc5_box_m.sh never runs {script}"
    for cmd, used in calls:
        assert used <= flags, f"{script}: {sorted(used - flags)} are not defined (has {sorted(flags)})"
        if cmd is not None and choices:
            assert cmd in choices, f"{script}: subcommand {cmd!r} not in {sorted(choices)}"


def test_the_model_name_box_m_sends_e4b_is_one_serve_paged_serves():
    """serve_paged answers to E4B_PAGED_MODEL and E4B_PAGED_SERVED_NAME's extras, and refuses any other name with 404."""
    in_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
    bash = shutil.which("bash")
    if bash is None:
        assert not in_ci, "CI must run this test, but found no bash"   # in CI a skip would be a silent pass
        pytest.skip("needs bash to evaluate sc5_model")
    mid = re.search(r"^MID=(\S+);", (REPO / "bench" / "sc1" / "sc1_run.sh").read_text(encoding="utf-8"), re.M).group(1)
    fn = re.search(r"^sc5_model\(\)\{.*\}$", BOX, re.M)
    assert fn, "sc5_model() moved"
    names = {}
    for fw in ("e4b", "vllm", "sglang"):
        names[fw] = subprocess.run([bash, "-c", f"MID={mid}; {fn.group(0)}; sc5_model {fw}"], capture_output=True,
                                   text=True).stdout.strip()
    # vLLM and SGLang answer to the --served-model-name their own start function passes
    for fw, fn_name in (("vllm", "m_vllm_start"), ("sglang", "m_sgl_start")):
        m = re.search(rf"^{fn_name}\(\)\{{(.*?)^m_\w+\(\)\{{", BOX, re.M | re.S)   # up to the next m_*() definition
        assert m, f"{fn_name}() moved"
        served = re.search(r"--served-model-name (\S+)", m.group(1))
        assert served, f"{fn_name} passes no --served-model-name"
        assert names[fw] == served.group(1), f"box M sends {fw} {names[fw]!r}; {fn_name} serves {served.group(1)!r}"
    try:
        import experts4bit_qlora.serve_paged as sp
    except ImportError as e:   # torch is absent on a developer machine; CI installs it, so there this must not skip
        assert not in_ci, f"CI must run this test, but serve_paged did not import: {e}"
        pytest.skip(f"serve_paged did not import: {e}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("E4B_PAGED_")}
    env.update({"E4B_PAGED_MODEL": mid})                      # what m_e4b_start sets; E4B_PAGED_SERVED_NAME is not set
    old = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        cfg = sp.PagedServeConfig.from_env()
    finally:
        os.environ.clear()
        os.environ.update(old)
    assert names["e4b"] in cfg.model_names, f"box M sends e4b {names['e4b']!r}; serve_paged serves {cfg.model_names}"
    assert "sc5" not in cfg.model_names                        # sc5-prove-4's request name is refused, as it was
