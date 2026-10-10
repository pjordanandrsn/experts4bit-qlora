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
  must be one serve_paged's own config serves.

sc5-prove-5 (2026-10-10, $1.531) proved, but one readout failed silently:
* **readouts:** SGLang's capacity printed "kv_tokens=?". The parser read internal_states[0], where SGLang 0.5.21 keeps
  no max_total_num_tokens; the top level and internal_states[0].memory_usage.token_capacity carry it. Every capacity
  parser must run on its framework's response shape (copied from sc5-prove-5's receipts) and, at the matched setting,
  yield a capacity the reducer accepts."""
import ast
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

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


# ---- (d) readouts --------------------------------------------------------------------------------------------------
# The responses as sc5-prove-5 recorded them (trimmed to the fields around each capacity), at the default setting, and the
# same shapes at the matched setting (box M's flags: e4b 64 x 1024, vLLM 4096 blocks of 16, SGLang 65536 tokens).
def _e4b_health(seqs: int, tps: int) -> str:
    return json.dumps({"status": "ready", "error": None, "model": "Qwen/Qwen3-30B-A3B", "served_model_names": ["Qwen/Qwen3-30B-A3B"],
                       "engine": {"max_seqs": seqs, "max_tokens_per_seq": tps}})


def _vllm_metrics(blocks: int) -> str:
    labels = ('_block_size_resolved="True",block_size="16",cache_dtype="auto",effective_attention_block_size="16",'
              'enable_prefix_caching="False",engine="0",gpu_memory_utilization="0.92",kv_cache_size_tokens="%d",'
              'mamba_block_size="None",num_cpu_blocks="None",num_gpu_blocks="%d",num_gpu_blocks_override="%s"'
              % (blocks * 16, blocks, "None" if blocks == 8001 else blocks))
    return ("# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n# TYPE vllm:cache_config_info gauge\n"
            "vllm:cache_config_info{%s} 1.0\nvllm:num_requests_running{engine=\"0\"} 0.0\n" % labels)


def _sglang_info(tokens: int, flag) -> str:
    state = {"tokenizer_path": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "page_size": 1, "max_total_tokens": flag,
             "max_running_requests": 64, "memory_usage": {"kvcache": 8.063, "token_capacity": tokens, "token_capacity_swa": None},
             "effective_max_running_requests_per_dp": 64}
    return json.dumps({"tokenizer_path": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "page_size": 1, "max_total_tokens": flag,
                       "max_running_requests": 64, "max_total_num_tokens": tokens, "internal_states": [state], "kv_events": None})


READOUTS = {   # framework -> (default response, its readout as sc5-prove-5 logged or should have, matched response)
    "e4b": (_e4b_health(32, 4096), (131072, 0), _e4b_health(64, 1024)),
    "vllm": (_vllm_metrics(8001), (128016, 16), _vllm_metrics(4096)),
    "sglang": (_sglang_info(88066, None), (88066, 1), _sglang_info(65536, 65536)),
}


def _parser(fw: str) -> str:
    """The python heredoc m_capacity runs for framework ``fw``."""
    body = BOX[BOX.index("m_capacity(){"):BOX.index("m_cell(){")]
    m = re.search(rf"^\s+{fw}\)\s.*?<<'PYC'\n(.*?)\nPYC$", body, re.M | re.S)
    assert m, f"m_capacity has no {fw} parser"
    return m.group(1)


def _readout(code: str, response: str, tmp_path: pathlib.Path):
    f = tmp_path / "response"
    f.write_text(response, encoding="utf-8")
    out = subprocess.run([sys.executable, "-", str(f)], input=code, capture_output=True, text=True, encoding="utf-8")
    words = out.stdout.split()
    return (int(words[0]), int(words[1])) if out.returncode == 0 and len(words) == 2 else None


@pytest.mark.parametrize("fw", sorted(READOUTS))
def test_every_capacity_parser_reads_its_frameworks_response(fw, tmp_path):
    default, want, matched = READOUTS[fw]
    assert _readout(_parser(fw), default, tmp_path) == want
    # the matched setting: the reducer VOIDs a block whose kv_tokens is missing or off the registration by more than kv_rounding
    reduce = importlib.util.spec_from_file_location("sc5_reduce", SC5 / "sc5_reduce.py")
    mod = importlib.util.module_from_spec(reduce)
    reduce.loader.exec_module(mod)
    assert int(_const("SC5_MATCHED_KV")) == mod.MATCHED_KV_TOKENS
    got = _readout(_parser(fw), matched, tmp_path)
    assert got is not None and abs(got[0] - mod.MATCHED_KV_TOKENS) <= got[1], (fw, got)


def test_the_sglang_parser_refuses_what_it_cannot_read_and_sc5_prove_5s_parser_is_caught(tmp_path):
    code = _parser("sglang")
    split = json.loads(_sglang_info(88066, None))
    split["internal_states"][0]["memory_usage"]["token_capacity"] = 65536   # two reports of one pool that disagree
    assert _readout(code, json.dumps(split), tmp_path) is None
    bare = json.loads(_sglang_info(88066, None))
    del bare["max_total_num_tokens"], bare["internal_states"][0]["memory_usage"]
    assert _readout(code, json.dumps(bare), tmp_path) is None
    # the mutant: sc5-prove-5's own parser, which read internal_states[0], reads nothing from SGLang 0.5.21's response
    prove5 = ('import json, sys\ns = json.load(open(sys.argv[1]))\n'
              's = s.get("internal_states", [s])[0] if isinstance(s.get("internal_states"), list) else s\n'
              'print(int(s["max_total_num_tokens"]), int(s.get("page_size", 1) or 1))\n')
    assert _readout(prove5, READOUTS["sglang"][0], tmp_path) is None


def test_the_proof_gates_on_every_capacity_readout():
    prove = BOX[BOX.index("prove_m(){"):]
    assert 'M_KV=""' in BOX[BOX.index("m_block(){"):BOX.index("box_m(){")] and "M_KV=${kv:-}" in BOX
    assert '[ -n "$M_KV" ] || { say "PROVE: $fw capacity readout failed' in prove


# ---- (d) the reading's cost guard: box_m stops after draw 1's first matched block per framework on a capacity miss -------
def _fn(name: str) -> str:
    m = re.search(rf"^{name}\(\)\{{.*?(?=^\S)", BOX, re.M | re.S)   # up to the next line that starts in column 0
    assert m, f"{name}() moved"
    return m.group(0)


_GUARD_STUBS = r"""
SC5_DRAWS=2; SC5_MEMS="default matched"; SC5_MATCHED_KV=65536; W=/nonexistent; SC5_D=/nonexistent; PY=true
phase(){ :; }; line(){ echo "LINE $*"; }; say(){ :; }; can_run(){ return 1; }; tee(){ cat > /dev/null; }
fetch_common(){ :; }; fetch(){ :; }; bake_qwen3(){ :; }; sc2_prompts(){ :; }; m_sgl_start(){ :; }; m_stop(){ :; }
finish(){ echo "FINISH $1"; exit "$1"; }
m_block(){ local i=$(( $1 - 1 )); echo "BLOCK $1 d$2 $3 $4 b$5"
  M_READY=${READY[$i]:-1} M_KV=${KVS[$i]-65536} M_RND=${RNDS[$i]:-0}; }
"""


def _box_m(kvs: dict, rnds=None, ready=None):
    """box_m with m_block scripted: block n (1-based) reads kvs.get(n, 65536); returns (rc, blocks run, the stop line)."""
    bash = shutil.which("bash")
    in_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
    if bash is None:
        assert not in_ci, "CI must run this test, but found no bash"
        pytest.skip("needs bash")
    def arr(d, default):
        return "(" + " ".join(f'"{d.get(n, default)}"' for n in range(1, 25)) + ")"
    script = (_GUARD_STUBS + f"KVS={arr(kvs, 65536)}; RNDS={arr(rnds or {}, 0)}; READY={arr(ready or {}, 1)}\n"
              + "\n".join(_fn(f) for f in ("m_draw_order", "m_matched_ok", "box_m")) + "\nbox_m\n")
    out = subprocess.run([bash, "-c", script], capture_output=True, text=True)
    blocks = [ln for ln in out.stdout.splitlines() if ln.startswith("BLOCK ")]
    stop = [ln for ln in out.stdout.splitlines() if "SC5_MATCHED_STOP" in ln]
    return out.returncode, blocks, (stop[0] if stop else None)


def test_draw_1s_first_three_matched_blocks_are_one_per_framework():
    first = re.search(r'^m_draw_order\(\)\{ case "\$1" in 1\) echo "([^"]+)"', BOX, re.M).group(1).split()[:3]
    assert sorted(f.split(":")[0] for f in first) == ["e4b", "sglang", "vllm"]


def test_the_reading_runs_every_block_when_the_matched_capacities_read_true():
    rc, blocks, stop = _box_m({8: 65536 + 16}, rnds={8: 16})          # vLLM at the edge of its block rounding: fine
    assert rc == 0 and stop is None and len(blocks) == 24, (rc, stop, blocks[-3:])


@pytest.mark.parametrize("n,fw,kv", [(9, "sglang", "88066"), (8, "vllm", ""), (7, "e4b", "131072")])
def test_the_reading_stops_after_block_9_when_a_matched_readout_misses(n, fw, kv):
    rc, blocks, stop = _box_m({n: kv})
    assert rc == 35 and len(blocks) == 9, (rc, blocks[-2:])
    assert stop and f"{fw}={kv or 'unread'}" in stop, stop


def test_the_guard_ignores_draw_2_and_blocks_whose_server_never_came_up():
    # block 8's server never came up (VOID under its own rule, no readout to judge); block 21 is draw 2's matched e4b
    rc, blocks, stop = _box_m({8: "", 21: "131072"}, ready={8: 0})
    assert rc == 0 and stop is None and len(blocks) == 24
