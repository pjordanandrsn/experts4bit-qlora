"""The TC1 harness extension (bench/tc1/tc1_arm.py) that P237 and P238 share: a generic OBSERVE-ONLY hook (``TC1_ARM_HOOK``) and three
registered options (``--vram-cap-gib``, ``--unsloth-offload-layers``, ``--layer-major``), plus ``build_parser()``.

- The selftest's extension block runs end to end. It covers the hook's output equality against a baseline, its edges 0..N with every
  training forward inside a step, the output-equality MUTANT, the partial hook, five refusals, and --layer-major's wiring.
- ``build_parser()`` is the pre-refactor parser, action for action, plus the three options. The fixture is the parser main() built at
  1f43d364 (amendment 81's merge), and the test re-derives it from that commit.
- Each option's refusals, at the command line and before anything loads, so CI drives them without a GPU.
- The pure helpers: the VRAM cap against a fake CUDA (getter present and absent, a cap above the card), the Unsloth offload kwargs
  from a signature (named, missing, **kwargs), and the block swap's resolved layers.
- Unset is unchanged: the selftest asserts that no default arm carries an extension key, and with no hook set nothing is loaded.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tc1" / "tc1_arm.py"
FIXTURE = REPO / "tests" / "fixtures" / "tc1_arm_parser_1f43d364.json"
BASE = "1f43d3648ad3be5ca6ed6a27086ef7b1d97048d9"            # amendment 81's merge: tc1_arm.py before this extension
NEW = {"vram_cap_gib", "unsloth_offload_layers", "layer_major"}
SNAPSHOT = r'''
import argparse, importlib.util, json, sys
class _Got(Exception):
    pass
spec = importlib.util.spec_from_file_location("tc1_arm_snapshot", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["tc1_arm_snapshot"] = m; spec.loader.exec_module(m)
def actions(parser):
    def plain(v):
        return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
    return [{"dest": a.dest, "option_strings": list(a.option_strings), "action": type(a).__name__, "default": plain(a.default),
             "type": getattr(a.type, "__name__", None if a.type is None else repr(a.type)),
             "choices": list(a.choices) if a.choices is not None else None, "required": bool(a.required), "nargs": a.nargs,
             "const": plain(a.const), "help": a.help} for a in parser._actions]
if len(sys.argv) > 2 and sys.argv[2] == "build_parser":
    print(json.dumps(actions(m.build_parser()), sort_keys=True))
else:
    real = argparse.ArgumentParser.parse_args
    def grab(self, *a, **k):
        raise _Got(self)
    argparse.ArgumentParser.parse_args = grab
    try:
        m.main()
    except _Got as g:
        print(json.dumps(actions(g.args[0]), sort_keys=True))
'''


def _env():
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))


def _run(*args):
    return subprocess.run([sys.executable, str(ARM), *args], capture_output=True, text=True, timeout=900, cwd=REPO, env=_env())


def _actions(path, mode=""):
    p = subprocess.run([sys.executable, "-c", SNAPSHOT, str(path), mode], capture_output=True, text=True, timeout=300, cwd=REPO, env=_env())
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _load_arm():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tc1_arm_hook_under_test", ARM)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tc1_arm_hook_under_test"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def selftest():
    p = _run("--selftest")
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    d = Path(re.search(r"SELFTEST OK dir=(\S+)", p.stdout).group(1))
    return p, d


def test_the_selftest_runs_the_extension_block(selftest):
    p, d = selftest
    ext = re.search(r"extension=(\{.*\})$", p.stdout.strip().splitlines()[-1]).group(1)
    assert "'hook': {'edges': 13, 'losses_equal': True}" in ext and "'refusals': 5" in ext and "'engine_calls': 13" in ext, ext
    assert "FAILING-CASE ext-hook-output-equality: a hook scaling the training loss by 1.001 moves losses[0]" in p.stdout
    hk = json.loads((d / "tiny_e4b_fused_attn4_ext_hook.json").read_text())
    base = json.loads((d / "tiny_e4b_fused_attn4_ext_base.json").read_text())
    assert hk["losses"] == base["losses"] and hk["arm_hook"]["record"]["edges"] == list(range(hk["steps"] + 1))
    assert "arm_hook" not in base and "layer_major" not in base and "step_contract" not in base
    lm = json.loads((d / "tiny_e4b_fused_attn4_ext_lm.json").read_text())
    assert lm["layer_major"]["guard"]["where"] == "before the timed window" and lm["step_contract"]["optimizer_steps"] == lm["steps"]


def test_no_default_selftest_arm_carries_an_extension_key(selftest):
    _, d = selftest
    keys = {"arm_hook", "vram_cap", "unsloth_offload_layers", "layer_major", "step_contract"}
    for f in d.glob("tiny*.json"):
        if "_ext_" in f.name:
            continue
        r = json.loads(f.read_text())
        assert not keys & set(r), (f.name, sorted(keys & set(r)))


def test_the_fixture_is_the_parser_main_built_at_the_base_commit(tmp_path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:bench/tc1/tc1_arm.py"], capture_output=True)
    if r.returncode != 0:
        if os.environ.get("CI") == "true":
            pytest.fail(f"the base commit {BASE[:12]} is not readable here (fetch-depth): {r.stderr.decode()[:200]}")
        pytest.skip(f"the base commit {BASE[:12]} is not in this clone")
    base = tmp_path / "tc1_arm_base.py"
    base.write_bytes(r.stdout)
    assert _actions(base) == json.loads(FIXTURE.read_text())


def test_build_parser_is_the_old_parser_plus_the_three_options():
    old = {a["dest"]: a for a in json.loads(FIXTURE.read_text())}
    new = {a["dest"]: a for a in _actions(ARM, "build_parser")}
    assert set(new) - set(old) == NEW and not set(old) - set(new)
    for dest, a in old.items():
        assert new[dest] == a, dest
    assert [d for d in new if d in old] == list(old), "the old options keep their order"
    assert new["vram_cap_gib"]["type"] == "float" and new["vram_cap_gib"]["default"] is None
    assert new["unsloth_offload_layers"]["default"] is None
    assert new["layer_major"]["choices"] == [0, 1] and new["layer_major"]["default"] == 0


def test_main_parses_through_build_parser():
    """main() builds no parser of its own: what it parses is build_parser()'s, action for action."""
    assert _actions(ARM) == _actions(ARM, "build_parser")


REFUSALS = [
    (("--framework", "e4b", "--arm", "fused", "--unsloth-offload-layers", "2", "--vram-cap-gib", "12"), "is an Unsloth option"),
    (("--framework", "unsloth", "--arm", "unsloth", "--unsloth-offload-layers", "auto"), "needs --vram-cap-gib"),
    (("--framework", "unsloth", "--arm", "unsloth", "--unsloth-offload-layers", "0", "--vram-cap-gib", "12"), "auto or a positive integer"),
    (("--framework", "unsloth", "--arm", "unsloth", "--unsloth-offload-layers", "x", "--vram-cap-gib", "12"), "auto or a positive integer"),
    (("--framework", "unsloth", "--arm", "unsloth", "--layer-major", "1"), "is an e4b option"),
    (("--framework", "e4b", "--arm", "fused", "--layer-major", "1", "--microbatch-timing", "1"), "no micro-batch boundary"),
]


@pytest.mark.parametrize("args,why", REFUSALS)
def test_each_option_refuses_before_anything_loads(tmp_path, args, why):
    p = _run(*args, "--prereg", "bench/tc1/TC1-PREREG.md", "--tag", "ext_refuse", "--out", str(tmp_path), "--adapter-dir", str(tmp_path / "a"))
    assert p.returncode == 3, (p.returncode, (p.stdout + p.stderr)[-2000:])
    rows = [json.loads(f.read_text()) for f in tmp_path.glob("*.json")]
    assert len(rows) == 1 and rows[0]["status"] == "refused" and why in rows[0]["reason"], rows


def test_parse_offload_layers():
    arm = _load_arm()
    assert arm.parse_offload_layers(None) is None and arm.parse_offload_layers("auto") == "auto" and arm.parse_offload_layers("AUTO") == "auto"
    assert arm.parse_offload_layers("12") == 12 and arm.parse_offload_layers(3) == 3
    for bad in ("0", "-1", "1.5", "", "auto2"):
        with pytest.raises(ValueError):
            arm.parse_offload_layers(bad)


class _FakeCuda:
    def __init__(self, total_gib=24.0, getter=True):
        self.total, self.set_to = int(total_gib * 2 ** 30), None
        if getter:
            self.get_per_process_memory_fraction = lambda dev=0: self.set_to

    def is_available(self):
        return True

    def get_device_properties(self, dev):
        return types.SimpleNamespace(total_memory=self.total)

    def set_per_process_memory_fraction(self, frac, dev=0):
        self.set_to = frac


def _args(tmp_path, **kw):
    a = types.SimpleNamespace(framework="unsloth", fam="tiny", model="m", revision="0" * 40, arm="unsloth", tag="t", steps=1, seq=8, accum=1,
                              micro_batch=1, offload=0, prereg="p", note=None, out=str(tmp_path), adapter_dir=str(tmp_path / "a"),
                              vram_cap_gib=None, unsloth_offload_layers=None, layer_major=0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_the_vram_cap_sets_the_fraction_and_reads_it_back(tmp_path):
    arm = _load_arm()
    cuda = _FakeCuda(24.0)
    rec = arm.apply_vram_cap(_args(tmp_path, vram_cap_gib=12.0), cuda=cuda)
    assert cuda.set_to == 0.5 and rec == {"given_gib": 12.0, "resolved": {"fraction": 0.5, "total_bytes": 24 * 2 ** 30, "getter_reads": 0.5}}
    assert arm.apply_vram_cap(_args(tmp_path), cuda=cuda) is None


def test_the_vram_cap_refuses_without_the_getter_and_above_the_card(tmp_path):
    """Without get_per_process_memory_fraction, Unsloth's planner silently reads 1.0, so the cap refuses rather than run uncapped."""
    arm = _load_arm()
    with pytest.raises(SystemExit) as e:
        arm.apply_vram_cap(_args(tmp_path, vram_cap_gib=12.0), cuda=_FakeCuda(24.0, getter=False))
    assert e.value.code == 10 and "planner would read 1.0" in json.loads(next(tmp_path.glob("*.json")).read_text())["reason"]
    for f in tmp_path.glob("*.json"):
        f.unlink()
    with pytest.raises(SystemExit) as e:
        arm.apply_vram_cap(_args(tmp_path, vram_cap_gib=30.0), cuda=_FakeCuda(24.0))
    assert e.value.code == 3


def test_the_unsloth_offload_kwargs_come_from_the_signature(tmp_path):
    arm = _load_arm()

    def named(model_name, offload_layers=0, max_memory=None, **kw):
        pass

    def via_kwargs(model_name, offload_layers=0, **kw):
        pass

    def without(model_name, max_memory=None, **kw):
        pass

    def no_max(model_name, offload_layers=0):
        pass
    a = _args(tmp_path, unsloth_offload_layers="auto", vram_cap_gib=12.0)
    kw, rec = arm.unsloth_offload_kwargs(named, a)
    assert kw == {"offload_layers": "auto", "max_memory": {0: "12GiB"}} and rec["max_memory_via"] == "named" and rec["resolved"] is None
    assert arm.unsloth_offload_kwargs(via_kwargs, _args(tmp_path, unsloth_offload_layers=4, vram_cap_gib=12.5))[0] == \
        {"offload_layers": 4, "max_memory": {0: "12.5GiB"}}
    for fn in (without, no_max):
        with pytest.raises(NotImplementedError):
            arm.unsloth_offload_kwargs(fn, a)
    with pytest.raises(NotImplementedError):
        arm.unsloth_offload_kwargs(named, _args(tmp_path, unsloth_offload_layers=4))      # no cap
    assert arm.unsloth_offload_kwargs(named, _args(tmp_path)) == ({}, None)
    assert "offload_layers" in inspect.signature(named).parameters


def test_the_block_swaps_resolved_layers():
    arm = _load_arm()
    bs = types.SimpleNamespace(indices=[3, 4, 5])
    assert arm.unsloth_offload_resolved(types.SimpleNamespace(_unsloth_block_swap=bs)) == {"n_layers": 3, "indices": [3, 4, 5]}
    peft = types.SimpleNamespace(base_model=types.SimpleNamespace(model=types.SimpleNamespace(_unsloth_block_swap=bs)))
    assert arm.unsloth_offload_resolved(peft)["indices"] == [3, 4, 5]
    assert arm.unsloth_offload_resolved(types.SimpleNamespace()) is None


def test_no_hook_set_loads_nothing(tmp_path, monkeypatch):
    arm = _load_arm()
    monkeypatch.delenv(arm.ARM_HOOK_ENV, raising=False)
    assert arm.load_arm_hook(_args(tmp_path)) is None
    assert arm.arm_hook_call(None, "step_edge", 0) is None
