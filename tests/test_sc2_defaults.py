# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``bench/sc2/sc2_defaults.py``: the resolved serving defaults of a tree, and the diff that refuses SC2f's and SC2h's
launch on any change (SC2f Amendment 3, SC2h Amendment 1).

Pinned: the rule's own cases; a dump of THIS tree resolves every lever it names without raising, on CPU, with the
caller's ``E4B_*`` variables ignored; a moved default in the tree is refused by its lever's name, and ``--allow`` passes
exactly that lever.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bench" / "sc2" / "sc2_defaults.py"
sys.path.insert(0, str(TOOL.parent))
import sc2_defaults as sd  # noqa: E402


def test_the_rule_self_test():
    assert sd.self_test() == 0


def test_a_dump_of_this_tree_resolves_every_lever_on_cpu(tmp_path):
    out = tmp_path / "d.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT), E4B_K25_PLAN="today", E4B_PAGED_MAX_SEQS="64")
    r = subprocess.run([sys.executable, str(TOOL), "dump", "--out", str(out)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    lev = d["levers"]
    assert Path(d["e4b_file"]).resolve() == (ROOT / "experts4bit_qlora").resolve()
    for k in ("family.resolve_int8_head_bw", "family.resolve_attn_split_cap", "family.resolve_prefill_glue",
              "route.resolve_k25_plan", "route._k19_mode_env", "config.max_seqs_requested", "fusion.E4B_FUSE_T1_GLUE"):
        assert k in lev, k
    assert not [k for k, v in lev.items() if isinstance(v, str) and v.startswith(("RAISES", "ABSENT"))]
    # the caller's variables are cleared before anything resolves: the two set above leave no trace
    assert lev["config.max_seqs_requested"] == "auto"
    assert "today" not in json.dumps(lev["route.resolve_k25_plan"])


def test_a_moved_default_refuses_by_name_and_allow_passes_it(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))       # dump clears E4B_* from the environment it sees
    ref = sd.dump()
    from experts4bit_qlora import serve_paged as sp
    real = sp.resolve_int8_head_bw
    monkeypatch.setattr(sp, "resolve_int8_head_bw", lambda value, model_type, shape, **k: ("0", "default-off: moved"))
    moved = sd.dump()
    monkeypatch.setattr(sp, "resolve_int8_head_bw", real)
    r = sd.diff(ref, moved)
    assert not r["ok"] and r["refused"] == ["family.resolve_int8_head_bw"]
    assert sd.diff(ref, moved, ["family.resolve_int8_head_bw"])["ok"]
    assert sd.diff(ref, sd.dump())["ok"]


@pytest.mark.parametrize("argv,rc", [(["--self-test"], 0)])
def test_the_cli(argv, rc):
    assert sd.main(argv) == rc
