#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""The resolved serving defaults of one e4b tree, dumped on CPU, and the diff that refuses a launch on any change
(SC2f Amendment 3, SC2h Amendment 1).

``dump`` imports the ``experts4bit_qlora`` it finds first on ``sys.path`` (point ``PYTHONPATH`` at a checkout) and records,
with every ``E4B_*`` / ``GNF4_*`` variable cleared, what each serving lever resolves to for the lanes' model:
- the server's config from the environment (``PagedServeConfig.from_env``, every field);
- the per-family resolutions for ``qwen3_moe``: the decode fusions, the prefill glue, the attention split cap, and the
  int8 lm_head at Qwen3-30B-A3B's head shape;
- the expert-path routes ``engines.hot_residency`` resolves from the environment (K19, the int4 prefill, the lean glue
  and dispatch, the wide tiles, the tile programs, K21, K25 and its decode and plan, T == 1 device grouping);
- a source scan (``scan``): every ``E4B_*`` / ``GNF4_*`` environment read in the package with its default literal, and
  every such name the source mentions, so an inline default with no resolver, or a knob added later, is a lever too.

A lever that a tree lacks, or whose resolver raises, records that fact as its value. ``diff A B`` compares two dumps
and exits 1 naming every lever whose value differs or that exists in only one, unless ``--allow LEVER=reason`` names it. The
lane's amendment names any lever it allows. Nothing here needs a GPU; the grouped-nf4-gemm installed beside the tree is
recorded, and the two dumps must record the same one.

    PYTHONPATH=<ref checkout>    python bench/sc2/sc2_defaults.py dump --out ref.json
    PYTHONPATH=<launch checkout> python bench/sc2/sc2_defaults.py dump --out launch.json
    python bench/sc2/sc2_defaults.py diff ref.json launch.json [--allow LEVER=reason ...]
    python bench/sc2/sc2_defaults.py --self-test
"""
import argparse
import ast
import dataclasses
import json
import os
import sys
from pathlib import Path

MODEL_TYPE = "qwen3_moe"
HEAD_SHAPE = (151936, 2048)          # Qwen3-30B-A3B's lm_head (vocab x hidden)
HOT_RESIDENCY_ENVS = ("_k19_mode_env", "_int4_prefill_mode_env", "_lean_glue_mode_env", "_prefill_lean_dispatch_env",
                      "_k19_prefill_big_m_env", "_wide_tiles_mode_env", "_tile_programs_env", "_k21_mode_env",
                      "_k25_mode_env", "_k25_lut_env", "_nf4_t1_device_grouping_env")


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def _call(fn, *a, **k):
    try:
        return _jsonable(fn(*a, **k))
    except Exception as e:  # noqa: BLE001 -- a raising resolver is a value the diff compares
        return f"RAISES {type(e).__name__}: {str(e)[:200]}"


PREFIXES = ("E4B_", "GNF4_")


def _const_names(tree) -> dict:
    """Module-level ``NAME = "E4B_..."`` constants, so ``os.environ.get(NAME, ...)`` resolves to its variable."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name) and node.value.value.startswith(PREFIXES):
                    out[t.id] = node.value.value
    return out


def scan(pkg_dir) -> dict:
    """Every ``E4B_*`` / ``GNF4_*`` environment read in the package's source with its default's source text, and every
    such name the source mentions at all. A read is a call whose first argument is (or names a constant equal to) such
    a string and whose callee is ``*.get`` / ``getenv`` / ``env`` / ``_env``; the default is the second argument's
    ``ast.unparse`` (``None`` when absent). Inline defaults with no resolver, and knobs added later, show up here."""
    reads: dict = {}
    names: set = set()
    for path in sorted(Path(pkg_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        consts = _const_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(PREFIXES) \
                    and node.value.replace("_", "").isalnum() and node.value.upper() == node.value:
                names.add(node.value)
            if not isinstance(node, ast.Call) or not node.args:
                continue
            f = node.func
            callee = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if callee not in ("get", "getenv", "env", "_env"):
                continue
            a0 = node.args[0]
            name = (a0.value if isinstance(a0, ast.Constant) and isinstance(a0.value, str)
                    else consts.get(a0.id) if isinstance(a0, ast.Name) else None)
            if not name or not name.startswith(PREFIXES):
                continue
            default = ast.unparse(node.args[1]) if len(node.args) > 1 else "None"
            reads.setdefault(name, set()).add(default)
    return {"reads": {k: sorted(v) for k, v in sorted(reads.items())}, "names": sorted(names)}


def dump() -> dict:
    for k in [k for k in os.environ if k.startswith(("E4B_", "GNF4_"))]:
        del os.environ[k]
    import experts4bit_qlora as e
    from experts4bit_qlora import serve_paged as sp
    from experts4bit_qlora.engines import hot_residency as hr
    out = {}
    cfg = sp.PagedServeConfig.from_env()
    for f in dataclasses.fields(cfg):
        out[f"config.{f.name}"] = _jsonable(getattr(cfg, f.name))
    unset = getattr(sp, "FUSION_UNSET", "default")
    if hasattr(sp, "resolve_fusion_modes"):
        modes, sources = sp.resolve_fusion_modes({k: unset for k in sp.FUSION_KNOBS}, MODEL_TYPE)
        for k in sorted(modes):
            out[f"fusion.{k}"] = [modes[k], sources.get(k)]
    for name, args in (("resolve_prefill_glue", (unset, MODEL_TYPE)), ("resolve_attn_split_cap", (unset, MODEL_TYPE)),
                       ("resolve_int8_head_bw", (unset, MODEL_TYPE, HEAD_SHAPE))):
        fn = getattr(sp, name, None)
        out[f"family.{name}"] = _call(fn, *args) if fn else "ABSENT"
    # K25's decode resolves process-wide at engine build (the server's order), and the env readers then report the
    # resolved value; resolving it first makes a dump the same in a fresh process and after another dump
    lut = getattr(hr, "resolve_k25_lut", None)
    lut_v = lut(model_type=MODEL_TYPE) if lut else None
    for name in HOT_RESIDENCY_ENVS:
        fn = getattr(hr, name, None)
        out[f"route.{name}"] = _call(fn) if fn else "ABSENT"
    out["route.resolve_k25_lut"] = _call(lambda: lut_v) if lut else "ABSENT"
    plan = getattr(hr, "resolve_k25_plan", None)
    out["route.resolve_k25_plan"] = (_call(plan, lut_v, model_type=MODEL_TYPE) if plan and lut_v is not None
                                     else "ABSENT")
    try:
        import importlib.metadata as md
        gnf4 = md.version("grouped-nf4-gemm")
    except Exception:  # noqa: BLE001
        gnf4 = None
    src = scan(os.path.dirname(e.__file__))
    for name, defaults in src["reads"].items():
        out[f"env.{name}"] = defaults
    for name in src["names"]:
        out.setdefault(f"env.{name}", "NAMED")      # a name the source mentions without a direct read with a default
    return {"e4b_version": getattr(e, "__version__", None), "e4b_file": os.path.dirname(e.__file__),
            "grouped_nf4_gemm": gnf4, "model_type": MODEL_TYPE, "levers": out}


def parse_allow(items) -> dict:
    """``--allow LEVER=reason`` -> ``{lever: reason}``; a missing or empty reason is refused (the reason is the record)."""
    out = {}
    for it in items:
        lever, sep, reason = str(it).partition("=")
        if not sep or not lever.strip() or not reason.strip():
            raise ValueError(f"--allow needs LEVER=reason, got {it!r}")
        out[lever.strip()] = reason.strip()
    return out


def diff(a: dict, b: dict, allow=()) -> dict:
    reasons = allow if isinstance(allow, dict) else {k: "" for k in allow}
    allow = set(reasons)
    la, lb = a["levers"], b["levers"]
    changed = sorted(k for k in set(la) & set(lb) if la[k] != lb[k])
    only_a, only_b = sorted(set(la) - set(lb)), sorted(set(lb) - set(la))
    named = changed + only_a + only_b
    refused = [k for k in named if k not in allow]
    why = []
    if a.get("grouped_nf4_gemm") != b.get("grouped_nf4_gemm"):
        why.append(f"grouped-nf4-gemm differs: {a.get('grouped_nf4_gemm')} vs {b.get('grouped_nf4_gemm')}")
    return {"ok": not refused and not why, "refused": refused,
            "allowed": {k: reasons[k] for k in sorted(set(named) & allow)},
            "changed": {k: [la[k], lb[k]] for k in changed}, "only_ref": only_a, "only_launch": only_b, "why": why}


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def self_test() -> int:
    base = {"grouped_nf4_gemm": "0.50.0", "levers": {"config.max_seqs": 16, "route._k25_mode_env": "auto",
                                                     "fusion.E4B_FUSE_T1_GLUE": ["auto", "default-allowlisted"]}}
    same = json.loads(json.dumps(base))
    moved = json.loads(json.dumps(base))
    moved["levers"]["route._k25_mode_env"] = "0"
    added = json.loads(json.dumps(base))
    added["levers"]["config.prefill_every"] = 1
    other_gnf4 = json.loads(json.dumps(base))
    other_gnf4["grouped_nf4_gemm"] = "0.51.0"
    cases = [
        ("identical dumps pass", diff(base, same)["ok"]),
        ("a moved default refuses by name", diff(base, moved)["refused"] == ["route._k25_mode_env"]),
        ("a new lever refuses by name", diff(base, added)["refused"] == ["config.prefill_every"]),
        ("a lever gone refuses by name", diff(added, base)["refused"] == ["config.prefill_every"]),
        ("--allow passes exactly the named lever", diff(base, added, ["config.prefill_every"])["ok"]
         and not diff(base, moved, ["config.prefill_every"])["ok"]),
        ("--allow keeps its reason", diff(base, added, parse_allow(["config.prefill_every=opt-in, default off"]))
         ["allowed"] == {"config.prefill_every": "opt-in, default off"}),
        ("--allow without a reason is refused", _raises(lambda: parse_allow(["config.prefill_every"]))
         and _raises(lambda: parse_allow(["config.prefill_every= "]))),
        ("a different kernels package refuses", not diff(base, other_gnf4)["ok"]),
        ("a raising resolver is a value", _call(lambda: 1 / 0).startswith("RAISES ZeroDivisionError")),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"sc2_defaults self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    c = sub.add_parser("diff")
    c.add_argument("ref")
    c.add_argument("launch")
    c.add_argument("--allow", action="append", default=[])
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "dump":
        rec = dump()
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
        print(f"SC2_DEFAULTS dumped {len(rec['levers'])} levers from e4b {rec['e4b_version']} at {rec['e4b_file']}")
        return 0
    if a.cmd == "diff":
        with open(a.ref) as f:
            ra = json.load(f)
        with open(a.launch) as f:
            rb = json.load(f)
        r = diff(ra, rb, parse_allow(a.allow))
        print("SC2_DEFAULTS_DIFF " + json.dumps(r, sort_keys=True))
        return 0 if r["ok"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
