#!/usr/bin/env python3
"""Lane P86's reducer (bench/p86/PREREG-p86.md; e4b#564). Where do e4b's and vLLM's decode steps go, kernel by kernel,
on one box with the same prompt token ids?

Reads, from the run directory:
- e4b:  e4b_b{1,16}_int4{,_r2}.json (`step_ms_clean`, the graph-replay window) and logs/census_e4b_b{1,16}.txt (the
        replay census, P42's torch-profiler table; parsed by bench/p42/p42_reduce.py's `parse_census`);
- vLLM: vllm_graph_b{1,16}{,_r2}.json (P37's slope arm: `decode_ms_per_step`, the min-of-3 estimator) and
        vllm_census_b{1,16}.json (p86_vllm_census.py: per-kernel us per decode step, by the same slope).

Each engine's kernels are sorted into FAMILIES by a registered name map (first match wins), and the families into
ROLES that exist in both engines. Per batch and role: e4b ms/step, vLLM ms/step, and the gap. "Host and launch" is
each engine's step time minus its kernel sum.

  VOID       an arm is missing or failed; a census covers no decode steps; the vLLM build is not the registered one;
             a census's kernel sum exceeds its own step time by more than 10 % (the census is not measuring the step)
  NOT_READ   more than 10 % of either engine's kernel time at B=16 falls in "other" (the name map does not cover the
             engine: the ranking would be a ranking of the map)
  else       READ: the roles ranked by gap (e4b - vLLM, ms/step) at B=16; the largest positive gap names the next lane

    python p86_reduce.py --dir <dir> --out verdict.json
    python p86_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

VLLM_VERSION = "0.30.0"
BATCHES = (16, 1)
SLACK = 1.10          # a census may not exceed its step time by more than this
OTHER_MAX = 0.10      # "other" may hold at most this share of an engine's kernel time at B=16

# e4b: P57's families (bench/p57/p57_reduce.py), plus the fp8 KV append, then "other"
E4B_FAMILIES = [
    ("k16 small-M GEMM", re.compile(r"gemm_int4_b32_smallm")),
    ("int4 GEMV", re.compile(r"_gemv_int4_b32")),
    ("bf16 GEMM (cutlass/cublas)", re.compile(r"cutlass|cublas|xmma|nvjet|gemvx|Sm90|sm80|sm90", re.I)),
    ("fp8 paged decode", re.compile(r"paged_decode|f8dot")),
    ("fp8 KV append", re.compile(r"_fp8_append|fp8_kv_append")),
    ("index / gather / scatter", re.compile(r"index|gather|scatter|_combine_rows", re.I)),
    ("activation quant", re.compile(r"_quant_x_rows")),
    ("split-K reduce", re.compile(r"_reduce_partials|reduce_kernel")),
    ("rope/norm folds", re.compile(r"rope_norm_heads|rmsnorm|rms_norm|_swiglu|swiglu_rows", re.I)),
]
# vLLM: by the kernel names its CUDA ops, Marlin, attention backends and torch.compile emit
VLLM_FAMILIES = [
    ("Marlin MoE", re.compile(r"moe.*marlin|marlin.*moe|fused_moe|moe_wna16|grouped_gemm|moegemm", re.I)),
    ("Marlin dense", re.compile(r"marlin|gptq|awq", re.I)),
    ("MoE routing", re.compile(r"moe_align|count_and_sort|topk|gating|moe_sum|permute|moe::|expert|index|gather|scatter", re.I)),
    ("attention", re.compile(r"flash|attn|attention|fmha|paged|reshape_and_cache|batchdecode|batchprefill|cascade|unified", re.I)),
    ("norm/rope/act", re.compile(r"rms_?norm|layer_?norm|rotary|rope|act_and_mul|silu|gelu", re.I)),
    ("torch.compile (triton)", re.compile(r"^triton_")),
    ("bf16 GEMM (cutlass/cublas)", re.compile(r"cutlass|cublas|nvjet|xmma|gemm|gemv|sm90|sm100|sm120|ampere", re.I)),
]
# roles present in both engines (e4b families, vLLM families)
ROLES = [
    ("quantized linear (experts + attention projections)",
     ("k16 small-M GEMM", "int4 GEMV", "activation quant", "split-K reduce"), ("Marlin MoE", "Marlin dense")),
    ("routing / gather glue", ("index / gather / scatter",), ("MoE routing",)),
    ("attention + KV write", ("fp8 paged decode", "fp8 KV append"), ("attention",)),
    ("norm / rope / activation", ("rope/norm folds",), ("norm/rope/act", "torch.compile (triton)")),
    ("dense bf16 GEMM (output head)", ("bf16 GEMM (cutlass/cublas)",), ("bf16 GEMM (cutlass/cublas)",)),
    ("other kernels", ("other",), ("other",)),
]


def _p42():
    here = Path(__file__).resolve().parent
    for cand in (here / "p42_reduce.py", here.parent / "p42" / "p42_reduce.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("p42_reduce", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("p42_reduce.py not found beside this reducer or in bench/p42")


def classify(rows, families):
    """rows: [{name, ms}] -> {family: ms}, every row in exactly one family ('other' last)."""
    out = {name: 0.0 for name, _ in families}
    out["other"] = 0.0
    top = {name: [] for name, _ in families}
    top["other"] = []
    for r in rows:
        fam = next((name for name, rx in families if rx.search(r["name"])), "other")
        out[fam] += r["ms"]
        top[fam].append((r["ms"], r["name"]))
    return out, {k: sorted(v, reverse=True)[:4] for k, v in top.items()}


def e4b_side(d: Path, B: int, parse_census):
    steps = []
    for sfx in ("", "_r2"):
        f = d / f"e4b_b{B}_int4{sfx}.json"
        try:
            steps.append(float(json.loads(f.read_text())["step_ms_clean"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            steps.append(None)
    c = d / "logs" / f"census_e4b_b{B}.txt"
    if not c.is_file():
        return {"steps": steps, "census": None}
    replays, rows = parse_census(c.read_text())
    rows = [{"name": r["name"], "ms": r["self_ms"] / replays} for r in rows] if replays else []
    return {"steps": steps, "census": {"replays": replays, "rows": rows}}


def vllm_side(d: Path, B: int):
    steps, versions = [], set()
    for sfx in ("", "_r2"):
        f = d / f"vllm_graph_b{B}{sfx}.json"
        try:
            r = json.loads(f.read_text())
            steps.append(float(r["decode_ms_per_step"]))
            versions.add(r.get("vllm_version"))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            steps.append(None)
    f = d / f"vllm_census_b{B}.json"
    try:
        c = json.loads(f.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {"steps": steps, "census": None, "versions": versions}
    versions.add(c.get("vllm_version"))
    rows, noise = decode_rows(c.get("kernels", []))
    return {"steps": steps, "versions": versions,
            "census": {"decode_steps": c.get("decode_steps"), "in_process": c.get("in_process"), "rows": rows,
                       "prefill_noise_ms": noise}}


def decode_rows(kernels):
    """A decode kernel runs every step, so it has MORE calls in the long run than in the short one. A kernel with the
    same count in both ran only in prefill/setup, and its per-step difference is run-to-run jitter (the A2000
    rehearsal: a prefill GEMM at 0.00 calls/step read 101.5 us/step). Those are excluded and reported as noise."""
    rows, noise = [], 0.0
    for k in kernels:
        if int(k.get("calls_long", 0)) > int(k.get("calls_short", 0)):
            rows.append({"name": k["name"], "ms": k["us_per_step"] / 1e3})
        else:
            noise += k["us_per_step"] / 1e3
    return rows, noise


def reduce(e4b: dict, vllm: dict) -> dict:
    v: dict = {"lane": "P86", "verdict": None, "reasons": [], "batches": {}}
    for B in BATCHES:
        E, V = e4b.get(B), vllm.get(B)
        if not E or not V or None in E["steps"] or None in V["steps"] or not E["census"] or not V["census"]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"B={B}: an arm is missing or failed")
            return v
        if not E["census"]["replays"] or not V["census"]["decode_steps"] or not E["census"]["rows"] or not V["census"]["rows"]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"B={B}: a census covers no decode steps")
            return v
        if V["versions"] != {VLLM_VERSION}:
            v["verdict"] = "VOID"
            v["reasons"].append(f"B={B}: vLLM {sorted(map(str, V['versions']))}, not the registered {VLLM_VERSION}")
            return v
        e_step, v_step = statistics.median(E["steps"]), statistics.median(V["steps"])
        e_fam, e_top = classify(E["census"]["rows"], E4B_FAMILIES)
        v_fam, v_top = classify(V["census"]["rows"], VLLM_FAMILIES)
        e_sum, v_sum = sum(e_fam.values()), sum(v_fam.values())
        for eng, s, st in (("e4b", e_sum, e_step), ("vLLM", v_sum, v_step)):
            if s > SLACK * st:
                v["verdict"] = "VOID"
                v["reasons"].append(f"B={B}: the {eng} census sums {s:.3f} ms/step against a {st:.3f} ms step (> {SLACK}x)")
                return v
        roles = []
        for role, ef, vf in ROLES:
            em, vm = sum(e_fam.get(f, 0.0) for f in ef), sum(v_fam.get(f, 0.0) for f in vf)
            roles.append({"role": role, "e4b_ms": em, "vllm_ms": vm, "gap_ms": em - vm})
        roles.append({"role": "host and launch (step - kernels)", "e4b_ms": e_step - e_sum, "vllm_ms": v_step - v_sum,
                      "gap_ms": (e_step - e_sum) - (v_step - v_sum)})
        v["batches"][B] = {"e4b_step_ms": e_step, "vllm_step_ms": v_step, "ratio": e_step / v_step,
                           "e4b_steps": E["steps"], "vllm_steps": V["steps"], "e4b_kernel_ms": e_sum, "vllm_kernel_ms": v_sum,
                           "e4b_families": e_fam, "vllm_families": v_fam, "e4b_top": e_top, "vllm_top": v_top, "roles": roles,
                           "vllm_prefill_noise_ms": V["census"].get("prefill_noise_ms", 0.0),
                           "e4b_other_share": e_fam["other"] / e_sum if e_sum else 1.0,
                           "vllm_other_share": v_fam["other"] / v_sum if v_sum else 1.0}
    b16 = v["batches"][16]
    if b16["e4b_other_share"] > OTHER_MAX or b16["vllm_other_share"] > OTHER_MAX:
        v["verdict"] = "NOT_READ"
        v["reasons"].append(f"B=16: 'other' holds {b16['e4b_other_share']:.0%} of e4b's and {b16['vllm_other_share']:.0%} of "
                            f"vLLM's kernel time (> {OTHER_MAX:.0%}): the name map does not cover the engine")
        return v
    ranked = sorted((r for r in b16["roles"]), key=lambda r: -r["gap_ms"])
    v["ranked_b16"] = [(r["role"], round(r["gap_ms"], 4)) for r in ranked]
    v["verdict"] = "READ"
    top = ranked[0]
    v["reasons"].append(f"B=16 step {b16['e4b_step_ms']:.3f} vs {b16['vllm_step_ms']:.3f} ms ({b16['ratio']:.3f}x); the largest gap is "
                        f"'{top['role']}' at {top['gap_ms']:+.3f} ms/step")
    v["next_lane_role"] = top["role"] if top["gap_ms"] > 0 else None
    return v


def load(d: Path):
    p42 = _p42()
    return ({B: e4b_side(d, B, p42.parse_census) for B in BATCHES}, {B: vllm_side(d, B) for B in BATCHES})


def _synthetic(e_rows, v_rows, e_step=11.6, v_step=8.3, version=VLLM_VERSION):
    e4b = {B: {"steps": [e_step, e_step], "census": {"replays": 8, "rows": [dict(r) for r in e_rows]}} for B in BATCHES}
    vllm = {B: {"steps": [v_step, v_step], "versions": {version},
                "census": {"decode_steps": 96, "in_process": True, "rows": [dict(r) for r in v_rows]}} for B in BATCHES}
    return e4b, vllm


E_ROWS = [{"name": "_gemv_int4_b32", "ms": 6.34}, {"name": "_gemm_int4_b32_smallm", "ms": 1.30},
          {"name": "void at::native::indexSelectSmallIndex", "ms": 0.95}, {"name": "_fp8_paged_decode_split_f8dot", "ms": 0.68},
          {"name": "void cutlass::Kernel2<cutlass_80_wmma>", "ms": 0.57}, {"name": "_reduce_partials", "ms": 0.37},
          {"name": "_quant_x_rows", "ms": 0.31}, {"name": "rope_norm_heads", "ms": 0.27}]
V_ROWS = [{"name": "void marlin_moe_wna16::Marlin<...>", "ms": 4.6}, {"name": "void marlin::Marlin<...>", "ms": 1.1},
          {"name": "moe_align_block_size_kernel", "ms": 0.3}, {"name": "flash_fwd_splitkv_kernel", "ms": 0.6},
          {"name": "nvjet_tst_bf16", "ms": 0.45}, {"name": "triton_red_fused_rms_norm_0", "ms": 0.4}]


def self_test() -> None:
    got = reduce(*_synthetic(E_ROWS, V_ROWS))
    assert got["verdict"] == "READ", got
    assert got["next_lane_role"] == "quantized linear (experts + attention projections)", got["ranked_b16"]
    b16 = got["batches"][16]
    assert abs(b16["e4b_kernel_ms"] - sum(r["ms"] for r in E_ROWS)) < 1e-9
    assert abs(sum(r["e4b_ms"] for r in b16["roles"]) - 11.6) < 1e-9          # the roles plus host sum to the step
    assert abs(sum(r["vllm_ms"] for r in b16["roles"]) - 8.3) < 1e-9
    # glue the biggest gap
    e_rows = [dict(r, ms=(r["ms"] if "gather" not in r["name"] and "index" not in r["name"] else 3.0)) for r in E_ROWS]
    assert reduce(*_synthetic(e_rows, V_ROWS, e_step=13.7))["next_lane_role"] == "routing / gather glue"
    # missing arm, wrong vLLM, census over its step, no decode steps: VOID
    e, v = _synthetic(E_ROWS, V_ROWS)
    e[1]["steps"] = [None, 4.2]
    assert reduce(e, v)["verdict"] == "VOID"
    assert reduce(*_synthetic(E_ROWS, V_ROWS, version="0.29.0"))["verdict"] == "VOID"
    assert reduce(*_synthetic(E_ROWS, V_ROWS, v_step=6.0))["verdict"] == "VOID"          # 7.45 ms of kernels in a 6.0 ms step
    e, v = _synthetic(E_ROWS, V_ROWS)
    v[16]["census"]["decode_steps"] = 0
    assert reduce(e, v)["verdict"] == "VOID"
    e, v = _synthetic(E_ROWS, V_ROWS)
    e[16]["census"] = None
    assert reduce(e, v)["verdict"] == "VOID"
    # an unmapped engine: NOT_READ
    v_rows = V_ROWS + [{"name": "some_new_kernel_xyz", "ms": 1.2}]
    assert reduce(*_synthetic(E_ROWS, v_rows, v_step=9.6))["verdict"] == "NOT_READ"
    # the family maps: every registered name lands where it should
    fam, _ = classify([{"name": n, "ms": 1.0} for n in ("void marlin_moe_wna16::Marlin<x>", "void marlin::Marlin<y>",
                                                         "moe_align_block_size_kernel", "topkGatingSoftmax", "flash_fwd_splitkv_kernel",
                                                         "reshape_and_cache_flash_kernel", "vllm::rms_norm_kernel",
                                                         "triton_poi_fused_add_0", "nvjet_tst_192x8", "foo")], VLLM_FAMILIES)
    assert fam == {"Marlin MoE": 1.0, "Marlin dense": 1.0, "MoE routing": 2.0, "attention": 2.0, "norm/rope/act": 1.0,
                   "torch.compile (triton)": 1.0, "bf16 GEMM (cutlass/cublas)": 1.0, "other": 1.0}, fam
    fam, _ = classify([{"name": n, "ms": 1.0} for n in ("_gemm_int4_b32_smallm", "_gemv_int4_b32", "_fp8_append_bt1_side",
                                                         "_fp8_paged_decode_split_f8dot", "void at::native::indexSelectLargeIndex",
                                                         "_quant_x_rows", "_reduce_partials", "rope_norm_heads", "xyz")], E4B_FAMILIES)
    assert fam["k16 small-M GEMM"] == fam["int4 GEMV"] == fam["fp8 KV append"] == fam["other"] == 1.0, fam
    # the decode filter: only kernels that gained calls in the long run are decode; the rest is reported noise
    rows, noise = decode_rows([{"name": "decode_k", "us_per_step": 50.0, "calls_short": 32, "calls_long": 128},
                               {"name": "prefill_gemm", "us_per_step": 101.5, "calls_short": 28, "calls_long": 28}])
    assert [r["name"] for r in rows] == ["decode_k"] and abs(noise - 0.1015) < 1e-12, (rows, noise)
    print("p86_reduce self-test OK (13 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(*load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1, default=list))
    print(f"P86_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for B, b in v.get("batches", {}).items():
        print(f"  B={B}: e4b {b['e4b_step_ms']:.3f} ms (kernels {b['e4b_kernel_ms']:.3f}) | vLLM {b['vllm_step_ms']:.3f} ms "
              f"(kernels {b['vllm_kernel_ms']:.3f}) | ratio {b['ratio']:.3f}")
        for r in b["roles"]:
            print(f"    {r['role'][:48]:48s} e4b {r['e4b_ms']:7.3f}  vLLM {r['vllm_ms']:7.3f}  gap {r['gap_ms']:+7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
