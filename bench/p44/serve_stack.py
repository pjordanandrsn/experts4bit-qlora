#!/usr/bin/env python3
"""bench/p44/serve_stack.py -- ONE table of the serving arms bo7 timed, and ONE builder for the served model.

Two consumers, two lanes, one definition of "the arm": `kl_serve.py` (P44-b) and `expert_residuals.py` (P44-a's
census) build the served model through :func:`build_served_model`; the P44-a K8 arms run `step_decomp.py` from the
shell and take their environment from ``python serve_stack.py env <family> <arm>``, so the flag set a K8 row was scored
under and the flag set a KL row was scored under are the same bytes. The table is bo7's `arm` lines
(`bench/hybrid-g9/throughput-20260904/bo7/logs/bo7_run.sh`): exp / calibrated-attention / fuse-word / extra env.

The lane hook (`bench/p42/hook/usercustomize.py`, referenced by the drivers and staged beside this file as ``hook/``) applies the int4 expert and
int4 attention levers right after ``enable_hybrid_tier`` returns, reading the environment; the fold and epilogue
fusions are env-gated and structural (a set flag either engages or refuses with a sentence). Nothing here changes a
kernel or a default.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# fuse word -> (E4B_FUSE_T1_GLUE, E4B_FUSE_T1_GLUE_R2, E4B_FUSE_ROUTER_EPI), bo7's fenv() verbatim
FUSE = {"0": (0, 0, 0), "r1": (1, 0, 0), "r12": (1, 1, 0), "epi": (0, 0, 1), "r1epi": (1, 0, 1), "all": (1, 1, 1)}

# family -> arm -> (E4B_SERVE_EXP_INT4, E4B_SERVE_ATTN_INT4_CALIB, fuse word, extra env)
ARMS = {
    "olmoe": {
        "nf4": (0, 0, "0", {}),
        "int4all": (1, 1, "all", {}),                                   # RTN experts + C4-calibrated int4 attention (bo7)
        # P44-PREREG arm 3: the streamed 64k SEQUENTIAL calibration that licensed Qwen3 (bo6c), calibrated on
        # wikitext-TRAIN so c4val1 is the outside text. The hook shares one E4B_CALIB_SOURCE, so the int4 attention
        # pack of this arm is wikitext-calibrated too (bo7's timed calibexp arms calibrated on C4) -- calibration text
        # only; the kernels and the speed are the same.
        "calibexp_all": (1, 1, "all", {"E4B_SERVE_EXP_INT4_CALIB": "1", "E4B_CALIB_NSEQ": "128",
                                       "E4B_CALIB_SOURCE": "wikitext"}),
    },
    "granite": {
        "nf4": (0, 0, "0", {}),
        "r12epi": (0, 0, "all", {}),
        "int4_r12epi": (1, 0, "all", {}),
        "calibexp_r12epi": (1, 0, "all", {"E4B_SERVE_EXP_INT4_CALIB": "1"}),   # hook default: 32 x 512 C4 tokens
    },
    "gptoss": {
        "nf4_r12": (0, 0, "r12", {}),
        "store_r12": (1, 0, "r12", {"E4B_INT4_KEEP_NF4": "1"}),           # native MXFP4 store for single rows
    },
    "gemma4": {
        "nf4": (0, 0, "0", {}),
        "r1epi": (0, 0, "r1epi", {}),                                       # round 2 refuses by design on this family
        "int4_r1epi": (1, 0, "r1epi", {}),
        "calattn_r1epi": (0, 1, "r1epi", {}),
    },
    "mixtral": {
        "nf4": (0, 0, "0", {}),
        "all": (1, 1, "all", {}),
    },
}

# model id + the revision each P44 runner pins (resolved 2026-09-19 from the Hub, before any box was rented)
MODELS = {
    "olmoe": ("allenai/OLMoE-1B-7B-0924-Instruct", "7f1c97f440f06ce36705e4f2b843edb5925f4498"),
    "granite": ("ibm-granite/granite-3.1-3b-a800m-instruct", "a02780686e08a03fe0d2679a293b5c74a90efa89"),
    "mixtral": ("mistralai/Mixtral-8x7B-Instruct-v0.1", "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"),
    "gemma4": ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
    "gptoss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
}

# P47 (#597, the Gemma-4 diagnostic): the SAME nf4 lever set (no int4 stores, no fusions) on five BUILDERS of the
# model. Every arm's env is the nf4 control's; what differs is how the model is constructed -- `BUILDERS` below.
ARMS["gemma4diag"] = {
    "served_nf4": (0, 0, "0", {}),          # P44-b's `nf4`: arena bake + placement + hybrid tier, all-VRAM (the 1.077 anchor)
    "loader_nf4": (0, 0, "0", {}),          # the training loader alone: load_moe_4bit_streaming, NF4 experts, no arena / tier
    "loader_bf16experts": (0, 0, "0", {}),  # the training loader with quantize_layers=set(): e4b's Gemma-4 modelling, bf16 experts
    "loader_nf4_lo": (0, 0, "0", {}),       # NF4 experts in the first half of the layers only (the rest bf16)
    "loader_nf4_hi": (0, 0, "0", {}),       # NF4 experts in the second half only
}
MODELS["gemma4diag"] = MODELS["gemma4"]
BUILDERS = {"gemma4diag": {"served_nf4": "served", "loader_nf4": "loader", "loader_bf16experts": "loader_unquant",
                           "loader_nf4_lo": "loader_lo", "loader_nf4_hi": "loader_hi"}}
LOADER_BUILDERS = ("loader", "loader_unquant", "loader_lo", "loader_hi")
# P48 (#597, after P47 put the nat in layers 0-14): ONE NF4 layer at a time. Family `gemma4layer`, arm `L<i>` for every
# text-decoder layer, builder `loader_only_<i>` = the training loader with quantize_layers={i} (bf16 experts everywhere
# else). 30 arms; the smallest row bounds e4b's Gemma-4 modelling cost from above (P47 amendment 1), the per-layer
# profile says WHERE block-64 NF4 hurts. No served arm -> no arena bake (`needs_arena`).
GEMMA4_TEXT_LAYERS = 30
ARMS["gemma4layer"] = {f"L{i:02d}": (0, 0, "0", {}) for i in range(GEMMA4_TEXT_LAYERS)}
MODELS["gemma4layer"] = MODELS["gemma4"]
BUILDERS["gemma4layer"] = {f"L{i:02d}": f"loader_only_{i}" for i in range(GEMMA4_TEXT_LAYERS)}
_ONLY = re.compile(r"^loader_only_(\d+)(?::([a-z0-9]+))?(?::b(\d+))?$")   # loader_only_<layer>[:<quant_type>][:b<blocksize>]
# P49 (#597, after P48 put 83 % of the gap in layer 0 alone): the expert FORMAT on layer 0, and NF4 at four depths for the
# activation probe. Same nf4 lever set everywhere (no int4 store, no fusion); the builder names the store.
ARMS["gemma4fmt"] = {
    "L00_nf4": (0, 0, "0", {}), "L00_fp4": (0, 0, "0", {}), "L00_int8": (0, 0, "0", {}), "L00_fp8": (0, 0, "0", {}),
    "L00_nf4b32": (0, 0, "0", {}),
    "L07_nf4": (0, 0, "0", {}), "L14_nf4": (0, 0, "0", {}), "L21_nf4": (0, 0, "0", {}), "L27_nf4": (0, 0, "0", {}),
}
MODELS["gemma4fmt"] = MODELS["gemma4"]
BUILDERS["gemma4fmt"] = {"L00_nf4": "loader_only_0:nf4", "L00_fp4": "loader_only_0:fp4", "L00_int8": "loader_only_0:int8",
                         "L00_fp8": "loader_only_0:fp8", "L00_nf4b32": "loader_only_0:nf4:b32",
                         "L07_nf4": "loader_only_7:nf4", "L14_nf4": "loader_only_14:nf4", "L21_nf4": "loader_only_21:nf4",
                         "L27_nf4": "loader_only_27:nf4"}


def parse_only(builder: str):
    """``loader_only_<i>[:<quant_type>][:b<blocksize>]`` -> (layer, quant_type, blocksize) or None."""
    m = _ONLY.match(builder)
    if not m:
        return None
    return int(m.group(1)), (m.group(2) or "nf4"), int(m.group(3) or 64)


def needs_arena(family: str) -> bool:
    """True when any arm of the family is built through the served stack (which needs the NF4 arena bake)."""
    return any(builder_for(family, arm) == "served" for arm in ARMS[family])


def builder_for(family: str, arm: str) -> str:
    """How this arm's model is built: ``served`` (P44-b, default) or one of the P47 loader builders."""
    return BUILDERS.get(family, {}).get(arm, "served")


def control_arm(family: str) -> str:
    """The family's control row: ``nf4`` where registered, else the FIRST registered arm (P47: ``served_nf4``)."""
    return "nf4" if "nf4" in ARMS[family] else next(iter(ARMS[family]))


LANE_KEYS = ("E4B_SERVE_EXP_INT4", "E4B_SERVE_EXP_INT4_CALIB", "E4B_SERVE_ATTN_INT4_CALIB", "E4B_SERVE_ATTN_INT4",
             "E4B_SERVE_LMHEAD_INT4_CALIB", "E4B_SERVE_DENSE_INT4_CALIB", "E4B_CALIB_SOURCE", "E4B_CALIB_NSEQ",
             "E4B_INT4_KEEP_NF4", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")


def arm_env(family: str, arm: str, model_id: str | None = None) -> dict[str, str]:
    """The environment bo7's `arm` helper exported for this arm (plus the recompile limits every lane sets)."""
    exp, ca, fuse, extra = ARMS[family][arm]
    g, r, e = FUSE[fuse]
    env = {"E4B_SERVE_EXP_INT4": str(exp), "E4B_SERVE_ATTN_INT4_CALIB": str(ca), "E4B_CALIB_SOURCE": "c4",
           "E4B_FUSE_T1_GLUE": str(g), "E4B_FUSE_T1_GLUE_R2": str(r), "E4B_FUSE_ROUTER_EPI": str(e),
           "E4B_MODEL_ID": model_id or MODELS[family][0],
           "E4B_RECOMPILE_LIMIT": "64", "E4B_ACCUM_RECOMPILE_LIMIT": "64"}
    env.update(extra)
    return env


def apply_env(env: dict[str, str]) -> None:
    """Clear every lane key, then set this arm's. The hook and the fusions read os.environ at apply time."""
    for k in LANE_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)


def routed_topk(cfg) -> int:
    """step_decomp._routed_topk, copied: the routed top-k under whatever name this family's config uses
    (multimodal configs keep the MoE fields under text_config)."""
    for c in (cfg, getattr(cfg, "text_config", None)):
        for key in ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "moe_topk", "top_k_experts", "top_k"):
            v = getattr(c, key, None)
            if isinstance(v, int) and v > 0:
                return v
    raise ValueError("cannot find the routed top-k in this config")


def calib_batches(tok, n_seq: int = 32, source: str = "c4", seq_len: int = 512, bsz: int = 4):
    """The lane hook's ``_calib_batches`` (hook v7), copied so the census runs the SAME activations the recipe
    calibrates on: C4 validation shard 00000 (or wikitext-2 train), 6 MB of text, ``n_seq`` windows of ``seq_len``."""
    import torch
    from datasets import load_dataset
    if source == "c4":
        ds = load_dataset("allenai/c4", data_files={"v": "en/c4-validation.00000-of-00008.json.gz"}, split="v")
        text = "\n\n".join(ds["text"][:4000])
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    text = text[:6_000_000]
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (ids.numel() - seq_len) // n_seq)
    rows = [ids[i * step:i * step + seq_len] for i in range(n_seq)]
    return [torch.stack(rows[i:i + bsz]) for i in range(0, n_seq, bsz)]


def build_served_model(model_id: str, arena: str, calib_path: str, *, hot_rows: int = 64, device: str = "cuda"):
    """The served model exactly as `step_decomp.py --placement-override all-vram --amort off --no-fuse-qkv` builds it:
    NF4 load through the arena, the placement solver run then every expert moved into VRAM, the hybrid tier enabled
    (the staged hook then applies the env lanes), amortisation off, and the three env-gated fusions called so a set
    flag either engages or refuses. Returns ``(model, info)``; ``info`` is the proof-of-execution census a row needs
    (how many attention projections are Int4Linear, how many expert layers carry int4 stores and of what kind, how
    many modules each fusion patched)."""
    import torch
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines.glue_fuse import fuse_t1_glue
    from experts4bit_qlora.engines.glue_r2 import fuse_t1_glue_r2
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.engines.hybrid import enable_hybrid_tier
    from experts4bit_qlora.engines.placement import solve_placement
    from experts4bit_qlora.engines.router_epilogue import fuse_router_epilogue

    torch.manual_seed(1689)
    model, _ = load_moe_4bit_streaming(model_id, device, torch.bfloat16, r=8, alpha=16, quant_type="nf4", arena=arena)
    model.eval()
    # the decode-shaped scorer carries HF's KV cache step to step; the paged runner never needed it, so make it explicit
    for cfg_ in (model.config, getattr(model.config, "text_config", None)):
        if cfg_ is not None:
            cfg_.use_cache = True
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    k = routed_topk(model.config)
    idx = json.loads(Path(arena + ".index.json").read_text())
    bpe = 0
    for seg in idx["segments"]:
        n = 1
        for d in seg["shape_per_expert"]:
            n *= d
        bpe += n * (4 if seg["dtype"] == "F32" else 1)
    torch.set_num_threads(8)
    man = solve_placement(n_layers=L, n_experts=E, bytes_per_expert=bpe,
                          vram_budget_bytes=int(1.2 * 2**30), dram_budget_bytes=int(6.0 * 2**30),
                          calibration=json.loads(Path(calib_path).read_text()), profile_path=None,
                          batch=1, top_k=k, cpu_us_fixed=None, cpu_us_per_row=None)
    pairs = sorted(tuple(pp) for t in ("vram", "dram", "nvme") for pp in man["tiers"][t])
    man["tiers"] = {"vram": [list(pp) for pp in pairs], "dram": [], "nvme": []}
    man["masses"] = {"vram_frac": 1.0, "dram_frac": 0.0, "nvme_frac": 0.0}
    n = enable_hybrid_tier(model, arena, man, hot_rows=hot_rows, threads=0, pool=True,
                           dispatch_diet=False, collapse_resident=True)
    if n != L:
        raise RuntimeError(f"enable_hybrid_tier patched {n}/{L} MoE layers")
    for m in mods:
        m._hot_residency.arm_amortization(False)
    info = {"moe_layers": L, "experts": E, "top_k": k,
            "fuse_t1_glue_n": fuse_t1_glue(model), "fuse_t1_glue_r2_n": fuse_t1_glue_r2(model),
            "fuse_router_epilogue_n": fuse_router_epilogue(model)}
    try:
        from experts4bit_qlora.engines.int4_attn import Int4Linear
        info["int4_attn_projections"] = sum(1 for m in model.modules() if isinstance(m, Int4Linear))
    except ImportError:
        info["int4_attn_projections"] = 0
    stores = [getattr(getattr(m, "_hot_residency", None), "_int4_stores", None) for m in mods]
    info["int4_expert_layers"] = sum(1 for s in stores if s)
    info["int4_store_kinds"] = sorted({str(s.get("kind", "int4_b32")) if isinstance(s, dict) else "int4_b32"
                                       for s in stores if s})
    return model, info


def text_layers(model_id: str) -> int:
    """The text tower's decoder-layer count from the pinned config (multimodal configs keep it under text_config)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    for c in (getattr(cfg, "text_config", None), cfg):
        n = getattr(c, "num_hidden_layers", None) if c is not None else None
        if isinstance(n, int) and n > 0:
            return n
    raise ValueError("cannot find num_hidden_layers in this config")


def quantize_layer_set(builder: str, n_layers: int):
    """P47's per-builder ``quantize_layers``: None (all), the empty set (none), the first half, the second half."""
    half = n_layers // 2
    po = parse_only(builder)
    if po:
        i = po[0]
        if i >= n_layers:
            raise ValueError(f"{builder}: layer {i} is outside the {n_layers} text-decoder layers")
        return {i}
    return {"loader": None, "loader_unquant": set(), "loader_lo": set(range(0, half)),
            "loader_hi": set(range(half, n_layers))}[builder]


def build_loader_model(model_id: str, builder: str, *, device: str = "cuda"):
    """P47's loader arms: the model exactly as a TRAINING step would hold it -- `load_moe_4bit_streaming` with no
    arena and no hybrid tier (P43 T2b's fixture), NF4 experts in the layers ``quantize_layer_set`` names and the
    checkpoint's bf16 everywhere else. Returns ``(model, info)``; ``info`` carries the proof of execution --
    ``verify_moe_4bit``'s quantised / unquantised stack counts beside the counts the builder expects -- so a
    builder whose layer set did not apply refuses its row (`kl_serve._builder_check`)."""
    import torch
    from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit
    po = parse_only(builder)
    if builder not in LOADER_BUILDERS and not po:
        raise ValueError(f"unknown loader builder {builder!r}; registered: {LOADER_BUILDERS} or loader_only_<i>[:<qt>][:b<bs>]")
    n = text_layers(model_id)
    ql = quantize_layer_set(builder, n)
    qt, bs = (po[1], po[2]) if po else ("nf4", 64)
    torch.manual_seed(1689)
    model, _ = load_moe_4bit_streaming(model_id, device, torch.bfloat16, r=8, alpha=16, offload=False, pin=True,
                                       prefetch=False, quant_type=qt, quantize_layers=ql, blocksize=bs)
    model.eval()
    for cfg_ in (model.config, getattr(model.config, "text_config", None)):
        if cfg_ is not None:
            cfg_.use_cache = True
    v = verify_moe_4bit(model)
    info = {"builder": builder, "moe_layers": n, "quantize_layers": "all" if ql is None else sorted(ql),
            "quant_type": qt, "blocksize": bs, "quantized_types": sorted({q["quant_type"] for q in v["quantized"]}),
            "quantized_blocksizes": sorted({int(getattr(model.get_submodule(q["module"]), "blocksize", 0)) for q in v["quantized"]}),
            "n_quantized": v["n_quantized"], "n_unquantized": v["n_unquantized"],
            "expected_quantized": n if ql is None else len(ql), "expected_unquantized": 0 if ql is None else n - len(ql),
            "quantized_modules": [q["module"] for q in v["quantized"]][:64],
            "unquantized_modules": [u["module"] for u in v["unquantized"]][:64],
            "int4_expert_layers": 0, "int4_attn_projections": 0,
            "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": 0, "fuse_router_epilogue_n": 0}
    return model, info


def build_arm_model(family: str, arm: str, model_id: str, arena: str, calib_path: str, *, device: str = "cuda"):
    """ONE entry for every KL arm: the served stack (P44-b) or a P47 loader builder, by `builder_for`."""
    b = builder_for(family, arm)
    if b == "served":
        model, info = build_served_model(model_id, arena, calib_path, device=device)
        info["builder"] = "served"
        return model, info
    return build_loader_model(model_id, b, device=device)


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) == 3 and a[0] == "env":
        print(" ".join(f"{k}={v}" for k, v in arm_env(a[1], a[2]).items()))
        return 0
    if len(a) == 2 and a[0] == "model":
        print(" ".join(MODELS[a[1]]))
        return 0
    if len(a) == 2 and a[0] == "arms":
        print(" ".join(ARMS[a[1]]))
        return 0
    if len(a) == 3 and a[0] == "builder":
        print(builder_for(a[1], a[2]))
        return 0
    if len(a) == 2 and a[0] == "needs_arena":
        print("1" if needs_arena(a[1]) else "0")
        return 0
    print("usage: serve_stack.py env <family> <arm> | model <family> | arms <family> | builder <family> <arm> | needs_arena <family>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
