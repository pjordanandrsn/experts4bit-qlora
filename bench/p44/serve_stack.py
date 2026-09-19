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
    print("usage: serve_stack.py env <family> <arm> | model <family> | arms <family>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
