#!/usr/bin/env python3
"""cli_fused_probe.py -- the $0 correctness leg of bench/cli-fused/DESIGN.md (any CUDA card; no timing is read).

    python cli_fused_probe.py --model DIR/p137-tiny --data data.json --out probe.json

The trainer (experts4bit_qlora.train.main) runs in this process three times, its module reloaded with each arm's environment.
Every arm uses OFFLOAD_EXPERTS=1 and STEPS=2 on the tiny Qwen3-MoE fixture (bench/p137/p137_tiny.py), with alpaca rows
from --data.

  L  LAYER_MAJOR=1, FAST_TRAIN unset   the fused layer-major path, which LAYER_MAJOR=1 now means
  A  FAST_TRAIN=1, LAYER_MAJOR=0       the fused path, stock accumulation loop
  R  LAYER_MAJOR=1, FAST_TRAIN=0       the reference loop under layer-major (today's behaviour)

STRICT checks (every one must hold; the probe exits 1 otherwise):
  0. the library self-pair: layer_major_step called twice in one process on the same micro-batches is bit-identical (so a
     nondeterministic kernel fails here as "not reproducible", never as "trainer != library");
  1. the trainer IS the library call. A model built directly (torch.manual_seed as the trainer's SEED,
     load_moe_4bit_streaming with offload, enable_fast_train(dgrad=True), non-reentrant gradient checkpointing),
     with layer_major_step run on the exact micro-batches of L's first step: the adapters are equal byte for byte to
     L's at that step's entry, and the per-micro-batch losses are torch.equal to L's;
  2. the patch count is the number of MoE layers in L and A, and 0 in R; R's step-0 losses differ from L's (nonzero);
  3. the stagings: L and R steady 2n - 1 per step; A steady GRAD_ACCUM x (2n - 1); A's step 0 GRAD_ACCUM x (2n - 1) + 1;
     L's step 0 above 2n (the drift guard's forwards run there);
  4. the refusals: FAST_TRAIN=1 with TRAIN_ARENA, and a FAST_TRAIN=1 run whose enable_fast_train patches nothing, each
     SystemExit before training;
  5. resident (no offload) FAST_TRAIN=1: the patch count is the number of MoE layers, no raise, and the absmax compressed
     as apply_absmax_dq's soft resident default leaves it (E4B_ABSMAX_DQ unset).
REPORTING: the size of R's step-0 difference from L's (the forward fusion's error; no receipt bounds it on the fixture).
"""
import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile

N_LAYERS = 2                     # the fixture's MoE layers


def _adapter_sha(model):
    import torch
    h = hashlib.sha256()
    for p in model.parameters():
        if p.requires_grad:
            h.update(p.detach().to("cpu").contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _patch_dataset(rows):
    """datasets.load_dataset("tatsu-lab/alpaca", split="train[a:b]") -> those rows of --data (no network)."""
    import datasets

    def load(name, split):
        m = re.fullmatch(r"train\[(\d*):(\d*)\]", split)
        assert name == "tatsu-lab/alpaca" and m, (name, split)
        a, b = int(m.group(1) or 0), int(m.group(2) or len(rows))
        return datasets.Dataset.from_list(rows[a:b])
    datasets.load_dataset = load


def _run_trainer(env, model_dir, record):
    import torch

    import experts4bit_qlora.engines.fast as fast
    import experts4bit_qlora.engines.layer_major as lm
    from experts4bit_qlora.engines import offload as off

    base = {"MODEL": model_dir, "OFFLOAD_EXPERTS": "1", "STEPS": "2", "GRAD_ACCUM": "4", "SEQ": "256",
            "TOKEN_BUDGET": "512", "N_TRAIN": "64", "EVAL_EVERY": "50", "DO_GEN": "0", "TRAIN_ATTENTION": "0",
            "R": "8", "ALPHA": "16", "SEED": "0", "E4B_OFFLOAD_STATS": "1", "OUT": tempfile.mkdtemp()}
    for k in ("FAST_TRAIN", "LAYER_MAJOR", "TRAIN_ARENA"):
        os.environ.pop(k, None)
    os.environ.update({**base, **env})
    if env.get("OFFLOAD_EXPERTS") == "0":
        os.environ.pop("E4B_OFFLOAD_STATS", None)
    import experts4bit_qlora.train as train
    train = importlib.reload(train)
    fast.FAST_TRAIN_STATS["patched"] = 0
    real_lm = lm.layer_major_step
    steps = []

    def lm_hook(model, batches, weights, handles=None):
        if "lm_first" not in record:
            record["lm_first"] = {"adapter_sha": _adapter_sha(model),
                                  "batches": [{k: v.detach().cpu() for k, v in b.items()} for b in batches],
                                  "weights": list(weights)}
        out = real_lm(model, batches, weights, handles=handles)
        if "lm_first_losses" not in record:
            record["lm_first_losses"] = [float(x).hex() for x in out]
        return out
    lm.layer_major_step = lm_hook
    real_step = torch.optim.AdamW.step

    def step_hook(self, *a, **k):
        steps.append(len(off._stats().copies))
        off.reset_offload_stats()
        return real_step(self, *a, **k)
    torch.optim.AdamW.step = step_hook
    sys.argv = ["train"]
    try:
        train.main()
    finally:
        lm.layer_major_step = real_lm
        torch.optim.AdamW.step = real_step
    record["patched"] = fast.FAST_TRAIN_STATS["patched"]
    record["stagings"] = steps
    record["absmax_compressed"] = _absmax_compressed()
    return record


def _absmax_compressed():
    """Whether any live module carries a compressed expert absmax. The mark sits on the expert stack, not the ExpertsLoRA
    wrapper; the offload arms never compress, so after the resident arm this reads that arm's model."""
    import gc

    import torch

    from experts4bit_qlora.absmax_dq import is_absmax_compressed
    return any(is_absmax_compressed(m) for m in gc.get_objects() if isinstance(m, torch.nn.Module))


def _library_losses(model_dir, first):
    """The direct library call on L's first micro-batches."""
    import torch

    from experts4bit_qlora.engines.fast import enable_fast_train
    from experts4bit_qlora.engines.layer_major import layer_major_step
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import trainable_lora_param_ids

    torch.manual_seed(0)
    model, _ = load_moe_4bit_streaming(model_dir, "cuda", torch.bfloat16, 8, 16, offload=True, pin=True, quant_type="nf4")
    expert_ids, _attn = trainable_lora_param_ids(model)
    for p in model.parameters():
        p.requires_grad_(id(p) in expert_ids)
    n = enable_fast_train(model, dgrad=True, absmax_dq=False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    sha = _adapter_sha(model)
    batches = [{k: v.to("cuda") for k, v in b.items()} for b in first["batches"]]
    losses = layer_major_step(model, batches, first["weights"])
    again = layer_major_step(model, batches, first["weights"])     # the self-pair: no optimizer step between, same adapters
    return {"patched": n, "adapter_sha": sha, "losses": [float(x).hex() for x in losses],
            "self_pair": [float(x).hex() for x in again]}


def _refuses(env, model_dir, zero_patch=False):
    import experts4bit_qlora.engines.fast as fast
    real = fast.enable_fast_train
    if zero_patch:
        fast.enable_fast_train = lambda *a, **k: 0
    try:
        _run_trainer(env, model_dir, {})
    except SystemExit as e:
        return str(e)[:200]
    finally:
        fast.enable_fast_train = real
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _patch_dataset(json.load(open(a.data)))
    n = N_LAYERS
    L = _run_trainer({"LAYER_MAJOR": "1"}, a.model, {})
    A = _run_trainer({"FAST_TRAIN": "1", "LAYER_MAJOR": "0"}, a.model, {})
    R = _run_trainer({"LAYER_MAJOR": "1", "FAST_TRAIN": "0"}, a.model, {})
    lib = _library_losses(a.model, L["lm_first"])
    accum = 4
    checks = {
        "0_library_self_pair_bitwise": lib["losses"] == lib["self_pair"],
        "1_adapters_equal_library": L["lm_first"]["adapter_sha"] == lib["adapter_sha"],
        "1_step0_losses_bitwise_library": L["lm_first_losses"] == lib["losses"],
        "2_patch_count": L["patched"] == n and A["patched"] == n and R["patched"] == 0 and lib["patched"] == n,
        "2_fused_differs_from_reference": R["lm_first_losses"] != L["lm_first_losses"],
        "3_stagings_L": len(L["stagings"]) == 2 and L["stagings"][0] > 2 * n and L["stagings"][1] == 2 * n - 1,
        "3_stagings_R": len(R["stagings"]) == 2 and R["stagings"][1] == 2 * n - 1,
        "3_stagings_A": A["stagings"] == [accum * (2 * n - 1) + 1, accum * (2 * n - 1)],
    }
    arena = _refuses({"FAST_TRAIN": "1", "TRAIN_ARENA": "/nonexistent/arena"}, a.model)
    zero = _refuses({"FAST_TRAIN": "1"}, a.model, zero_patch=True)
    checks["4_refuses_arena"] = bool(arena) and "TRAIN_ARENA" in arena
    checks["4_refuses_zero_patch"] = bool(zero) and "patched no ExpertsLoRA" in zero
    try:
        Res = _run_trainer({"FAST_TRAIN": "1", "LAYER_MAJOR": "0", "OFFLOAD_EXPERTS": "0"}, a.model, {})
        res_err = None
    except Exception as e:                                       # the bug class the review caught: a raise at FAST_TRAIN
        Res, res_err = {"patched": None, "absmax_compressed": None}, f"{type(e).__name__}: {str(e)[:200]}"
    checks["5_resident_no_raise_patched"] = res_err is None and Res["patched"] == n
    checks["5_resident_absmax_as_apply_left_it"] = res_err is None and Res["absmax_compressed"] is True
    rep = {"R_vs_L_step0_max_abs": max(abs(float.fromhex(x) - float.fromhex(y))
                                       for x, y in zip(R["lm_first_losses"], L["lm_first_losses"]))}
    out = {"checks": checks, "pass": all(checks.values()), "reporting": rep,
           "L": {k: L[k] for k in ("patched", "stagings", "lm_first_losses")},
           "A": {k: A[k] for k in ("patched", "stagings")},
           "R": {k: R[k] for k in ("patched", "stagings", "lm_first_losses")},
           "library": {k: lib[k] for k in ("patched", "losses", "self_pair")},
           "resident": {"patched": Res["patched"], "absmax_compressed": Res["absmax_compressed"], "error": res_err},
           "refusals": {"arena": arena, "zero_patch": zero}}
    json.dump(out, open(a.out, "w"), indent=1)
    print("CLI_FUSED_PROBE " + json.dumps({"pass": out["pass"], "checks": checks, "reporting": rep}), flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
