# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P106's toggle check (bench/p106/PREREG-p106.md): is ``GdnToggle("torch")`` in a process WITH the kernels the same
arithmetic as a process WITHOUT them, bit for bit, on this card?

--save PATH     run in a kernel-free process: the logits of tiny random hybrids (dense Qwen3.5 [LIN, ATT, LIN, LIN] and
                Qwen3.5-MoE [LIN, ATT, LIN, LIN], 4 seeds each), through transformers' DynamicCache with 32-token chunked
                prefill and 6 cached decode steps, and through e4b's PagedModelRunner (SDPA standing in for the fp8 kernel)
--compare PATH  run in a process with fla (+ causal-conv1d): the same, once under GdnToggle("torch") (must equal the saved
                logits bit for bit), once under GdnToggle("resolved") (must differ), then both again (each must repeat);
                ``--json-out`` writes the verdict and the per-step counts for the reducer

The helpers are test_linear_state_dense_parity_gpu.py's, staged beside this file (and gdn_toggle.py with them).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_linear_state_dense_parity_gpu import _model as dense_model, _paged, _prompts, _reference  # noqa: E402

LIN, ATT = "linear_attention", "full_attention"
SEEDS = range(4)


def moe_model(layer_types, seed):
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=len(layer_types), num_attention_heads=4,
                               num_key_value_heads=2, head_dim=64, moe_intermediate_size=64,
                               shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                               linear_value_head_dim=32, linear_conv_kernel_dim=4, layer_types=list(layer_types),
                               max_position_embeddings=512)
    torch.manual_seed(seed)
    return Qwen3_5MoeForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def run_all():
    out = {}
    for kind, build in (("dense", dense_model), ("moe", moe_model)):
        for seed in SEEDS:
            prompts = _prompts(seed)
            model = build([LIN, ATT, LIN, LIN], seed)
            ref, tok = _reference(model, prompts)
            _, got = _paged(model, prompts, tok, compact=True, stand_in=True)
            for s in prompts:
                out[f"{kind}/{seed}/hf/{s}"] = torch.stack(ref[s]).cpu()
                out[f"{kind}/{seed}/paged/{s}"] = torch.stack(got[s]).cpu()
            del model
    return out


def diff(a, b):
    n_diff, worst = 0, 0.0
    for k in a:
        d = (a[k] - b[k]).abs()
        n_diff += int((d > 0).sum())
        worst = max(worst, float(d.max() / b[k].abs().max()))
    return n_diff, worst


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save")
    g.add_argument("--compare")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    import transformers.models.qwen3_5.modeling_qwen3_5 as m_dense
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m_moe
    from gdn_toggle import GdnToggle
    if a.save:
        tg = GdnToggle([m_dense, m_moe])
        print("TOGGLE_RESOLVED", json.dumps(tg.resolved_record()))
        torch.save(run_all(), a.save)
        print("SAVED", a.save)
        return 0
    ref = torch.load(a.compare)
    tg = GdnToggle([m_dense, m_moe])
    print("TOGGLE_RESOLVED", json.dumps(tg.resolved_record()))
    ok, steps = True, []
    for step, which in enumerate(("torch", "resolved", "torch", "resolved")):
        tg.use(which)
        rec = tg.record()
        got = run_all()
        n_diff, worst = diff(got, ref)
        hf = {k: v for k, v in got.items() if "/hf/" in k}
        pg = {k: v for k, v in got.items() if "/paged/" in k}
        n_hf, w_hf = diff(hf, {k: ref[k] for k in hf})
        n_pg, w_pg = diff(pg, {k: ref[k] for k in pg})
        mods = sorted(set(rec.values()))
        print(f"STEP {step} {which}: impl modules {mods} | differing logits vs kernel-free: hf {n_hf} (worst rel {w_hf:.3e}) "
              f"paged {n_pg} (worst rel {w_pg:.3e})")
        if which == "torch":
            good = n_diff == 0 and all(v.startswith("transformers.") for v in rec.values())
        else:
            good = n_diff > 0 and not all(v.startswith("transformers.") for v in rec.values())
        ok &= good
        steps.append({"step": step, "path": which, "impl_modules": mods, "differing_hf": n_hf, "worst_rel_hf": w_hf,
                      "differing_paged": n_pg, "worst_rel_paged": w_pg, "ok": good})
    print("TOGGLE_PROBE", "OK" if ok else "FAIL")
    if a.json_out:
        json.dump({"ok": bool(ok), "steps": steps, "resolved": tg.resolved_record(), "n_logits": sum(v.numel() for v in ref.values())},
                  open(a.json_out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
