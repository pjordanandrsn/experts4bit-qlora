# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P106 rehearsal of ``p106_box.main()`` on the NAS A2000 ($0): the whole box, end to end, on a tiny random Qwen3.5-MoE
hybrid. That covers argument parsing, the engine's environment, the switch, Q (lockstep, null pair), M (the mutant),
T (TTFT through the scheduler) and the record.

Qwen3.6-35B-A3B itself does not fit: the home lab's resident GPU services hold ~8 of the A2000's 12 GB, and the model's
non-expert weights need ~5. Three pieces are replaced, nothing else:
- ``serve_paged.build_engine``: the same construction around the tiny model, from the same ``PagedServeConfig`` (an fp8
  KV pool sized to the attention layers, ``PagedModelRunner``, ``ContinuousScheduler``);
- ``wikitext_prompts``: random token windows of the requested lengths (the tiny vocabulary has no tokenizer);
- ``loaded_commit``: the pinned revision (no hub checkout).

Decode attention is SDPA over the pool's fp8 bytes (``--stand-in-attention``; sm_86 has no fp8 kernel).

    python box_rehearsal.py OUT.json      (with p106_box.py, gdn_toggle.py and p98_box.py importable)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(os.path.dirname(HERE)), "p98")):
    sys.path.insert(0, d)
import p106_box  # noqa: E402

REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
LIN, ATT = "linear_attention", "full_attention"


def tiny_engine(cfg):
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    import experts4bit_qlora.serve_paged as sp
    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler
    mc = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=64, moe_intermediate_size=64,
                              shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
                              linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                              linear_value_head_dim=32, linear_conv_kernel_dim=4, layer_types=[LIN, ATT, LIN, LIN],
                              max_position_embeddings=8192)
    torch.manual_seed(0)
    model = Qwen3_5MoeForCausalLM(mc).to(cfg.device, torch.bfloat16).eval()
    paged_attention.register(model)
    kv = Fp8PagedKV(kv_layers(model, 4), 2, 64, batch=cfg.max_seqs, max_tokens_per_seq=cfg.max_tokens_per_seq,
                    batched_append=True, device=cfg.device)
    runner = PagedModelRunner(model, kv, device=cfg.device)
    sched = ContinuousScheduler(runner=runner, max_seqs=cfg.max_seqs, kv_slots=cfg.max_seqs,
                                chunk_tokens=cfg.chunk_tokens, max_prefill_tokens_per_step=cfg.prefill_budget)
    info = {"moe_layers": 4, "experts": 4, "top_k": 2, "model_type": "qwen3_5_moe_text (tiny, rehearsal)",
            "graph_status": None}
    return sp.EngineParts(scheduler=sched, tokenizer=None, eos_ids=frozenset(), info=info, runner=runner)


def windows(tok, ks, length):
    out = []
    for k in ks:
        g = torch.Generator().manual_seed(100 + k)
        out.append(torch.randint(0, 256, (length,), generator=g).tolist())
    return out


def main():
    import experts4bit_qlora.serve_paged as sp
    sp.build_engine = tiny_engine
    p106_box.wikitext_prompts = windows
    p106_box.loaded_commit = lambda model, revision: revision
    sys.argv = ["p106_box.py", "--model", "tiny-qwen3_5_moe", "--revision", REV, "--arena", "-", "--calib", "-",
                "--out", sys.argv[1], "--stand-in-attention", "--windows", "2", "--null-windows", "1", "--prompt", "512",
                "--cont", "16", "--chunk", "128", "--mutant-cont", "4", "--ttft-lengths", "256,1024", "--ttft-rounds", "2"]
    return p106_box.main()


if __name__ == "__main__":
    sys.exit(main())
