# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P108 rehearsal of ``p108_box.measure()`` on the NAS A2000 ($0), on the GPU: a tiny random Gemma-4 MoE (five sliding
layers at window 16 and one full layer, so the window binds on 40-token prompts), 8 windows in groups of 4, with the
decode attention stood in by SDPA over the pool's own fp8 bytes honouring the window (sm_86 has no fp8 kernel).
Gemma-4-26B-A4B itself does not fit the 12 GB card. The record is reduced with the lane's reducer, the loaded commit
set to the pinned revision; it VOIDs on its window count (8 of 32) by design.

    python box_rehearsal.py OUT_DIR
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import p108_box  # noqa: E402
import p108_reduce  # noqa: E402


def tiny(seed=0):
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    cfg = Gemma4TextConfig(vocab_size=256, hidden_size=128, intermediate_size=128, num_hidden_layers=6,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32, global_head_dim=64,
                           num_global_key_value_heads=1, layer_types=["sliding_attention"] * 5 + ["full_attention"],
                           sliding_window=16, enable_moe_block=True, num_experts=4, top_k_experts=2,
                           moe_intermediate_size=64, attention_k_eq_v=True, hidden_size_per_layer_input=0,
                           vocab_size_per_layer_input=256, max_position_embeddings=512)
    torch.manual_seed(seed)
    return Gemma4ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def main():
    out = sys.argv[1]
    g = torch.Generator().manual_seed(1)
    windows = [torch.randint(0, 256, (40 + 24,), generator=g).tolist() for _ in range(8)]
    rec = p108_box.measure(tiny(), windows, prompt=40, cont=24, chunk=16, group=4, device="cuda", stand_in=True)
    rec = dict(rec, loaded_commit=p108_reduce.REV, gpu=torch.cuda.get_device_name(0))
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "box.json"), "w").write(json.dumps(rec, indent=1))
    pw = rec["per_window"]
    for arm in pw:
        if arm == "R":
            continue
        d = [x["nll"] - r["nll"] for x, r in zip(pw[arm], pw["R"])]
        kl = sum(x.get("kl", 0.0) for x in pw[arm]) / len(pw[arm])
        print(f"P108_REHEARSAL {arm}: n {len(d)} bias {sum(d) / len(d):+.4f} spread {sum(map(abs, d)) / len(d):.4f} kl {kl:.2e}")
    e = rec["engagement"]
    print(f"P108_REHEARSAL engagement: calls {e['calls']}/{e['expected_calls']} by_window {e['by_window']} rep_identical "
          f"{e['rep_identical']} timing {rec['timing_s']}")


if __name__ == "__main__":
    main()
