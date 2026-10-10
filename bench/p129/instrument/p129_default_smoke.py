"""P129 default-flip smoke (A2000, correctness only; no timings): the 2-layer Qwen3-MoE set up as TC1's e4b arm, fp32 and bf16
adapters.

- default: ``E4B_TRAIN_FUSE_QKV`` unset, ``enable_fast_train`` fuses every layer, refuses none; three AdamW steps, losses finite.
- off: ``E4B_TRAIN_FUSE_QKV=0``, nothing fused, every q/k/v base kept; the same three steps. Each step's loss within 2**-6 of the
  off side's (a semantic error moves it by far more; rounding by far less).
- round trip, the ``attn_only`` arm's shape: default, ``enable_fast_train`` then ``disable_fast_train``, one step. Bit for bit the
  off side's ``enable`` then ``disable`` step: the loss and every adapter gradient.
- re-enable: the round-tripped model enabled again fuses every layer, and its first step's loss is bit for bit the default's.
"""
import json
import os

import torch

from experts4bit_qlora import disable_fast_train, enable_fast_train, load_moe_4bit_streaming
from experts4bit_qlora.engines import train_qkv_fuse as tq
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit

D = os.environ.get("P129_MODEL_DIR", "qwen3moe-2l")
QKV = ("q_proj", "k_proj", "v_proj")
torch.autograd.set_multithreading_enabled(False)
g = torch.Generator().manual_seed(2)
IDS = torch.randint(0, 4096, (2, 270), generator=g).cuda()


def knob(v):
    if v is None:
        os.environ.pop("E4B_TRAIN_FUSE_QKV", None)
    else:
        os.environ["E4B_TRAIN_FUSE_QKV"] = v


def build(adapter_dtype):
    torch.manual_seed(0)
    model, _ = load_moe_4bit_streaming(D, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
    model.to("cuda")
    quantize_attention_projections_4bit(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    add_attention_lora(model, 16, 16, adapter_dtype)
    torch.manual_seed(1)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02)
    model.train()
    return model


def step(m, opt=None):
    m.zero_grad(set_to_none=True)
    out = m(input_ids=IDS, labels=IDS)
    out.loss.backward()
    if opt is not None:
        opt.step()
    torch.cuda.synchronize()
    return out.loss.detach().float().item()


def grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if "lora" in n and p.grad is not None}


def bases_kept(m):
    return all(getattr(L.self_attn, n).base is not None for L in m.model.layers for n in QKV)


def train3(m):
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if "lora" in n], lr=1e-4)
    return [step(m, opt) for _ in range(3)]


res = {"model_dir": D}
for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    r = res[dt_name] = {}
    knob(None)
    m = build(dt)
    nl = len(m.model.layers)
    enable_fast_train(m)
    r["default_fused"], r["default_refused"] = tq.TRAIN_QKV_STATS["fused"], dict(tq.TRAIN_QKV_STATS["refused"])
    r["default_losses"] = train3(m)
    del m
    torch.cuda.empty_cache()
    knob("0")
    m = build(dt)
    enable_fast_train(m)
    r["off_fused"], r["off_bases_kept"] = sum(hasattr(L.self_attn, "qkv_proj") for L in m.model.layers), bases_kept(m)
    r["off_losses"] = train3(m)
    del m
    torch.cuda.empty_cache()
    m = build(dt)                                                   # off side, enable then disable: today's attn_only arm
    enable_fast_train(m)
    disable_fast_train(m)
    off_rt_loss, off_rt_g = step(m), grads(m)
    del m
    torch.cuda.empty_cache()
    knob(None)
    m = build(dt)                                                   # default, enable then disable
    enable_fast_train(m)
    r["rt_fused_before_disable"] = tq.TRAIN_QKV_STATS["fused"]
    disable_fast_train(m)
    r["rt_after_disable"] = {"fused_stat": tq.TRAIN_QKV_STATS["fused"], "bases_kept": bases_kept(m),
                             "qkv_proj_left": sum(hasattr(L.self_attn, "qkv_proj") for L in m.model.layers)}
    rt_loss, rt_g = step(m), grads(m)
    r["rt_loss"], r["rt_loss_off"] = rt_loss, off_rt_loss
    r["rt_loss_bitwise"] = rt_loss == off_rt_loss
    r["rt_grads_bitwise"] = rt_g.keys() == off_rt_g.keys() and all(torch.equal(rt_g[k], off_rt_g[k]) for k in rt_g)
    r["rt_n_grads"] = len(rt_g)
    m.zero_grad(set_to_none=True)
    enable_fast_train(m)                                            # re-enable: fuses again
    r["reenable_fused"] = tq.TRAIN_QKV_STATS["fused"]
    r["reenable_loss"] = step(m)
    r["reenable_loss_bitwise_default_step0"] = r["reenable_loss"] == r["default_losses"][0]
    del m
    torch.cuda.empty_cache()
    rel = [abs(a - b) / abs(b) for a, b in zip(r["default_losses"], r["off_losses"])]
    r["loss_rel_diff"] = rel
    r["checks"] = {
        "default fuses every layer, refuses none": r["default_fused"] == nl and r["default_refused"] == {},
        "off fuses nothing and keeps every base": r["off_fused"] == 0 and r["off_bases_kept"],
        "losses finite": all(map(lambda v: v == v and abs(v) != float("inf"), r["default_losses"] + r["off_losses"])),
        "each step's loss within 2**-6 of off": all(x <= 2.0 ** -6 for x in rel),
        "disable restores every layer": r["rt_fused_before_disable"] == nl and r["rt_after_disable"] == {
            "fused_stat": 0, "bases_kept": True, "qkv_proj_left": 0},
        "round trip bit for bit (loss, every adapter grad)": r["rt_loss_bitwise"] and r["rt_grads_bitwise"] and r["rt_n_grads"] > 0,
        "re-enable fuses every layer, step 0 bit for bit": r["reenable_fused"] == nl and r["reenable_loss_bitwise_default_step0"],
    }
res["PASS"] = all(all(res[k]["checks"].values()) for k in ("fp32", "bf16"))
out = os.environ.get("P129_OUT", "p129_default_smoke.json")
with open(out, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: (v["checks"] if isinstance(v, dict) and "checks" in v else v) for k, v in res.items()}, indent=1))
