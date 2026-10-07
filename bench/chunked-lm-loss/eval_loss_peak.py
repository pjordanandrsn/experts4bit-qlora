"""Diagnostic: the held-out loss's own CUDA peak at Qwen3's vocabulary, stock (transformers' ForCausalLMLoss under no_grad) vs
chunked_lm_loss_from_logits -- allocator bytes above the logits, and the two losses. Not a timing."""
import importlib.util, json, sys
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
spec = importlib.util.spec_from_file_location("clm", sys.argv[1]); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
V = 151936
res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "vocab": V, "rows": []}
for T in (2048, 4096):
    g = torch.Generator(device="cuda").manual_seed(T)
    logits = (torch.randn(1, T, V, device="cuda", generator=g, dtype=torch.float32) * 2.0).to(torch.bfloat16)
    labels = torch.randint(0, V, (1, T), device="cuda", generator=g)
    labels[0, :37] = -100
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    out = {"tokens": T, "logits_gib": logits.numel() * 2 / 2**30}
    for name, fn in (("stock", lambda: ForCausalLMLoss(logits, labels, V)),
                     ("chunked512", lambda: C.chunked_lm_loss_from_logits(logits, labels, chunk=512)),
                     ("chunked1024", lambda: C.chunked_lm_loss_from_logits(logits, labels, chunk=1024))):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            loss = fn()
        torch.cuda.synchronize()
        out[name] = {"loss": float(loss), "peak_above_logits_gib": round((torch.cuda.max_memory_allocated() - base) / 2**30, 3)}
        del loss
    out["chunked512_minus_stock"] = out["chunked512"]["loss"] - out["stock"]["loss"]
    out["ulps_fp32"] = abs(out["chunked512_minus_stock"]) / (torch.finfo(torch.float32).eps * abs(out["stock"]["loss"]))
    res["rows"].append(out)
    print(json.dumps(out), flush=True)
    del logits, labels
json.dump(res, open(sys.argv[2], "w"), indent=1)
