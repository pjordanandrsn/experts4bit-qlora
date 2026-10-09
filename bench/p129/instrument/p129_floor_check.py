"""P129 Amendment 3 harness check (RTX A2000, the two-layer random Qwen3-MoE at Qwen3-30B-A3B's layer dimensions, set up as TC1's e4b arm):
tc1_arm.qkv_floor_rows gives every mode for every row; A0 reproduces the stock rows exactly, with lora_B zero (native init) and non-zero;
D3 shares k's matmul with v on every call; the bases come back afterwards (the stock rows again, bitwise); and the envelope numbers for the
fused path B on the same rows."""
import json, os, sys
import torch
sys.path.insert(0, os.environ["TC1_DIR"])
import tc1_arm as T
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
from experts4bit_qlora.engines.train_qkv_fuse import enable_train_fuse_qkv
D = os.environ.get("P129_MODEL_DIR", "qwen3moe-2l")             # make_model.py writes it
g = torch.Generator().manual_seed(11)
rows = [torch.randint(0, 4096, (n,), generator=g).tolist() for n in (40, 57, 73, 88, 96, 105, 120, 64)]
kw = lambda x: {}
res = {}
for label, b_nonzero in (("lora_B_zero", False), ("lora_B_nonzero", True)):
    torch.manual_seed(0)
    model, cfg = load_moe_4bit_streaming(D, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
    model.to("cuda")
    quantize_attention_projections_4bit(model)
    model.config.use_cache = False
    add_attention_lora(model, 16, 16, torch.bfloat16)
    if b_nonzero:
        torch.manual_seed(5)
        for n_, p in model.named_parameters():
            if "lora_B" in n_:
                p.data.normal_(0, 0.02)
    enable_fast_train(model, verbose=False)
    model.train()
    _, stock = T.eval_loss(model, rows, kw, False)
    fl = T.qkv_floor_rows(model, rows, kw, False)
    _, after = T.eval_loss(model, rows, kw, False)
    n = enable_train_fuse_qkv(model)
    _, fused = T.eval_loss(model, rows, kw, False)
    D1 = fl["rows"]["D1"]
    e = {k: sum(abs(x - d) for x, d in zip(v, D1)) / len(D1) for k, v in {"A": stock, "D2": fl["rows"]["D2"], "D3": fl["rows"]["D3"],
                                                                           "D4": fl["rows"]["D4"], "B": fused}.items()}
    res[label] = {"modes": sorted(fl["rows"]), "rows_ok": all(len(v) == len(rows) for v in fl["rows"].values()),
                  "A0_equals_stock": fl["rows"]["A0"] == stock, "restored_equals_stock": after == stock,
                  "d3": [fl["d3_hits"], fl["d3_expected"]], "n_attention": fl["n_attention"], "fused": n,
                  "e": {k: round(v, 6) for k, v in e.items()}, "envelope": round(max(e[k] for k in ("A", "D2", "D3", "D4")), 6),
                  "stock": stock, "floor": fl["rows"], "B": fused}
    print(label, json.dumps({k: v for k, v in res[label].items() if k not in ("stock", "floor", "B")}), flush=True)
    del model; torch.cuda.empty_cache()
ok = all(r["A0_equals_stock"] and r["restored_equals_stock"] and r["rows_ok"] and r["d3"][0] == r["d3"][1] and r["modes"] == sorted(T.QKV_FLOOR_MODES)
         for r in res.values())
print("P129-FLOOR-TEST", "PASS" if ok else "FAIL")
json.dump(res, open(os.environ.get("P129_OUT", "floor_test.json"), "w"))
