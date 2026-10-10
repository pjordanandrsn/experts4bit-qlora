"""Guard (a) for the conversion-mapping layer_state (branch fam235/q235-reference-generic): real Qwen3-30B-A3B weights, on CPU.

1. Every layer's state dict from the new layer_state against the original Qwen3-only assembly, key for key and bit for bit.
2. The new reference replays the calibration's CPU command (6 layers, resident, 6 threads): its layer_sha256 and loss must equal
   ref6_resident_cpu/reference.json. The streamed form must equal it too.
"""
import hashlib, importlib.util, json, os, sys, time
import torch

E4B, M, CAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
torch.set_num_threads(6)
spec = importlib.util.spec_from_file_location("lsr", os.path.join(E4B, "bench/q235/instrument/layer_stream_ref.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


def legacy(shards, i, num_experts, device, dtype):            # verbatim from tests/test_q235_layer_stream_ref.py
    pre = f"model.layers.{i}."
    sd = {}
    for name in [k for k in shards.where if k.startswith(pre) and ".mlp.experts." not in k]:
        sd[name[len(pre):]] = shards.get(name, device).to(dtype)
    if shards.has(pre + "mlp.experts.gate_up_proj"):
        sd["mlp.experts.gate_up_proj"] = shards.get(pre + "mlp.experts.gate_up_proj", device).to(dtype)
        sd["mlp.experts.down_proj"] = shards.get(pre + "mlp.experts.down_proj", device).to(dtype)
    elif shards.has(pre + "mlp.experts.0.gate_proj.weight"):
        g = torch.stack([shards.get(f"{pre}mlp.experts.{e}.gate_proj.weight", device) for e in range(num_experts)])
        u = torch.stack([shards.get(f"{pre}mlp.experts.{e}.up_proj.weight", device) for e in range(num_experts)])
        sd["mlp.experts.gate_up_proj"] = torch.cat([g, u], dim=1).to(dtype)
        del g, u
        sd["mlp.experts.down_proj"] = torch.stack(
            [shards.get(f"{pre}mlp.experts.{e}.down_proj.weight", device) for e in range(num_experts)]).to(dtype)
    return sd


def sha(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


from transformers import AutoConfig, AutoModelForCausalLM
import transformers
rec = {"e4b_tree": E4B, "checkpoint": M, "torch": torch.__version__, "transformers": transformers.__version__, "threads": 6,
       "layer_stream_ref_sha256": hashlib.sha256(open(os.path.join(E4B, "bench/q235/instrument/layer_stream_ref.py"), "rb").read()).hexdigest()}
cfg = AutoConfig.from_pretrained(M)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
shards, conv = t.Shards(M), t.Conversions(model)
t0 = time.time(); per_layer = []
for i in range(cfg.num_hidden_layers):
    new = t.layer_state(shards, i, conv, "cpu", torch.bfloat16)
    old = legacy(shards, i, cfg.num_experts, "cpu", torch.bfloat16)
    same = new.keys() == old.keys() and all(new[k].dtype == old[k].dtype and new[k].shape == old[k].shape
                                            and torch.equal(new[k].view(torch.uint8), old[k].view(torch.uint8)) for k in old)
    per_layer.append({"layer": i, "identical": bool(same), "keys": len(new),
                      "gate_up_sha256": sha(new["mlp.experts.gate_up_proj"]), "down_sha256": sha(new["mlp.experts.down_proj"])})
    del new, old
    print(f"layer {i}: identical={same}", flush=True)
rec["layer_state_old_vs_new"] = {"layers": len(per_layer), "all_identical": all(p["identical"] for p in per_layer),
                                 "per_layer": per_layer, "seconds": round(time.time() - t0, 1)}
tok = torch.load(os.path.join(CAL, "tokens.pt"), weights_only=True)
rec["tokens_sha256"] = hashlib.sha256(open(os.path.join(CAL, "tokens.pt"), "rb").read()).hexdigest()
want = json.load(open(os.path.join(CAL, "ref6_resident_cpu", "reference.json")))
for mode in ("resident", "streamed"):
    r = t.run(M, tok["input_ids"], tok["attention_mask"], tok["labels"], "cpu", layers=6, resident=(mode == "resident"))
    rec[f"ref6_{mode}_cpu"] = {"layer_sha256": r["layer_sha256"], "loss": r["loss"], "seconds": r["seconds"],
                               "layer_sha256_match": r["layer_sha256"] == want["layer_sha256"], "loss_match": r["loss"] == want["loss"]}
    print(mode, rec[f"ref6_{mode}_cpu"]["layer_sha256_match"], rec[f"ref6_{mode}_cpu"]["loss_match"], flush=True)
rec["target"] = {"file": "ref6_resident_cpu/reference.json", "loss": want["loss"], "layer_sha256": want["layer_sha256"],
                 "torch": want.get("torch")}
rec["verdict"] = ("PASS" if rec["layer_state_old_vs_new"]["all_identical"]
                  and all(rec[f"ref6_{m}_cpu"]["layer_sha256_match"] and rec[f"ref6_{m}_cpu"]["loss_match"] for m in ("resident", "streamed"))
                  else "FAIL")
json.dump(rec, open(OUT, "w"), indent=1)
print("VERDICT", rec["verdict"], flush=True)
