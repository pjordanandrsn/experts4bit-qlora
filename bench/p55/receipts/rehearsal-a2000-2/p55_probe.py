import json, os, sys, traceback
name = sys.argv[1]
out = {"arm": name, "env": {k: os.environ.get(k) for k in
                            ("E4B_LOAD_SYNC_DEBUG", "CUDA_LAUNCH_BLOCKING")}}
try:
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    model, cfg = load_moe_4bit_streaming(
        os.environ["P55_MODEL"], "cuda", __import__("torch").bfloat16,
        r=8, alpha=16, quant_type="nf4", revision=os.environ["P55_REVISION"])
    out["status"] = "OK"
    out["n_layers"] = int(getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", -1))
except BaseException as exc:                      # every outcome is data here
    out["status"] = "FAILED"
    out["exc_type"] = type(exc).__name__
    out["exc_module"] = type(exc).__module__
    out["exc_msg"] = str(exc)
    out["notes"] = list(getattr(exc, "__notes__", []))
    out["tb"] = traceback.format_exc()
json.dump(out, open(f"result_{name}.json", "w"), indent=1)
print(f"P55 {name}: {out['status']}" + ("" if out["status"] == "OK" else f" {out['exc_type']}: {out['exc_msg'][:200]}"))
