"""A real 2-layer checkpoint: only the non-layer tensors and layers 0..n-1, re-sharded, with its own index."""
import json, os, re, sys
from safetensors import safe_open
from safetensors.torch import save_file
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
keep = {}
for k, f in idx["weight_map"].items():
    m = re.match(r"model\.layers\.(\d+)\.", k)
    if m is None or int(m.group(1)) < n:
        keep.setdefault(f, []).append(k)
wm, i = {}, 0
for f, keys in sorted(keep.items()):
    i += 1
    out = f"model-{i:05d}-cut.safetensors"
    with safe_open(os.path.join(src, f), "pt") as fh:
        save_file({k: fh.get_tensor(k) for k in keys}, os.path.join(dst, out), metadata={"format": "pt"})
    wm.update({k: out for k in keys})
json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
for f in os.listdir(src):
    if not f.endswith(".safetensors") and f not in ("model.safetensors.index.json", "config.json") and not os.path.exists(os.path.join(dst, f)):
        os.symlink(os.path.join(src, f), os.path.join(dst, f))
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["num_hidden_layers"] = n
if "max_window_layers" in cfg:
    cfg["max_window_layers"] = min(cfg["max_window_layers"], n)
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
print("cut", n, "layers:", len(wm), "tensors in", i, "shards")
