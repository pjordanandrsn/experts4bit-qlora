"""A depth-truncated view of a sharded checkpoint: the first N decoder layers, by a filtered index and links to the shards that hold
them. Nothing is copied; the source cache is read-only here."""
import json, os, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["num_hidden_layers"] = n
if "layer_types" in cfg:
    cfg["layer_types"] = cfg["layer_types"][:n]
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
ix = json.load(open(os.path.join(src, "model.safetensors.index.json")))
keep = {}
for k, f in ix["weight_map"].items():
    if k.startswith("model.layers."):
        if int(k.split(".")[2]) >= n:
            continue
    keep[k] = f
json.dump({"metadata": ix.get("metadata", {}), "weight_map": keep}, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
for f in sorted(set(keep.values())):
    link = os.path.join(dst, f)
    if not os.path.lexists(link):
        os.symlink(os.path.realpath(os.path.join(src, f)), link)
for f in os.listdir(src):
    if f.endswith((".json", ".txt")) and f not in ("config.json", "model.safetensors.index.json") and not os.path.lexists(os.path.join(dst, f)):
        os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(dst, f))
print(json.dumps({"layers": n, "tensors": len(keep), "shards": sorted(set(keep.values()))}))
