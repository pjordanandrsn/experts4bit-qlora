"""Amendment 3 prerequisite check: the in-process null and plants (one fresh process, q235_box.py --repeat --plants at c9d28ac9)
against the 2026-10-10 depth test's three separate fresh processes (450b07ff), on Qwen3-30B-A3B at full depth.

    python inproc_check.py INPROC_REPEAT_DIR DEPTH_DIR OUT.json
"""
import json
import sys

inproc, depth, out = sys.argv[1:4]
rep = json.load(open(f"{inproc}/box.json"))
box = json.load(open(f"{depth}/box/box.json"))
rec = {"null_equals_depth_box_bitwise": rep["layer_sha256"] == box["layer_sha256"],
       "null_loss": rep["repeat"]["loss"], "depth_box_loss": box["repeat"]["loss"], "plants": {}}
for kind in ("routing", "epilogue"):
    p = json.load(open(f"{inproc}/plant_{kind}/box.json"))
    d = json.load(open(f"{depth}/plant_{kind}/box.json"))
    rec["plants"][kind] = {"layer": p["plant"]["layer"], "equals_depth_plant_bitwise": p["layer_sha256"] == d["layer_sha256"],
                           "loss": p["repeat"]["loss"], "depth_loss": d["repeat"]["loss"],
                           "pre_plant_equal_null": p["pre_plant_equal_null"], "planted_layer_differs": p["planted_layer_differs"],
                           "forward_s": p["repeat"]["forward_s"]}
rec["null_forward_s"] = rep["repeat"].get("forward_s")
rec["load_seconds"] = rep["load"]["seconds"]
rec["all_bitwise"] = rec["null_equals_depth_box_bitwise"] and all(v["equals_depth_plant_bitwise"] for v in rec["plants"].values())
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(rec))
