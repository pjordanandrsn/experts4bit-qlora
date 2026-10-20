"""Build today's default server on the cut model, as the P161 box builds it, and read the build's K25 witnesses: the
decode record, the plan record, and the box's own shape spy (p161_box.install_shape_spy, staged beside this file)."""
import json, os, sys, time
plan = sys.argv[1]
os.environ["E4B_K25_PLAN"] = plan
os.environ.pop("E4B_K25_LUT", None)                                   # qwen3_moe's default decode (prmt), as the box
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p161_box                                                      # the box's spy, at the bytes the box stages
p161_box.install_shape_spy()                                         # before the build, as the box installs it
import torch, nf4_grouped
from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
hr.DEVICE_GROUPING[0] = True; hr.FORCE_SINGLETON_GROUPS[0] = False   # the graph server sets these; sm_86 cannot capture
cfg = PagedServeConfig.from_env()
c0, s0, p0, sh0 = dict(nf4_grouped.dispatch_counts()), dict(hr.K25_LUT_SEEN), dict(hr.K25_PLAN_SEEN), p161_box._shapes()
t0 = time.time()
parts = build_engine(cfg)
c1 = dict(nf4_grouped.dispatch_counts())
info = parts.info
L = len(hr.target_modules(parts.runner.model))
rec = {"plan_env": plan, "layers": L, "k25_lut": info.get("k25_lut"), "k25_plan": info.get("k25_plan"),
       "k25_seen_before": s0, "k25_seen_build": dict(hr.K25_LUT_SEEN),
       "k25_plan_seen_before": p0, "k25_plan_seen_build": dict(hr.K25_PLAN_SEEN),
       "k25_shapes_before": sh0, "k25_shapes_build": p161_box._shapes(),
       "dispatch_build": {k: c1.get(k, 0) - c0.get(k, 0) for k in c1 if c1.get(k, 0) - c0.get(k, 0)},
       "graph_status": {str(k): v for k, v in (info.get("graph_status") or {}).items()},
       "buckets": list(cfg.buckets), "max_seqs": cfg.max_seqs, "build_s": round(time.time() - t0, 1),
       "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
k25 = rec["k25_plan_seen_build"].get(f"{plan}|le256", 0)
rec["per_layer"] = {"k25": k25 / L, "kernel_calls": sum(rec["k25_shapes_build"].values()) / L}
print("LAYERCUT " + json.dumps(rec, default=str), flush=True)
json.dump(rec, open(f"layercut_{plan}.json", "w"), indent=1, default=str)
