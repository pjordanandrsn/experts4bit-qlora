"""Build today's default server on the cut model, as the P158 box builds it, and read the two build counts."""
import json, os, sys, time
lut = sys.argv[1]
os.environ["E4B_K25_LUT"] = lut
import torch, nf4_grouped
from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
hr.DEVICE_GROUPING[0] = True; hr.FORCE_SINGLETON_GROUPS[0] = False   # the graph server sets these; sm_86 cannot capture
cfg = PagedServeConfig.from_env()
c0, s0 = dict(nf4_grouped.dispatch_counts()), dict(hr.K25_LUT_SEEN)
t0 = time.time()
parts = build_engine(cfg)
c1 = dict(nf4_grouped.dispatch_counts())
info = parts.info
L = len(hr.target_modules(parts.runner.model))
rec = {"lut_env": lut, "layers": L, "k25_lut": info.get("k25_lut"), "k25_seen_before": s0, "k25_seen_build": dict(hr.K25_LUT_SEEN),
       "dispatch_build": {k: c1.get(k, 0) - c0.get(k, 0) for k in c1 if c1.get(k, 0) - c0.get(k, 0)},
       "graph_status": {str(k): v for k, v in (info.get("graph_status") or {}).items()},
       "fusions": {k: info.get(k) for k in ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")},
       "int8_head_bw": {k: v for k, v in (info.get("int8_head_bw") or {}).items() if k != "engagement"},
       "buckets": list(cfg.buckets), "max_seqs": cfg.max_seqs, "build_s": round(time.time() - t0, 1),
       "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
seen = rec["k25_seen_build"].get(f"{lut}|le256", 0)
rec["per_layer"] = {"k25": seen / L, "bw_prmt32": rec["dispatch_build"].get("bw_prmt32", 0) / L}
rec["matches_closed_forms"] = (rec["k25_seen_build"] == {f"{lut}|le256": 20 * L}
                               and rec["dispatch_build"].get("bw_prmt32") == 10 * L)
print("LAYERCUT " + json.dumps(rec, default=str), flush=True)
json.dump(rec, open(f"layercut_{lut}.json", "w"), indent=1, default=str)
