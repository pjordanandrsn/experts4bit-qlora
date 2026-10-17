"""DeepSeek-V4.1-Flash-1017, gate B: memory, residency and transfer bounds. Analytic: no weights, no GPU.

Bytes come from the checkpoint's shard headers (gate A's ds41_metadata.py), rates from the records named in README.md.
Every tok/s figure is a TRANSFER BOUND (bytes over the slowest link it must cross), never a measurement.

    python ds41_gate_b.py [--headers DIR] [--context 131072]
"""
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GB = 1e9

# Read from the reference code (gate A, revision 1bae0e10): the bf16 caches the model allocates.
KV_SOURCES = {2: 2, 8: 2, 14: 2, 20: 1}         # kv_source layer -> its compress ratio (config compress_ratios)
LATENT, INDEX_DIM, WINDOW, LAYERS = 512, 128, 128, 40
ENGRAM_ROWS_PER_TOKEN = 2 * (4 - 1) * 8           # 2 tables x (max_ngram 4 - 1) x 8 heads (model.py:344)
ENGRAM_ROW_BYTES = 256 + 8                        # one FP8 row + its 8 E8M0 scales, two separate tensors
TOPK, N_EXPERTS = 6, 384

# Measured rates (README.md names each record). GB/s.
HOSTS = {
    "A2000 host (gen3 x8)": {"h2d": 5.8, "disk": 1.63, "ram_free": 51.5},
    "rented, Q235's 239 GB floor (box 1 rates)": {"h2d": 21.76, "disk": 6.31, "ram_free": 239.0},
    "rented, large RAM (box 1: 1,081 GB)": {"h2d": 21.76, "disk": 6.31, "ram_free": 1081.0},
}
GEN5 = (40.0, 53.0)                               # unmeasured: the gen-5 host's listed range (assumption)
GPUS = (12, 24, 48, 96)
RESERVE_GB = 1.0                                  # CUDA context, activations, workspace at batch 1: ASSUMPTION
OS_GB = 8.0                                       # host RAM kept for the OS and buffers: ASSUMPTION


def component_bytes(headers):
    spec = importlib.util.spec_from_file_location("ds41_metadata", os.path.join(HERE, "..", "ds41-gate-a", "ds41_metadata.py"))
    md = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md)
    base = f"https://huggingface.co/{md.REPO}/resolve/{md.REVISION}/"
    index = json.loads(md.fetch(base + "model.safetensors.index.json"))
    out = {}
    for shard in sorted(set(index["weight_map"].values())):
        h = md.header(base, shard, headers)
        h.pop("__metadata__", None)
        for n, v in h.items():
            c = "head" if n == "head.weight" else "embedding" if n == "embed.weight" else md.component(n)
            out[c] = out.get(c, 0) + v["data_offsets"][1] - v["data_offsets"][0]
    return out


def kv_bytes(context):
    compressed = sum(context // r * (LATENT + INDEX_DIM) * 2 for r in KV_SOURCES.values())
    return compressed + LAYERS * WINDOW * LATENT * 2


def decode_bound(host, gpu_gb, b, context, h2d=None):
    """Routed experts per token, served from GPU cache, then host RAM, then disk; Engram rows from RAM when they fit."""
    h2d = h2d or host["h2d"]
    experts = b["routed experts (40 layers)"]
    per_expert = experts / (LAYERS * N_EXPERTS)
    token = TOPK * LAYERS * per_expert
    resident = b["gpu_resident"] + kv_bytes(context) / GB + 2 * TOPK * per_expert / GB + RESERVE_GB
    if resident > gpu_gb:
        return {"feasible": False, "binding": f"VRAM: needs {resident:.1f} GB before any expert cache"}
    gpu_hit = min(1.0, (gpu_gb - resident) * GB / experts)          # uniform routing: ASSUMPTION
    ram = host["ram_free"] - OS_GB - b["embedding"] / GB
    engram_in_ram = ram >= b["Engram tables"] / GB
    if engram_in_ram:
        ram -= b["Engram tables"] / GB
    ram_hit = min(1.0 - gpu_hit, max(0.0, ram) * GB / experts)
    disk_bytes = token * (1 - gpu_hit - ram_hit)
    link_bytes = token * (1 - gpu_hit)
    t = max(link_bytes / (h2d * GB), disk_bytes / (host["disk"] * GB))
    binding = "disk" if disk_bytes / host["disk"] >= link_bytes / h2d else "H2D"
    return {"feasible": True, "binding": binding, "tok_s_bound": 1 / t, "gpu_hit": gpu_hit, "ram_hit": ram_hit,
            "engram": "host RAM" if engram_in_ram else f"disk: {2 * ENGRAM_ROWS_PER_TOKEN} random reads per token (IOPS UNKNOWN)",
            "gb_per_token_disk": disk_bytes / GB, "gb_per_token_link": link_bytes / GB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers", default=None)
    ap.add_argument("--context", type=int, default=131072)
    a = ap.parse_args()
    b = component_bytes(a.headers)
    b["gpu_resident"] = sum(b[k] for k in ("attention (+ compressor, indexer)", "shared experts + router", "head",
                                         "Engram projections", "hyper-connections", "norms")) / GB
    per_expert = b["routed experts (40 layers)"] / (LAYERS * N_EXPERTS)
    print(json.dumps({
        "gpu_resident_weights_gb (text decode, as stored)": round(b["gpu_resident"], 2),
        "routed_experts_gb": round(b["routed experts (40 layers)"] / GB, 2), "per_expert_mb": round(per_expert / 1e6, 2),
        "routed_expert_gb_per_token_uncached": round(TOPK * LAYERS * per_expert / GB, 3),
        "engram_tables_gb": round(b["Engram tables"] / GB, 2),
        "engram_bytes_per_token": ENGRAM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES,
        "kv_bf16_bytes_per_token": sum((LATENT + INDEX_DIM) * 2 // r for r in KV_SOURCES.values()),
        "kv_bf16_gb": {str(c): round(kv_bytes(c) / GB, 3) for c in (32768, 131072, 1 << 20)},
        "host_ram_full_residency_gb": round((b["routed experts (40 layers)"] + b["Engram tables"] + b["embedding"]) / GB + OS_GB, 1),
    }, indent=1))
    for name, host in HOSTS.items():
        for g in GPUS:
            if name.startswith("A2000") and g != 12:
                continue
            r = decode_bound(host, g, b, a.context)
            row = f"{name:44s} {g:3d} GB  "
            if not r["feasible"]:
                print(row + "INFEASIBLE  " + r["binding"])
                continue
            extra = ""
            if r["binding"] == "H2D" and host["h2d"] < GEN5[0]:
                lo, hi = (decode_bound(host, g, b, a.context, h2d=x)["tok_s_bound"] for x in GEN5)
                extra = f"  (gen5 {lo:.1f}-{hi:.1f})"
            print(row + f"bound {r['tok_s_bound']:5.2f} tok/s{extra}  binding {r['binding']:4s}  gpu hit {r['gpu_hit']:.1%}"
                  f"  ram hit {r['ram_hit']:.1%}  Engram {r['engram']}")
    for g in GPUS:
        fixed = b["gpu_resident"] + LAYERS * WINDOW * LATENT * 2 / GB + 2 * TOPK * per_expert / GB + RESERVE_GB
        per_tok = sum((LATENT + INDEX_DIM) * 2 / r for r in KV_SOURCES.values())
        print(f"{g:3d} GB card: longest context with no expert cache = {max(0.0, (g - fixed) * GB / per_tok) / 1e6:,.2f} M tokens")
    cold = dict(HOSTS["A2000 host (gen3 x8)"], disk=0.743)
    r = decode_bound(cold, 12, b, a.context)
    print(f"A2000 host at its cold-load disk rate 0.743 GB/s: bound {r['tok_s_bound']:.2f} tok/s")
    for name, host in HOSTS.items():
        r = decode_bound(host, 12, b, a.context)
        if r["feasible"] and not r["engram"].startswith("host"):
            print(f"{name}: Engram's disk reads bind only below {2 * ENGRAM_ROWS_PER_TOKEN * r['tok_s_bound']:.0f} random reads/s"
                  " at that bound (unmeasured on this host)")
    stage = 2 * LAYERS - 1
    for m in (1, 4):
        gbs = m * stage * b["routed experts (40 layers)"] / LAYERS / GB
        print(f"training, whole-layer staging, {m} micro-batch(es): {m * stage} stagings = {gbs:,.1f} GB per step;"
              f" {gbs / 21.76:,.1f} s at 21.76 GB/s H2D, {gbs / 6.31:,.1f} s from disk at 6.31 GB/s")
    print(f"training, one whole layer staged: {b['routed experts (40 layers)'] / LAYERS / GB:.2f} GB + {b['gpu_resident']:.2f} GB resident"
          f" = {b['routed experts (40 layers)'] / LAYERS / GB + b['gpu_resident']:.2f} GB before activations, LoRA and optimizer")


if __name__ == "__main__":
    main()
