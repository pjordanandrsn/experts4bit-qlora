"""DeepSeek-V4.1-Flash-1017, gate B: memory, residency and transfer bounds. Analytic: no weights, no GPU.

Bytes come from the checkpoint's shard headers (gate A's ds41_metadata.py), rates from the records named in README.md.
Every tok/s figure is a TRANSFER BOUND (bytes over the slowest link they cross), never a measurement.

    python ds41_gate_b.py [--headers DIR] [--context 131072]
"""
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GB, GIB = 1e9, 1 << 30

# Read from the reference code (gate A, revision 1bae0e10): the bf16 caches the model allocates.
KV_SOURCES = {2: 2, 8: 2, 14: 2, 20: 1}         # kv_source layer -> its compress ratio (config compress_ratios)
LATENT, INDEX_DIM, WINDOW, LAYERS = 512, 128, 128, 40
ENGRAM_READS_PER_TOKEN = 2 * 2 * (4 - 1) * 8      # 2 tables x (weight row + scale row) x (max_ngram 4 - 1) x 8 heads
ENGRAM_BYTES_PER_TOKEN = 2 * (4 - 1) * 8 * (256 + 8)
TOPK, N_EXPERTS = 6, 384

# Hosts: measured rates and memory (README.md names each record). GB/s and GB.
#   gpu_free: measured free VRAM in GB (else the card's GiB size); ram: host RAM the model may use; os: reserve kept back.
HOSTS = {
    "A2000 host": {"h2d": 5.8, "disk": 1.63, "ram": 51.5, "os": 0.0, "cards": {12: 10698 * (1 << 20) / GB}},
    "rented, 239 GB floor": {"h2d": 21.76, "disk": 6.31, "ram": 239.0, "os": 8.0},
    "rented, 1,081 GB (box 1)": {"h2d": 21.76, "disk": 6.31, "ram": 1081.0, "os": 8.0},
}
GEN5 = (40.0, 53.0)                               # unmeasured: the gen-5 host's listed range (assumption)
CARDS = (12, 24, 48, 96)                          # GiB, as sold
RESERVE_GB = 1.0                                  # CUDA context, activations, workspace at batch 1: ASSUMPTION


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


def placement(host, gpu_gb, b, context, engram_on_disk, h2d=None):
    """Routed experts per token from the GPU cache, then host RAM, then disk. The Engram tables sit in RAM or on disk."""
    h2d = h2d or host["h2d"]
    experts, tables = b["routed experts (40 layers)"], b["Engram tables"]
    per_expert = experts / (LAYERS * N_EXPERTS)
    token = TOPK * LAYERS * per_expert
    resident = b["gpu_resident"] + kv_bytes(context) / GB + 2 * TOPK * per_expert / GB + RESERVE_GB
    if resident > gpu_gb:
        return {"feasible": False, "binding": f"VRAM: needs {resident:.2f} GB of {gpu_gb:.2f} before any expert cache"}
    ram = host["ram"] - host["os"] - b["embedding"] / GB - (0 if engram_on_disk else tables / GB)
    if ram < 0:
        return {"feasible": False, "binding": "host RAM: the Engram tables do not fit"}
    gpu_hit = min(1.0, (gpu_gb - resident) * GB / experts)          # uniform routing: ASSUMPTION
    ram_hit = min(1.0 - gpu_hit, ram * GB / experts)
    disk_bytes, link_bytes = token * (1 - gpu_hit - ram_hit), token * (1 - gpu_hit)
    t_link, t_disk = link_bytes / (h2d * GB), disk_bytes / (host["disk"] * GB)
    bound = 1 / max(t_link, t_disk)
    return {"feasible": True, "binding": "disk" if t_disk >= t_link else "H2D", "tok_s_bound": bound, "gpu_hit": gpu_hit,
            "ram_hit": ram_hit, "engram": "disk" if engram_on_disk else "host RAM",
            "engram_binds_below_reads_per_s": ENGRAM_READS_PER_TOKEN * bound if engram_on_disk else None}


def best(host, gpu_gb, b, context, h2d=None):
    rows = [placement(host, gpu_gb, b, context, d, h2d) for d in (False, True)]
    ok = [r for r in rows if r["feasible"]]
    return (max(ok, key=lambda r: r["tok_s_bound"]) if ok else rows[0]), rows


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
        "engram_bytes_per_token": ENGRAM_BYTES_PER_TOKEN, "engram_random_reads_per_token": ENGRAM_READS_PER_TOKEN,
        "kv_bf16_bytes_per_token": sum((LATENT + INDEX_DIM) * 2 // r for r in KV_SOURCES.values()),
        "kv_bf16_gb": {str(c): round(kv_bytes(c) / GB, 3) for c in (32768, 131072, 1 << 20)},
        "host_ram_full_residency_gb": {
            "Engram tables on disk": round((b["routed experts (40 layers)"] + b["embedding"]) / GB + 8, 1),
            "Engram tables in RAM": round((b["routed experts (40 layers)"] + b["Engram tables"] + b["embedding"]) / GB + 8, 1)},
    }, indent=1))
    print(f"\ndecode, batch 1, context {a.context:,}: the better placement, then the other")
    for name, host in HOSTS.items():
        for card in CARDS:
            if name.startswith("A2000") and card != 12:
                continue
            gpu_gb = host.get("cards", {}).get(card, card * GIB / GB)
            r, both = best(host, gpu_gb, b, a.context)
            head = f"{name:26s} {card:3d} GiB ({gpu_gb:6.2f} GB)  "
            if not r["feasible"]:
                print(head + "INFEASIBLE  " + r["binding"])
                continue
            extra = ""
            if r["binding"] == "H2D" and host["h2d"] < GEN5[0]:
                lo, hi = (best(host, gpu_gb, b, a.context, h2d=x)[0]["tok_s_bound"] for x in GEN5)
                extra = f" (gen5 {lo:.1f}-{hi:.1f})"
            other = [o for o in both if o is not r and o["feasible"]]
            alt = f"  | Engram {'on disk' if other[0]['engram'] == 'disk' else 'in RAM'}: {other[0]['tok_s_bound']:.2f}" if other else ""
            binds = (f"; Engram binds below {r['engram_binds_below_reads_per_s']:.0f} reads/s"
                     if r["engram_binds_below_reads_per_s"] else "")
            print(head + f"{r['tok_s_bound']:5.2f} tok/s{extra}  {r['binding']:4s}  Engram {r['engram']:8s}  hits GPU "
                  f"{r['gpu_hit']:.1%} RAM {r['ram_hit']:.1%}{binds}{alt}")
    cold = dict(HOSTS["A2000 host"], disk=0.743)
    print(f"A2000 host at its cold-load disk rate 0.743 GB/s: {best(cold, HOSTS['A2000 host']['cards'][12], b, a.context)[0]['tok_s_bound']:.2f} tok/s")
    print()
    per_tok = sum((LATENT + INDEX_DIM) * 2 / r for r in KV_SOURCES.values())
    fixed = b["gpu_resident"] + LAYERS * WINDOW * LATENT * 2 / GB + 2 * TOPK * per_expert / GB + RESERVE_GB
    for name, gpu_gb in [("A2000 (measured free)", HOSTS["A2000 host"]["cards"][12])] + [(f"{c} GiB", c * GIB / GB) for c in CARDS]:
        print(f"{name:22s} longest context with no expert cache: {max(0.0, (gpu_gb - fixed) * GB / per_tok) / 1e6:,.2f} M tokens")
    print()
    stage = 2 * LAYERS - 1
    layer_gb = b["routed experts (40 layers)"] / LAYERS / GB
    for m in (1, 4):
        gbs = m * stage * layer_gb
        print(f"training, whole-layer staging, {m} micro-batch(es): {m * stage} stagings = {gbs:,.1f} GB per step;"
              f" {gbs / 21.76:,.1f} s at 21.76 GB/s H2D, {gbs / 6.31:,.1f} s from disk at 6.31 GB/s")
    print(f"training, one whole layer staged: {layer_gb:.2f} GB + {b['gpu_resident']:.2f} GB resident"
          f" = {layer_gb + b['gpu_resident']:.2f} GB before activations, LoRA and optimizer")


if __name__ == "__main__":
    main()
