#!/usr/bin/env python3
"""Per-layer expert-weight statistics + block-64 NF4 round-trip error for a fused-MoE safetensors checkpoint (CPU, numpy).
Streams one (layer, projection) tensor at a time. Output: JSON lines per (layer, proj)."""
import json, struct, sys, os, time
import numpy as np

D = sys.argv[1]; OUT = sys.argv[2]
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=np.float32)
BOUND = (NF4[1:] + NF4[:-1]) / 2

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def nf4_roundtrip(x, block=64):
    flat = x.reshape(-1)
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    b = flat.reshape(-1, block)
    amax = np.abs(b).max(axis=1, keepdims=True)
    amax[amax == 0] = 1.0
    n = b / amax
    q = np.searchsorted(BOUND, n)
    deq = NF4[q] * amax
    err = (deq - b).reshape(-1)[: flat.size - pad]
    return err

def tensors(D):
    idx = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]
    files = {}
    for k, f in idx.items():
        if ".experts." in k:
            files.setdefault(f, []).append(k)
    for f, keys in sorted(files.items()):
        p = os.path.join(D, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
            for k in sorted(keys, key=lambda s: (int(s.split(".layers.")[1].split(".")[0]), s)):
                meta = hdr[k]
                s, e = meta["data_offsets"]
                fh.seek(base + s)
                raw = fh.read(e - s)
                yield k, meta["dtype"], meta["shape"], raw

out = open(OUT, "w")
t0 = time.time()
for k, dt, shape, raw in tensors(D):
    layer = int(k.split(".layers.")[1].split(".")[0]); proj = k.split(".")[-1]
    if dt == "BF16":
        x = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
    elif dt == "F32":
        x = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    else:
        x = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    E = shape[0]
    rms_all = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rec = {"layer": layer, "proj": proj, "dtype": dt, "shape": shape, "rms": rms_all, "absmax": float(np.abs(x).max()),
           "n_nonfinite": int((~np.isfinite(x)).sum()), "frac_zero": float((x == 0).mean())}
    per_e_rms, per_e_amax, per_e_err = [], [], []
    for e in range(E):
        xe = x[e]
        r = float(np.sqrt(np.mean(xe.astype(np.float64) ** 2))); a = float(np.abs(xe).max())
        err = nf4_roundtrip(xe)
        per_e_rms.append(r); per_e_amax.append(a); per_e_err.append(float(np.sqrt(np.mean(err.astype(np.float64) ** 2)) / max(r, 1e-30)))
    per_e_rms, per_e_amax, per_e_err = map(np.array, (per_e_rms, per_e_amax, per_e_err))
    rec.update({"expert_rms_min": float(per_e_rms.min()), "expert_rms_med": float(np.median(per_e_rms)), "expert_rms_max": float(per_e_rms.max()),
                "expert_amax_over_rms_med": float(np.median(per_e_amax / per_e_rms)), "expert_amax_over_rms_max": float((per_e_amax / per_e_rms).max()),
                "nf4_rel_err_med": float(np.median(per_e_err)), "nf4_rel_err_max": float(per_e_err.max()), "nf4_rel_err_mean": float(per_e_err.mean()),
                "frac_gt6rms": float((np.abs(x) > 6 * rms_all).mean())})
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(f"L{layer:02d} {proj:12s} {dt} {shape} rms {rms_all:.4g} amax/rms {rec['expert_amax_over_rms_med']:.1f} nf4err med {rec['nf4_rel_err_med']:.4f} max {rec['nf4_rel_err_max']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    del x
