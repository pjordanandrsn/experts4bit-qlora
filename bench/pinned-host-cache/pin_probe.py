"""Name the fixed pinned host term: what RssShmem holds at a q235_box load end beyond the expert homes.

Loads exactly as q235_box.py --repeat does (its own build()), then records, at the load end, after
torch._C._host_emptyCache(), and after the step-0 forward:
  - /proc/self/status (VmRSS, RssAnon, RssFile, RssShmem);
  - torch.cuda.host_memory_stats() (allocated_bytes = blocks owned, active + cached; active_bytes = checked out);
  - every pinned CPU tensor storage reachable from Python, deduplicated by storage, with its nbytes and the
    power-of-two block the caching host allocator rounds it to, grouped by owner;
  - /proc/self/smaps mappings with Rss >= 100 MB (path, size, rss, shared/private split).
The forward's loss is compared bitwise with the memslope record for the same slice, so a safe emptyCache shows as
an unchanged loss."""
import gc
import importlib.util
import json
import os
import sys
import time

import torch

box_path, ckpt, tokens_path, out_path = sys.argv[1:5]
ref_loss = float(sys.argv[5]) if len(sys.argv) > 5 else None
spec = importlib.util.spec_from_file_location("q235_box", box_path)
box = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box)


def status():
    d = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"):
            d[k] = int(v.split()[0]) * 1024
    return d


def pow2(n):
    return 1 << (n - 1).bit_length() if n > 0 else 0


def owners(model):
    """storage data_ptr -> a name: module params/buffers by qualified name, offload homes by layer."""
    names = {}
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.device.type == "cpu" and t.numel():
            names.setdefault(t.untyped_storage().data_ptr(), n)
    for mn, m in model.named_modules():
        h = getattr(m, "_e4b_offload", None) or getattr(m, "offload_handle", None)
        for cand in (h, m):
            home = getattr(cand, "home", None)
            if isinstance(home, dict):
                for k, t in home.items():
                    if isinstance(t, torch.Tensor):
                        names.setdefault(t.untyped_storage().data_ptr(), f"home:{mn}.{k}")
            arena = getattr(cand, "arena_cpu", None) or getattr(cand, "_arena_cpu", None)
            if isinstance(arena, dict):
                for k, t in arena.items():
                    names.setdefault(t.untyped_storage().data_ptr(), f"arena:{mn}.{k}")
    return names


def pinned(model):
    seen = {}
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.device.type == "cpu" and o.is_pinned():
                s = o.untyped_storage()
                seen.setdefault(s.data_ptr(), (s.nbytes(), tuple(o.shape), str(o.dtype)))
        except Exception:
            pass
    names = owners(model)
    rows = []
    for ptr, (nb, shape, dt) in seen.items():
        nm = names.get(ptr, "unowned")
        rows.append({"owner": nm, "nbytes": nb, "pow2": pow2(nb), "shape": list(shape), "dtype": dt})
    groups = {}
    for r in rows:
        g = "home" if r["owner"].startswith(("home:", "arena:")) else ("param/buffer" if r["owner"] != "unowned" else "unowned")
        a = groups.setdefault(g, {"count": 0, "nbytes": 0, "pow2": 0})
        a["count"] += 1
        a["nbytes"] += r["nbytes"]
        a["pow2"] += r["pow2"]
    big = sorted((r for r in rows if not r["owner"].startswith(("home:", "arena:"))), key=lambda r: -r["nbytes"])[:20]
    return {"count": len(rows), "nbytes": sum(r["nbytes"] for r in rows), "pow2": sum(r["pow2"] for r in rows),
            "groups": groups, "largest_non_home": big}


def smaps(min_rss=100 << 20):
    maps, cur = [], None
    for line in open("/proc/self/smaps"):
        parts = line.split()
        if parts and "-" in parts[0] and len(parts) >= 5 and not parts[0].endswith(":"):
            cur = {"range": parts[0], "perms": parts[1], "path": " ".join(parts[5:]) if len(parts) > 5 else "[anon]"}
            maps.append(cur)
        elif cur is not None and parts and parts[0].endswith(":") and len(parts) >= 2 and parts[1].isdigit():
            cur[parts[0][:-1]] = int(parts[1]) * 1024
    keep = []
    for m in maps:
        if m.get("Rss", 0) >= min_rss:
            keep.append({k: m.get(k) for k in ("range", "perms", "path", "Size", "Rss", "Shared_Clean", "Shared_Dirty",
                                                "Private_Clean", "Private_Dirty", "Anonymous")})
    by_path = {}
    for m in maps:
        b = by_path.setdefault(m["path"], {"n": 0, "Rss": 0, "Shared": 0})
        b["n"] += 1
        b["Rss"] += m.get("Rss", 0)
        b["Shared"] += m.get("Shared_Clean", 0) + m.get("Shared_Dirty", 0)
    top = sorted(by_path.items(), key=lambda kv: -kv[1]["Rss"])[:15]
    return {"mappings_over_100MB": keep, "rss_by_path_top": [{"path": p, **v} for p, v in top]}


def snap(tag, model):
    torch.cuda.synchronize()
    gc.collect()
    hs = torch.cuda.host_memory_stats()
    return {"tag": tag, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": status(),
            "host_memory_stats": {k: v for k, v in hs.items() if "bytes" in k or "count" in k},
            "pinned": pinned(model), "smaps": smaps(),
            "device_allocated": torch.cuda.memory_allocated()}


rec = {"argv": sys.argv[1:], "torch": torch.__version__}
t0 = time.time()
model, load = box.build(ckpt, "cuda", 16, 16, "bf16", "fast")
rec["load_seconds"] = round(time.time() - t0, 1)
rec["load"] = {k: load[k] for k in ("offload", "patched") if k in load}
snaps = [snap("load_end", model)]
torch._C._host_emptyCache()
snaps.append(snap("after_host_emptyCache", model))
tok = torch.load(tokens_path, weights_only=True)
ids, mask, labels = (tok[k].to("cuda") for k in ("input_ids", "attention_mask", "labels"))
with torch.enable_grad():
    loss = float(model(input_ids=ids, attention_mask=mask, labels=labels).loss.detach())
rec["forward_loss"] = loss
rec["ref_loss"] = ref_loss
rec["loss_equal_ref"] = (loss == ref_loss) if ref_loss is not None else None
snaps.append(snap("after_forward", model))
rec["snaps"] = snaps
json.dump(rec, open(out_path, "w"), indent=1, default=str)
s = {x["tag"]: (x["status"]["RssShmem"], x["host_memory_stats"].get("allocated_bytes.current"),
                x["host_memory_stats"].get("active_bytes.current"), x["pinned"]["pow2"]) for x in snaps}
print(json.dumps({"loss": loss, "ref": ref_loss, "summary(RssShmem, owned=active+cached, active, pinned_pow2)": s}))
