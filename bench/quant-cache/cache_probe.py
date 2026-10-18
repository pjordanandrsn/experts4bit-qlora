#!/usr/bin/env python3
"""cache_probe.py -- the $0 correctness leg of bench/quant-cache/DESIGN.md (E4B_QUANT_CACHE). Any CUDA card; no timing is
read.

    python cache_probe.py --model DIR/p137-tiny --out probe.json

Every load runs in its own process (``--role``), so a hit reads only what the cache wrote. STRICT checks:
  1. bytes: per MoE layer, the sha256 of the four packed tensors (from the offload homes) is equal for a fresh load (no
     cache), the miss that writes the cache, and the hit that reads it;
  2. training: with enable_fast_train(dgrad=True) and gradient checkpointing, layer_major_step on fixed micro-batches gives
     bit-identical per-micro-batch losses and the same stagings fresh and on a hit;
  3. host memory, per pin mode (default, E4B_OFFLOAD_PIN_SLAB=1, E4B_OFFLOAD_ARENA=1): on a hit the homes' byte count and
     is_pinned() per home equal a fresh load's, and RssShmem + RssAnon after the load is at most 64 MiB above it
     (a hit reads no bf16 rows, so it may sit below);
  4. the hit's log says hit, and the miss's says it wrote;
  5. refusals and fallbacks: a flipped byte refuses (QuantCacheCorrupt); a bitsandbytes upgrade (a key change) is a miss;
     a manifest recorded on another GPU re-quantizes one layer, compares, and the hit stands; a forced mismatch there is a
     miss that writes a device-keyed copy; an int8 store with the variable set refuses.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROWS = 2


def _rss():
    out = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("RssAnon", "RssShmem", "RssFile"):
            out[k] = int(v.split()[0]) * 1024
    return out


def role_load(model, mode_env, train):
    import torch

    from experts4bit_qlora.engines import offload as off
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    os.environ["E4B_OFFLOAD_STATS"] = "1"
    logs = []
    import experts4bit_qlora.loader as L
    real_log = L.log
    L.log = lambda m: (logs.append(m), real_log(m))
    torch.manual_seed(0)
    model_, _ = load_moe_4bit_streaming(model, "cuda", torch.bfloat16, 8, 16, offload=True, pin=True, quant_type="nf4")
    handles = off.offload_handles(model_)
    layers = []
    for h in handles:
        d = hashlib.sha256()
        for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
            d.update(h.home[n].detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        layers.append(d.hexdigest())
    rec = {"layers": layers, "home_bytes": sum(h._stage_nbytes for h in handles),
           "pinned": [bool(off._is_pinned(t)) for h in handles for t in h.home.values()],
           "pin_mode": sorted({str(h.pin_mode) for h in handles}), "rss": _rss(),
           "cache_log": [m for m in logs if "[quant-cache]" in m]}
    if train:
        from transformers import AutoTokenizer

        from experts4bit_qlora.engines.fast import enable_fast_train
        from experts4bit_qlora.engines.layer_major import layer_major_step
        from experts4bit_qlora.lora import trainable_lora_param_ids
        tok = AutoTokenizer.from_pretrained(model)
        ids, _ = trainable_lora_param_ids(model_)
        for p in model_.parameters():
            p.requires_grad_(id(p) in ids)
        model_.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        enable_fast_train(model_, dgrad=True, absmax_dq=False)
        model_.config.use_cache = False
        model_.train()
        texts = ["The quick brown fox jumps over the lazy dog.", "Layer-major accumulation stages each layer once.",
                 "Quantized experts live in pinned host memory.", "A cache hit must read the same bytes."]
        mbs = []
        for t in texts:
            enc = tok(t, return_tensors="pt")
            mbs.append({"input_ids": enc["input_ids"].cuda(), "attention_mask": enc["attention_mask"].cuda(),
                        "labels": enc["input_ids"].cuda()})
        off.reset_offload_stats()
        losses = layer_major_step(model_, mbs, [0.25] * 4)
        rec["losses"] = [float(x).hex() for x in losses]
        rec["stagings"] = len(off._stats().copies)
    return rec


def _child(args, env):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--role", "load", "--model", args.model] + args.extra,
                       env={**os.environ, **env}, capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-3000:]
    for line in r.stdout.splitlines():
        if line.startswith("PROBE_ROLE "):
            return json.loads(line[len("PROBE_ROLE "):]), None
    return None, tail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--role")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args(argv)
    if a.role == "load":
        print("PROBE_ROLE " + json.dumps(role_load(a.model, None, a.train)), flush=True)
        return 0
    a.extra = []
    base = {k: "" for k in ("E4B_QUANT_CACHE", "E4B_QUANT_CACHE_REBUILD", "E4B_OFFLOAD_PIN_SLAB", "E4B_OFFLOAD_ARENA")}
    root = tempfile.mkdtemp(prefix="qcache-probe-")
    checks, rep, errs = {}, {}, {}
    a.extra = ["--train"]
    fresh, errs["fresh"] = _child(a, {**base})
    miss, errs["miss"] = _child(a, {**base, "E4B_QUANT_CACHE": root})
    hit, errs["hit"] = _child(a, {**base, "E4B_QUANT_CACHE": root})
    ok = fresh and miss and hit
    checks["1_bytes_fresh_miss_hit_equal"] = bool(ok) and fresh["layers"] == miss["layers"] == hit["layers"]
    checks["2_training_bitwise"] = bool(ok) and fresh["losses"] == hit["losses"] and fresh["stagings"] == hit["stagings"]
    checks["4_logs"] = bool(ok) and any("wrote" in m for m in miss["cache_log"]) and any(
        m.strip().startswith("[quant-cache] hit") for m in hit["cache_log"])
    a.extra = []
    for mode, env in (("default", {}), ("slab", {"E4B_OFFLOAD_PIN_SLAB": "1"}), ("arena", {"E4B_OFFLOAD_ARENA": "1"})):
        f, errs[f"fresh_{mode}"] = _child(a, {**base, **env})
        h, errs[f"hit_{mode}"] = _child(a, {**base, **env, "E4B_QUANT_CACHE": root})
        good = f is not None and h is not None
        dr = (h["rss"].get("RssAnon", 0) + h["rss"].get("RssShmem", 0)) - (f["rss"].get("RssAnon", 0) + f["rss"].get("RssShmem", 0)) if good else None
        rep[f"rss_delta_{mode}"] = dr
        checks[f"3_memory_{mode}"] = good and h["home_bytes"] == f["home_bytes"] and h["pinned"] == f["pinned"] and \
            h["pin_mode"] == f["pin_mode"] and dr <= 64 << 20 and h["layers"] == f["layers"]
    # 5. refusals and fallbacks (each in its own copy of the cache)
    key = [d for d in os.listdir(root) if not d.startswith(".tmp-")][0]

    def copy_root():
        r2 = tempfile.mkdtemp(prefix="qcache-probe-")
        shutil.copytree(os.path.join(root, key), os.path.join(r2, key))
        return r2
    r2 = copy_root()
    f0 = os.path.join(r2, key, sorted(x for x in os.listdir(os.path.join(r2, key)) if x.endswith(".safetensors"))[0])
    b = bytearray(open(f0, "rb").read())
    b[-1] ^= 0xFF
    open(f0, "wb").write(bytes(b))
    _, tail = _child(a, {**base, "E4B_QUANT_CACHE": r2})
    checks["5_flipped_byte_refuses"] = tail is not None and "QuantCacheCorrupt" in tail
    r3 = copy_root()
    man = os.path.join(r3, key, "manifest.json")
    m = json.load(open(man))
    m["device"]["name"] = "Another GPU"
    json.dump(m, open(man, "w"))
    dv, errs["device"] = _child(a, {**base, "E4B_QUANT_CACHE": r3})
    checks["5_other_gpu_requantizes_and_the_hit_stands"] = dv is not None and any(
        "the hit stands" in x for x in dv["cache_log"]) and dv["layers"] == fresh["layers"]
    r4 = copy_root()
    man4 = os.path.join(r4, key, "manifest.json")
    m4 = json.load(open(man4))
    m4["device"]["name"] = "Another GPU"
    json.dump(m4, open(man4, "w"))
    a.extra = ["--force-mismatch"]
    fm, errs["forced"] = _child(a, {**base, "E4B_QUANT_CACHE": r4})
    a.extra = []
    keys4 = [d for d in os.listdir(r4) if not d.startswith(".tmp-")]
    checks["5_forced_mismatch_is_a_miss_that_writes_a_device_key"] = fm is not None and len(keys4) == 2 and \
        fm["layers"] == fresh["layers"] and any("wrote" in x for x in fm["cache_log"])
    import experts4bit_qlora.quant_cache as qc
    r5 = tempfile.mkdtemp(prefix="qcache-probe-")
    os.environ["E4B_QUANT_CACHE"] = r5
    real = qc.key_fields

    def bumped(**k):
        f = real(**k)
        f["bitsandbytes"] = f["bitsandbytes"] + ".post-upgrade"
        return f
    shutil.copytree(os.path.join(root, key), os.path.join(r5, key))
    qc.key_fields = bumped
    try:
        from experts4bit_qlora.loader import _open_quant_cache  # noqa: F401  (the key is built through key_fields)
        import torch
        from experts4bit_qlora.loader import load_moe_4bit_streaming
        logs = []
        import experts4bit_qlora.loader as L
        rl = L.log
        L.log = lambda msg: (logs.append(msg), rl(msg))
        load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, 8, 16, offload=True, pin=True, quant_type="nf4")
        L.log = rl
        checks["5_bnb_upgrade_is_a_miss"] = any(x.strip().startswith("[quant-cache] miss") for x in logs)
        try:
            load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, 8, 16, offload=True, pin=True, quant_type="int8")
            checks["5_int8_refuses"] = False
        except ValueError as e:
            checks["5_int8_refuses"] = "4-bit" in str(e)
    finally:
        qc.key_fields = real
        os.environ.pop("E4B_QUANT_CACHE", None)
    out = {"checks": checks, "pass": all(checks.values()), "reporting": rep,
           "fresh": fresh and {k: fresh[k] for k in ("layers", "home_bytes", "pin_mode", "losses", "stagings")},
           "errors": {k: v for k, v in errs.items() if v}}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print("QCACHE_PROBE " + json.dumps({"pass": out["pass"], "checks": checks, "reporting": rep}), flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    if "--force-mismatch" in sys.argv:
        sys.argv.remove("--force-mismatch")
        import experts4bit_qlora.loader as _L

        def _forced(cache, i, base):              # the forced-mismatch arm: the re-quantized layer "differs"
            cache.device_mismatch_to_miss()
        _L._quant_cache_device_check = _forced
    sys.exit(main())
