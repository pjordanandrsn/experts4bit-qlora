"""Kimi-K3, ALL 93 layers, real weights, on one 12 GB A2000. No cloud.

The 4-layer composition test (k3_compose_forward.py) proved the seams hold. This runs
the whole model. Three things made it possible, all landed today:

  * grouped-nf4-gemm#24 -- the k slot buffer is shared across layers. Private slots
    are 281 MB each at K3's geometry: 25.8 GB of VRAM across 92 engines, on a 12 GB
    card. Shared, it is 281 MB total.
  * experts4bit-qlora#48 -- dense weights served from the checkpoint by byte offset,
    so the 114.4 GB dense side needs no host RAM. This box has 22 GB free; pinning
    was never an option.
  * experts4bit-qlora#50 -- the loader's shape gate, which is what makes K3's 69 KDA
    layers loadable at all (A_log ships per-head at 96 values, zero-padded to 128).
    Imported here rather than reimplemented, so this run exercises the shipped code.

Per token this reads 25.83 GB of expert rows plus 107.3 GB of dense weights off the
SATA-SSD pool -- ~133 GB, which at the pool's measured 1.29 GB/s is a couple of
minutes. Slow is the correct trade: the alternative is renting a box to avoid waiting.

Peak host RAM is ONE layer, because layers are materialized one at a time and their
big tensors are placeholders that dense_offload replaces with disk homes without ever
reading them. That matters beyond tidiness -- this NAS runs the household's Plex, arr
stack and mail, and an OOM here is an outage.
"""
import json
import math
import os
import statistics
import time

import torch


SNAP = "/z532/hf-models/moonshotai_Kimi-K3"
ARENA = os.environ.get("K3_ARENA", "/z531/k3-arena/k3.arena")
DENSE = os.environ.get("K3_DENSE", "/workspace/k3-dense")
CKPT_PREFIX = "language_model.model."
TOPK_SLOTS = 16
HOT_ROWS = int(os.environ.get("K3_HOT_ROWS", "48"))
MODE = os.environ.get("K3_MODE", "gen")            # gen | ppl
NEW = int(os.environ.get("K3_NEW_TOKENS", "4"))
PPL_TOKENS = int(os.environ.get("K3_PPL_TOKENS", "128"))
# GB of dense weights to hold in PINNED host RAM instead of serving from disk.
# Pinned pages are NOT reclaimable -- unlike the ZFS ARC they displace -- and this
# box also runs the household's Plex/arr/mailcow, so the default is 0 and the
# budget is deliberately explicit.
PIN_GB = float(os.environ.get("K3_PIN_GB", "0"))
VERIFY_CACHE = os.environ.get("K3_VERIFY_CACHE", "1") == "1"
DEV = "cuda"

from experts4bit_qlora.formats.dense_disk import DenseDiskSource            # noqa: E402
from experts4bit_qlora.engines.dense_offload import (                       # noqa: E402
    MIN_BYTES, dense_offload_report, enable_dense_offload)
from experts4bit_qlora.loader import _fit                           # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM           # noqa: E402


def free_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return -1.0


t_start = time.time()
print(f"host RAM available: {free_gb():.1f} GB")
if free_gb() < 6:
    raise SystemExit("under 6 GB free -- refusing to start on a box running the "
                     "household's services")

# ---- build all 93 layers on meta -------------------------------------------
cfg = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
tc = getattr(cfg, "text_config", cfg)
print(f"instantiating ALL {tc.num_hidden_layers} layers on meta ...")
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
model.eval()

# K3's PreTrainedModel forces flash_attention_2 at construction
# (modeling_kimi_linear.py:1110-1117). flash-attn is not installed; eager is the
# reference, not a downgrade -- FA2 is exact. Set it on EVERY module's config.
_n = 0
for _m in model.modules():
    _c = getattr(_m, "config", None)
    if _c is not None and getattr(_c, "_attn_implementation", None) is not None:
        _c._attn_implementation = "eager"
        _n += 1
print(f"attn -> eager on {_n} module configs")

base = model.language_model.model
layers = base.layers
kinds = {}
for i, blk in enumerate(layers):
    k = (type(blk.self_attn).__name__.replace("Kimi", "").replace("Attention", "")
         + ("+moe" if hasattr(blk, "block_sparse_moe") else "+dense"))
    kinds[k] = kinds.get(k, 0) + 1
print(f"layer census: {kinds}")

# ---- materialize the dense side, one layer at a time -----------------------
# Big layer tensors become EMPTY placeholders: dense_offload only reads their shape
# and dtype to build a disk home, never their data, and the gate below proves that
# every one of them really did become a disk home. Anything dense_offload leaves
# resident -- 1-D, or under MIN_BYTES -- must hold real bytes, so it is read.
setup_src = DenseDiskSource(DENSE)
print(f"{setup_src!r}")
names = [n for n, _t in list(model.named_parameters()) + list(model.named_buffers())]
expert_names = {n for n in names if ".block_sparse_moe.experts." in n}
dense_names = [n for n in names if n not in expert_names]
print(f"tensors: {len(names)} total, {len(expert_names)} expert, "
      f"{len(dense_names)} dense")

placeholders = real = narrowed = 0
absent = []
t0 = time.time()
for n in dense_names:
    mod_path, attr = n.rsplit(".", 1)
    mod = model.get_submodule(mod_path)
    store = mod._parameters if attr in mod._parameters else mod._buffers
    t = store[attr]
    nbytes = t.numel() * t.element_size()
    in_layer = ".layers." in n
    big = t.dim() >= 2 and nbytes >= MIN_BYTES
    if in_layer and big:
        # Verify the bytes EXIST before promising a placeholder will be filled from
        # them. Downstream, a missing key makes dense_offload silently pin the
        # uninitialised placeholder as a host home (dense_offload.py:175-182) -- real
        # weights replaced by garbage, no error.
        if n not in setup_src.tensors:
            raise SystemExit(f"{n} is a managed placeholder with no checkpoint bytes")
        # untouched pages: virtual only, freed at evict, never read
        store[attr] = (torch.nn.Parameter(torch.empty(t.shape, dtype=t.dtype),
                                          requires_grad=False)
                       if attr in mod._parameters else
                       torch.empty(t.shape, dtype=t.dtype))
        placeholders += 1
        continue
    # The extract was built from the checkpoint's own non-expert keys, and this
    # model's parameter names ARE those keys -- the 4-layer run confirmed the map is
    # the identity (its `not in weight_map` check printed nothing).
    if n not in setup_src.tensors:
        absent.append(n)
        continue
    got = setup_src.fetch(n).clone()            # clone: fetch aliases shared staging
    got, cut = _fit(n, got, t)                  # PR #50, on real K3 data
    narrowed += cut
    # Same routing the 4-layer run used: layer tensors and the two 2.35 GB vocab
    # tensors stay on the host (a base forward gathers one embedding row and never
    # touches lm_head); everything else -- final norm, vision tower -- goes to the
    # GPU, where 0.8 GB of vision is affordable and deviating is not worth the risk.
    tgt = "cpu" if (in_layer or "lm_head" in n or "embed_tokens" in n) else DEV
    got = got.to(tgt)
    store[attr] = (torch.nn.Parameter(got, requires_grad=False)
                   if attr in mod._parameters else got)
    real += 1
print(f"materialized in {time.time()-t0:.0f}s: {placeholders} placeholders, "
      f"{real} real ({narrowed} narrowed by the shape gate), {len(absent)} absent")
setup_src.close()                               # drop its staging buffer
# The model declares 2629 dense tensors; the checkpoint ships 2628. The odd one out is
# a NON-PERSISTENT buffer the module computes for itself, and it is in the vision
# tower, which a text forward never enters. Leaving it on meta is the honest handling:
# synthesizing a weight would be inventing data. But a missing tensor anywhere on the
# LANGUAGE path is a real bug, so only the vision tower gets this pass.
text_absent = [n for n in absent if not n.startswith("vision_tower")]
if text_absent:
    raise SystemExit(f"language-path tensors missing from the extract: {text_absent}")
if absent:
    print(f"  NOT IN CHECKPOINT ({len(absent)}, all vision_tower, left on meta -- the "
          f"vision tower is NOT exercised by this forward): {absent}")

stray = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
         if t.is_meta and n not in expert_names and n not in set(absent)]
if stray:
    raise SystemExit(f"dense tensors still on meta: {stray[:6]}")
print(f"host RAM available after materialize: {free_gb():.1f} GB")

# ---- dense side served from disk, one layer resident at a time -------------
serve_src = DenseDiskSource(DENSE)
# Dense homes, in two policies. A layer is ~1.169 GB, and every dense byte is read
# exactly once per forward, so there is no reuse asymmetry to exploit -- the only
# question is how many layers ride the fast path. Disk is 1.47 GB/s here; pinned host
# RAM over this box's Gen3 x8 link measures 6.17 GB/s, so a pinned layer's bytes move
# 4.2x faster. Prefetch cannot help instead: the disk is already at >98% of its
# roofline, so there is no idle device to overlap against.
PER_LAYER_GB = 1.169447638
n_pin = min(len(layers), int(PIN_GB / PER_LAYER_GB)) if PIN_GB > 0 else 0
print(f"dense policy: {n_pin} layer(s) PINNED in host RAM "
      f"({n_pin*PER_LAYER_GB:.1f} GB locked), {len(layers)-n_pin} served from disk")

import experts4bit_qlora.engines.dense_offload as _do  # noqa: E402
_all_layers = _do.decoder_layers(base)
_orig_decoder_layers = _do.decoder_layers


def _only(keep):
    """enable_dense_offload installs its forward hooks only for layers it CONSTRUCTS
    a handle for, so the two policies cannot be applied by calling it twice on the
    same set. Narrowing what it SEES lets each call install its own hooks through the
    library's own code path, rather than reimplementing hook installation here."""
    names = {n for n, _l in keep}
    return lambda _m: [(n, l) for n, l in _all_layers if n in names]


hs = []
if n_pin:
    # Pinned layers need REAL bytes resident to copy into pinned memory, so their big
    # tensors were placeholders and must be filled first.
    # Filter on the SAME predicate the placeholder loop used, so "every placeholder in
    # a pinned layer got real bytes" is checked rather than inferred. No byte total can
    # see this: nbytes is t.numel()*t.element_size() on both branches, identical whether
    # a tensor holds checkpoint data or garbage.
    fill = miss = 0
    for _name, _layer in _all_layers[:n_pin]:
        for n, t in list(_layer.named_parameters()) + list(_layer.named_buffers()):
            if t.is_meta or t.dim() < 2 or t.numel() * t.element_size() < MIN_BYTES:
                continue
            key = f"{CKPT_PREFIX}{_name}.{n}"
            loc = serve_src.tensors.get(key)
            if loc is None or tuple(loc.shape) != tuple(t.shape):
                miss += 1
                continue
            with torch.no_grad():
                t.copy_(serve_src.fetch(key))
            fill += 1
    if miss:
        raise SystemExit(f"{miss} placeholder(s) in the {n_pin} pinned layer(s) have no "
                         f"checkpoint bytes -- pinning them would pin uninitialised memory")
    print(f"  filled {fill} placeholder(s) with real bytes for the pinned layers")
    _do.decoder_layers = _only(_all_layers[:n_pin])
    hs += enable_dense_offload(base, DEV, pin=True, prefetch=False, log=print)
if n_pin < len(_all_layers):
    _do.decoder_layers = _only(_all_layers[n_pin:])
    hs += enable_dense_offload(base, DEV, pin=False, prefetch=False,
                               source=serve_src, key_prefix=CKPT_PREFIX, log=print)
_do.decoder_layers = _orig_decoder_layers

rep = dense_offload_report(hs)
print("dense offload:", json.dumps(rep))
if rep["tensors"] != placeholders:
    raise SystemExit(f"dense homes do not account for every managed tensor: "
                     f"homes={rep['tensors']} vs {placeholders} placeholders")
# `bytes == host_bytes + disk_bytes` holds per handle BY CONSTRUCTION
# (dense_offload.py:173/182/186), so no byte total can see the partition -- comparing
# them was a check that could not fail. Check the partition directly instead: a
# disk-policy layer holding a HOST home means its checkpoint key was missing and
# dense_offload pinned the uninitialised placeholder. `hs` is pinned-first then disk.
stray = [i + n_pin for i, h in enumerate(hs[n_pin:]) if h.host_bytes]
if stray:
    raise SystemExit(f"layers {stray[:6]} fell back to a HOST copy of an uninitialised "
                     f"placeholder -- keys missing from {DENSE}")
print(f"  gate: {rep['tensors']} homes, {rep['disk_bytes']/1e9:.2f} GB from disk + "
      f"{rep['host_bytes']/1e9:.2f} GB pinned")
for h in hs:
    h.evict()
print(f"host RAM available after evict: {free_gb():.1f} GB")

# ---- experts from the arena: 92 engines, ONE slot buffer, ONE tier ----------
from mxfp4_residency import Mxfp4NvmeResidencyK3   # noqa: E402
from nvme_arena import load_index                  # noqa: E402
from nvme_residency import ColdTier                # noqa: E402

index = load_index(ARENA)
tier = ColdTier(ARENA, hot_rows=HOT_ROWS, pinned=True, index=index)
engines = []
try:
    t0 = time.time()
    for name, blk in model.named_modules():
        if not name.endswith("block_sparse_moe"):
            continue
        lay = int(name.split(".layers.")[1].split(".")[0])
        eng = Mxfp4NvmeResidencyK3(
            ARENA, lay, k_slots=TOPK_SLOTS, device=DEV, index=index, tier=tier,
            store=engines[0].store if engines else None)     # <- grouped-nf4-gemm#24
        engines.append(eng)
        blk.moe_infer = (lambda e: (lambda x, ids, w: e.forward(x, ids, w)))(eng)
    st = engines[0].store
    print(f"{len(engines)} MoE engines in {time.time()-t0:.0f}s, sharing {st!r}")
    print(f"  slots: {st.bytes/1e9:.3f} GB shared, vs "
          f"{st.bytes*len(engines)/1e9:.1f} GB if each had its own")
    assert all(e.store is st for e in engines) and st.users == len(engines)
    print(f"VRAM after setup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ---- shared: tokenizer, and the ONE correct projection ----------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get(
        "K3_TOKENIZER", "/workspace/k3-tokenizer"), trust_remote_code=True)
    lm_head = model.language_model.lm_head          # 2.35 GB, deliberately on the host

    def project(h):
        """Hidden -> logits. NO norm here: KimiLinearModel.forward already applied
        self.norm at modeling_kimi_linear.py:1219, so `last_hidden_state` is normed.
        The prefill driver re-applied it, which left the argmax alone (' Paris' both
        ways) while sharpening the distribution from 71.2% to 98.6% -- a wrong number
        that a spot check cannot see. One norm, once."""
        with torch.no_grad():
            return lm_head(h.to("cpu").to(lm_head.weight.dtype)).float()

    def counters():
        ts = tier.stats()
        return {"disk_reads": ts.get("disk_reads", 0), "claims": st.claims,
                "dense_reads": serve_src.stats()["reads"],
                "dense_bytes": serve_src.stats()["read_bytes"]}

    def delta(a, b):
        return {k: b[k] - a[k] for k in a}

    # ---- contract asserts: cheap, and both have silent failure modes ------
    assert TOPK_SLOTS == tc.num_experts_per_token, (
        f"k_slots {TOPK_SLOTS} != num_experts_per_token {tc.num_experts_per_token}: the "
        "engine dispatches on `router_indices.numel() > k`, so a mismatch silently "
        "routes a prefill into the decode path and dies in want_buf.copy_")
    n_kda = 0
    for i in range(tc.num_hidden_layers):
        if tc.is_kda_layer(i):
            a, b = layers[i].self_attn.A_log, layers[i].self_attn.dt_bias
            assert tuple(a.shape) == (96,) and tuple(b.shape) == (12288,), \
                (i, tuple(a.shape), tuple(b.shape))
            n_kda += 1
    print(f"contract: k_slots==top_k=={TOPK_SLOTS}; {n_kda} KDA layers carry "
          f"A_log(96,) + dt_bias(12288,) -- the shape gate's 69 narrows, verified")

    if MODE == "route":
        # Does the cached decode path SELECT DIFFERENT EXPERTS than a fresh prefill?
        # The two paths diverge at cos 0.9996 flat (1 step == 3 steps, so not
        # accumulation). With 16-of-896 routing, a tiny hidden-state difference can flip
        # a selection at a near-tie -- a DISCRETE amplifier that smooth-accumulation
        # models miss. If selections differ, that is the whole explanation and the
        # divergence is intrinsic. If routing is identical and logits still differ,
        # something else is wrong.
        REC = {}

        def _rec(nm, fn):
            def g(x, ids, w):
                REC.setdefault(nm, []).append(ids.detach().reshape(-1, 16).cpu())
                return fn(x, ids, w)
            return g

        for _nm, _blk in model.named_modules():
            if _nm.endswith("block_sparse_moe"):
                _blk.moe_infer = _rec(_nm, _blk.moe_infer)

        prompt = os.environ.get("K3_PROMPT", "The capital city of France is")
        ids = tok(prompt, return_tensors="pt").input_ids
        print(f"\nprompt {prompt!r} -> {ids.shape[1]} tokens")
        with torch.no_grad():
            out = base(inputs_embeds=base.embed_tokens(ids).to(DEV), use_cache=True)
        cache = out.past_key_values
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        nxt = int(project(h[0, -1:])[0].argmax())
        print(f"  prefill done, next token {tok.decode([nxt])!r}")

        REC.clear()                       # record ONLY the cached decode step
        with torch.no_grad():
            base(inputs_embeds=base.embed_tokens(torch.tensor([[nxt]])).to(DEV),
                 past_key_values=cache, use_cache=True)
        cached = {k: v[0][-1].tolist() for k, v in REC.items()}

        REC.clear()                       # and ONLY the fresh prefill's last position
        full = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        with torch.no_grad():
            base(inputs_embeds=base.embed_tokens(full).to(DEV))
        fresh = {k: v[0][-1].tolist() for k, v in REC.items()}

        common = sorted(set(cached) & set(fresh))
        diff_layers, total_diff = [], 0
        for k in common:
            a, b = set(cached[k]), set(fresh[k])
            d = len(a ^ b) // 2
            if d:
                diff_layers.append((k.split(".layers.")[1].split(".")[0], d))
                total_diff += d
        print(f"\n{'='*66}\nROUTING DIFF -- cached decode vs fresh prefill, same position"
              f"\n{'='*66}")
        print(f"  layers compared: {len(common)}")
        print(f"  layers whose expert SET differs: {len(diff_layers)} of {len(common)}")
        print(f"  total expert substitutions: {total_diff} (of {len(common)*16} slots)")
        if diff_layers:
            print(f"  first differing: {diff_layers[:12]}")
            print("\n  -> routing flips are present, which explains divergence larger "
                  "than smooth bf16 accumulation predicts. Intrinsic, not a cache bug.")
        else:
            print("\n  -> routing is IDENTICAL at every layer. The cos 0.9996 is NOT "
                  "explained by expert flips -- investigate the kernels themselves.")
        result = {"mode": "route", "layers": len(common),
                  "layers_differing": len(diff_layers), "substitutions": total_diff,
                  "detail": diff_layers}
    elif MODE == "ppl":
        # ---- perplexity: one prefill, a prediction at every position ------
        # Cheaper per token than generation by a wide margin: the 108.76 GB dense read
        # is paid ONCE for the whole window, and expert reads saturate at the arena's
        # 82,432 rows once 16*T draws cover 896 experts per layer.
        text = os.environ.get("K3_PPL_TEXT") or (
            "The capital city of France is Paris. It sits on the river Seine and has "
            "been the political and cultural centre of the country for centuries. The "
            "city is known for its museums, its boulevards, and the cathedral of Notre "
            "Dame, which stands on an island in the middle of the river. Millions of "
            "visitors arrive each year to see the Louvre and the Eiffel Tower, and the "
            "surrounding region remains the most densely populated part of France.")
        ids = tok(text, return_tensors="pt").input_ids[:, :PPL_TOKENS]
        T = ids.shape[1]
        print(f"\nperplexity window: {T} tokens")
        c0 = counters()
        with torch.no_grad():
            out = base(inputs_embeds=base.embed_tokens(ids).to(DEV))
        torch.cuda.synchronize()
        hs_out = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        dt = time.time() - t_start

        nll, ntok, per_tok, top1 = 0.0, 0, [], []
        for lo in range(0, T - 1, 64):            # chunked: [64, 163840] fp32 at a time
            hi = min(lo + 64, T - 1)
            lg = project(hs_out[0, lo:hi])
            ce = torch.nn.functional.cross_entropy(
                lg, ids[0, lo + 1:hi + 1], reduction="none")
            per_tok += ce.tolist()                # NLL of token t+1 given 0..t, t = lo..hi-1
            top1 += lg.argmax(-1).tolist()
            nll += ce.sum().item()
            ntok += hi - lo
        loss = nll / ntok
        print(f"\n{'='*66}\nPERPLEXITY -- {len(layers)} layers, {T} tokens, "
              f"{ntok} predictions\n{'='*66}")
        print(f"  mean NLL {loss:.4f}   perplexity {math.exp(loss):.3f}")
        print(f"  counters: {json.dumps(delta(c0, counters()))}")
        result = {"mode": "ppl", "tokens": T, "predictions": ntok,
                  "nll": loss, "perplexity": math.exp(loss),
                  "token_ids": ids[0].tolist(), "per_token_nll": per_tok,
                  "top1_ids": top1}
    else:
        # ---- generation: prefill, then N-1 cached decode steps ------------
        # Every MoE call's (expert ids, weights) is RECORDED along the cached path, so the
        # cache check below can REPLAY exactly that routing into a fresh prefill. Top-16 of
        # 896 in bf16 flips on near-ties between the chunked and recurrent kernels (the route
        # mode found 45-47 of 92 layers differ), and each flip swaps a whole expert, so an
        # unreplayed comparison measures the tie-break process, not the cache. With routing
        # replayed, what is left is the kernels and the cache -- which is what the gate is for.
        ROUTE = {"mode": "off", "rec": {}, "replay": {}, "seen": {}}

        def _route_hook(nm, fn):
            def g(x, ids, w):
                m = ROUTE["mode"]
                if m == "rec":
                    ROUTE["rec"].setdefault(nm, []).append(
                        (ids.detach().cpu().clone(), w.detach().cpu().clone()))
                elif m == "replay":
                    rid, rw = ROUTE["replay"][nm]
                    if tuple(rid.shape) != tuple(ids.shape):
                        raise SystemExit(f"replay shape {tuple(rid.shape)} != {tuple(ids.shape)}"
                                         f" at {nm}: the recorded path and this forward "
                                         "do not cover the same positions")
                    ROUTE["seen"][nm] = ids.detach().cpu().clone()   # what THIS router chose
                    ids = rid.to(ids.device, ids.dtype)
                    w = rw.to(w.device, w.dtype)
                elif m == "seen":
                    ROUTE["seen"][nm] = ids.detach().cpu().clone()
                return fn(x, ids, w)
            return g

        for _nm, _blk in model.named_modules():
            if _nm.endswith("block_sparse_moe"):
                _blk.moe_infer = _route_hook(_nm, _blk.moe_infer)

        def route_diff(a, b):
            """Per layer, per row: how many of the 16 expert slots differ as SETS."""
            layers, subs, rows = 0, 0, 0
            for k in sorted(set(a) & set(b)):
                ra, rb = a[k].reshape(-1, 16), b[k].reshape(-1, 16)
                d = 0
                for i in range(ra.shape[0]):
                    di = len(set(ra[i].tolist()) ^ set(rb[i].tolist())) // 2
                    d += di
                    rows += bool(di)
                layers += bool(d)
                subs += d
            return {"layers_differing": layers, "substitutions": subs,
                    "rows_differing": rows, "layers": len(set(a) & set(b))}

        prompt = os.environ.get("K3_PROMPT", "The capital city of France is")
        CONTROL = os.environ.get("K3_CONTROL", "")          # "" | "drop_state"
        ids = tok(prompt, return_tensors="pt").input_ids
        print(f"\nprompt {prompt!r} -> {ids.shape[1]} tokens {ids[0].tolist()}"
              + (f"   CONTROL={CONTROL}" if CONTROL else ""))
        c0 = counters()
        ROUTE["mode"] = "rec"
        t0 = time.time()
        with torch.no_grad():
            out = base(inputs_embeds=base.embed_tokens(ids).to(DEV), use_cache=True)
        torch.cuda.synchronize()
        prefill_s = time.time() - t0
        ROUTE["mode"] = "off"
        hs_out = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        cache = out.past_key_values
        if cache is None:
            raise SystemExit("prefill returned no cache -- check text_config.use_cache")
        prefill_counters = delta(c0, counters())
        print(f"prefill {ids.shape[1]} tokens in {prefill_s:.1f}s   "
              f"cache seq_len {cache.get_seq_length()}   "
              f"counters {json.dumps(prefill_counters)}")

        # A second, identical prefill in the same process separates one-time costs (Triton
        # JIT compiles on a fresh cache, cold ARC) from the steady-state prefill. Its output
        # is discarded and it is not recorded.
        prefill_repeat_s = None
        if os.environ.get("K3_PREFILL_REPEAT", "0") == "1":
            t0 = time.time()
            with torch.no_grad():
                base(inputs_embeds=base.embed_tokens(ids).to(DEV), use_cache=True)
            torch.cuda.synchronize()
            prefill_repeat_s = time.time() - t0
            print(f"prefill repeat (warm, discarded): {prefill_repeat_s:.1f}s")

        # The prefill hands you the FIRST new token for free (position T-1). N new
        # tokens therefore needs N-1 decode forwards, not N.
        logits = project(hs_out[0, -1:])
        nxt = int(logits[0].argmax())
        p1 = torch.softmax(logits[0], -1).max().item()
        gen, per_step, kept, probs = [nxt], [], [logits[0].clone()], [p1]
        print(f"  token 1/{NEW}: {tok.decode([nxt])!r} (p={p1*100:.2f}%) -- free from prefill")

        for i in range(1, NEW):
            if CONTROL == "drop_state" and i == NEW - 1:
                # NEGATIVE CONTROL: a cache that lost every KDA layer's recurrent state.
                # The gate below must FAIL on this; if it passes, it cannot see a cache bug.
                n_drop = 0
                for _li, _st in enumerate(cache.recurrent_states):
                    if _st is not None:
                        _st.zero_()
                        n_drop += 1
                print(f"  CONTROL drop_state: zeroed the recurrent state of {n_drop} KDA layers")
            seq_before = cache.get_seq_length()
            c = counters()
            ROUTE["mode"] = "rec"
            t0 = time.time()
            with torch.no_grad():
                out = base(inputs_embeds=base.embed_tokens(
                    torch.tensor([[gen[-1]]])).to(DEV),
                    past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            step_s = time.time() - t0
            ROUTE["mode"] = "off"
            cache = out.past_key_values
            seq_after = cache.get_seq_length()
            # A cache that silently attends to the wrong positions produces FLUENT
            # NONSENSE with no error -- cache_position=[0] against 6 cached yields the
            # same mask SHAPE as the correct [6], with 1 of 7 positions unmasked. So
            # assert the length actually advanced.
            assert seq_after == seq_before + 1, (i, seq_before, seq_after)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            lg = project(h[0, -1:])
            nxt = int(lg[0].argmax())
            pi = torch.softmax(lg[0], -1).max().item()
            gen.append(nxt)
            kept.append(lg[0].clone())
            probs.append(pi)
            d = delta(c, counters())
            per_step.append({"s": step_s, **d})
            print(f"  token {i+1}/{NEW}: {tok.decode([nxt])!r} "
                  f"(p={pi*100:.2f}%)  "
                  f"{step_s:.1f}s  expert_rows {d['disk_reads']}  "
                  f"dense {d['dense_bytes']/1e9:.1f} GB  claims {d['claims']}")

        print(f"\n{'='*66}\nGENERATED: {tok.decode(gen)!r}\n{'='*66}")
        print(f"  ids {gen}")
        if per_step:
            print(f"  decode step: median {statistics.median([p['s'] for p in per_step]):.1f}s, "
                  f"{per_step[0]['dense_bytes']/1e9:.1f} GB dense + "
                  f"{per_step[0]['disk_reads']} expert rows each")

        result = {"mode": "gen", "prompt": prompt, "generated": gen,
                  "text": tok.decode(gen), "probs": probs, "prefill_s": prefill_s,
                  "prefill_repeat_s": prefill_repeat_s, "prefill_counters": prefill_counters,
                  "per_step": per_step, "control": CONTROL or None}

        # ---- the cache gate: ROUTING FIRST, then logits under replayed routing ----
        # The cached path's last logits against a fresh, cache-free prefill of the same
        # sequence. Two comparisons:
        #   free     -- the fresh prefill routes for itself (the old check; reported, NOT
        #               gated: its gap is dominated by discrete routing flips);
        #   replayed -- the fresh prefill is handed the cached path's own (ids, weights)
        #               at every MoE layer and position. Routing is then identical by
        #               construction, so the logits must agree to kernel precision.
        # GATE, fixed before any replayed number existed: cos >= 0.9999 AND argmax agrees.
        # Reference points (July): a correct single KDA layer, chunked vs recurrent, cos
        # 0.999983; a cache with the recurrent state DROPPED, free routing, cos 0.9957.
        GATE_COS = 0.9999
        if VERIFY_CACHE and len(gen) > 1:
            full = torch.cat([ids, torch.tensor([gen[:-1]])], dim=1)
            cl = kept[-1]
            rec = {k: (torch.cat([a for a, _ in v], 0), torch.cat([b for _, b in v], 0))
                   for k, v in ROUTE["rec"].items()}
            n_rows = {k: v[0].shape[0] for k, v in rec.items()}
            if set(n_rows.values()) != {full.shape[1]}:
                raise SystemExit(f"recorded rows {sorted(set(n_rows.values()))} != "
                                 f"{full.shape[1]} positions in the fresh prefill")

            def compare(tag):
                with torch.no_grad():
                    ref = base(inputs_embeds=base.embed_tokens(full).to(DEV))
                rh = ref.last_hidden_state if hasattr(ref, "last_hidden_state") else ref[0]
                rl = project(rh[0, -1:])[0]
                cos = torch.nn.functional.cosine_similarity(rl[None], cl[None]).item()
                rel = ((rl - cl).abs().max() / rl.abs().max()).item()
                agree = int(rl.argmax()) == int(cl.argmax())
                print(f"  {tag:8s} cos {cos:.6f}   rel {rel:.3e}   argmax agrees: {agree}")
                return {"cos": cos, "rel": rel, "argmax_agrees": agree}

            print(f"\ncache gate: fresh prefill of all {full.shape[1]} tokens (no cache) vs "
                  f"the cached path's last step")
            gate = {}
            if os.environ.get("K3_VERIFY_FREE", "1") == "1":
                ROUTE["seen"] = {}
                ROUTE["mode"] = "seen"
                gate["free"] = compare("free")
                ROUTE["mode"] = "off"
                gate["free"]["routing"] = route_diff(
                    {k: v[0] for k, v in rec.items()}, ROUTE["seen"])
                print(f"           routing vs the cached path: {gate['free']['routing']}")
            ROUTE["replay"] = rec
            ROUTE["seen"] = {}
            ROUTE["mode"] = "replay"
            gate["replayed"] = compare("replayed")
            ROUTE["mode"] = "off"
            # What the fresh router WOULD have chosen, layer by layer, given the replayed
            # routing upstream: local near-tie flips, not their downstream cascade.
            gate["replayed"]["local_flips"] = route_diff(
                {k: v[0] for k, v in rec.items()}, ROUTE["seen"])
            print(f"           local flips the replay overrode: "
                  f"{gate['replayed']['local_flips']}")
            r = gate["replayed"]
            gate["threshold_cos"] = GATE_COS
            gate["verdict"] = "PASS" if (r["cos"] >= GATE_COS and r["argmax_agrees"]) else "FAIL"
            print(f"  GATE (replayed routing, cos >= {GATE_COS} and argmax agrees): "
                  f"{gate['verdict']}")
            result["cache_gate"] = gate

    print(f"\n  VRAM peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  dense policy: {rep['host_bytes']/1e9:.2f} GB pinned + "
          f"{rep['disk_bytes']/1e9:.2f} GB disk")
    print(f"  host RAM available: {free_gb():.1f} GB")
    print(f"  total wall {time.time()-t_start:.0f}s")
    result.update({"vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "pinned_gb": rep["host_bytes"] / 1e9,
                   "disk_gb": rep["disk_bytes"] / 1e9,
                   "wall_s": time.time() - t_start})
    import importlib.metadata as _md, platform, hashlib
    def _v(p):
        try:
            return _md.version(p)
        except Exception:
            return None
    result["provenance"] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "packages": {p: _v(p) for p in ("experts4bit-qlora", "grouped-nf4-gemm", "torch",
                                        "triton", "transformers", "tiktoken", "bitsandbytes")},
        "python": platform.python_version(), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "driver_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "arena": ARENA, "dense": DENSE, "snapshot": SNAP,
        "env": {k: v for k, v in os.environ.items() if k.startswith("K3_")}}
    out = os.path.join(os.environ.get("K3_OUT", "/workspace/k3-rel-20260928"),
                       f"k3_{MODE}_n{NEW if MODE == 'gen' else 0}_pin{int(PIN_GB)}"
                       f"{('_' + os.environ['K3_CONTROL']) if os.environ.get('K3_CONTROL') else ''}.json")
    json.dump(result, open(out, "w"), indent=1)
    print(f"  saved -> {out}")
finally:
    tier.close()
    serve_src.close()
