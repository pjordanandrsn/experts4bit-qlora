# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""A bf16 reference forward for a MoE checkpoint that does not fit the device: one decoder layer resident at a time.

Q235's fidelity reference (bench/q235/PREREG-q235.md). The model is transformers' own modeling, built on ``meta``; each layer is
assembled on the host from the checkpoint's shards, copied to the device, applied, and released, so the device holds one layer's
weights (about 5 GB at Qwen3-235B-A22B or Mixtral-8x22B, 1.25 GB at Qwen3-30B-A3B) plus the activations. Each checkpoint key is
converted exactly as ``from_pretrained`` converts it, by transformers' own conversion mapping (qwen2_moe's rule for Qwen3: each
expert's gate and up stacked, then concatenated along dim 1, down stacked; mixtral's: ``.block_sparse_moe.`` renamed to ``.mlp.``,
w1/w3/w2 as gate/up/down). That way the layer computes what the resident model computes. The mapping is never e4b's: a reference
that borrowed the system under test's mapping would agree with it on a shared mistake, such as a swapped gate/up role.

``--resident`` loads every layer first and keeps them (the depth-truncated resident reference, ``--layers k``); the forward is the
same code in the same order, so streamed and resident are expected to agree bit for bit on one device.

Records for the given token ids: each layer's output (a file per layer), each layer's router top-k sets, and the per-token loss
(transformers' causal-LM convention: labels shifted by one, -100 ignored). With ``--layers k`` the head is applied after layer k.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask


class Shards:
    """Read tensors by name from a safetensors checkpoint (the index's shards), one tensor at a time."""

    def __init__(self, path):
        from safetensors import safe_open
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.where = json.load(open(idx))["weight_map"]
        else:
            self.where = {k: "model.safetensors" for k in safe_open(os.path.join(path, "model.safetensors"), "pt").keys()}
        self._open = {}
        self._safe_open = safe_open

    def get(self, name, device):
        f = self.where[name]
        if f not in self._open:
            self._open = {f: self._safe_open(os.path.join(self.path, f), "pt", device="cpu")}   # one shard open at a time
        return self._open[f].get_tensor(name).to(device)

    def has(self, name):
        return name in self.where


class Conversions:
    """transformers' checkpoint-to-module conversion for ``model``: the renamings and converters ``from_pretrained`` applies
    (``get_model_conversion_mapping``), split as ``core_model_loading`` splits them, with the meta state dict the loader checks
    renamed keys against."""

    def __init__(self, model):
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming
        mapping = get_model_conversion_mapping(model)
        self.renamings = [m for m in mapping if isinstance(m, WeightRenaming)]
        self.converters = [m for m in mapping if isinstance(m, WeightConverter)]
        self.by_pattern = {p: c for c in self.converters for p in c.source_patterns}
        self.meta = model.state_dict()
        self.prefix = model.base_model_prefix
        self.model = model


def layer_state(shards, i, conv, device, dtype):
    """Layer ``i``'s state dict in the modeling's layout, converted from the checkpoint's own spelling as ``from_pretrained`` converts
    it (``conv``: :class:`Conversions`). Keys are taken in ``dot_natural_key`` order and renamed by ``rename_source_key``. The keys a
    ``WeightConverter`` claims (per-expert gate/up/down) are gathered into that converter's own operations (stack, then gate before
    up). A key that lands nowhere in the module tree raises."""
    from copy import deepcopy
    from transformers.core_model_loading import dot_natural_key, rename_source_key
    pre = f"model.layers.{i}."
    sd, pending = {}, {}
    for key in sorted((k for k in shards.where if k.startswith(pre)), key=dot_natural_key):
        name, pattern = rename_source_key(key, conv.renamings, conv.converters, conv.prefix, conv.meta)
        if name not in conv.meta and key in conv.meta:                    # the loader's fallback: keep the key as it is
            name, pattern = key, None
        if not name.startswith(pre) or name not in conv.meta:
            raise KeyError(f"checkpoint key {key!r} converts to {name!r}, which the module tree does not have")
        tensor = shards.get(key, device)
        if pattern is None:
            sd[name[len(pre):]] = tensor.to(dtype)
        else:
            pending.setdefault(name, deepcopy(conv.by_pattern[pattern])).add_tensor(name, key, pattern, tensor)
    for name, converter in pending.items():
        for full, tensor in converter.convert(name, model=conv.model, config=conv.model.config).items():
            sd[full[len(pre):]] = tensor.to(dtype)
    return sd


def run(path, token_ids, attention_mask, labels, device, *, layers=None, resident=False, out_dir=None, dtype=torch.bfloat16):
    cfg = AutoConfig.from_pretrained(path)
    n = cfg.num_hidden_layers if layers is None else int(layers)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    shards = Shards(path)
    conv = Conversions(model)
    dev = torch.device(device)
    t0 = time.time()
    emb = shards.get("model.embed_tokens.weight", dev).to(dtype)
    ids, am = token_ids.to(dev), attention_mask.to(dev)
    h = torch.nn.functional.embedding(ids, emb)
    del emb
    L = ids.shape[1]
    cache_position = torch.arange(L, device=dev)
    position_ids = cache_position.unsqueeze(0).expand(ids.shape[0], -1)
    rotary = type(model.model.rotary_emb)(config=cfg).to(dev)
    pe = rotary(h, position_ids)
    mask = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=am, past_key_values=None, position_ids=position_ids)
    topk, layer_sha = [], []

    def load(i):
        layer = model.model.layers[i]
        layer.to_empty(device=dev)
        # assembled on the host and copied in, so the device never holds a second (stacked) copy of the layer's experts
        missing, unexpected = layer.load_state_dict(layer_state(shards, i, conv, "cpu", dtype), strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        return layer

    if resident:
        for i in range(n):
            load(i)
    with torch.no_grad():
        for i in range(n):
            layer = model.model.layers[i] if resident else load(i)
            got = {}
            hk = layer.mlp.gate.register_forward_hook(lambda mod, inp, out: got.__setitem__("idx", out[2].detach().cpu()))
            h = layer(h, attention_mask=mask, position_ids=position_ids, position_embeddings=pe)
            if isinstance(h, tuple):
                h = h[0]
            hk.remove()
            topk.append(got["idx"])
            hb = h.detach().to("cpu").contiguous()
            layer_sha.append(hashlib.sha256(hb.view(torch.uint8).numpy().tobytes()).hexdigest())
            if out_dir:
                torch.save(hb, os.path.join(out_dir, f"layer{i:03d}.pt"))
            if not resident:
                layer.to("meta")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        norm_w = shards.get("model.norm.weight", dev).to(dtype)
        model.model.norm.to_empty(device=dev)
        model.model.norm.weight.data.copy_(norm_w)
        h = model.model.norm(h)
        head = shards.get("lm_head.weight" if shards.has("lm_head.weight") else "model.embed_tokens.weight", dev).to(dtype)
        losses = []
        lab = labels.to(dev)
        for b in range(h.shape[0]):                                         # one sequence of logits at a time
            logits = torch.nn.functional.linear(h[b], head).float()
            tgt = lab[b, 1:]
            lp = torch.nn.functional.cross_entropy(logits[:-1], tgt, ignore_index=-100, reduction="none")
            losses.append(lp.cpu())
        per_token = torch.stack(losses)
    valid = (labels[:, 1:] != -100)
    mean = float(per_token[valid].mean())
    return {"layers": n, "resident": resident, "device": str(dev), "dtype": str(dtype), "loss": mean,
            "per_token_loss": per_token, "router_topk": topk, "layer_sha256": layer_sha, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("tokens", help="a torch file with input_ids, attention_mask, labels")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t = torch.load(a.tokens, weights_only=True)
    os.makedirs(a.out, exist_ok=True)
    r = run(a.checkpoint, t["input_ids"], t["attention_mask"], t["labels"], a.device, layers=a.layers, resident=a.resident, out_dir=a.out)
    torch.save({"per_token_loss": r.pop("per_token_loss"), "router_topk": r.pop("router_topk")}, os.path.join(a.out, "reference.pt"))
    r["torch"] = torch.__version__
    json.dump(r, open(os.path.join(a.out, "reference.json"), "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("layers", "resident", "device", "loss", "seconds")}))


if __name__ == "__main__":
    main()
