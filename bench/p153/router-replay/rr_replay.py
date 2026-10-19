"""p153-w16-router-replay ($0, A2000, correctness and accounting only; no timings): bound P153's unattributed W16 cost's
token-divergence term.

At W16 the int8 head is passed at every bucket-16 step, so S1's W16 steps differ from S0's only in the TOKENS they
decode (the head is rounded at every row count, and 12 of 16 rows diverge by 160 tokens). On an MoE, different tokens
mean different router picks, so different expert work per step. The lm_head is downstream of every router, so both
arms' recorded streams go through the SAME model, teacher-forced, and only the tokens differ.

- **Streams:** p153-5090-1's public arm records. Each row is the 512-token P109 prompt plus that arm's recorded 160-token
  W16 continuation (gen[0..158] as inputs; gen[0] came from the prefill).
- **Model:** Qwen/Qwen3-30B-A3B @ad44e777 in bf16, streamed LAYER BY LAYER from the LAN model store, mounted READ-ONLY.
  Each shard is sha256-checked against the store's manifest when first opened, and its handle is closed after its last
  layer. Nothing is copied or deleted, and one decoder layer (~1.2 GiB) is resident on the GPU at a time.
- **The read:** for each decode step whose cost the W16 slope counts (inputs gen[31..158], the steps producing tokens
  33..160), each layer's DISTINCT experts across the 16 rows' top-8 picks, for S0 and for S1. A grouped expert GEMM at
  16 rows reads each distinct expert's weights once, so S1/S0 of that count bounds the term.
- **Fidelity check (reported):** the teacher-forced argmax at each decode input vs the recorded next token. The replay
  is bf16, and the served model is NF4.

usage: python rr_replay.py --s0 arm_S0a.json --s1 arm_S1a.json --prompts prompts.json --model DIR --manifest SHA256 \
       --out rr.json
"""
import argparse
import hashlib
import json
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig
import transformers.models.qwen3_moe.modeling_qwen3_moe as qm

ap = argparse.ArgumentParser()
ap.add_argument("--s0", required=True)
ap.add_argument("--s1", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--model", required=True, help="the model store's checkpoint directory (read-only)")
ap.add_argument("--manifest", required=True, help="the store's sha256 manifest for that checkpoint")
ap.add_argument("--out", required=True)
ap.add_argument("--rev", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
ap.add_argument("--first", type=int, default=31, help="first decode input index counted (gen[31] -> token 33)")
a = ap.parse_args()
man_text = open(a.manifest, encoding="utf-8").read()
assert f"# revision:  {a.rev}" in man_text, "the manifest is not for the registered revision"
want_sha = {ln.split()[1]: ln.split()[0] for ln in man_text.splitlines() if ln and not ln.startswith("#")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


for name in ("config.json", "model.safetensors.index.json"):
    assert sha256(f"{a.model}/{name}") == want_sha[name], name
dev = torch.device("cuda")
torch.set_grad_enabled(False)

prompts = json.load(open(a.prompts))["rows"]
streams = {}
for arm, path in (("S0", a.s0), ("S1", a.s1)):
    gen = json.load(open(path))["workloads"]["W16"]["tokens"]["160"]
    assert len(gen) == len(prompts) == 16 and all(len(g) == 160 for g in gen)
    streams[arm] = gen
P = len(prompts[0])
assert all(len(p) == P for p in prompts)
ids = torch.tensor([prompts[r] + streams[arm][r][:159] for arm in ("S0", "S1") for r in range(16)], dtype=torch.long)
B, T = ids.shape                                                  # 32 x (P + 159)
dec = list(range(P - 1, T))                                       # positions whose output is gen[k]: P-1 -> gen[0]
counted = [P + k for k in range(a.first, 159)]                    # decode inputs gen[first..158] at positions P+k

cfg = AutoConfig.from_pretrained(a.model)
cfg._attn_implementation = "sdpa"
cfg._experts_implementation = "eager"         # a standalone layer: the plain per-expert loop
L, E, K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
index = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]


def layer_of(name):
    if name.startswith("model.layers."):
        return int(name.split(".")[2])
    return -1 if name.startswith("model.embed_tokens") else L    # embeddings first; norm and lm_head last


last_use = {}
for name, shard in index.items():
    last_use[shard] = max(last_use.get(shard, -2), layer_of(name))
opened, verified = {}, []


def tensor(name):
    shard = index[name]
    if shard not in opened:
        t0 = time.time()
        path = f"{a.model}/{shard}"
        got = sha256(path)
        assert got == want_sha[shard], f"{shard}: sha256 {got} != the manifest's {want_sha[shard]}"
        opened[shard] = safe_open(path, framework="pt", device="cpu")
        verified.append({"shard": shard, "sha256": got, "hash_s": round(time.time() - t0, 1)})
        print(f"RR shard {shard} sha256 OK", flush=True)
    return opened[shard].get_tensor(name)


def release(upto):
    """Close (never delete: the store is read-only) every shard whose last needed layer is <= upto."""
    for shard in [s for s in opened if last_use[s] <= upto]:
        del opened[shard]


h = torch.nn.functional.embedding(ids.to(dev), tensor("model.embed_tokens.weight").to(dev, torch.bfloat16))
release(-1)
rot = qm.Qwen3MoeRotaryEmbedding(cfg).to(dev)
pos = torch.arange(T, device=dev)[None].expand(B, -1)
cos_sin = rot(h, pos)
picks = torch.empty(L, B, len(counted), K, dtype=torch.int16)
cidx = torch.tensor(counted, device=dev)

for li in range(L):
    pre = f"model.layers.{li}."
    with torch.device("meta"):
        layer = qm.Qwen3MoeDecoderLayer(cfg, li)
    layer = layer.to_empty(device=dev).to(torch.bfloat16)
    sd = {k[len(pre):]: tensor(k) for k in index if k.startswith(pre) and ".mlp.experts." not in k}
    sd["mlp.experts.gate_up_proj"] = torch.stack([torch.cat([tensor(f"{pre}mlp.experts.{e}.gate_proj.weight"),
                                                             tensor(f"{pre}mlp.experts.{e}.up_proj.weight")], 0)
                                                  for e in range(E)])
    sd["mlp.experts.down_proj"] = torch.stack([tensor(f"{pre}mlp.experts.{e}.down_proj.weight") for e in range(E)])
    missing, unexpected = layer.load_state_dict({k: v.to(dev, torch.bfloat16) for k, v in sd.items()}, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    del sd
    got = {}
    hook = layer.mlp.gate.register_forward_hook(lambda mod, inp, out: got.__setitem__("idx", out[2]))
    h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
    hook.remove()
    idx = got["idx"].reshape(B, T, K)
    picks[li] = idx[:, cidx].to(torch.int16).cpu()
    del layer
    torch.cuda.empty_cache()
    release(li)
    print(f"RR layer {li} done", flush=True)

# fidelity: the teacher-forced argmax at the decode positions vs the recorded next tokens
norm =qm.Qwen3MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev)
norm.weight.data = tensor("model.norm.weight").to(dev, torch.bfloat16)
head_w = tensor("lm_head.weight").to(dev, torch.bfloat16)
release(L)
hd = norm(h[:, dec])
agree = {}
for ai, arm in enumerate(("S0", "S1")):
    am = torch.cat([(hd[ai * 16:(ai + 1) * 16, j:j + 32] @ head_w.T).argmax(-1) for j in range(0, len(dec), 32)], 1).cpu()
    want = torch.tensor(streams[arm], dtype=torch.long)                              # gen[0..159] at dec positions
    agree[arm] = float((am == want).float().mean())

torch.save({"picks": picks, "agree": agree, "counted": counted}, a.out + ".pt")     # the expensive part, kept first

# the read: distinct experts per (layer, step) across the 16 rows, per arm
distinct = {}
for ai, arm in enumerate(("S0", "S1")):
    pk = picks[:, ai * 16:(ai + 1) * 16].long()                                       # [L, 16, S, K]
    one = torch.zeros(L, len(counted), E, dtype=torch.bool)
    for r in range(16):
        one.scatter_(2, pk[:, r], True)
    distinct[arm] = one.sum(-1).float()                                               # [L, S]
d0, d1 = distinct["S0"], distinct["S1"]
out = {
    "model": f"Qwen/Qwen3-30B-A3B@{a.rev}", "dtype": "bfloat16 (the served model is NF4; both arms identical)",
    "rows": 16, "prompt_tokens": P, "steps_counted": len(counted), "first_counted_input": a.first,
    "layers": L, "experts": E, "top_k": K,
    "distinct_mean": {"S0": float(d0.mean()), "S1": float(d1.mean())},
    "distinct_ratio_S1_over_S0": float(d1.sum() / d0.sum()),
    "per_layer_ratio": [float(x) for x in (d1.sum(1) / d0.sum(1))],
    "per_step_ratio_quantiles": [float(x) for x in torch.quantile(d1.sum(0) / d0.sum(0), torch.tensor([0.05, 0.5, 0.95]))],
    "steps_S1_more": int((d1.sum(0) > d0.sum(0)).sum()), "steps_S1_fewer": int((d1.sum(0) < d0.sum(0)).sum()),
    "teacher_forced_argmax_agreement": agree,
    "rows_identical_S1_S0": sum(x == y for x, y in zip(streams["S0"], streams["S1"])),
    "shards_verified": verified, "peak_gpu_mib": int(torch.cuda.max_memory_allocated() / 2**20),
}
json.dump(out, open(a.out, "w"), indent=1)
print("RR_DONE " + json.dumps({k: out[k] for k in ("distinct_mean", "distinct_ratio_S1_over_S0",
                                                   "teacher_forced_argmax_agreement", "steps_S1_more",
                                                   "steps_S1_fewer", "peak_gpu_mib")}), flush=True)
