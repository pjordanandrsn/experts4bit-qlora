# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Gate G1 harness — per-layer CPU-router round trip at batch 1.

Three arms:

  replica   M synthetic decoder layers (attention stand-in -> router ->
            expert stand-in that CONSUMES the device index tensor). Every
            non-router op is sync-free by construction, so any blocking
            call nsys attributes to the decode loop belongs to the router.
            This is the arm the G1 numbers come from.
  legacy    identical replica, unpatched reference router + the classic
            ids.cpu().tolist() consume before the expert kernel — the
            #105/#108 stall class Phase 1 kills, measured side by side.
  model     real OLMoE decode (transformers, bf16) with enable_cpu_router:
            served-call trip stats in a real forward. The expert path's own
            syncs are NOT the router's and are reported separately.

Run under nsys for the sync audit:
  nsys profile -t cuda,osrt -o g1_trace python bench/bench_cpu_router.py --arm replica
  nsys stats --report cuda_api_sum g1_trace.nsys-rep

Outputs one JSON receipt per arm (--out).
"""

import argparse
import json
import time

import torch

from experts4bit_qlora.engines import cpu_router as cr


class OlmoeTopKRouter(torch.nn.Module):
    """Reference-math router at OLMoE geometry; class name drives dispatch."""

    def __init__(self, E, H, k, device, dtype):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = k, E, H
        self.norm_topk_prob = False
        self.weight = torch.nn.Parameter(
            torch.randn(E, H, device=device, dtype=dtype) * 0.02
        )

    def forward(self, hidden_states):
        h = hidden_states.reshape(-1, self.hidden_dim)
        logits = torch.nn.functional.linear(h, self.weight)
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        val, idx = torch.topk(probs, self.top_k, dim=-1)
        return logits, val.to(logits.dtype), idx


class Replica(torch.nn.Module):
    """One synthetic decode layer: 2-matmul attention stand-in, router,
    expert stand-in reading the device index tensor (dependency is real)."""

    def __init__(self, E, H, k, device, dtype):
        super().__init__()
        self.router = OlmoeTopKRouter(E, H, k, device, dtype)
        self.wa = torch.nn.Parameter(torch.randn(H, H, device=device, dtype=dtype) * 0.02)
        self.wb = torch.nn.Parameter(torch.randn(H, H, device=device, dtype=dtype) * 0.02)
        self.emb = torch.nn.Parameter(torch.randn(E, H, device=device, dtype=dtype) * 0.02)

    def forward(self, h, legacy=False):
        h = torch.tanh(h @ self.wa) @ self.wb          # "attention"
        _, w, idx = self.router(h)
        if legacy:
            ids = idx.cpu().tolist()                    # the killed pattern
            idx = torch.tensor(ids, device=h.device).reshape(idx.shape)
        picked = self.emb.index_select(0, idx.reshape(-1))
        picked = picked.reshape(idx.shape[0], idx.shape[1], -1)
        return h + (picked * w.unsqueeze(-1).to(picked.dtype)).sum(dim=1)


def run_replica(args, legacy):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    layers = [Replica(args.experts, args.hidden, args.topk, dev, torch.bfloat16)
              for _ in range(args.layers)]
    model = torch.nn.ModuleList(layers)
    if not legacy:
        n = cr.enable_cpu_router(model, max_rows=8, timing=not args.audit)
        assert n == args.layers, f"patched {n} != {args.layers}"
    h = torch.randn(args.batch, args.hidden, device=dev, dtype=torch.bfloat16)
    step_ms = []
    with torch.no_grad():
        if args.audit:
            # sync-audit mode for nsys: NO host synchronization anywhere in
            # the token loop — any blocking call in the trace belongs to
            # the code under test, not the harness
            for _ in range(args.tokens):
                x = h
                for lay in layers:
                    x = lay(x, legacy=legacy)
            torch.cuda.synchronize()                    # once, at the end
            step_ms.append(0.0)
        else:
            for t in range(args.tokens + args.warmup):
                torch.cuda.synchronize()                # step boundary only
                t0 = time.perf_counter()
                x = h
                for lay in layers:
                    x = lay(x, legacy=legacy)
                torch.cuda.synchronize()
                if t >= args.warmup:
                    step_ms.append((time.perf_counter() - t0) * 1e3)
    out = {"arm": "legacy" if legacy else "replica",
           "layers": args.layers, "tokens": args.tokens, "batch": args.batch,
           "geometry": {"E": args.experts, "H": args.hidden, "k": args.topk}}
    step_ms.sort()
    out["step_ms_p50"] = step_ms[len(step_ms) // 2]
    out["per_layer_us_p50"] = out["step_ms_p50"] * 1e3 / args.layers
    if not legacy:
        out["router"] = cr.router_trip_stats(model)
    return out


def run_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    # transformers' grouped-mm chooser admits capability >= 9.0 but the op
    # itself accepts ONLY 9.0 — on sm_120 the expert forward crashes.
    # Pin the eager experts path; the router (what this arm measures) is
    # independent of the experts implementation.
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "eager"
    n = cr.enable_cpu_router(model, max_rows=8, timing=True)
    ids = tok("The three tiers of the memory hierarchy are",
              return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False)
    text = tok.decode(out[0][-16:])
    stats = cr.router_trip_stats(model)
    return {"arm": "model", "model": args.model, "patched": n,
            "router": stats, "tail_text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["replica", "legacy", "model"],
                    default="replica")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--audit", action="store_true",
                    help="no harness syncs in the loop (for nsys)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.arm == "model":
        res = run_model(args)
    else:
        res = run_replica(args, legacy=(args.arm == "legacy"))
    res["torch"] = torch.__version__
    res["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
