"""P139's numerics premise ($0): K33 (``_gemv_nf4_bw``) at Mixtral-8x7B's expert shapes, every plan in K33's list.

Shapes: gate_up (28672, 4096) and down (4096, 14336), 8 experts, 2 routed rows (Mixtral's top 2 at one request). At
each plan, K33's own contract (``kernel/k33_bench.py``): ``bw_prmt32`` bitwise ``bw_tree``, and each within the fp32
reference's tolerance (error <= 1.05x the scalar route's, or <= 2^-9 of the output scale). Also: the served W1 call's
token-row form (``gather_div=2``: one token row read for both slots) bitwise its expanded call, and whether bw is
bitwise the scalar route (reported). No timing.

    python k33_mixtral_numerics.py OUT.json            # CUDA
    TRITON_INTERPRET=1 python k33_mixtral_numerics.py OUT.json --interpret   # CPU, the default plan only
"""
import json
import os
import sys
import time

import torch

import nf4_grouped as ng

SHAPES = {"gate_up": (28672, 4096), "down": (4096, 14336)}
E, ROWS = 8, 2
TOL_X, TOL_ABS = 1.05, 2.0 ** -9
PLANS = [(16, 256, 4, 1), (16, 512, 4, 1), (16, 1024, 4, 1), (16, 256, 8, 1), (16, 512, 8, 1),
         (32, 256, 4, 1), (32, 512, 4, 1), (32, 512, 8, 1), (32, 1024, 8, 1), (16, 256, 4, 2), (16, 512, 4, 2),
         (32, 256, 4, 2)]


def env(**kv):
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main(out_path, interpret=False):
    dev = "cpu" if interpret else "cuda"
    plans = [ng._BW_PLAN_DEFAULT] if interpret else PLANS
    g = torch.Generator(device=dev).manual_seed(139)
    rec = {"device": "interpreter" if interpret else torch.cuda.get_device_name(),
           "gnf4": getattr(ng, "__version__", None), "gnf4_file": ng.__file__, "shapes": SHAPES, "experts": E,
           "rows": ROWS, "plans": [list(p) for p in plans], "by_shape": {}}
    t0 = time.time()
    for name, (N, K) in SHAPES.items():
        B = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g, device=dev)
        A = torch.rand(E, N, K // 64, generator=g, device=dev) + 0.5
        ids = torch.randperm(E, generator=g, device=dev)[:ROWS].to(torch.int32)
        x = torch.randn(ROWS, K, generator=g, device=dev).to(torch.bfloat16)
        env(GNF4_GEMV_BW="0", GNF4_GEMV_DOTPAD="0", GNF4_PDL="0", GNF4_GEMV_BW_DECODE=None)
        scalar = ng.gemm_4bit_grouped(x, B, A, [1] * ROWS, ids)
        ref = torch.stack([ng.dequant_ref(B[e], A[e], N, K).float() @ x[r].float()
                           for r, e in enumerate(ids.tolist())])
        scale = ref.abs().max().clamp_min(1e-6)
        err_scalar = ((scalar.float() - ref).abs() / scale).max().item()
        per = {}
        for plan in plans:
            outs = {}
            try:
                for d in ("tree", "prmt32"):
                    env(GNF4_GEMV_BW="1", GNF4_GEMV_BW_DECODE=d, GNF4_PDL="0")
                    outs[d] = ng.gemm_4bit_grouped(x, B, A, [1] * ROWS, ids, bw_config=plan)
            except Exception as exc:  # noqa: BLE001 -- a plan that cannot run is reported, not selectable
                per[",".join(map(str, plan))] = {"refused": f"{type(exc).__name__}: {str(exc)[:160]}"}
                continue
            err = {d: ((o.float() - ref).abs() / scale).max().item() for d, o in outs.items()}
            per[",".join(map(str, plan))] = {
                "bitwise_tree_prmt32": bool(torch.equal(outs["tree"], outs["prmt32"])),
                "bitwise_scalar": bool(torch.equal(outs["prmt32"], scalar)),
                "err": err, "tolerance": all(e_ <= max(TOL_X * err_scalar, TOL_ABS) for e_ in err.values())}
        # the served W1 form: ONE token row, both slots reading it (gather_div=2), against the expanded call
        x1 = x[:1]
        env(GNF4_GEMV_BW="1", GNF4_GEMV_BW_DECODE="prmt32", GNF4_PDL="0")
        try:
            gd = ng.gemm_4bit_grouped(x1, B, A, [1] * ROWS, ids, gather_div=ROWS)
            ex = ng.gemm_4bit_grouped(x1.repeat_interleave(ROWS, 0), B, A, [1] * ROWS, ids)
            gather = {"bitwise_expanded": bool(torch.equal(gd, ex))}
        except TypeError as exc:
            gather = {"refused": f"{type(exc).__name__}: {exc}"}
        rec["by_shape"][name] = {"N": N, "K": K, "err_scalar": err_scalar, "plans": per, "gather_div": gather}
        print(f"K33_NUMERICS {name} {N}x{K}: " + json.dumps(
            {p: (v.get("refused") or [v["bitwise_tree_prmt32"], v["tolerance"], v["bitwise_scalar"]])
             for p, v in per.items()}) + f" gather {gather}", flush=True)
        del B, A
        if not interpret:
            torch.cuda.empty_cache()
    env(GNF4_GEMV_BW=None, GNF4_GEMV_BW_DECODE=None, GNF4_GEMV_DOTPAD=None, GNF4_PDL=None)
    ok = all(v.get("bitwise_tree_prmt32") and v.get("tolerance")
             for s in rec["by_shape"].values() for v in s["plans"].values() if "refused" not in v)
    rec["all_runnable_plans_pass"] = ok
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"K33_NUMERICS_DONE pass={ok} wall={rec['wall_s']}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], interpret="--interpret" in sys.argv))
