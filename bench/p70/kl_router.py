#!/usr/bin/env python3
"""kl_router.py -- P70 (e4b#726): what does rounding the fused router's decode weights to the model's dtype cost?

P64's scorer (`bench/p64/kl_a16.py`), staged beside this file and imported UNCHANGED, with four of its module-level
names replaced before its run loop starts (the loop looks each one up when it runs): the pass table, the toggle, the
expected decode counts and the counter installer. So the served stack, its build, the one-row scoring through P59's
scorer, the storage of the logits and the per-pass census are P64's code, byte for byte; only WHICH arithmetic a pass
flips differs.

The lever: `experts4bit_qlora.engines.router_epilogue.CAST_WEIGHTS` (env `E4B_ROUTER_EPI_CAST`, default off). At
<= 64 rows -- every decode step here -- the fused router epilogue returns the kernel's fp32 routing weights; above 64
rows the model's own router runs and returns them in bf16 (`router_top_value.to(router_logits.dtype)`). The cast
makes the fused path return the bf16 weights the upstream router would, so every row count computes one function.

PASSES (registered in bench/p70/P70-PREREG.md; the served stack is the same object throughout):

    ep32        the served stack as shipped: fused router, fp32 weights at every decode step
    ep16        CAST_WEIGHTS on: the same weights rounded to bf16 -- the upstream function. THE LEVER.
    ep32_rep    ep32 again, after the toggle: the in-process determinism control (must be bit-identical to ep32)
    ep32_pc96   ep32 with a 96-token prefill chunk: an arithmetic-order floor sample. Its prefill forwards carry 96 rows,
                so the router function in the prefill is the primary's (the original router, bf16 weights).
    ep32_pc384  ep32 with the prefix in one 384-row forward: the second floor sample, same router function.
    ep32_pc64   ep32 with a 64-token prefill chunk -- P64's floor sample. Its prefill forwards carry 64 rows, so they run
                the FUSED router (fp32 weights) where the primary's run the original (bf16). Informational: it measures
                how much of P64's floor was this very switch. Never a floor sample here.

The census adds, per pass and phase, the patched routers' calls by row class (<= 64 rows / more) and by the dtype of
the weights they returned. That is the engagement proof: under ep32 every decode call returns fp32, under ep16 bf16.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p64"), os.path.join(HERE, "..", "p44"),
             os.path.join(HERE, "..", "p59")):
    if cand not in sys.path:                        # staged flat on the box; bench/p64, bench/p44, bench/p59 in the repo
        sys.path.insert(0, cand)

import kl_a16 as K  # noqa: E402  (P64's scorer, reused unchanged)

PREFIX, TOKENS = 384, 512                           # the runner's registered prefix and the prompt dump's row length
DECODE_PER_ROW = TOKENS - PREFIX

#: pass -> (cast the fused router's weights, attention A16 (never here), prefill chunk). The ORDER below is the
#: order a run spends its clock: the primary pair on both texts, the determinism control, the two floor samples,
#: then the informational P64-floor pass.
PASSES = {
    "ep32": (False, False, 128),
    "ep16": (True, False, 128),
    "ep32_rep": (False, False, 128),
    "ep32_pc96": (False, False, 96),
    "ep32_pc384": (False, False, 384),
    "ep32_pc64": (False, False, 64),
}
SCHEDULE = (("ep32", "wikitext"), ("ep16", "wikitext"), ("ep32", "c4val1"), ("ep16", "c4val1"),
            ("ep32_rep", "wikitext"), ("ep32_rep", "c4val1"),
            ("ep32_pc96", "wikitext"), ("ep32_pc96", "c4val1"),
            ("ep32_pc384", "wikitext"), ("ep32_pc384", "c4val1"),
            ("ep32_pc64", "wikitext"), ("ep32_pc64", "c4val1"))

_P64_TOGGLES = K.Toggles
_P64_INSTALL = K.install_counters


def _re():
    from experts4bit_qlora.engines import router_epilogue as re_mod
    return re_mod


class RouterToggles:
    """The pass's arithmetic: CAST_WEIGHTS for this lane; P64's toggles held at the shipped default in every pass (its
    constructor's refusals -- DECODE_A16 on at start, Int4Linear off its registered route -- apply unchanged)."""

    def __init__(self):
        self.p64 = _P64_TOGGLES()
        self.re = _re()
        if self.re.CAST_WEIGHTS[0] is not False:
            raise SystemExit("router_epilogue.CAST_WEIGHTS is on at process start (E4B_ROUTER_EPI_CAST set?): the served "
                             "stack must be built and scored with the shipped default; the lane flips it per pass")

    def set(self, cast: bool, _attn_a16: bool = False):
        self.p64.restore()
        self.re.CAST_WEIGHTS[0] = bool(cast)

    def restore(self):
        self.re.CAST_WEIGHTS[0] = False
        self.p64.restore()

    def state(self) -> dict:
        return dict(self.p64.state(), CAST_WEIGHTS=self.re.CAST_WEIGHTS[0])


def _dtype_name(t: torch.Tensor) -> str:
    return {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}.get(t.dtype, str(t.dtype))


def install_router_counters(model, census) -> int:
    """A forward hook on every router the router-epilogue fusion patched (an instance `forward`): count each call by
    row class and by the dtype of the weights it returned, in the census's current phase. Counts only."""
    re_mod = _re()
    n = 0
    for m in model.modules():
        if "Router" not in type(m).__name__ or "forward" not in m.__dict__:
            continue

        def hook(mod, args, out, _re_mod=re_mod):
            pos = _re_mod._out_positions(out)
            if pos is None:
                census.c[f"router_unreadable_{census.phase}"] += 1
                return
            w = out[pos[1]]
            size = "le64" if w.shape[0] <= _re_mod._MAX_DECODE_ROWS else "gt64"
            census.c[f"router_{size}_{_dtype_name(w)}_{census.phase}"] += 1
        m.register_forward_hook(hook)
        n += 1
    return n


def install_counters(model, census) -> dict:
    out = _P64_INSTALL(model, census)
    out["router_hooks"] = install_router_counters(model, census)
    return out


def expected_counts(pass_name: str, info: dict, steps: int) -> dict:
    """The exact counts a pass must show. P64's decode counts at its shipped arithmetic (every pass here is P64's
    `a8`), plus every router class x dtype x phase, zeros included. The reducer applies the same function."""
    L, n_attn = info["moe_layers"], info.get("int4_attn_projections", 0)
    H = (info.get("counters") or {}).get("router_hooks", 0)
    cast, _, chunk = PASSES[pass_name]
    rows = steps // DECODE_PER_ROW
    n_prefill = rows * -(-PREFIX // chunk)
    small = chunk <= _re()._MAX_DECODE_ROWS
    c = {"expert_quant_decode": 2 * L * steps, "expert_dequant_decode": 0, "attn_quant_decode": n_attn * steps}
    for size in ("le64", "gt64"):
        for dt in ("fp32", "bf16"):
            for ph in ("decode", "prefill"):
                c[f"router_{size}_{dt}_{ph}"] = 0
    fused_dt = "bf16" if cast else "fp32"            # the fused path returns fp32 unless the cast is on
    c[f"router_le64_{fused_dt}_decode"] = H * steps
    c[f"router_le64_{fused_dt}_prefill" if small else "router_gt64_bf16_prefill"] = H * n_prefill
    return c


def apply_overrides():
    """Point P64's run loop at this lane's pass table, toggle, counts and counters."""
    K.PASSES, K.SCHEDULE = PASSES, SCHEDULE
    K.Toggles, K.expected_counts, K.install_counters = RouterToggles, expected_counts, install_counters


# ----------------------------------------------------------------------------------------------- the proving run
def _routing_map(w: torch.Tensor, i: torch.Tensor):
    """A router's output as the function it is: for each row, the expert ids in ascending order and the weight each
    expert gets. The top-k SLOT order is not part of it -- the kernel's order can differ from torch.topk's where
    two probabilities tie within fp32 rounding (amendment 1)."""
    order = i.argsort(-1)
    return i.gather(-1, order), w.gather(-1, order)


def _same_experts(a, b) -> bool:
    return bool(torch.equal(_routing_map(*a)[0], _routing_map(*b)[0]))


def _map_bit_equal_frac(a, b) -> float:
    """Fraction of (row, expert) weights bit-equal between two routings with the same experts; 0.0 if they differ."""
    (ia, wa), (ib, wb) = _routing_map(*a), _routing_map(*b)
    if not torch.equal(ia, ib) or wa.dtype != wb.dtype:
        return 0.0
    return float((wa == wb).float().mean())


def _prove_cast(out_path: str) -> int:
    """On this card's REAL `int4_b32.router_epilogue`, no checkpoint: a transformers Qwen3MoeTopKRouter at Qwen3-30B-
    A3B's shape (hidden 2048, 128 experts, top-8), bf16, patched by `fuse_router_epilogue`, against an unpatched copy.

    Every comparison is at the SAME row count as upstream's (amendment 1, 2026-09-25): a bf16 GEMM's last-place
    logits depend on the row count (measured: 1 vs 64 vs 80 rows all differ on the A2000), so a fused 64-row call
    compared with rows of an 80-row upstream call measures the GEMM, not the router. p70-prove-2 did exactly that.
    Routings are compared as maps expert -> weight (`_routing_map`), not slot by slot.

    Asserted: switch off -> fp32 weights at <= 64 rows, on -> bf16; at 64 rows and at 1 row, the same experts as
    upstream at that row count, both states; above 64 rows the original forward, both states, equal to upstream.
    Reported, not asserted: the fraction of cast-on weights bit-equal to upstream's (the registered design), the
    slot-order agreement, and whether the GEMM itself varies with the row count on this card."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeTopKRouter
    re_mod = _re()
    assert re_mod.CAST_WEIGHTS[0] is False, "CAST_WEIGHTS on at process start"
    os.environ["E4B_FUSE_ROUTER_EPI"] = "1"
    cfg = Qwen3MoeConfig(hidden_size=2048, num_experts=128, num_experts_per_tok=8, norm_topk_prob=True)
    torch.manual_seed(726)
    ref = Qwen3MoeTopKRouter(cfg).to("cuda", torch.bfloat16)
    with torch.no_grad():
        ref.weight.normal_(0, 0.02)
    holder = torch.nn.Module()
    holder.gate = Qwen3MoeTopKRouter(cfg).to("cuda", torch.bfloat16)
    holder.gate.load_state_dict(ref.state_dict())
    n = re_mod.fuse_router_epilogue(holder)
    res = {"gpu": torch.cuda.get_device_name(0), "patched": n, "amendment": 1}
    x = (torch.randn(80, 2048) * 0.5).to("cuda", torch.bfloat16)
    lin = lambda t: torch.nn.functional.linear(t, ref.weight)  # noqa: E731
    with torch.no_grad():
        up = {m: ref(x[:m])[1:] for m in (1, 64, 80)}          # (weights, indices) at each row count
        res["gemm_varies_with_rows_64_vs_80"] = not torch.equal(lin(x[:64]), lin(x)[:64])
        res["gemm_varies_with_rows_1_vs_64"] = not torch.equal(lin(x[:1]), lin(x[:64])[:1])
        out = {}
        for cast in (False, True):
            re_mod.CAST_WEIGHTS[0] = cast
            out[cast] = {m: holder.gate(x[:m])[1:] for m in (1, 64, 80)}
        re_mod.CAST_WEIGHTS[0] = False
    torch.cuda.synchronize()
    off, on = out[False], out[True]
    res.update({
        "off_dtype_le64": str(off[64][0].dtype), "on_dtype_le64": str(on[64][0].dtype),
        "off_dtype_gt64": str(off[80][0].dtype), "on_dtype_gt64": str(on[80][0].dtype),
        "same_experts_as_upstream_64": _same_experts(off[64], up[64]) and _same_experts(on[64], up[64]),
        "same_experts_as_upstream_1": _same_experts(off[1], up[1]) and _same_experts(on[1], up[1]),
        "gt64_is_upstream_both_states": bool(torch.equal(off[80][0], up[80][0]) and torch.equal(off[80][1], up[80][1])
                                             and torch.equal(on[80][0], up[80][0]) and torch.equal(on[80][1], up[80][1])),
        "on_bit_equal_to_upstream_fraction_64": _map_bit_equal_frac(on[64], up[64]),
        "on_bit_equal_to_upstream_fraction_1": _map_bit_equal_frac(on[1], up[1]),
        "off_minus_upstream_max_abs_64": float((_routing_map(*off[64])[1].float() - _routing_map(*up[64])[1].float()).abs().max())
                                         if _same_experts(off[64], up[64]) else None,
        "slot_order_equals_upstream_64": bool(torch.equal(on[64][1], up[64][1])),
    })
    ok = (n == 1 and res["off_dtype_le64"] == "torch.float32" and res["on_dtype_le64"] == "torch.bfloat16"
          and res["off_dtype_gt64"] == res["on_dtype_gt64"] == "torch.bfloat16"
          and res["same_experts_as_upstream_64"] and res["same_experts_as_upstream_1"]
          and res["gt64_is_upstream_both_states"])
    res["all_passed"] = bool(ok)
    json.dump(res, open(out_path, "w"), indent=1)
    print("PROVECAST " + json.dumps(res), flush=True)
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------- the CPU self-test
def _self_test() -> int:
    """CPU, a tiny bf16 Qwen3-MoE from transformers with its routers patched by the real `fuse_router_epilogue` (a
    torch stand-in for the kernel entry point, as tests/test_router_epilogue.py uses) and the 64-row threshold scaled
    to 4 rows: (1) P64's own self-test passes (the scorer, unchanged); (2) for every chunk class -- fused prefill
    (chunk <= threshold) and original prefill (chunk > threshold) -- the router census equals `expected_counts` exactly,
    with the cast off and on; (3) the cast leaves the prefill-last logits bit-identical when the prefill runs the
    original router, and changes decode logits; (4) the toggle restores the shipped default."""
    import types
    rc = K._self_test()
    if rc != 0:
        return rc
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    re_mod = _re()
    saved = (sys.modules.get("int4_b32"), re_mod._MAX_DECODE_ROWS, os.environ.get("E4B_FUSE_ROUTER_EPI"))
    stub = types.ModuleType("int4_b32")

    def router_epilogue(logits, k, norm, *, select_on_logits=False, bias=None):
        probs = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(probs, k, dim=-1)
        return probs, (v / v.sum(dim=-1, keepdim=True)) if norm else v, i
    stub.router_epilogue = router_epilogue
    sys.modules["int4_b32"] = stub
    os.environ["E4B_FUSE_ROUTER_EPI"] = "1"
    re_mod._MAX_DECODE_ROWS = 4
    try:
        cfg = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=16,
                             num_experts_per_tok=4, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=16, vocab_size=512, max_position_embeddings=256, norm_topk_prob=True)
        cfg._attn_implementation = "eager"
        torch.manual_seed(0)
        model = Qwen3MoeForCausalLM(cfg).eval().to(torch.bfloat16)
        H = re_mod.fuse_router_epilogue(model)
        assert H == cfg.num_hidden_layers, f"patched {H} routers, expected {cfg.num_hidden_layers}"
        census = K.Census()

        def pre_hook(mod, args, kwargs):
            ids_ = kwargs.get("input_ids", args[0] if args else None)
            census.phase = "decode" if ids_ is not None and ids_.shape[-1] == 1 else "prefill"
        model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        assert install_router_counters(model, census) == H
        B, T, prefix = 2, 20, 12
        ids = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(5))
        info = {"moe_layers": 0, "counters": {"router_hooks": H}}      # L = 0: the int4 counts are not in this model
        results = {}
        global PREFIX, DECODE_PER_ROW
        saved_consts = (PREFIX, DECODE_PER_ROW)
        PREFIX, DECODE_PER_ROW = prefix, T - prefix
        try:
            for chunk, cast in ((4, False), (4, True), (6, False), (6, True), (12, False)):
                name = f"t_c{chunk}_{'cast' if cast else 'fp32'}"
                PASSES[name] = (cast, False, chunk)
                re_mod.CAST_WEIGHTS[0] = cast
                census.reset()
                pre, dec = K.score_rows(model, ids, prefix, chunk, "cpu", B)
                re_mod.CAST_WEIGHTS[0] = False
                got = {k: v for k, v in census.snapshot().items() if k.startswith("router_")}
                want = {k: v for k, v in expected_counts(name, info, B * (T - prefix)).items() if k.startswith("router_")}
                assert all(got.get(k, 0) == v for k, v in want.items()) and set(got) <= set(want), (name, got, want)
                results[name] = (pre, dec)
                del PASSES[name]
        finally:
            PREFIX, DECODE_PER_ROW = saved_consts
        pre_off, dec_off = results["t_c6_fp32"]
        pre_on, dec_on = results["t_c6_cast"]
        assert torch.equal(pre_off, pre_on), "the cast touched a prefill whose forwards run the original router"
        assert not torch.equal(dec_off, dec_on), "the cast changed nothing at decode: the lever did not engage"
        tog = RouterToggles()
        tog.set(True)
        assert re_mod.CAST_WEIGHTS[0] is True and tog.state()["CAST_WEIGHTS"] is True
        tog.restore()
        assert re_mod.CAST_WEIGHTS[0] is False
    finally:
        re_mod._MAX_DECODE_ROWS = saved[1]
        re_mod.CAST_WEIGHTS[0] = False
        if saved[0] is None:
            sys.modules.pop("int4_b32", None)
        else:
            sys.modules["int4_b32"] = saved[0]
        if saved[2] is None:
            os.environ.pop("E4B_FUSE_ROUTER_EPI", None)
        else:
            os.environ["E4B_FUSE_ROUTER_EPI"] = saved[2]
    print("self-test OK (P70): P64's scorer self-test; the router census equals expected_counts for fused and original "
          "prefills, cast off and on; the cast leaves an original-router prefill bit-identical and changes decode")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    if "--prove-cast" in sys.argv[1:]:
        ap = argparse.ArgumentParser()
        ap.add_argument("--prove-cast", action="store_true")
        ap.add_argument("--out", default="prove_cast.json")
        return _prove_cast(ap.parse_args().out)
    for flag in ("--anchor", "--build-only", "--prove-flag"):
        if flag in sys.argv[1:]:
            raise SystemExit(f"{flag} is P64's (kl_a16.py), not a P70 mode: P70 scores the int4 stack only")
    apply_overrides()
    return K.main()


if __name__ == "__main__":
    sys.exit(main())
