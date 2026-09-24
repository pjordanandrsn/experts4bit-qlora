#!/usr/bin/env python3
"""kl_a16.py -- P64 (e4b#709): what does the int4 serve lane's int8 activation step cost at decode?

One served stack, built ONCE in this process, scored under several PASSES that differ only in the arithmetic of
the T == 1 forwards. Every pass scores the same token rows the same way: P59's scorer
(`bench/p59/kl_b16.py:batched_teacher_forced`), called with ONE row at a time, so every decode forward is T == 1
-- the shape where the int4 expert store runs `gemv_int4_b32` on `quant_x_rows`' int8 activations
(`experts4bit_qlora/engines/hot_residency.py`, the singleton route). Each row prefills its first `--prefix` tokens
in chunks of the pass's prefill chunk (T > 1, the dequant + bf16 route in every pass), then decodes the rest one
token per forward with a carried KV cache, the ground-truth token fed back. Logits are kept per pass and text; the
KL and NLL are read by `p64_reduce.py` with kl_fidelity's primitives.

PASSES (registered in bench/p64/P64-PREREG.md; the served stack is the same object throughout):

    a8        the served stack as shipped: experts W4A8 (quant_x_rows + GEMV), attention W4A8 (Int4Linear GEMV)
    a16       E4B_INT4_DECODE_A16 on (hot_residency.DECODE_A16): T == 1 experts take the prefill branch --
              dequant each routed expert, bf16 matmul, no quant_x_rows. Attention unchanged. THE LEVER.
    a16_all   a16 + the attention projections at one row sent to Int4Linear's cached-bf16 branch (the branch its
              prefill uses) by setting the class's GEMV_ROWS_MAX and SMALLM_ROWS_MAX to 0 for the pass. Lane-side
              instrument, not a library flag.
    a8_rep    a8 again, after the toggles: the in-process determinism control (must be bit-identical to a8).
    a8_pc64   a8 with a 64-token prefill chunk (primary: 128): the same arithmetic in another valid order --
    a8_pc384  and in one 384-token forward: the arithmetic-order FLOOR (METHODOLOGY section 13.1's method).
    nf4       the anchor, built in its own process (`--anchor`): NF4 experts, bf16 attention, no folds.

PROOF OF EXECUTION per pass and text (a census the reducer re-checks; a lever that did not engage refuses the pass):
counters on `int4_b32.quant_x_rows` and `int4_pack_ref.dequant_int4_ref` -- patched AFTER the build, so only the
expert dispatch (which imports them at call time) is counted, never Int4Linear, which bound its own at install --
and on every Int4Linear's own `_qx`, split by phase (a forward with one token is `decode`, anything else
`prefill`). The expected counts are exact: at decode, a8 = 2 x L expert quantises and 0 expert dequants per step;
a16 = 0 and 2 x L x top_k; a16_all additionally 0 attention quantises.

    python kl_a16.py --self-test                     # CPU: per-row scoring == kl_fidelity's decode scorer; counters
    python kl_a16.py --model ... --arena ... --calib ... --prompts wikitext=p_wt.json,c4val1=p_c4.json \\
        --passes a8,a16,... --out out/                                           # on the box, env set by the runner
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p44"), os.path.join(HERE, "..", "p59")):
    if cand not in sys.path:                        # staged flat on the box; bench/, bench/p44, bench/p59 in the repo
        sys.path.insert(0, cand)

from kl_b16 import batched_teacher_forced  # noqa: E402  (P59's scorer, reused unchanged)

#: pass -> (expert A16 flag, attention A16, prefill chunk). The ORDER below is the order a run spends its clock:
#: the primary pair on both texts first, then the determinism control, the floor, the attention arm, the second
#: floor sample.
PASSES = {
    "a8": (False, False, 128),
    "a16": (True, False, 128),
    "a16_all": (True, True, 128),
    "a8_rep": (False, False, 128),
    "a8_pc64": (False, False, 64),
    "a8_pc384": (False, False, 384),
    "nf4": (False, False, 128),
}
SCHEDULE = (("a8", "wikitext"), ("a16", "wikitext"), ("a8", "c4val1"), ("a16", "c4val1"),
            ("a8_rep", "wikitext"), ("a8_rep", "c4val1"),
            ("a8_pc64", "wikitext"), ("a8_pc64", "c4val1"),
            ("a16_all", "wikitext"), ("a16_all", "c4val1"),
            ("a8_pc384", "wikitext"), ("a8_pc384", "c4val1"))
ANCHOR_SCHEDULE = (("nf4", "wikitext"), ("nf4", "c4val1"))


class Census:
    """Call counters split by forward phase. `phase` is set by a forward pre-hook on the model."""

    def __init__(self):
        self.phase = "prefill"
        self.c = defaultdict(int)

    def reset(self):
        self.c = defaultdict(int)

    def snapshot(self) -> dict:
        return dict(sorted(self.c.items()))


def install_counters(model, census: Census) -> dict:
    """Count, by phase: forwards, the expert dispatch's quant_x_rows / dequant_int4_ref calls, and every
    Int4Linear's own activation quantise. Wrappers only count; the arithmetic is the wrapped function's."""
    def pre_hook(mod, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        census.phase = "decode" if ids is not None and ids.shape[-1] == 1 else "prefill"
        census.c[f"forwards_{census.phase}"] += 1
    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    out = patch_expert_counters(census)
    out["attn_int4linear_counted"] = 0
    try:
        from experts4bit_qlora.engines.int4_attn import Int4Linear
    except ImportError:
        Int4Linear = None
    if Int4Linear is not None:
        for m in model.modules():
            if isinstance(m, Int4Linear):
                own = m._qx

                def qx(x, _own=own):
                    census.c[f"attn_quant_{census.phase}"] += 1
                    return _own(x)
                m._qx = qx
                out["attn_int4linear_counted"] += 1
    return out


def patch_expert_counters(census: Census) -> dict:
    """The two kernel entry points the expert dispatch imports at call time, wrapped to count by phase."""
    out = {"expert_quant_counter": False, "expert_dequant_counter": False}
    try:
        import int4_b32
        orig_q = int4_b32.quant_x_rows

        def q(*a, **k):
            census.c[f"expert_quant_{census.phase}"] += 1
            return orig_q(*a, **k)
        int4_b32.quant_x_rows = q
        out["expert_quant_counter"] = True
    except ImportError:
        pass
    try:
        import int4_pack_ref
        orig_d = int4_pack_ref.dequant_int4_ref

        def d(*a, **k):
            census.c[f"expert_dequant_{census.phase}"] += 1
            return orig_d(*a, **k)
        int4_pack_ref.dequant_int4_ref = d
        out["expert_dequant_counter"] = True
    except ImportError:
        pass
    return out


def attn_pack_sha256(model) -> str | None:
    """sha256 over every Int4Linear's packed + scales bytes in module order: the attention half of the stack, which
    no artifact pins (#674) -- recorded so two builds of the stack can be compared by bytes, not by config."""
    try:
        from experts4bit_qlora.engines.int4_attn import Int4Linear
    except ImportError:
        return None
    h, n = hashlib.sha256(), 0
    for name, m in model.named_modules():
        if isinstance(m, Int4Linear):
            h.update(name.encode())
            for t in (m.packed, m.scales):
                h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
            n += 1
    return f"sha256:{h.hexdigest()}" if n else None


class Toggles:
    """Set a pass's arithmetic, and restore the stack exactly as built. The expert flag is hot_residency's
    one-element list; the attention route is Int4Linear's two class constants, read on every forward."""

    def __init__(self):
        from experts4bit_qlora.engines import hot_residency as hr
        self.hr = hr
        try:
            from experts4bit_qlora.engines.int4_attn import Int4Linear
        except ImportError:
            Int4Linear = None
        self.L = Int4Linear
        self.base = (hr.DECODE_A16[0], getattr(Int4Linear, "GEMV_ROWS_MAX", None), getattr(Int4Linear, "SMALLM_ROWS_MAX", None))
        if self.base[0] is not False:
            raise SystemExit("hot_residency.DECODE_A16 is on at process start (E4B_INT4_DECODE_A16 set?): the served "
                             "stack must be built and scored with the shipped default; the lane flips it per pass")
        if Int4Linear is not None and self.base[1] != 1:
            raise SystemExit(f"Int4Linear.GEMV_ROWS_MAX is {self.base[1]}, not 1: not the route this lane registers")

    def set(self, expert_a16: bool, attn_a16: bool):
        self.hr.DECODE_A16[0] = bool(expert_a16)
        if self.L is not None:
            self.L.GEMV_ROWS_MAX = 0 if attn_a16 else self.base[1]
            self.L.SMALLM_ROWS_MAX = 0 if attn_a16 else self.base[2]

    def restore(self):
        self.set(False, False)

    def state(self) -> dict:
        return {"DECODE_A16": self.hr.DECODE_A16[0], "GEMV_ROWS_MAX": getattr(self.L, "GEMV_ROWS_MAX", None),
                "SMALLM_ROWS_MAX": getattr(self.L, "SMALLM_ROWS_MAX", None)}


def load_prompts(spec: str) -> dict:
    """`wikitext=a.json,c4val1=b.json` -> {text: (ids [B, T] long, prompts_sha256, record)}; digests checked."""
    out = {}
    for part in spec.split(","):
        name, path = part.split("=", 1)
        rec = json.load(open(path))
        rows = rec["prompts"]
        assert len({len(p) for p in rows}) == 1, f"{name}: rows must be equal length"
        assert len({tuple(p) for p in rows}) == len(rows), f"{name}: rows must be distinct"
        sha = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        assert sha == rec["prompts_sha256"], f"{name}: prompt file digest mismatch"
        out[name] = (torch.tensor(rows, dtype=torch.long), sha, rec)
    return out


def score_rows(model, ids: torch.Tensor, prefix: int, chunk: int, device, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows scored one at a time (B = 1, so every decode forward is T == 1) through P59's scorer."""
    pre, dec = [], []
    for b in range(rows):
        p, d = batched_teacher_forced(model, ids[b:b + 1], prefix, device, prefill_chunk=chunk)
        pre.append(p[0])
        dec.append(d[0])
    return torch.stack(pre), torch.stack(dec)


def _storable(t: torch.Tensor, dtype) -> tuple[torch.Tensor, str]:
    """Keep the logits in the model's own dtype when that round-trips exactly (kl_serve's rule: lossless)."""
    if dtype != torch.float32:
        c = t.to(dtype)
        if torch.equal(c.float(), t):
            return c, str(dtype)
    return t, "torch.float32"


def expected_counts(pass_name: str, info: dict, steps: int) -> dict:
    """The exact decode-phase counts a pass must show (the reducer applies the same function)."""
    L, k, n_attn = info["moe_layers"], info["top_k"], info.get("int4_attn_projections", 0)
    exp_a16, attn_a16, _ = PASSES[pass_name]
    if pass_name == "nf4":
        return {"expert_quant_decode": 0, "expert_dequant_decode": 0, "attn_quant_decode": 0}
    return {"expert_quant_decode": 0 if exp_a16 else 2 * L * steps,
            "expert_dequant_decode": 2 * L * k * steps if exp_a16 else 0,
            "attn_quant_decode": 0 if attn_a16 else n_attn * steps}


def run(a) -> int:
    import serve_stack
    prompts = load_prompts(a.prompts) if a.prompts else {}
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model, info = serve_stack.build_served_model(a.model, a.arena, a.calib, device=a.device)
    info["build_s"] = round(time.time() - t0, 1)
    info["env"] = {k: v for k, v in os.environ.items() if k.startswith(("E4B_", "GNF4_"))}
    info["attn_pack_sha256"] = attn_pack_sha256(model)
    if a.build_only:
        # the pack build (the hook dumps the artifact inside the build when E4B_INT4_DUMP_ARTIFACT_DIR is set);
        # nothing is scored. The registered build is P55x's step_decomp invocation; this is its fallback.
        json.dump(info, open(os.path.join(a.out, "build.json"), "w"), indent=1, default=str)
        print("P64BUILDONLY " + json.dumps({k: info.get(k) for k in ("int4_expert_layers", "int4_attn_projections", "build_s")}), flush=True)
        return 0 if info.get("int4_expert_layers") else 40
    schedule = ANCHOR_SCHEDULE if a.anchor else SCHEDULE
    want = [x.strip() for x in a.passes.split(",")] if a.passes else sorted({p for p, _ in schedule})
    schedule = [(p, t) for p, t in schedule if p in want and t in prompts]
    if a.anchor:
        if info.get("int4_expert_layers") or info.get("int4_attn_projections"):
            print("P64 REFUSED: the NF4 anchor carries int4 stores or Int4Linear -- the lane env leaked", flush=True)
            return 44
    else:
        if not info.get("int4_expert_layers"):
            print("P64 REFUSED: no int4 expert store engaged -- the served stack is not the int4 lane", flush=True)
            return 40
    toggles = Toggles()
    census = Census()
    info["counters"] = install_counters(model, census)
    if not a.anchor and not (info["counters"]["expert_quant_counter"] and info["counters"]["expert_dequant_counter"]):
        print("P64 REFUSED: the engagement counters could not be installed", flush=True)
        return 40
    json.dump(info, open(os.path.join(a.out, "build.json"), "w"), indent=1, default=str)
    print("P64BUILD " + json.dumps({k: info.get(k) for k in ("moe_layers", "top_k", "int4_expert_layers", "int4_store_kinds",
                                                              "int4_attn_projections", "fuse_t1_glue_n", "fuse_t1_glue_r2_n",
                                                              "fuse_router_epilogue_n", "build_s", "attn_pack_sha256")}), flush=True)
    rc = 0
    last_s = {}
    for pass_name, text in schedule:
        exp_a16, attn_a16, chunk = PASSES[pass_name]
        kind = "a16" if exp_a16 else "a8"
        est = last_s.get(kind) or (3.0 * max(last_s.values()) if last_s else 0.0)   # an unseen kind: 3x the slowest seen
        if a.deadline_epoch and time.time() + est + a.margin_s > a.deadline_epoch:
            print(f"P64 SKIPPED {pass_name}/{text}: needs ~{est:.0f}s, deadline in {a.deadline_epoch - time.time():.0f}s (host-limited)", flush=True)
            json.dump({"pass": pass_name, "text": text, "skipped": "deadline"}, open(os.path.join(a.out, f"{pass_name}.{text}.census.json"), "w"))
            continue
        ids, sha, rec = prompts[text]
        rows = min(a.rows, ids.shape[0])
        steps = rows * (ids.shape[1] - a.prefix)
        toggles.set(exp_a16, attn_a16)
        census.reset()
        t1 = time.time()
        try:
            pre, dec = score_rows(model, ids, a.prefix, chunk, a.device, rows)
        finally:
            state = toggles.state()
            toggles.restore()
        wall = time.time() - t1
        last_s[kind] = wall
        got = census.snapshot()
        want_c = expected_counts(pass_name, info, steps)
        engaged = all(got.get(k, 0) == v for k, v in want_c.items())
        dec_s, dec_dtype = _storable(dec, torch.bfloat16)     # bf16 when the model's own logits were bf16 (exact)
        pre_s, _ = _storable(pre, torch.bfloat16)
        torch.save({"prefill_last": pre_s, "decode": dec_s, "prefix": a.prefix, "prompts_sha256": sha, "pass": pass_name,
                    "text": text, "stored_dtype": dec_dtype, "rows": rows},
                   os.path.join(a.out, f"{pass_name}.{text}.pt"))
        cen = {"pass": pass_name, "text": text, "expert_a16": exp_a16, "attn_a16": attn_a16, "prefill_chunk": chunk,
               "prefix": a.prefix, "rows": rows, "decode_steps": steps, "toggle_state_during_pass": state,
               "counts": got, "expected_decode_counts": want_c, "engaged_as_registered": engaged,
               "wall_s": round(wall, 1), "s_per_decode_step_incl_prefill": round(wall / max(1, steps), 5),
               "prompts_sha256": sha, "stored_dtype": dec_dtype, "decode_logits_shape": list(dec.shape),
               "source": rec.get("source"), "attn_pack_sha256": info.get("attn_pack_sha256")}
        json.dump(cen, open(os.path.join(a.out, f"{pass_name}.{text}.census.json"), "w"), indent=1)
        print("P64PASS " + json.dumps({k: cen[k] for k in ("pass", "text", "engaged_as_registered", "wall_s", "decode_steps")}
                                      | {"counts": {k: got.get(k, 0) for k in want_c}}), flush=True)
        if not engaged:
            print(f"P64 REFUSED {pass_name}/{text}: decode counts {got} != registered {want_c} -- the lever did not engage as registered", flush=True)
            rc = rc or 40
        del pre, dec, pre_s, dec_s
    return rc


def _prove_flag(out_path: str) -> int:
    """On this card's REAL kernels, no model: Qwen3-30B-A3B's expert shapes (hidden 2048, moe_intermediate 768),
    16 random experts packed onto the int4-b32 grid, one token's 8 routed rows through hot_residency's singleton
    route. Checked: flag off runs quant_x_rows + the GEMV and never the dequant loop; flag on runs no quant_x_rows,
    2 x 8 dequants, and equals the prefill branch on the same rows bit for bit; the off route still captures in a CUDA
    graph and replays bit-identical to eager; the on route refuses under capture with its sentence. Reported, not
    asserted: each state's relative error against the fp32 product of the same int4 values and bf16 activations."""
    import torch.nn.functional as F
    from int4_pack_ref import pack_int4_b32
    import int4_pack_ref
    from experts4bit_qlora.engines import hot_residency as hr
    dev, E, H, inter, k = "cuda", 16, 2048, 768, 8
    g = torch.Generator().manual_seed(1689)
    gu_w = torch.randn(E, 2 * inter, H, generator=g) * 0.02
    dn_w = torch.randn(E, H, inter, generator=g) * 0.02

    def stack(W):
        pk, sc = zip(*[pack_int4_b32(W[e]) for e in range(E)])
        return torch.stack(pk).to(dev), torch.stack(sc).to(dev)
    (gp, gs), (dp, ds) = stack(gu_w), stack(dn_w)
    stores = {"gu": {"packed": gp, "scales": gs, "N": 2 * inter, "K": H}, "dn": {"packed": dp, "scales": ds, "N": H, "K": inter}}
    fgu = torch.empty(0, 0, 0, dtype=torch.uint8, device=dev)
    fdn = torch.empty(0, 0, 0, dtype=torch.uint8, device=dev)
    fa = torch.empty(0, 0, 0, device=dev)
    x = (torch.randn(1, H, generator=g) * 0.5).to(torch.bfloat16).expand(k, H).contiguous().to(dev)
    ids = torch.randperm(E, generator=g)[:k].to(dev)
    census = Census()
    census.phase = "decode"
    patch_expert_counters(census)

    def call(**kw):
        return hr._fused_over_stack(x, ids, fgu, fa, fdn, fa, (2 * inter, H, H, inter), True, F.silu, int4_stores=stores, **kw)
    res = {"gpu": torch.cuda.get_device_name(0), "shapes": {"E": E, "hidden": H, "moe_intermediate": inter, "top_k": k}}
    base = hr.DECODE_A16[0]
    assert base is False, "DECODE_A16 on at process start"
    census.reset()
    off = call(singleton_groups=True)
    torch.cuda.synchronize()
    res["off_counts"] = census.snapshot()
    hr.DECODE_A16[0] = True
    census.reset()
    on = call(singleton_groups=True)
    torch.cuda.synchronize()
    res["on_counts"] = census.snapshot()
    hr.DECODE_A16[0] = False
    census.reset()
    pre = call()
    res["prefill_counts"] = census.snapshot()
    res["on_equals_prefill_branch"] = bool(torch.equal(on, pre))
    res["on_differs_from_off"] = not bool(torch.equal(on, off))
    ref = torch.empty(k, H, dtype=torch.float64)
    for r, e in enumerate(ids.tolist()):
        wg = int4_pack_ref.dequant_int4_ref(gp[e].cpu(), gs[e].cpu(), 2 * inter, H).double()
        wd = int4_pack_ref.dequant_int4_ref(dp[e].cpu(), ds[e].cpu(), H, inter).double()
        gu = x[r].cpu().double() @ wg.t()
        ga, up = gu.chunk(2)
        ref[r] = (F.silu(ga) * up) @ wd.t()
    for name, o in (("off", off), ("on", on)):
        res[f"{name}_rel_err_vs_fp64_of_same_int4"] = float((o.cpu().double() - ref).norm() / ref.norm())
    # the default route still captures (the certified pattern) and replays bit-identical to eager
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            cap = call(singleton_groups=True)
        gr.replay()
        torch.cuda.synchronize()
        res["off_captures_and_replays_equal"] = bool(torch.equal(cap, off))
    except Exception as e:  # recorded: a default-path capture failure is a finding, not this flag's
        res["off_captures_and_replays_equal"] = False
        res["off_capture_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    hr.DECODE_A16[0] = True
    try:
        gr2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr2):
            call(singleton_groups=True)
        res["on_refuses_under_capture"] = False
    except RuntimeError as e:
        res["on_refuses_under_capture"] = "eager-only" in str(e)
        res["on_capture_message"] = str(e)[:200]
    finally:
        hr.DECODE_A16[0] = False
    ok = (res["off_counts"].get("expert_quant_decode") == 2 and res["off_counts"].get("expert_dequant_decode", 0) == 0
          and res["on_counts"].get("expert_quant_decode", 0) == 0 and res["on_counts"].get("expert_dequant_decode") == 2 * k
          and res["on_equals_prefill_branch"] and res["on_differs_from_off"] and res["on_refuses_under_capture"]
          and res["off_captures_and_replays_equal"])
    res["all_passed"] = bool(ok)
    json.dump(res, open(out_path, "w"), indent=1)
    print("PROVEFLAG " + json.dumps({k_: res[k_] for k_ in ("gpu", "all_passed", "on_equals_prefill_branch", "on_refuses_under_capture",
                                                            "off_captures_and_replays_equal", "off_rel_err_vs_fp64_of_same_int4",
                                                            "on_rel_err_vs_fp64_of_same_int4")}), flush=True)
    return 0 if ok else 1


def _self_test() -> int:
    """CPU, tiny random Qwen3-MoE (no e4b stack): (1) one-row scoring through P59's scorer equals kl_fidelity's
    per-row decode scorer at every decode position and at the prefill-last position, for every prefill chunk the
    lane registers (scaled); (2) the phase census counts ceil(prefix / chunk) prefill forwards and T - prefix decode
    forwards per row; (3) the stored logits round-trip; (4) the NLL index the reducer uses reads the right target."""
    import kl_fidelity
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=16,
                         num_experts_per_tok=4, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, vocab_size=512, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).eval()
    B, T, prefix = 3, 24, 12
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    census = Census()
    install_counters(model, census)
    worst = 0.0
    for chunk in (4, 5, 12):                               # 128 / 64 / 384 at the lane's scale; 5 leaves an uneven last chunk
        census.reset()
        pre, dec = score_rows(model, ids, prefix, chunk, "cpu", B)
        c = census.snapshot()
        assert c.get("forwards_prefill") == B * -(-prefix // chunk), (chunk, c)
        assert c.get("forwards_decode") == B * (T - prefix), (chunk, c)
        for b in range(B):
            ref = kl_fidelity.decode_teacher_forced_logits(model, ids[b])   # [T, V]: position i predicts token i+1
            worst = max(worst, (dec[b] - ref[prefix:].float()).abs().max().item(),
                        (pre[b] - ref[prefix - 1].float()).abs().max().item())
    assert worst < 2e-3, f"one-row scoring differs from kl_fidelity's decode scorer by {worst:.3e}"
    s, dt = _storable(dec, torch.bfloat16)
    assert torch.equal(s.float(), dec) or dt == "torch.float32"
    # NLL index: decode logit j (fed ids[prefix + j]) predicts ids[prefix + j + 1]
    lp = torch.log_softmax(dec[0].double(), -1)
    nll0 = -lp[0, ids[0, prefix + 1]].item()
    ref0 = -torch.log_softmax(kl_fidelity.decode_teacher_forced_logits(model, ids[0])[prefix].double(), -1)[ids[0, prefix + 1]].item()
    assert abs(nll0 - ref0) < 1e-3, (nll0, ref0)
    print(json.dumps({"one_row_vs_decode_scorer_max_abs_diff": worst, "chunks": [4, 5, 12], "rows": B, "T": T, "prefix": prefix,
                      "census_last": census.snapshot(), "nll_index_check": [nll0, ref0]}, indent=1))
    print("self-test OK: one-row scoring == kl_fidelity's decode scorer for every chunk; phase census exact; NLL index right")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="P64: decode-scored KL A/B of the int8 activation step (e4b#709)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--anchor", action="store_true", help="the NF4 anchor process (its own build; passes: nf4)")
    ap.add_argument("--build-only", action="store_true", help="build the served stack (the hook dumps the pack) and exit")
    ap.add_argument("--prove-flag", action="store_true", help="the flag on this card's real kernels, no model (the proving run)")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--arena")
    ap.add_argument("--calib")
    ap.add_argument("--prompts", help="text=path.json[,text=path.json]")
    ap.add_argument("--passes", default=None, help="comma list; default every registered pass of the process kind")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--prefix", type=int, default=384)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deadline-epoch", type=int, default=0)
    ap.add_argument("--margin-s", type=int, default=600)
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.prove_flag:
        return _prove_flag(a.out or "prove_flag.json")
    for k in (("arena", "calib", "out") if a.build_only else ("arena", "calib", "prompts", "out")):
        if getattr(a, k) is None:
            ap.error(f"--{k} is required")
    for p in (a.passes or "").split(","):
        if p and p not in PASSES:
            ap.error(f"unregistered pass {p!r}; registered: {list(PASSES)}")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
