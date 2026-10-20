"""DQ11 MLP-adapter BUILD, Phase 1: per-module correctness of the fused MLP adapters against the unfused path's neutral
floor (``DQ11-MLP-BUILD-PREREG.md`` D3, ``DQ11-MLP-BUILD-AMENDMENT-1.md``).

One read takes one PEFT ``MistralMLP`` whose projections are LoRA on bitsandbytes NF4 bases, a captured MLP input ``x``
and the loss gradient ``gy`` at its output, and returns eleven tensors:
- the three projection outputs;
- ``dx``, the shared MLP input's gradient;
- ``dh``, ``down_proj``'s input's gradient;
- each adapter's A and B gradients.

The reads:

- ``peft_read``: PEFT's own unfused path (the truth).
- ``ref_read(draws)``: a reference for the same path that repeats PEFT 0.19.1's ``Linear4bit`` LoRA forward op for op
  and changes only what each redraw names:
  - **R1:** the forward rank GEMM split along K;
  - **R2:** the base and the delta added in FP32, then cast;
  - **R3:** the A and B gradient GEMMs split along the tokens;
  - **R4:** the LoRA input-gradient GEMM split along the rank;
  - **R5:** dx's two LoRA terms summed in the other order.

  With no redraw it must equal ``peft_read`` bit for bit (else ``REF_MISMATCH``).
- ``fused_read``: the engine (``experts4bit_qlora.engines.dense_lora_fuse``) on the same module.

The floor of each tensor is the worst of the 31 draws: the largest max-|draw - truth| and the largest relative RMS. The
fused read passes iff every tensor is at or below its floor on both. A tensor no draw moved must match bit for bit.
"""
from __future__ import annotations

import collections
import contextlib
import itertools

import torch
import torch.nn.functional as F

DRAWS = ("R1", "R2", "R3", "R4", "R5")
#: Amendment 1: B = randn * sigma, the smallest sigma at which every checked tensor's adapter share is >= SHARE_MIN.
SIGMAS = (0.01, 0.02, 0.05, 0.1)
SHARE_MIN = 0.01
B_SEED = 3407
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TENSORS = ("out_gate", "out_up", "out_down", "dx", "dh",
           "dA_gate", "dB_gate", "dA_up", "dB_up", "dA_down", "dB_down")
#: Autograd's order for dx's four terms on PEFT's Linear4bit LoRA path (Amendment 1's probe): up base, up LoRA, gate
#: LoRA, gate base. R5 swaps the two LoRA terms.
DX_ORDER = ("up_base", "up_lora", "gate_lora", "gate_base")
DX_ORDER_R5 = ("up_base", "gate_lora", "up_lora", "gate_base")


def draw_sets() -> list:
    """The 31 non-empty combinations of R1-R5, each one draw."""
    return [frozenset(c) for k in range(1, len(DRAWS) + 1) for c in itertools.combinations(DRAWS, k)]


def _adapter(proj):
    (name,) = [a for a in proj.active_adapters if a in proj.lora_A.keys()]
    return proj.lora_A[name], proj.lora_B[name], float(proj.scaling[name])


def _params(mlp) -> list:
    """``[A_gate, B_gate, A_up, B_up, A_down, B_down]``, the adapter weights in TENSORS order."""
    out = []
    for p in PROJECTIONS:
        a, b, _s = _adapter(getattr(mlp, p))
        out += [a.weight, b.weight]
    return out


def _named(outs, dx, dh, grads) -> dict:
    return dict(zip(TENSORS, [o.detach() for o in outs] + [dx, dh] + list(grads)))


def peft_read(mlp, x, gy) -> dict:
    """PEFT's own unfused path: the projection outputs from forward hooks, ``dh`` at ``down_proj``'s input."""
    seen = {}
    hooks = [getattr(mlp, p).register_forward_hook(lambda m, a, o, p=p: seen.__setitem__(p, o)) for p in PROJECTIONS]
    hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m, a: seen.__setitem__("h", a[0])))
    try:
        xl = x.detach().clone().requires_grad_(True)
        y = mlp(xl)
    finally:
        for h in hooks:
            h.remove()
    params = _params(mlp)
    dx, dh, *grads = torch.autograd.grad(y, [xl, seen["h"], *params], gy)
    return _named([seen[p] for p in PROJECTIONS], dx, dh, grads)


class _LoraLinear(torch.autograd.Function):
    """``F.linear(inp, w)`` for a LoRA A or B matmul, with the redraws that change its gradient GEMMs. The parts no redraw
    names come from autograd on the same forward, so they are what autograd computes on PEFT's path.
    - ``split_k`` (R1, A only): the forward split along K, two halves summed;
    - ``split_tokens`` (R3): the weight gradient as the sum of the two token halves;
    - ``split_rank`` (R4, A only): the input gradient as the sum over the two halves of the rank."""

    @staticmethod
    def _fwd(inp, w, split_k):
        if not split_k:
            return F.linear(inp, w)
        k = w.shape[1] // 2
        return F.linear(inp[..., :k], w[:, :k]) + F.linear(inp[..., k:], w[:, k:])

    @staticmethod
    def forward(ctx, inp, w, split_k, split_tokens, split_rank):
        ctx.save_for_backward(inp, w)
        ctx.flags = (split_k, split_tokens, split_rank)
        return _LoraLinear._fwd(inp, w, split_k)

    @staticmethod
    def backward(ctx, g):
        inp, w = ctx.saved_tensors
        split_k, split_tokens, split_rank = ctx.flags
        with torch.enable_grad():
            i_, w_ = inp.detach().requires_grad_(True), w.detach().requires_grad_(True)
            d_inp, d_w = torch.autograd.grad(_LoraLinear._fwd(i_, w_, split_k), [i_, w_], g)
        g2, i2 = g.reshape(-1, g.shape[-1]), inp.reshape(-1, inp.shape[-1])
        if split_tokens:
            t = g2.shape[0] // 2
            d_w = g2[:t].t() @ i2[:t] + g2[t:].t() @ i2[t:]
        if split_rank:
            r = w.shape[0] // 2
            d_inp = (g2[:, :r] @ w[:r] + g2[:, r:] @ w[r:]).reshape(inp.shape)
        return d_inp, d_w, None, None, None


def _ref_proj(proj, x_base, x_lora, draws, via_function=False):
    """PEFT's ``Linear4bit`` LoRA forward (0.21.2, the fixture lock's; identical in 0.19.1), op for op, with R1-R4 where
    named. ``via_function`` takes the :class:`_LoraLinear` path with every split off (a self-check)."""
    a, b, s = _adapter(proj)
    result = proj.base_layer(x_base).clone()
    xf = x_lora.to(a.weight.dtype)
    if via_function or draws & {"R1", "R3", "R4"}:
        t = _LoraLinear.apply(xf, a.weight, "R1" in draws, "R3" in draws, "R4" in draws)
        u = _LoraLinear.apply(t, b.weight, False, "R3" in draws, False)
    else:
        u = b(a(xf))                                                    # lora_B(lora_A(x)), as PEFT calls them
    delta = u * s
    if "R2" in draws:
        return (result.to(delta.dtype) + delta).to(result.dtype)
    return result + delta.to(result.dtype)


def ref_read(mlp, x, gy, draws=frozenset(), *, via_function=False) -> dict:
    """The reference for the unfused path under ``draws``. dx's four terms (gate base, gate LoRA, up base, up LoRA) reach
    four separate leaves and are summed in :data:`DX_ORDER`, or :data:`DX_ORDER_R5` under R5."""
    draws = frozenset(draws)
    leaves = {k: x.detach().clone().requires_grad_(True) for k in ("gate_base", "gate_lora", "up_base", "up_lora")}
    out_g = _ref_proj(mlp.gate_proj, leaves["gate_base"], leaves["gate_lora"], draws, via_function)
    a_g = mlp.act_fn(out_g)
    out_u = _ref_proj(mlp.up_proj, leaves["up_base"], leaves["up_lora"], draws, via_function)
    h = a_g * out_u
    y = _ref_proj(mlp.down_proj, h, h, draws, via_function)
    params = _params(mlp)
    keys = list(leaves)
    got = torch.autograd.grad(y, [*leaves.values(), h, *params], gy)
    parts, dh, grads = dict(zip(keys, got[:4])), got[4], got[5:]
    order = DX_ORDER_R5 if "R5" in draws else DX_ORDER
    dx = parts[order[0]]
    for k in order[1:]:
        dx = dx + parts[k]
    return _named([out_g, out_u, y], dx, dh, grads)


def fused_read(mlp, x, gy) -> dict:
    """The engine's path on a fused ``mlp``: gate's and up's outputs are recorded from the engine's first ``_FusedLoRA``
    call, ``h`` from ``down_proj``'s base layer, which the fused forward calls with it."""
    from experts4bit_qlora.engines import dense_lora_fuse as dlf

    if not hasattr(mlp, "_e4b_dense_lora"):
        raise ValueError("fused_read: the MLP is not fused")
    seen, real = {}, dlf._FusedLoRA

    class _Recording:
        @staticmethod
        def apply(*args):
            out = real.apply(*args)
            seen.setdefault("gate_up", out)
            return out

    # A hook on the BASE layer is not one the engine falls back on: the fused forward calls base_layer as a module.
    hook = mlp.down_proj.base_layer.register_forward_pre_hook(lambda m, a: seen.__setitem__("h", a[0]))
    dlf._FusedLoRA = _Recording
    try:
        xl = x.detach().clone().requires_grad_(True)
        y = mlp(xl)
    finally:
        dlf._FusedLoRA = real
        hook.remove()
    params = _params(mlp)
    dx, dh, *grads = torch.autograd.grad(y, [xl, seen["h"], *params], gy)
    g, u = seen["gate_up"]
    return _named([g, u, y], dx, dh, grads)


def adapter_terms(mlp, x, gy) -> dict:
    """Each checked tensor's adapter term in FP32, on the unfused path (Amendment 1's shares), and its own norm.
    ``g`` in ``s.g.B`` is each projection's own output gradient. The A and B gradients are wholly adapter terms."""
    xl = x.detach().clone().requires_grad_(True)
    outs = {}
    hooks = [getattr(mlp, p).register_forward_hook(lambda m, a, o, p=p: outs.__setitem__(p, o)) for p in PROJECTIONS]
    hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m, a: outs.__setitem__("h", a[0])))
    try:
        y = mlp(xl)
    finally:
        for hk in hooks:
            hk.remove()
    params = _params(mlp)
    got = torch.autograd.grad(y, [outs["gate_proj"], outs["up_proj"], xl, outs["h"], *params], gy)
    g_gate, g_up, dx, dh, grads = got[0], got[1], got[2], got[3], got[4:]
    terms, totals = {}, {}
    with torch.no_grad():
        for p, name, inp in (("gate_proj", "out_gate", x), ("up_proj", "out_up", x), ("down_proj", "out_down", outs["h"])):
            a, b, s = _adapter(getattr(mlp, p))
            xf = inp.reshape(-1, inp.shape[-1]).float()
            terms[name] = s * (xf @ a.weight.float().t()) @ b.weight.float().t()
            totals[name] = outs[p].float()
        lora_dx = 0
        for p, g in (("gate_proj", g_gate), ("up_proj", g_up)):
            a, b, s = _adapter(getattr(mlp, p))
            lora_dx = lora_dx + (s * g.reshape(-1, g.shape[-1]).float() @ b.weight.float()) @ a.weight.float()
        terms["dx"], totals["dx"] = lora_dx, dx.float()
        a, b, s = _adapter(mlp.down_proj)
        terms["dh"] = (s * gy.reshape(-1, gy.shape[-1]).float() @ b.weight.float()) @ a.weight.float()
        totals["dh"] = dh.float()
        for name, gr in zip(TENSORS[5:], grads):
            terms[name] = totals[name] = gr.float()
    return {k: {"adapter_norm": float(torch.linalg.vector_norm(terms[k])),
                "norm": float(torch.linalg.vector_norm(totals[k]))} for k in TENSORS}


def shares(terms) -> dict:
    """Each tensor's adapter share: its adapter term's norm over its own (1 for the A and B gradients by construction)."""
    out = {}
    for k, v in terms.items():
        out[k] = (v["adapter_norm"] / v["norm"]) if v["norm"] > 0 else 0.0
    return out


def set_b(mlps, sigma, seed=B_SEED) -> None:
    """Amendment 1's B: ``Generator(seed)``, ``randn(shape) * sigma`` in FP32, drawn in ``named_parameters`` order over
    the checked MLPs (in the order given), then moved to each weight's device."""
    g = torch.Generator("cpu").manual_seed(seed)
    for mlp in mlps:
        for name, p in mlp.named_parameters():
            if ".lora_B." in "." + name:
                with torch.no_grad():
                    p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) * sigma).to(p.device))


def deviation(a, b) -> dict:
    """``max_abs`` = max |a - b| and ``rel_rms`` = ||a - b|| / ||b||, in FP32."""
    d = a.float() - b.float()
    nb = float(torch.linalg.vector_norm(b.float()))
    nd = float(torch.linalg.vector_norm(d))
    return {"max_abs": float(d.abs().max()) if d.numel() else 0.0, "rel_rms": (nd / nb) if nb > 0 else (0.0 if nd == 0 else float("inf"))}


def floor(truth, draws_read) -> dict:
    """Per tensor, the worst over the draws (``{frozenset: read}``) of max_abs and rel_rms, and which draws moved it."""
    out = {}
    for k in TENSORS:
        devs = {d: deviation(r[k], truth[k]) for d, r in draws_read.items()}
        out[k] = {"max_abs": max(v["max_abs"] for v in devs.values()),
                  "rel_rms": max(v["rel_rms"] for v in devs.values()),
                  "moved_by": sorted("+".join(sorted(d)) for d, v in devs.items() if v["max_abs"] > 0)}
    return out


def floor_streamed(truth, read) -> dict:
    """:func:`floor` over the 31 draws with ``read(draw)`` called one draw at a time, each discarded once folded in (at
    the fixture's size a draw holds about 210 MB)."""
    out = {k: {"max_abs": 0.0, "rel_rms": 0.0, "moved_by": []} for k in TENSORS}
    for d in draw_sets():
        r = read(d)
        for k in TENSORS:
            dev = deviation(r[k], truth[k])
            out[k]["max_abs"] = max(out[k]["max_abs"], dev["max_abs"])
            out[k]["rel_rms"] = max(out[k]["rel_rms"], dev["rel_rms"])
            if dev["max_abs"] > 0:
                out[k]["moved_by"].append("+".join(sorted(d)))
        del r
    for v in out.values():
        v["moved_by"].sort()
    return out


def coverage(fl) -> dict:
    """Which single redraws appear among the draws that moved each tensor (the registration's coverage map)."""
    return {k: sorted({r for d in v["moved_by"] for r in d.split("+")}) for k, v in fl.items()}


def bar(truth, fused, fl, terms) -> dict:
    """The bar: every tensor at or below its floor on max_abs and rel_rms (a floor of 0 means bit for bit). ``e_min`` is
    the smallest adapter-term error, as a fraction of the adapter term, that the floor lets the bar see (Amendment 1)."""
    rows = {}
    for k in TENSORS:
        dev = deviation(fused[k], truth[k])
        f = fl[k]
        ok = dev["max_abs"] <= f["max_abs"] and dev["rel_rms"] <= f["rel_rms"]
        an = terms[k]["adapter_norm"]
        e_min = (f["rel_rms"] * terms[k]["norm"] / an) if an > 0 else float("inf")
        rows[k] = {**dev, "floor_max_abs": f["max_abs"], "floor_rel_rms": f["rel_rms"], "pass": ok, "e_min": e_min}
    return rows


def choose_sigma(shares_by_sigma):
    """Amendment 1's sigma: the smallest of :data:`SIGMAS` whose shares (a list of per-layer ``{tensor: share}``, read
    on the unfused path only) are all at least :data:`SHARE_MIN`; ``None`` means ``VOID_LORA_SHARE``."""
    for sigma in SIGMAS:
        layers = shares_by_sigma.get(sigma)
        if layers and all(v >= SHARE_MIN for sh in layers for v in sh.values()):
            return sigma
    return None


def vacuous(read) -> list:
    """The A and B gradients that are all zero or not finite (Amendment 1: ``VOID_LORA_SHARE``)."""
    return [k for k in TENSORS[5:] if not bool(torch.isfinite(read[k]).all()) or not bool(read[k].any())]


def outcome(rows_by_layer, rank) -> str:
    """``PHASE1_FAIL: <layer:tensor ...>`` if any tensor is over its floor; else ``PHASE1_PASS``, with
    ``RANK1_UNRESOLVED: <layer:tensor ...>`` naming every tensor whose ``e_min`` exceeds 1/rank (Amendment 1)."""
    over = [f"{layer}:{k}" for layer, rows in rows_by_layer.items() for k, v in rows.items() if not v["pass"]]
    if over:
        return "PHASE1_FAIL: " + " ".join(over)
    hidden = [f"{layer}:{k}" for layer, rows in rows_by_layer.items() for k, v in rows.items() if v["e_min"] > 1 / rank]
    return "PHASE1_PASS" + (", RANK1_UNRESOLVED: " + " ".join(hidden) if hidden else "")


@contextlib.contextmanager
def lora_forward_counter(model):
    """Count PEFT LoRA forwards per module name, by wrapping each LoRA class's ``forward`` (a class attribute, never a
    hook: since e4b#1814 a hook on a projection makes a fused MLP fall back). Yields a ``Counter``; restores on exit."""
    names = {id(m): n for n, m in model.named_modules()}
    counts = collections.Counter()
    classes = {type(m) for m in model.modules() if hasattr(m, "lora_A") and hasattr(m, "base_layer")}
    own = {cls: cls.__dict__.get("forward") for cls in classes}
    for cls in classes:
        def wrapped(self, *args, _real=cls.forward, **kwargs):
            counts[names.get(id(self), "?")] += 1
            return _real(self, *args, **kwargs)
        cls.forward = wrapped
    try:
        yield counts
    finally:
        for cls, f in own.items():
            if f is None:
                del cls.forward
            else:
                cls.forward = f


def bit_diagnostics(a, b) -> dict:
    """For a tensor the bar holds bit for bit: how many elements differ, how many of those flip sign, and the largest
    distance in units of the last place over the same-signed ones (the integer views' difference; across a sign flip it
    means nothing, so a flip is counted, never measured)."""
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[a.dtype]
    differ = a != b
    n = int(differ.sum())
    if n == 0:
        return {"differing": 0, "sign_flips": 0, "max_ulp": 0}
    flips = differ & (torch.signbit(a) != torch.signbit(b))
    same = differ & ~flips
    ia, ib = a.contiguous().view(view).long(), b.contiguous().view(view).long()
    max_ulp = int((ia - ib).abs()[same].max()) if bool(same.any()) else 0
    return {"differing": n, "sign_flips": int(flips.sum()), "max_ulp": max_ulp}


def read_layers(mlps, xs, gys, *, enable, disable, stats, sigma=None, stage=None, evict=None, adapters=None):
    """Phase 1's per-module read over ``mlps`` (``{layer: mlp}``) at captured ``xs`` / ``gys``.

    ``sigma=None`` runs Amendment 1's ladder (``set_b`` then the unfused shares) and stops at the first qualifying
    sigma; ``sigma=False`` keeps the adapters as they are (the reported state). ``enable()`` fuses the whole model and
    returns the adapter count (``adapters`` expected, else ``ENGAGE_COUNT``); ``disable()`` undoes it; ``stats`` is the
    engine's ``DENSE_MLP_LORA_STATS``. ``stage`` / ``evict`` bracket each layer's reads (dense streaming)."""
    stage, evict = stage or (lambda i: None), evict or (lambda i: None)
    record = {"layers": {}, "sigma": None, "ladder": {}}
    order = list(mlps)

    def layer_shares():
        out = {}
        for i in order:
            stage(i)
            try:
                out[i] = shares(adapter_terms(mlps[i], xs[i], gys[i]))
            finally:
                evict(i)
        return out

    if sigma is None:
        for s in SIGMAS:
            set_b([mlps[i] for i in order], s)
            record["ladder"][s] = layer_shares()
            if choose_sigma({s: list(record["ladder"][s].values())}) == s:
                record["sigma"] = s
                break
        if record["sigma"] is None:
            record["outcome"] = "VOID_LORA_SHARE"
            return record
    else:
        record["ladder"]["as-is"] = layer_shares()
    rows_by_layer = {}
    for i in order:
        mlp, x, gy = mlps[i], xs[i], gys[i]
        stage(i)
        try:
            truth = peft_read(mlp, x, gy)
            bad = vacuous(truth)
            if bad:
                record["outcome"] = f"VOID_LORA_SHARE: {i}:" + " ".join(bad)
                return record
            mismatch = [k for via in (False, True) for k, v in ref_read(mlp, x, gy, via_function=via).items()
                        if not torch.equal(v, truth[k])]
            if mismatch:
                record["outcome"] = f"REF_MISMATCH: {i}:" + " ".join(sorted(set(mismatch)))
                return record
            fl = floor_streamed(truth, lambda d: ref_read(mlp, x, gy, d))
            terms = adapter_terms(mlp, x, gy)
            n = enable()
            try:
                if (adapters is not None and n != adapters) or stats.get("refused"):
                    record["outcome"] = f"ENGAGE_COUNT: {n}, refused {sorted(stats.get('refused', {}))}"
                    return record
                fused = fused_read(mlp, x, gy)
                if stats.get("fallback", 0):
                    record["outcome"] = f"VOID_FALLBACK: {stats['fallback']}"
                    return record
            finally:
                disable()
        finally:
            evict(i)
        rows = bar(truth, fused, fl, terms)
        for k, row in rows.items():
            row["adapter_norm"], row["norm"] = terms[k]["adapter_norm"], terms[k]["norm"]
            if fl[k]["max_abs"] == 0:
                row["bitwise_required"] = bit_diagnostics(fused[k], truth[k])
        record["layers"][i] = {"rows": rows, "coverage": coverage(fl), "moved_by": {k: v["moved_by"] for k, v in fl.items()}}
        rows_by_layer[i] = rows
    rank = _adapter(mlps[order[0]].gate_proj)[0].weight.shape[0]
    record["outcome"] = outcome(rows_by_layer, rank)
    record["e_min_max"] = max(r["e_min"] for rows in rows_by_layer.values() for r in rows.values())
    return record


# --- Amendment 4: a floor without its own candidate, an FP64 accuracy judge -------------------------------------------

#: Amendment 4 (a): ``(result.float() + delta).to(bf16)``, an FP32 add then one cast, is the fused epilogue's own
#: arithmetic, so every draw containing it leaves the floor.
EPILOGUE_REDRAW = "R2"
OUTPUTS = ("out_gate", "out_up", "out_down")


def landed(mlp, x, gy, today, fused) -> list:
    """Amendment 4 (a), mechanically: the single redraws strictly closer to the fused read than to today's path, on both
    max-abs and relative RMS, on any projection output."""
    out = []
    for r in DRAWS:
        read = ref_read(mlp, x, gy, {r})
        for k in OUTPUTS:
            to_fused, to_today = deviation(read[k], fused[k]), deviation(read[k], today[k])
            if to_fused["max_abs"] < to_today["max_abs"] and to_fused["rel_rms"] < to_today["rel_rms"]:
                out.append(r)
                break
        del read
    return out


def post_a_draws(excluded) -> list:
    """The draws that contain none of ``excluded`` (with R2 alone excluded: the 15 combinations of R1, R3, R4, R5)."""
    excluded = set(excluded)
    return [d for d in draw_sets() if not (d & excluded)]


def _base_weight(base):
    """The exact values a base layer multiplies by: an NF4 ``Linear4bit``'s dequantized weight, else its weight."""
    w = base.weight
    if getattr(w, "quant_state", None) is not None:
        import bitsandbytes.functional as BF

        return BF.dequantize_4bit(w.data, w.quant_state)
    return w.detach()


def fp64_read(mlp, x, gy) -> dict:
    """Amendment 4 (b): the module in FP64 with no intermediate rounding. The base weights are the dequantized NF4 values
    upcast; A, B, ``x`` and ``gy`` are upcast; ``act_fn`` runs in FP64; every output and gradient comes from autograd in
    FP64. The eleven tensors, as the other reads give them."""
    d = torch.float64
    xl = x.detach().to(d).requires_grad_(True)
    weights, params = {}, []
    for p in PROJECTIONS:
        proj = getattr(mlp, p)
        a, b, s = _adapter(proj)
        A = a.weight.detach().to(d).requires_grad_(True)
        B = b.weight.detach().to(d).requires_grad_(True)
        weights[p] = (_base_weight(proj.base_layer).to(d), A, B, s)
        params += [A, B]

    def project(p, inp):
        w, A, B, s = weights[p]
        return inp @ w.t() + s * ((inp @ A.t()) @ B.t())

    g = project("gate_proj", xl)
    u = project("up_proj", xl)
    h = mlp.act_fn(g) * u
    y = project("down_proj", h)
    dx, dh, *grads = torch.autograd.grad(y, [xl, h, *params], gy.to(d))
    return _named([g, u, y], dx, dh, grads)


MARGIN_B = 0.05          # Amendment 4 (b): the fused path passes iff its error <= (1 + m) x the unfused band


def error(a, ref) -> dict:
    """``a`` against the FP64 reference, in FP64: ``rel_rms`` (= ||a - ref|| / ||ref||, which decides) and ``max_abs``
    (recorded)."""
    e = a.double() - ref
    nr = float(torch.linalg.vector_norm(ref))
    ne = float(torch.linalg.vector_norm(e))
    return {"rel_rms": (ne / nr) if nr > 0 else (0.0 if ne == 0 else float("inf")),
            "max_abs": float(e.abs().max()) if e.numel() else 0.0}


def floors_streamed(today, ref, read, draws) -> tuple:
    """One pass over ``draws`` (``read(draw)`` one at a time). Returns:
    - (a)'s floor: each draw's deviation from today's path;
    - (b)'s per-path errors against the FP64 reference, for today's path and every draw (``{path: {tensor: error}}``)."""
    fl = {k: {"max_abs": 0.0, "rel_rms": 0.0, "moved_by": []} for k in TENSORS}
    errs = {"today": {k: error(today[k], ref[k]) for k in TENSORS}}
    for d in draws:
        r = read(d)
        name = "+".join(sorted(d))
        errs[name] = {}
        for k in TENSORS:
            dev = deviation(r[k], today[k])
            fl[k]["max_abs"] = max(fl[k]["max_abs"], dev["max_abs"])
            fl[k]["rel_rms"] = max(fl[k]["rel_rms"], dev["rel_rms"])
            if dev["max_abs"] > 0:
                fl[k]["moved_by"].append(name)
            errs[name][k] = error(r[k], ref[k])
        del r
    for v in fl.values():
        v["moved_by"].sort()
    return fl, errs


def band(errs, k, *, leave_out=None) -> tuple:
    """The unfused band for tensor ``k``: the largest relative-RMS error over the paths in ``errs`` (less ``leave_out``),
    and the path that set it."""
    rows = [(e[k]["rel_rms"], p) for p, e in errs.items() if p != leave_out]
    return max(rows) if rows else (None, None)


def calibrate(errs_by_layer, margin=MARGIN_B) -> list:
    """Amendment 4 (b)'s false-reject control: each unfused path in turn is a neutral pseudo-candidate against the band of
    the others, with the same margin, on every tensor at every layer. Returns the failures (empty = the bar can judge)."""
    fails = []
    for layer, errs in errs_by_layer.items():
        for p in errs:
            for k in TENSORS:
                b, _by = band(errs, k, leave_out=p)
                if b is not None and errs[p][k]["rel_rms"] > (1 + margin) * b:
                    fails.append(f"{layer}:{k}:{p}")
    return fails


def bar_b(ref, fused, errs, terms, margin=MARGIN_B) -> dict:
    """Amendment 4 (b): a tensor passes iff the fused relative-RMS error against the FP64 reference is at most
    (1 + m) x the unfused band (today's path and the post-(a) draws). ``e_min`` = ((1 + m) x band x ||T_ref||) /
    adapter-term norm (Amendment 1, restated on (b))."""
    rows = {}
    for k in TENSORS:
        e = error(fused[k], ref[k])
        b, by = band(errs, k)
        ok = e["rel_rms"] <= (1 + margin) * b
        an = terms[k]["adapter_norm"]
        e_min = ((1 + margin) * b * float(torch.linalg.vector_norm(ref[k])) / an) if an > 0 else float("inf")
        rows[k] = {"err": e, "band_rel_rms": b, "band_by": by, "margin": margin, "pass": ok, "e_min": e_min,
                   "today_err": errs["today"][k]}
    return rows


def outcome_a4(rows_a, rows_b, rank, calibration_fails, paths) -> str:
    """Amendment 4: (b) decides after its calibration.
    - Fewer than two unfused paths, or any calibration failure: ``INDETERMINATE``.
    - Any (b) failure: ``PHASE1_FAIL``.
    - Else ``PHASE1_PASS``, naming where (a) failed or had no floor (``ACCURACY_OVER_REORDER``) and every ``e_min``
      above 1/rank (``RANK1_UNRESOLVED``)."""
    def names(rows, bad):
        return [f"{layer}:{k}" for layer, r in rows.items() for k, v in r.items() if bad(v)]

    if paths < 2:
        return "INDETERMINATE: fewer than two unfused paths to calibrate against"
    if calibration_fails:
        return "INDETERMINATE: calibration failed " + " ".join(calibration_fails)
    b_fail = names(rows_b, lambda v: not v["pass"])
    if b_fail:
        return "PHASE1_FAIL: " + " ".join(b_fail)
    out = "PHASE1_PASS"
    acc = names(rows_a, lambda v: v is None or not v["pass"]) if rows_a else [f"{layer}:*" for layer in rows_b]
    if acc:
        out += ", ACCURACY_OVER_REORDER: " + " ".join(acc)
    hidden = names(rows_b, lambda v: v["e_min"] > 1 / rank)
    if hidden:
        out += ", RANK1_UNRESOLVED: " + " ".join(hidden)
    return out


def _to(read, device) -> dict:
    return {k: v.to(device) for k, v in read.items()}


def read_layers_a4(mlps, xs, gys, *, enable, disable, stats, sigma=None, stage=None, evict=None, adapters=None):
    """Phase 1's per-module read under Amendment 4, in two passes over ``mlps`` (``{layer: mlp}``).

    - **Pass 1, per layer:** today's path, the vacuity check, both reference self-checks, the fused read (``enable`` /
      ``disable``, ``fallback`` = 0) and the landing check. Today's path and the fused read are kept on the CPU.
    - **Then:** the excluded redraws are R2 plus every one that landed at any layer, and the post-(a) draw set follows.
    - **Pass 2, per layer:** the FP64 reference, the adapter terms, and one streamed pass for (a)'s floor and (b)'s
      band; then both bars.

    ``sigma`` and the ladder are Amendment 1's, as in :func:`read_layers`."""
    stage, evict = stage or (lambda i: None), evict or (lambda i: None)
    record = {"amendment": 4, "layers": {}, "sigma": None, "ladder": {}, "landed": {}}
    order = list(mlps)

    def layer_shares():
        out = {}
        for i in order:
            stage(i)
            try:
                out[i] = shares(adapter_terms(mlps[i], xs[i], gys[i]))
            finally:
                evict(i)
        return out

    if sigma is None:
        for s in SIGMAS:
            set_b([mlps[i] for i in order], s)
            record["ladder"][s] = layer_shares()
            if choose_sigma({s: list(record["ladder"][s].values())}) == s:
                record["sigma"] = s
                break
        if record["sigma"] is None:
            record["outcome"] = "VOID_LORA_SHARE"
            return record
    else:
        record["ladder"]["as-is"] = layer_shares()
    kept = {}
    for i in order:
        mlp, x, gy = mlps[i], xs[i], gys[i]
        stage(i)
        try:
            today = peft_read(mlp, x, gy)
            bad = vacuous(today)
            if bad:
                record["outcome"] = f"VOID_LORA_SHARE: {i}:" + " ".join(bad)
                return record
            mismatch = [k for via in (False, True) for k, v in ref_read(mlp, x, gy, via_function=via).items()
                        if not torch.equal(v, today[k])]
            if mismatch:
                record["outcome"] = f"REF_MISMATCH: {i}:" + " ".join(sorted(set(mismatch)))
                return record
            n = enable()
            try:
                if (adapters is not None and n != adapters) or stats.get("refused"):
                    record["outcome"] = f"ENGAGE_COUNT: {n}, refused {sorted(stats.get('refused', {}))}"
                    return record
                fused = fused_read(mlp, x, gy)
                if stats.get("fallback", 0):
                    record["outcome"] = f"VOID_FALLBACK: {stats['fallback']}"
                    return record
            finally:
                disable()
            record["landed"][i] = landed(mlp, x, gy, today, fused)
            kept[i] = (_to(today, "cpu"), _to(fused, "cpu"))
            del today, fused
        finally:
            evict(i)
    excluded = sorted({EPILOGUE_REDRAW} | {r for v in record["landed"].values() for r in v})
    draws = post_a_draws(excluded)
    record["excluded"], record["draws"] = excluded, len(draws)
    record["epilogue_landed"] = all(EPILOGUE_REDRAW in v for v in record["landed"].values())
    rows_a, rows_b, errs_by_layer = {}, {}, {}
    for i in order:
        mlp, x, gy = mlps[i], xs[i], gys[i]
        stage(i)
        try:
            ref = fp64_read(mlp, x, gy)
            terms = adapter_terms(mlp, x, gy)
            device = x.device
            today, fused = (_to(t, device) for t in kept.pop(i))
            fl, errs = floors_streamed(today, ref, lambda d: ref_read(mlp, x, gy, d), draws)
            errs_by_layer[i] = errs
            if draws:
                rows_a[i] = bar(today, fused, fl, terms)
                for k, row in rows_a[i].items():
                    row["adapter_norm"], row["norm"] = terms[k]["adapter_norm"], terms[k]["norm"]
                    if fl[k]["max_abs"] == 0:
                        row["bitwise_required"] = bit_diagnostics(fused[k], today[k])
            rows_b[i] = bar_b(ref, fused, errs, terms)
            record["layers"][i] = {"a": rows_a.get(i, "NO_FLOOR"), "b": rows_b[i], "errors": errs,
                                   "coverage": coverage(fl) if draws else {}, "moved_by": {k: v["moved_by"] for k, v in fl.items()}}
            del ref, today, fused
        finally:
            evict(i)
    record["calibration_fails"] = calibrate(errs_by_layer)
    rank = _adapter(mlps[order[0]].gate_proj)[0].weight.shape[0]
    record["outcome"] = outcome_a4(rows_a, rows_b, rank, record["calibration_fails"], 1 + len(draws))
    record["e_min_max"] = max(r["e_min"] for rows in rows_b.values() for r in rows.values())
    return record
