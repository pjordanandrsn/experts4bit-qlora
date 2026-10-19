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
    """For a tensor the bar holds bit for bit: how many elements differ, and the largest distance in units of the last
    place (the integer views' difference, for same-signed values)."""
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[a.dtype]
    differ = a != b
    n = int(differ.sum())
    if n == 0:
        return {"differing": 0, "max_ulp": 0}
    ia, ib = a.contiguous().view(view).long(), b.contiguous().view(view).long()
    return {"differing": n, "max_ulp": int((ia - ib).abs()[differ].max())}


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
