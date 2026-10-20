"""DQ11 MLP-adapter BUILD, Phase 2's (b) part (``DQ11-MLP-BUILD-AMENDMENT-5.md`` sections 2-4).

It reads the same eleven tensors as Phase 1 (``dq11_mlp_phase1``) on the same captured ``x`` and ``gy``, with three
changes:

- **Leveled families.** Each family splits ONE op of the unfused path into ``c`` equal chunks, computed in that op's own
  dtype and summed left to right in that dtype, before the cast the path already has. No family moves, adds or removes a
  cast. The LoRA GEMMs are FP32 on this path, so the partials and their sums are FP32.
  - **R1(c):** the forward A GEMM, split along K (the in-features), c in {1, 2, 4, 8}. c = 2 is Phase 1's R1.
  - **R6(c):** the forward B GEMM, split along the rank, c in {1, 2, 4}.
  - **R9(c):** the B GEMM's input gradient (gy.B), split along its K (the out-features), c in {1, 2, 4, 8}.
  - **R3(c):** the A and B weight-gradient GEMMs, split along the tokens, c in {1, 2, 4}. c = 2 is Phase 1's R3.
  - **R4(c):** the A GEMM's input gradient (.A), split along the rank, c in {1, 2, 4}. c = 2 is Phase 1's R4.
  - **R5:** dx's four bf16 terms in the other order (Phase 1's R5).
  R2 (the base and the delta added in FP32, then one cast) is the fused epilogue's own arithmetic and is not a family.
- **Distinct paths.** Per tensor, U_T is the set of distinct results (sha256 of the bytes) among today's path and the
  draws that keep no landed family level for that tensor. k_T = |U_T|. Below 9, the tensor is ``UNJUDGED_AT_K`` and its
  calibration is not run. At 9 or more, the leave-one-out calibration runs over U_T; if it fails, the tensor is
  ``CALIBRATION_INDETERMINATE``; else (b) judges it PASS or FAIL.
- **Landing per tensor.** The unit tested is a family at one level (every other family at c = 1). It lands on T if, on
  T, it is closer to the fused path than to today's on both metrics, at any checked layer. The distances are recorded
  for every such draw and tensor.
"""
from __future__ import annotations

import hashlib
import itertools
import random

import torch
import torch.nn.functional as F

import dq11_mlp_phase1 as p1

TENSORS = p1.TENSORS
OUTPUTS = p1.OUTPUTS
FAMILIES = {"R1": (1, 2, 4, 8), "R6": (1, 2, 4), "R9": (1, 2, 4, 8), "R3": (1, 2, 4), "R4": (1, 2, 4), "R5": (False, True)}
DEFAULT = {"R1": 1, "R6": 1, "R9": 1, "R3": 1, "R4": 1, "R5": False}
#: The stratified subset (section 3): for every tensor, the full level grid of two families that move it.
#: The outputs: R1 x R6; dh: R9 x R4; dx: R9 x R4 x R5; dA and dB: R1 x R3.
GRIDS = (("R1", "R6"), ("R9", "R4"), ("R9", "R4", "R5"), ("R1", "R3"))
SEEDED = 24          # the seeded sample of the full product on top of the grids; fixed in Amendment 5 after the shakedown
SEED = p1.B_SEED
K_MIN = 9            # Amendment 4a's floor, applied to k_T (Amendment 5 section 2)
MARGIN = p1.MARGIN_B
UNJUDGED = "UNJUDGED_AT_K"
INDETERMINATE = "CALIBRATION_INDETERMINATE"


# --- draws ------------------------------------------------------------------------------------------------------------

def draw(**levels) -> tuple:
    """A draw as a sorted tuple of the families it moves off their default, e.g. ``draw(R1=4, R5=True)``."""
    bad = [f for f in levels if f not in FAMILIES or levels[f] not in FAMILIES[f]]
    if bad:
        raise ValueError(f"unknown family or level: {bad}")
    return tuple(sorted((f, c) for f, c in levels.items() if c != DEFAULT[f]))


def levels(d) -> dict:
    out = dict(DEFAULT)
    out.update(dict(d))
    return out


def name(d) -> str:
    return "+".join("R5" if f == "R5" else f"{f}x{c}" for f, c in d) or "today"


def single_levels() -> list:
    """Every family at each of its non-default levels with every other family at its default: the units landing tests."""
    return [draw(**{f: c}) for f, cs in FAMILIES.items() for c in cs if c != DEFAULT[f]]


def stratified(seeded=SEEDED, seed=SEED) -> list:
    """The draws (today's path excluded): the union of :data:`GRIDS`' full level grids, other families at default, plus
    ``seeded`` draws sampled from the rest of the full product with ``random.Random(seed)``. Sorted, so fixed."""
    out = set()
    for grid in GRIDS:
        for combo in itertools.product(*(FAMILIES[f] for f in grid)):
            out.add(draw(**dict(zip(grid, combo))))
    full = [draw(**dict(zip(FAMILIES, combo))) for combo in itertools.product(*FAMILIES.values())]
    rest = sorted(set(full) - out - {()})
    out |= set(random.Random(seed).sample(rest, min(seeded, len(rest))))
    out.discard(())
    return sorted(out)


# --- the leveled reference --------------------------------------------------------------------------------------------

def _bounds(n, c):
    if n % c:
        raise ValueError(f"PHASE2_REFUSAL: {n} does not split into {c} equal chunks")
    step = n // c
    return [(i * step, (i + 1) * step) for i in range(c)]


def _left_sum(parts):
    acc = parts[0]
    for p in parts[1:]:
        acc = acc + p
    return acc


def _chunked_linear(inp, w, c):
    """``F.linear(inp, w)`` split along K (``w.shape[1]``) into ``c`` chunks, each in ``inp``'s dtype, summed left to
    right in that dtype. ``c = 1`` is ``F.linear`` itself."""
    if c == 1:
        return F.linear(inp, w)
    return _left_sum([F.linear(inp[..., a:b], w[:, a:b]) for a, b in _bounds(w.shape[1], c)])


class _LoraLinearC(torch.autograd.Function):
    """A LoRA A or B matmul with leveled splits:
    - ``ck``: the forward, split along K (``w.shape[1]``): R1 on A, R6 on B;
    - ``ct``: the weight gradient, split along the tokens: R3;
    - ``cg``: the input gradient, split along ``w.shape[0]``: R4 on A (the rank), R9 on B (the out-features).
    The parts no split names come from autograd on the same forward, as Phase 1's ``_LoraLinear`` does, so with every
    split at 1 the values are what autograd computes on PEFT's path."""

    @staticmethod
    def forward(ctx, inp, w, ck, ct, cg):
        ctx.save_for_backward(inp, w)
        ctx.levels = (ck, ct, cg)
        return _chunked_linear(inp, w, ck)

    @staticmethod
    def backward(ctx, g):
        inp, w = ctx.saved_tensors
        ck, ct, cg = ctx.levels
        with torch.enable_grad():
            i_, w_ = inp.detach().requires_grad_(True), w.detach().requires_grad_(True)
            d_inp, d_w = torch.autograd.grad(_chunked_linear(i_, w_, ck), [i_, w_], g)
        g2, i2 = g.reshape(-1, g.shape[-1]), inp.reshape(-1, inp.shape[-1])
        if ct > 1:
            d_w = _left_sum([g2[a:b].t() @ i2[a:b] for a, b in _bounds(g2.shape[0], ct)])
        if cg > 1:
            d_inp = _left_sum([g2[:, a:b] @ w[a:b] for a, b in _bounds(w.shape[0], cg)]).reshape(inp.shape)
        return d_inp, d_w, None, None, None


def _ref_proj(proj, x_base, x_lora, lv, via_function=False):
    """PEFT 0.21.2's ``Linear4bit`` LoRA forward, op for op (as ``dq11_mlp_phase1._ref_proj``), with the leveled splits
    of ``lv``. Every cast stays where PEFT puts it: the LoRA input to the adapter's dtype, the delta to the base's."""
    a, b, s = p1._adapter(proj)
    result = proj.base_layer(x_base).clone()
    xf = x_lora.to(a.weight.dtype)
    if via_function or any(lv[f] != DEFAULT[f] for f in ("R1", "R6", "R9", "R3", "R4")):
        t = _LoraLinearC.apply(xf, a.weight, lv["R1"], lv["R3"], lv["R4"])
        u = _LoraLinearC.apply(t, b.weight, lv["R6"], lv["R3"], lv["R9"])
    else:
        u = b(a(xf))
    delta = u * s
    return result + delta.to(result.dtype)


def ref_read(mlp, x, gy, d=(), *, via_function=False) -> dict:
    """The unfused path's reference under draw ``d``: dx's four terms reach four leaves and are summed in bf16 in
    Phase 1's ``DX_ORDER``, or ``DX_ORDER_R5`` under R5."""
    lv = levels(d)
    leaves = {k: x.detach().clone().requires_grad_(True) for k in ("gate_base", "gate_lora", "up_base", "up_lora")}
    out_g = _ref_proj(mlp.gate_proj, leaves["gate_base"], leaves["gate_lora"], lv, via_function)
    a_g = mlp.act_fn(out_g)
    out_u = _ref_proj(mlp.up_proj, leaves["up_base"], leaves["up_lora"], lv, via_function)
    h = a_g * out_u
    y = _ref_proj(mlp.down_proj, h, h, lv, via_function)
    params = p1._params(mlp)
    keys = list(leaves)
    got = torch.autograd.grad(y, [*leaves.values(), h, *params], gy)
    parts, dh, grads = dict(zip(keys, got[:4])), got[4], got[5:]
    order = p1.DX_ORDER_R5 if lv["R5"] else p1.DX_ORDER
    dx = parts[order[0]]
    for k in order[1:]:
        dx = dx + parts[k]
    return p1._named([out_g, out_u, y], dx, dh, grads)


#: Phase 1's single redraws and their leveled equals; the leveled reference must reproduce each bit for bit.
PHASE1_EQUAL = {"R1": draw(R1=2), "R3": draw(R3=2), "R4": draw(R4=2), "R5": draw(R5=True)}


def digest(t) -> str:
    """sha256 over a tensor's dtype, shape and bytes: two paths are the same iff their results are equal bit for bit."""
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256(f"{t.dtype}:{tuple(t.shape)}:".encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# --- landing (section 2) ----------------------------------------------------------------------------------------------

def landing(mlp, x, gy, today, fused) -> dict:
    """Each family at each level (the others at default) against the fused read and today's path, on every tensor:
    ``{name: {tensor: {"to_fused", "to_today", "lands"}}}``. A family level lands on T iff it is strictly closer to the
    fused read than to today's on T, on both max-abs and relative RMS (so a level that does not move T never lands)."""
    out = {}
    for d in single_levels():
        r = ref_read(mlp, x, gy, d)
        row = {}
        for k in TENSORS:
            tf, tt = p1.deviation(r[k], fused[k]), p1.deviation(r[k], today[k])
            row[k] = {"to_fused": tf, "to_today": tt,
                      "lands": tf["max_abs"] < tt["max_abs"] and tf["rel_rms"] < tt["rel_rms"]}
        out[name(d)] = row
        del r
    return out


def landed_by_tensor(landing_by_layer) -> dict:
    """``{tensor: sorted family-level names that land on it at any checked layer}``."""
    out = {k: set() for k in TENSORS}
    for rows in landing_by_layer.values():
        for n, row in rows.items():
            for k, v in row.items():
                if v["lands"]:
                    out[k].add(n)
    return {k: sorted(v) for k, v in out.items()}


def excluded_for(d, landed_names) -> bool:
    """A draw leaves U_T when it contains any family level that lands on T."""
    return any(name((fc,)) in landed_names for fc in d)


# --- the streamed (b) pass, distinct paths, calibration ---------------------------------------------------------------

def paths_streamed(today, ref, read, draws, landed) -> dict:
    """One pass over ``draws``. Per tensor: the distinct results kept in U_T (today's path first) with their errors
    against the FP64 reference, the draws that gave each, and (a)'s floor over the kept draws (reported only)."""
    out = {}
    for k in TENSORS:
        h = digest(today[k])
        out[k] = {"paths": {h: {"err": p1.error(today[k], ref[k]), "draws": ["today"]}},
                  "floor": {"max_abs": 0.0, "rel_rms": 0.0}, "excluded_draws": 0}
    for d in draws:
        r = read(d)
        n = name(d)
        for k in TENSORS:
            row = out[k]
            if excluded_for(d, landed[k]):
                row["excluded_draws"] += 1
                continue
            h = digest(r[k])
            if h in row["paths"]:
                row["paths"][h]["draws"].append(n)
            else:
                row["paths"][h] = {"err": p1.error(r[k], ref[k]), "draws": [n]}
            dev = p1.deviation(r[k], today[k])
            row["floor"]["max_abs"] = max(row["floor"]["max_abs"], dev["max_abs"])
            row["floor"]["rel_rms"] = max(row["floor"]["rel_rms"], dev["rel_rms"])
        del r
    return out


def judge(paths, fused_err, *, margin=MARGIN, k_min=K_MIN) -> dict:
    """One tensor at one layer. ``paths`` = ``{digest: {"err": ...}}`` (U_T, distinct). Returns k, the bound, the band,
    the calibration's failures (run only at k >= k_min) and the status: UNJUDGED_AT_K, CALIBRATION_INDETERMINATE, PASS
    or FAIL."""
    errs = {h: v["err"]["rel_rms"] for h, v in paths.items()}
    k = len(errs)
    band = max(errs.values())
    row = {"k": k, "false_reject_bound": 1 / (k + 1), "band_rel_rms": band, "margin": margin,
           "fused_err": fused_err, "calibration_fails": []}
    if k < k_min:
        row["status"] = UNJUDGED
        return row
    for h, e in errs.items():
        others = max(v for g, v in errs.items() if g != h)
        if e > (1 + margin) * others:
            row["calibration_fails"].append(h[:12])
    if row["calibration_fails"]:
        row["status"] = INDETERMINATE
        return row
    row["status"] = "PASS" if fused_err["rel_rms"] <= (1 + margin) * band else "FAIL"
    return row


def outcome(rows_by_layer, rank) -> dict:
    """The (b) part's outcome over every checked tensor (section 7's rungs 3-5 read it). Every list names
    ``layer:tensor``."""
    def names(pred):
        return [f"{layer}:{k}" for layer, rows in rows_by_layer.items() for k, v in rows.items() if pred(v)]

    return {"fail": names(lambda v: v["status"] == "FAIL"),
            "calibration_indeterminate": names(lambda v: v["status"] == INDETERMINATE),
            "unjudged_at_k": names(lambda v: v["status"] == UNJUDGED),
            "passed": names(lambda v: v["status"] == "PASS"),
            "rank1_unresolved": names(lambda v: v["e_min"] > 1 / rank),
            "accuracy_over_reorder": names(lambda v: not v["a"]["pass"]),
            "k_min_seen": min(v["k"] for rows in rows_by_layer.values() for v in rows.values())}


def read_layers(mlps, xs, gys, *, enable, disable, stats, counter, sigma=None, stage=None, evict=None, adapters=None,
                draws=None) -> dict:
    """The (b) part, in two passes over ``mlps`` (``{layer: mlp}``), as Phase 1's ``read_layers_a4``.

    - **The adapter state:** Amendment 1's sigma ladder, unless ``sigma`` is given.
    - **Pass 1, per layer:** today's path, the vacuity check, the reference self-checks (bit-for-bit equal to PEFT, through
      the Function too, and to Phase 1's own single redraws), the fused read (engagement: ``adapters`` engaged, nothing
      refused, fallback 0, no PEFT MLP LoRA forward through ``counter``), and the landing tests.
    - **Pass 2, per layer:** the FP64 reference, the adapter terms, one streamed pass over the stratified draws, and each
      tensor judged on its distinct paths."""
    stage, evict = stage or (lambda i: None), evict or (lambda i: None)
    draws = stratified() if draws is None else draws
    record = {"amendment": 5, "layers": {}, "sigma": None, "ladder": {}, "landing": {}, "draws": [name(d) for d in draws]}
    order = list(mlps)

    def layer_shares():
        out = {}
        for i in order:
            stage(i)
            try:
                out[i] = p1.shares(p1.adapter_terms(mlps[i], xs[i], gys[i]))
            finally:
                evict(i)
        return out

    if sigma is None:
        for s in p1.SIGMAS:
            p1.set_b([mlps[i] for i in order], s)
            record["ladder"][s] = layer_shares()
            if p1.choose_sigma({s: list(record["ladder"][s].values())}) == s:
                record["sigma"] = s
                break
        if record["sigma"] is None:
            record["void"] = "VOID_LORA_SHARE"
            return record
    else:
        record["sigma"] = sigma
        p1.set_b([mlps[i] for i in order], sigma)
        record["ladder"][sigma] = layer_shares()
    engagement = {"adapters": [], "refused": {}, "fallback": 0, "mlp_lora_forwards": 0}
    record["engagement"] = engagement
    kept = {}
    for i in order:
        mlp, x, gy = mlps[i], xs[i], gys[i]
        stage(i)
        try:
            today = p1.peft_read(mlp, x, gy)
            bad = p1.vacuous(today)
            if bad:
                record["void"] = f"VOID_LORA_SHARE: {i}:" + " ".join(bad)
                return record
            mismatch = [f"self:{k}" for via in (False, True) for k, v in ref_read(mlp, x, gy, via_function=via).items()
                        if not torch.equal(v, today[k])]
            for r1, d in PHASE1_EQUAL.items():
                old, new = p1.ref_read(mlp, x, gy, {r1}), ref_read(mlp, x, gy, d)
                mismatch += [f"{r1}:{k}" for k in TENSORS if not torch.equal(old[k], new[k])]
                del old, new
            if mismatch:
                record["void"] = f"REF_MISMATCH: {i}:" + " ".join(sorted(set(mismatch)))
                return record
            n = enable()
            try:
                engagement["adapters"].append(n)
                engagement["refused"].update(stats.get("refused", {}))
                with counter() as counts:
                    fused = p1.fused_read(mlp, x, gy)
                engagement["mlp_lora_forwards"] += sum(v for m, v in counts.items() if m.rsplit(".", 1)[-1] in p1.PROJECTIONS)
                engagement["fallback"] = max(engagement["fallback"], int(stats.get("fallback", 0) or 0))
            finally:
                disable()
            record["landing"][i] = landing(mlp, x, gy, today, fused)
            kept[i] = (p1._to(today, "cpu"), p1._to(fused, "cpu"))
            del today, fused
        finally:
            evict(i)
    engagement["ok"] = (not engagement["refused"] and engagement["fallback"] == 0 and engagement["mlp_lora_forwards"] == 0
                        and (adapters is None or all(n == adapters for n in engagement["adapters"])))
    record["landed"] = landed_by_tensor(record["landing"])
    rows_by_layer = {}
    for i in order:
        mlp, x, gy = mlps[i], xs[i], gys[i]
        stage(i)
        try:
            ref = p1.fp64_read(mlp, x, gy)
            terms = p1.adapter_terms(mlp, x, gy)
            device = x.device
            today, fused = (p1._to(t, device) for t in kept.pop(i))
            streamed = paths_streamed(today, ref, lambda d: ref_read(mlp, x, gy, d), draws, record["landed"])
            rows = {}
            for k in TENSORS:
                s = streamed[k]
                row = judge(s["paths"], p1.error(fused[k], ref[k]))
                ref_norm = float(torch.linalg.vector_norm(ref[k]))
                an = terms[k]["adapter_norm"]
                row.update(ref_norm=ref_norm, adapter_norm=an, norm=terms[k]["norm"],
                           e_min=((1 + MARGIN) * row["band_rel_rms"] * ref_norm / an) if an > 0 else float("inf"),
                           excluded_draws=s["excluded_draws"],
                           paths={h[:16]: {"rel_rms": v["err"]["rel_rms"], "max_abs": v["err"]["max_abs"],
                                           "draws": v["draws"]} for h, v in s["paths"].items()})
                dev = p1.deviation(fused[k], today[k])
                row["a"] = {"fused_to_today": dev, "floor": s["floor"],
                            "pass": dev["max_abs"] <= s["floor"]["max_abs"] and dev["rel_rms"] <= s["floor"]["rel_rms"]}
                rows[k] = row
            rows_by_layer[i] = rows
            record["layers"][i] = rows
            del ref, today, fused
        finally:
            evict(i)
    rank = p1._adapter(mlps[order[0]].gate_proj)[0].weight.shape[0]
    record["outcome"] = outcome(rows_by_layer, rank)
    record["e_min_max"] = max(v["e_min"] for rows in rows_by_layer.values() for v in rows.values())
    return record
