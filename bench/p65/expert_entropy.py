#!/usr/bin/env python3
"""bench/p65/expert_entropy.py -- P65's per-expert activation entropy, read from the calibration tap's moments.

THE DEFINITION (the one the lane ranks by) is Colla-Q's activation-entropy proxy (Shin & Ryu, arXiv:2609.18131,
Eq. 5-7), computed on each expert's OUTPUT, ``y = W_dn h`` -- the paper evaluates "each expert using its
continuous-valued output activations (i.e., expert FFN outputs)":

    rho = exp(2 H_within) / exp(2 H_total)  =  sigma2_within / sigma2_total          (Gaussian: H = 1/2 ln 2 pi e s2)

    sigma2_within = 1/(|T||C|) sum_t sum_c (y_tc - mu_c)^2      mu_c = mean_t y_tc   (per-channel mean removed)
    sigma2_total  = 1/(|T||C|) sum_t sum_c (y_tc - mu_bar)^2    mu_bar = mean_c mu_c  (one global mean removed)

over the token-slots T an expert was routed and the channels C of its output. ``rho`` is in [0, 1]; lower means the
expert's channels carry distinct, predictable responses (the paper's "better expert"), higher means a weaker expert,
and Colla-Q gives weaker experts more bits. ``dH_nats = 1/2 ln rho = H_within - H_total`` is the same number as an
entropy difference. An identity makes the quantity concrete: ``sigma2_total - sigma2_within = Var_c(mu_c)``, so
``1 - rho`` is the share of the output's variance (about its global mean) that is a per-channel constant offset.

WHY THIS AND NOT ANOTHER ENTROPY. The issue asks for "activation entropy" because Colla-Q reports that an allocation
built on it holds across calibration domains better than a routing-statistics allocation (their Table 4). Testing that
claim needs their quantity, not a neighbour of it. Per-row Shannon entropies of a normalised |activation| vector, the
other obvious candidate, cannot be recovered from any accumulated moment -- the tap would have to keep every row -- and
nothing in the paper says it would behave the same. So the ratio is the registered signal. One descriptive neighbour
is recorded beside it because it costs nothing extra: ``h_energy``, the Shannon entropy of the per-channel energy
shares ``p_c = E[y_c^2] / sum_c E[y_c^2]``, normalised by ``ln |C|`` (1 = energy spread evenly over channels). It is
NOT read by any P65 decision rule.

HOW IT IS COMPUTED, AND WHY THAT IS EXACT. The tap (``_CALIB_SINK``, ``engines/hot_residency.py``) already feeds the
census the down-projection INPUT rows ``h`` of every expert, as the running mean ``H = 2 E[h h^T]``
(``gptq_pack.HessianAccumulator``), and ``calibrate_expert_hessians(..., activation_means=)`` now also returns
``m = E[h]``. Every term of Eq. 7 is linear in the first two moments of ``y``:

    E[y_c^2] = (W_dn (H/2) W_dn^T)_cc = 1/2 sum_k (W_dn H)_ck (W_dn)_ck        mu_c = (W_dn m)_c
    sigma2_within = mean_c (E[y_c^2] - mu_c^2)      sigma2_total = mean_c E[y_c^2] - (mean_c mu_c)^2

so the census never materialises ``y`` and the population normalisation is the paper's (1/|T||C|). The diagonal is the
same ``(W @ H) * W`` product P44's ``_trace_quad`` sums, left unsummed over rows, in fp32 on the census device; the
C-length vectors are then combined in fp64 on the CPU. Cheap: one extra ``[C,K] @ [K,K]`` per expert for the down role
(Mixtral's largest, 4096 x 14336^2, is ~1.7 TFLOP). Deterministic: fixed shapes, no sampling, no atomics in the
reduction; the census re-computes one row and asserts the bits match (``--selfcheck``).

The expert's INPUT gets the same ratio (``of = "input x"``, from ``diag(H_gu)/2`` and ``E[x]``), recorded on the gate/up
row. Descriptive only, like ``h_energy``.

Per-channel variances ``E[y_c^2] - mu_c^2`` can round below zero for a channel that is constant to fp32 precision; they
are clamped to 0 and COUNTED (``neg_var_channels``) so a row whose ratio rests on cancellation says so.
"""
from __future__ import annotations

import math

import torch


def entropy_ratio(m2, mu) -> dict:
    """Colla-Q's ratio from per-channel second moments ``m2 = E[a_c^2]`` and means ``mu = E[a_c]`` (length-C vectors).

    Returns the census fields: ``rho``, ``dH_nats``, ``sigma2_within``, ``sigma2_total``, ``h_energy`` (descriptive),
    ``channels`` and ``neg_var_channels``. ``rho`` is None when the total variance is not positive (a constant output).
    """
    m2 = torch.as_tensor(m2, dtype=torch.float64).flatten().cpu()
    mu = torch.as_tensor(mu, dtype=torch.float64).flatten().cpu()
    if m2.shape != mu.shape or m2.numel() == 0:
        raise ValueError(f"entropy_ratio: m2 {tuple(m2.shape)} and mu {tuple(mu.shape)} must be equal, non-empty vectors")
    C = m2.numel()
    var_c = m2 - mu * mu
    neg = int((var_c < 0).sum())
    var_c = var_c.clamp_min(0.0)
    s2_within = float(var_c.mean())
    s2_total = float(m2.mean() - mu.mean() ** 2)
    rho = s2_within / s2_total if s2_total > 0 else None
    if rho is not None:
        rho = min(max(rho, 0.0), 1.0)          # clamped variances can push it a hair outside [0, 1]
    tot = float(m2.sum())
    if tot > 0 and C > 1:
        p = (m2.clamp_min(0.0) / tot)
        nz = p[p > 0]
        h_energy = float(-(nz * nz.log()).sum()) / math.log(C)
    else:
        h_energy = None
    return {"rho": rho, "dH_nats": (0.5 * math.log(rho) if rho else None),
            "sigma2_within": s2_within, "sigma2_total": s2_total, "h_energy": h_energy,
            "channels": C, "neg_var_channels": neg}


def output_moments(W: torch.Tensor, H: torch.Tensor, mean_in: torch.Tensor, *, dev: str = "cpu"):
    """``(E[y_c^2], E[y_c])`` for ``y = W u``, from ``H = 2 E[u u^T]`` (the tap's running mean) and ``mean_in = E[u]``.

    ``W`` is ``[C, K]`` (the census's (out, in) orientation), ``H`` ``[K, K]``, ``mean_in`` ``[K]``. The diagonal is taken
    in fp32 on ``dev`` (the product P44's ``_trace_quad`` sums); the mean in fp64 on the CPU."""
    if W.dim() != 2 or H.shape != (W.shape[1], W.shape[1]) or mean_in.numel() != W.shape[1]:
        raise ValueError(f"output_moments: W {tuple(W.shape)}, H {tuple(H.shape)}, mean {tuple(mean_in.shape)} do not compose")
    with torch.no_grad():
        W32 = W.to(dev, torch.float32)
        H32 = H.to(dev, torch.float32)
        m2 = (0.5 * ((W32 @ H32) * W32).sum(1)).to("cpu", torch.float64)
        del W32, H32
        mu = W.to("cpu", torch.float64) @ mean_in.to("cpu", torch.float64).flatten()
    return m2, mu


def input_moments(H: torch.Tensor, mean_in: torch.Tensor):
    """``(E[u_c^2], E[u_c])`` of the input itself, from ``H = 2 E[u u^T]`` and ``E[u]``."""
    return (0.5 * torch.diagonal(H).to("cpu", torch.float64)), mean_in.to("cpu", torch.float64).flatten()


def entropy_fields(role: str, W: torch.Tensor, H, mean_in, *, dev: str = "cpu"):
    """``(fields, (m2, mu))`` for one census row, or None when the expert saw no rows (no Hessian, no mean).

    ``fields`` is the row's ``entropy`` dict: on ``dn`` rows the expert's output ``y = W_dn h`` (THE P65 signal), on
    ``gu`` rows its input ``x`` (descriptive). ``(m2, mu)`` are the moment vectors it was read from, returned so the
    census can combine two halves of a text exactly (:func:`combine_moments`) without re-reading anything."""
    if H is None or mean_in is None:
        return None
    if role == "dn":
        m2, mu = output_moments(W, H, mean_in, dev=dev)
        of = "output y = W_dn h"
    elif role == "gu":
        m2, mu = input_moments(H, mean_in)
        of = "input x"
    else:
        raise ValueError(f"entropy_fields: role {role!r} is not gu or dn")
    out = entropy_ratio(m2, mu)
    out["of"] = of
    return out, (m2, mu)


def combine_moments(parts):
    """Exact combination of per-half moment vectors: ``parts`` = [(rows, m2, mu), ...] -> (rows, m2, mu) of the union.
    Both moments are means over rows, so the union's are the row-weighted means."""
    parts = [(int(n), m2, mu) for n, m2, mu in parts if n > 0]
    if not parts:
        return 0, None, None
    N = sum(n for n, _, _ in parts)
    m2 = sum(m2.to(torch.float64) * (n / N) for n, m2, _ in parts)
    mu = sum(mu.to(torch.float64) * (n / N) for n, _, mu in parts)
    return N, m2, mu


def direct_entropy_ratio(A: torch.Tensor) -> dict:
    """The ratio from raw activation rows ``A [T, C]``, straight from Eq. 7 (fp64). The census's selfcheck and the
    tests hold :func:`entropy_ratio` of the moments to this."""
    A = A.to(torch.float64)
    mu_c = A.mean(0)
    s2_within = float(((A - mu_c) ** 2).mean())
    s2_total = float(((A - A.mean()) ** 2).mean())
    rho = s2_within / s2_total if s2_total > 0 else None
    return {"rho": rho, "sigma2_within": s2_within, "sigma2_total": s2_total}
