# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_DENSE_LT_PIN`` (the serving roadmap's lever K, step K1; opt-in, off by default): a pinned cuBLASLt GEMM for
named dense projections at 2 to 8 decode rows. Design: ``bench/dense-small-m/DESIGN.md``.

At 2 to 8 rows, cuBLAS's own heuristic picks a slower kernel than at one row on some of Qwen3-30B-A3B's dense shapes,
while cuBLASLt's own heuristic list holds an algorithm that does not slow down. This module runs those calls through one
cuBLASLt algorithm configuration PINNED per key, and never anything else silently:

- **The knob:** ``E4B_DENSE_LT_PIN`` names projections from :data:`SITES` (comma-separated). Unset or empty is off.
- **The key:** (the cuBLASLt version actually loaded, the device's compute capability, its SM count, the projection's
  (N, K), M). A pin is a full algorithm configuration (:data:`LT_PINS`), filled by a registered selection on the reading
  box, never at serve time.
- **The rows:** a call of 2..8 rows with a pin for its M routes. One with no pin passes to ``F.linear``, counted. At
  build, every captured decode bucket in 2..8 must have a pin per named projection, or the knob is refused by name.
  One row and more than 8 rows keep their route, uncounted here.
- **The library:** ctypes against the ONE ``libcublasLt`` mapped in this process (read from ``/proc/self/maps`` after a
  warm ``F.linear``), opened ``RTLD_NOLOAD``, its version asserted. No new dependency.
- **The workspace:** one persistent buffer, sized at build from ``cublasLtMatmulAlgoCheck``, nothing allocated per call
  except the output. A capture is legal. One stream is enforced: an eager pinned call on another stream is refused.
- **Determinism:** reduction schemes 0, 2 and 4 only (separate-kernel reductions); INPLACE (1) is refused.
- **Sites:** ``o_proj`` is an exact ``nn.Linear`` swap. ``router`` routes only through the router epilogue's patched
  forward (``engines/router_epilogue``, ``E4B_FUSE_ROUTER_EPI``), so every router must be patched by it, as the
  softmax-then-top-k kind, without a bias. Otherwise naming it is refused.

Tokens change at 2..8 rows (split-K reorders the sum), so a default needs a served read with a quality bar.
"""
from __future__ import annotations

import ctypes
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ENV = "E4B_DENSE_LT_PIN"
SITES = ("o_proj", "router")
MIN_M, MAX_M = 2, 8
REDUCTION_OK = (0, 2, 4)                     # NONE, COMPUTE_TYPE, OUTPUT_TYPE: separate-kernel reductions; INPLACE (1) refused

# cublasLt.h / library_types.h / cublas_api.h (CUDA 12) enumerators this module uses
_CUDA_R_32F, _CUDA_R_16BF = 0, 14
_COMPUTE_32F = 68
_OP_N, _OP_T = 0, 1
_DESC_TRANSA, _DESC_TRANSB = 3, 4
_PREF_MAX_WORKSPACE_BYTES, _PREF_REDUCTION_SCHEME_MASK = 1, 3
_REDUCTION_MASK = 0x6                        # COMPUTE_TYPE (2) | OUTPUT_TYPE (4); INPLACE (1) excluded; NONE is split-K 1
#: cublasLtMatmulAlgoConfigAttributes_t: name -> (attribute id, ctypes type); "algo_id" is set by AlgoInit, read back too
_ALGO_ATTRS = {"algo_id": (0, ctypes.c_int32), "tile": (1, ctypes.c_uint32), "split_k": (2, ctypes.c_int32),
               "reduction_scheme": (3, ctypes.c_uint32), "cta_swizzling": (4, ctypes.c_uint32),
               "custom_option": (5, ctypes.c_uint32), "stages": (6, ctypes.c_uint32),
               "inner_shape": (7, ctypes.c_uint16), "cluster_shape": (8, ctypes.c_uint16)}
CONFIG_FIELDS = tuple(_ALGO_ATTRS)

#: The pins: ``{(lt_version, (cc_major, cc_minor), sm_count, N, K, M): {field: int for field in CONFIG_FIELDS}}``.
#: EMPTY until a registered selection on the reading box fills it by a code PR (as P222's pair became
#: ``_K25_PLANS["m1"]``). With the table empty, the knob is refused at build for want of a pin.
LT_PINS: dict = {}

#: Calls at a named site with 2..8 rows, as ``{"<site>|M<m>|<prefill|decode|eager>|<pinned|passed>": calls}``, counted
#: in the Python forward (eager calls and captures count, replays do not).
SEEN: dict = {}


class _Algo(ctypes.Structure):               # cublasLtMatmulAlgo_t
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heuristic(ctypes.Structure):          # cublasLtMatmulHeuristicResult_t
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def sites_env(value=None) -> tuple:
    """``E4B_DENSE_LT_PIN`` as a tuple of site names in :data:`SITES` order; empty when off. An unknown name, or a
    repeated one, is refused."""
    v = os.environ.get(ENV, "") if value is None else value
    names = [x.strip().lower() for x in v.split(",") if x.strip()]
    bad = [x for x in names if x not in SITES]
    if bad or len(set(names)) != len(names):
        raise ValueError(f"{ENV}={v!r}: expected a comma-separated subset of {SITES}, each at most once")
    return tuple(s for s in SITES if s in names)


def mapped_cublaslt(maps_text=None) -> str:
    """The ONE ``libcublasLt`` file mapped in this process, read from ``/proc/self/maps``. None mapped, or two different
    files, is refused: the pin must call the same library instance torch uses, never load another."""
    if maps_text is None:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
            maps_text = f.read()
    paths = sorted({line.split()[-1] for line in maps_text.splitlines()
                    if len(line.split()) >= 6 and "libcublasLt.so" in line.split()[-1]})
    if len(paths) != 1:
        raise RuntimeError(f"{ENV}: expected exactly one libcublasLt mapped in this process (torch's), found {paths}")
    return paths[0]


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{ENV}: {what} returned cublasStatus {rc}")


class LtLib:
    """The ctypes binding to the mapped ``libcublasLt``: one handle, and the calls the pin makes."""

    def __init__(self, path):
        mode = getattr(os, "RTLD_NOLOAD", 0) | getattr(os, "RTLD_LAZY", 1)
        self.path, self.lib = path, ctypes.CDLL(path, mode=mode)
        L, vp, sz, i32, u64, i64 = self.lib, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint64, ctypes.c_int64
        L.cublasLtGetVersion.restype, L.cublasLtGetVersion.argtypes = sz, []
        for name, args in (("cublasLtCreate", [ctypes.POINTER(vp)]),
                           ("cublasLtMatmulDescCreate", [ctypes.POINTER(vp), i32, i32]),
                           ("cublasLtMatmulDescSetAttribute", [vp, i32, vp, sz]),
                           ("cublasLtMatrixLayoutCreate", [ctypes.POINTER(vp), i32, u64, u64, i64]),
                           ("cublasLtMatmulAlgoInit", [vp, i32, i32, i32, i32, i32, i32, i32, ctypes.POINTER(_Algo)]),
                           ("cublasLtMatmulAlgoConfigSetAttribute", [ctypes.POINTER(_Algo), i32, vp, sz]),
                           ("cublasLtMatmulAlgoConfigGetAttribute", [ctypes.POINTER(_Algo), i32, vp, sz,
                                                                     ctypes.POINTER(sz)]),
                           ("cublasLtMatmulAlgoCheck", [vp, vp, vp, vp, vp, vp, ctypes.POINTER(_Algo),
                                                        ctypes.POINTER(_Heuristic)]),
                           ("cublasLtMatmulPreferenceCreate", [ctypes.POINTER(vp)]),
                           ("cublasLtMatmulPreferenceSetAttribute", [vp, i32, vp, sz]),
                           ("cublasLtMatmulAlgoGetHeuristic", [vp, vp, vp, vp, vp, vp, vp, i32,
                                                               ctypes.POINTER(_Heuristic), ctypes.POINTER(i32)]),
                           ("cublasLtMatmul", [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
                                               ctypes.POINTER(_Algo), vp, sz, vp])):
            fn = getattr(L, name)
            fn.restype, fn.argtypes = ctypes.c_int, args
        self.handle = vp()
        _check(L.cublasLtCreate(ctypes.byref(self.handle)), "cublasLtCreate")

    def version(self) -> int:
        return int(self.lib.cublasLtGetVersion())

    def problem(self, N, K, M):
        """The descriptors of ``F.linear``'s GEMM, ``D[M, N] = X[M, K] @ W[N, K]^T``, as cuBLAS sees it column-major:
        ``D^T (N x M) = W^T-as-stored (K x N, transposed) times X^T (K x M)``; fp32 compute, bf16 in and out."""
        L, vp = self.lib, ctypes.c_void_p
        desc = vp()
        _check(L.cublasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, _CUDA_R_32F), "MatmulDescCreate")
        for attr, val in ((_DESC_TRANSA, _OP_T), (_DESC_TRANSB, _OP_N)):
            v = ctypes.c_int32(val)
            _check(L.cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(v), ctypes.sizeof(v)), "DescSetAttribute")
        lay = {}
        for name, rows, cols, ld in (("a", K, N, K), ("b", K, M, K), ("d", N, M, N)):
            h = vp()
            _check(L.cublasLtMatrixLayoutCreate(ctypes.byref(h), _CUDA_R_16BF, rows, cols, ld), "MatrixLayoutCreate")
            lay[name] = h
        return {"desc": desc, "a": lay["a"], "b": lay["b"], "d": lay["d"], "N": N, "K": K, "M": M}

    def algo(self, cfg) -> _Algo:
        a = _Algo()
        _check(self.lib.cublasLtMatmulAlgoInit(self.handle, _COMPUTE_32F, _CUDA_R_32F, _CUDA_R_16BF, _CUDA_R_16BF,
                                               _CUDA_R_16BF, _CUDA_R_16BF, int(cfg["algo_id"]), ctypes.byref(a)),
               "MatmulAlgoInit")
        for field, (attr, ct) in _ALGO_ATTRS.items():
            if field == "algo_id":
                continue
            v = ct(int(cfg[field]))
            _check(self.lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), attr, ctypes.byref(v),
                                                                 ctypes.sizeof(v)), f"AlgoConfigSetAttribute({field})")
        return a

    def config(self, a: _Algo) -> dict:
        out = {}
        for field, (attr, ct) in _ALGO_ATTRS.items():
            v, n = ct(0), ctypes.c_size_t(0)
            _check(self.lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(a), attr, ctypes.byref(v),
                                                                 ctypes.sizeof(v), ctypes.byref(n)),
                   f"AlgoConfigGetAttribute({field})")
            out[field] = int(v.value)
        return out

    def check(self, prob, a: _Algo) -> int:
        """``cublasLtMatmulAlgoCheck`` for these descriptors: the workspace the configuration needs, or a refusal."""
        h = _Heuristic()
        rc = self.lib.cublasLtMatmulAlgoCheck(self.handle, prob["desc"], prob["a"], prob["b"], prob["d"], prob["d"],
                                              ctypes.byref(a), ctypes.byref(h))
        if rc != 0 or h.state != 0:
            raise RuntimeError(f"{ENV}: cuBLASLt rejects the pinned configuration {self.config(a)} for "
                               f"(N, K, M) = ({prob['N']}, {prob['K']}, {prob['M']}) (status {rc}, state {h.state})")
        return int(h.workspaceSize)

    def heuristics(self, prob, limit=64, max_workspace=32 << 20) -> list:
        """cuBLASLt's own heuristic list for these descriptors, INPLACE reduction excluded: ``[(config, workspace)]``.
        The registered selection reads this; serving never does."""
        L, vp = self.lib, ctypes.c_void_p
        pref = vp()
        _check(L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PreferenceCreate")
        ws, mask = ctypes.c_uint64(max_workspace), ctypes.c_uint32(_REDUCTION_MASK)
        _check(L.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WORKSPACE_BYTES, ctypes.byref(ws),
                                                      ctypes.sizeof(ws)), "PreferenceSetAttribute(workspace)")
        _check(L.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_REDUCTION_SCHEME_MASK, ctypes.byref(mask),
                                                      ctypes.sizeof(mask)), "PreferenceSetAttribute(reduction mask)")
        res, got = (_Heuristic * limit)(), ctypes.c_int(0)
        _check(L.cublasLtMatmulAlgoGetHeuristic(self.handle, prob["desc"], prob["a"], prob["b"], prob["d"], prob["d"],
                                                pref, limit, res, ctypes.byref(got)), "AlgoGetHeuristic")
        out = []
        for i in range(got.value):
            cfg = self.config(res[i].algo)
            if res[i].state == 0 and cfg["reduction_scheme"] in REDUCTION_OK:
                out.append((cfg, int(res[i].workspaceSize)))
        return out

    def matmul(self, prob, a: _Algo, x, w, out, ws, ws_bytes, stream):
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        _check(self.lib.cublasLtMatmul(self.handle, prob["desc"], ctypes.byref(alpha), w.data_ptr(), prob["a"],
                                       x.data_ptr(), prob["b"], ctypes.byref(beta), out.data_ptr(), prob["d"],
                                       out.data_ptr(), prob["d"], ctypes.byref(a), ws, ws_bytes, stream),
               "cublasLtMatmul")


class LtRuntime:
    """One process's pinned path: the library, the key, the plans per (N, K, M), the shared workspace and the stream."""

    def __init__(self, lib: LtLib, key: tuple):
        self.lib, self.key, self.plans, self.ws, self.ws_bytes, self.stream = lib, key, {}, None, 0, None
        self.device = None

    def plan(self, N, K, M) -> dict:
        k = (*self.key, N, K, M)
        cfg = LT_PINS.get(k)
        if cfg is None:
            raise RuntimeError(f"{ENV}: no pin for key (cuBLASLt {k[0]}, cc {k[1]}, {k[2]} SMs, N {N}, K {K}, M {M}); "
                               "the pin table holds only what a registered selection put there")
        if set(cfg) != set(CONFIG_FIELDS) or cfg["reduction_scheme"] not in REDUCTION_OK:
            raise RuntimeError(f"{ENV}: the pin for {k} is malformed or uses a reduction scheme outside {REDUCTION_OK}")
        prob = self.lib.problem(N, K, M)
        a = self.lib.algo(cfg)
        need = self.lib.check(prob, a)
        self.plans[(N, K, M)] = (prob, a, need)
        return self.plans[(N, K, M)]

    def finish(self, device):
        """The ONE workspace (the largest any plan needs) and the serving stream, recorded."""
        self.ws_bytes = max([p[2] for p in self.plans.values()] or [0])
        self.device = torch.device(device)
        self.ws = torch.empty(max(self.ws_bytes, 1), dtype=torch.uint8, device=self.device)
        self.stream = torch.cuda.current_stream(self.device)

    def __call__(self, x, w, site):
        M, K = x.shape
        N = w.shape[0]
        prob, a, _ = self.plans[(N, K, M)]
        cur = torch.cuda.current_stream(x.device)
        if cur != self.stream and not torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{ENV}: {site} was called on a stream other than the one recorded at build; the pinned "
                               "path's one workspace is shared and must not run on two streams")
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        self.lib.matmul(prob, a, x, w, out, self.ws.data_ptr(), self.ws_bytes, cur.cuda_stream)
        return out


def _phase() -> str:
    try:
        from .paged_attention import current_context
        ctx = current_context()
    except Exception:                            # noqa: BLE001  (outside the paged server: no phase)
        ctx = None
    mode = getattr(ctx, "mode", None)
    return mode if mode in ("prefill", "decode") else "eager"


def _count(site, M, route):
    key = f"{site}|M{M}|{_phase()}|{route}"
    SEEN[key] = SEEN.get(key, 0) + 1


class PinnedCall:
    """One site's GEMM ``rows @ weight^T``: 2..8 rows with a pin route through the runtime; 2..8 rows without one go to
    ``fallback`` (counted); every other row count goes to ``fallback``, uncounted. ``fallback`` is what the site ran
    before the pin (``F.linear``, or K35's one-row module), so nothing else moves. Operands at a routed or counted call
    must be bf16 CUDA tensors, else the call is refused by name."""

    def __init__(self, runtime: LtRuntime, site: str, pinned_ms):
        self.runtime, self.site, self.pinned_ms = runtime, site, frozenset(pinned_ms)

    def __call__(self, x2d, weight, fallback):
        M = x2d.shape[0]
        if not MIN_M <= M <= MAX_M:
            return fallback()
        if (x2d.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
                or x2d.device.type != self.runtime.device.type):
            raise RuntimeError(f"{ENV}: {self.site} got {M} rows as {x2d.dtype} on {x2d.device} (weight "
                               f"{weight.dtype}); the pinned site takes bf16 operands on the build's device only")
        if M not in self.pinned_ms:
            _count(self.site, M, "passed")
            return fallback()
        _count(self.site, M, "pinned")
        return self.runtime(x2d.contiguous(), weight, self.site)


class PinnedLinear(nn.Module):
    """Wraps the projection module that was there (an exact ``nn.Linear`` without a bias, or K35's one-row
    ``DenseGemvLinear`` over one): its weight is shared, and every call the pin does not route goes to it unchanged."""

    def __init__(self, inner: nn.Module, call: PinnedCall):
        super().__init__()
        self.in_features, self.out_features = inner.in_features, inner.out_features
        self.inner, self._call = inner, call
        self.weight = inner.weight

    def forward(self, x):
        x2d = x.reshape(-1, self.in_features)
        y = self._call(x2d, self.weight, lambda: self.inner(x))
        return y if y.dim() == x.dim() else y.reshape(*x.shape[:-1], self.out_features)


def router_linear(mod, rows):
    """The router epilogue's GEMM: the pin's call when the router carries one, else ``F.linear`` exactly as before."""
    pin = getattr(mod, "_e4b_lt_pin", None)
    if pin is None:
        return F.linear(rows, mod.weight)
    return pin(rows, mod.weight, lambda: F.linear(rows, mod.weight))


def _o_proj_sites(model):
    for name, mod in model.named_modules():
        lin = getattr(mod, "o_proj", None)
        if isinstance(lin, nn.Module) and "Attention" in type(mod).__name__:
            yield f"{name}.o_proj" if name else "o_proj", mod, lin


def _routers(model):
    return [(name, mod) for name, mod in model.named_modules() if "Router" in type(mod).__name__]


def apply_dense_lt_pin(model, buckets, *, device="cuda", value=None, lib_factory=None, maps_text=None) -> dict:
    """Resolve ``E4B_DENSE_LT_PIN`` at engine build: after the fusions (the router epilogue) and before any graph is
    captured. ``buckets`` are the decode buckets the server will capture. Returns the ``/health`` record. Every refusal is
    by name; nothing degrades silently."""
    sites = sites_env(value)
    if not sites:
        return {"env": (os.environ.get(ENV) if value is None else value) or None, "on": False, "sites": []}
    if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        raise RuntimeError(f"{ENV}: the pinned path is cuBLASLt on CUDA only")
    if not sys.platform.startswith("linux") and maps_text is None:
        raise RuntimeError(f"{ENV}: reading the mapped cuBLASLt needs /proc/self/maps (Linux)")
    dev = torch.device(device)
    F.linear(torch.ones(2, 8, dtype=torch.bfloat16, device=dev), torch.ones(8, 8, dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize(dev)                  # torch's cuBLAS (and its cuBLASLt) is loaded by now
    lib = (lib_factory or LtLib)(mapped_cublaslt(maps_text))
    props = torch.cuda.get_device_properties(dev)
    key = (lib.version(), (props.major, props.minor), props.multi_processor_count)
    rt = LtRuntime(lib, key)
    graphed = sorted({int(b) for b in buckets if MIN_M <= int(b) <= MAX_M})
    targets = []                                 # (site, label, weight, install)
    if "o_proj" in sites:
        found = list(_o_proj_sites(model))
        if not found:
            raise RuntimeError(f"{ENV}: o_proj named, but no attention module holds an o_proj")
        for label, mod, lin in found:
            plain = type(lin) is nn.Linear and lin.bias is None
            k35 = type(lin).__name__ == "DenseGemvLinear"            # E4B_DENSE_GEMV_BW: one row only, so they compose
            if not (plain or k35):
                raise RuntimeError(f"{ENV}: {label} is {type(lin).__name__} (bias {getattr(lin, 'bias', None) is not None});"
                                   " the pin takes an exact nn.Linear without a bias, or K35's one-row module over one "
                                   "(int4 / int8 attention cannot combine)")
            targets.append(("o_proj", label, lin.weight,
                            lambda call, _m=mod, _l=lin: setattr(_m, "o_proj", PinnedLinear(_l, call))))
    if "router" in sites:
        routers = _routers(model)
        if not routers:
            raise RuntimeError(f"{ENV}: router named, but the model has no router module")
        for label, mod in routers:
            kind = getattr(mod, "_e4b_router_epi_kind", None)
            if kind != "softmax_topk" or getattr(mod, "bias", None) is not None:
                raise RuntimeError(f"{ENV}: router named, but {label} is not on the router epilogue's softmax-then-top-k "
                                   f"forward (kind {kind!r}, bias {getattr(mod, 'bias', None) is not None}); the pin "
                                   "routes the router's GEMM only there")
            targets.append(("router", label, mod.weight,
                            lambda call, _m=mod: setattr(_m, "_e4b_lt_pin", call)))
    for site, label, w, _ in targets:
        if w.dtype != torch.bfloat16 or w.dim() != 2 or w.stride(1) != 1:
            raise RuntimeError(f"{ENV}: {label}'s weight is {w.dtype} {tuple(w.shape)}, not a bf16 [N, K] with "
                               "contiguous rows")
    shapes = sorted({(site, int(w.shape[0]), int(w.shape[1])) for site, _, w, _ in targets})
    pinned = {}
    for site, N, K in shapes:
        ms = sorted(M for M in range(MIN_M, MAX_M + 1) if (*key, N, K, M) in LT_PINS)
        missing = [M for M in graphed if M not in ms]
        if missing:
            raise RuntimeError(f"{ENV}: {site} (N {N}, K {K}) has no pin for the captured decode bucket(s) {missing} "
                               f"under key (cuBLASLt {key[0]}, cc {key[1]}, {key[2]} SMs); a graphed bucket must have one")
        for M in ms:
            rt.plan(N, K, M)
        pinned[(site, N, K)] = ms
    rt.finish(dev)
    for site, label, w, install in targets:
        install(PinnedCall(rt, site, pinned[(site, int(w.shape[0]), int(w.shape[1]))]))
    return {"env": os.environ.get(ENV) if value is None else value, "on": True, "sites": list(sites),
            "key": {"cublaslt": key[0], "cc": list(key[1]), "sm_count": key[2]}, "library": lib.path,
            "modules": {s: sum(1 for t in targets if t[0] == s) for s in sites},
            "pinned_m": {f"{s}|{N}x{K}": ms for (s, N, K), ms in pinned.items()},
            "graphed_buckets": graphed, "workspace_bytes": rt.ws_bytes, "_stream": rt.stream}


def engagement() -> dict:
    """:data:`SEEN`, sorted: per (site, M, phase), the calls the pin routed and the ones it passed to ``F.linear``."""
    return dict(sorted(SEEN.items()))
