"""Profile-only instrumentation of the expert offload boundary (``E4B_EXPERT_PROFILE=<out.jsonl>``).

Answers one question, without changing any behavior: is the offload transfer wall concentrated in
a small set of layer/expert pairs? Offload answers whether the model can fit; expert-streaming
profiling asks whether the transfer wall is concentrated enough to route around. No cache, no
scheduler, no residency-policy change lives here — a hot-expert policy is gated on what this
profile shows (docs/EXPERT_STREAMING_PROFILE.md).

What is actually measurable today, stated honestly: staging is LAYER-granular —
:meth:`_ExpertOffload._copy_home_to_device` moves a layer's entire fused stack per visit, so no
per-expert H2D transfer exists in the current design. This module therefore records

  * per-layer H2D staging: count, bytes, wall of each copy (CUDA events recorded on the staging
    stream, reduced with ONE synchronize at flush — never in the hot path, same discipline as
    ``_OffloadStats``), tagged by policy (sync / cold_miss / prefetch);
  * per-(layer, expert) routing: how many forwards routed to the expert (``hits``) and how many
    token-slots it received (``tokens_routed``), accumulated as on-device bincounts in a forward
    pre-hook (one ``bincount`` per layer forward; no host sync until flush).

Per-expert *stall attribution* is a projection, not a measurement — the summarizer
(scripts/summarize_expert_streaming.py) states its attribution rule explicitly.

Usage: set ``E4B_EXPERT_PROFILE`` to the output JSONL path; ``experts4bit_qlora.train`` /
``experts4bit_qlora.infer`` call :func:`attach` after the model is built (no-op when unset).
Rows are written once, at process exit (atexit) or an explicit :func:`flush`.
"""

import atexit
import json
import os
from datetime import datetime, timezone

import torch

_ENV = "E4B_EXPERT_PROFILE"
_STATE = None  # single active profile per process (matches the env-var interface)


def enabled() -> bool:
    return bool(os.environ.get(_ENV))


class _LayerProbe:
    """Accumulators for one ExpertsLoRA layer: routing bincounts + staging copy events."""

    def __init__(self, layer_id, module):
        self.layer_id = layer_id
        # BARE expert modules are the norm for the families with a custom epilogue --
        # gpt-oss and DeepSeek-V4 are both built without an ExpertsLoRA wrapper, because
        # the generic adapter re-implements the expert math with a plain SwiGLU. Those are
        # exactly the models whose routing you most want to profile, so `base` has to fall
        # back to the module itself rather than assuming a wrapper.
        base = getattr(module, "base", module)
        self.num_experts = base.num_experts
        self.storage_mode = base.quant_type
        # Per-expert share of one staging copy, in bytes (packed slices + absmax slices) — the
        # quantity a pinning policy would keep resident per expert.
        n = self.num_experts

        # Under offload the base tensors are 0-element placeholders at attach time, so read the
        # real byte sizes from the offload handle's CPU home copies when present; fall back to the
        # (resident) base tensors otherwise. The summarizer also re-derives this from staged bytes
        # if it still lands at 0, so an old placeholder-zeroed profile stays salvageable.
        home = getattr(getattr(module, "_offload", None), "home", None)

        def _slice_bytes(name):
            t = home.get(name) if home is not None else getattr(base, name, None)
            return 0 if t is None else t.numel() * t.element_size() // n

        self.per_expert_bytes = sum(
            _slice_bytes(name) for name in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")
        )
        self.hits = None  # lazy: allocated on the device of the first routed index tensor
        self.tokens = None
        self.forwards = 0
        self.copies = []  # (start_evt, end_evt, nbytes, policy) — CUDA path only
        self.copy_count_no_cuda = 0
        self.h2d_bytes = 0

    def record_routing(self, top_k_index):
        idx = top_k_index.reshape(-1)
        if self.hits is None:
            self.hits = torch.zeros(self.num_experts, dtype=torch.long, device=idx.device)
            self.tokens = torch.zeros(self.num_experts, dtype=torch.long, device=idx.device)
        counts = torch.bincount(idx, minlength=self.num_experts)
        self.tokens += counts
        self.hits += (counts > 0).long()
        self.forwards += 1


def attach(model) -> bool:
    """Install the probes on every ExpertsLoRA in ``model``. Returns True iff profiling is on.

    Behavior-neutral: a forward pre-hook per expert layer (one bincount), and — when the layer is
    offloaded — a timing wrapper around its handle's ``_copy_home_to_device`` that only brackets
    the existing call with CUDA events. Nothing about staging order, residency, or math changes.
    """
    global _STATE
    if not enabled():
        return False
    from .._vendor.experts import ExpertsNbit
    from ..lora import ExpertsLoRA

    out_path = os.environ[_ENV]
    # A wrapped base is never dispatched (ExpertsLoRA calls base._project, not base.forward),
    # so hooking both would double-count and mis-number the layers. Probe the module that
    # actually receives the call: the wrapper when there is one, the bare stack otherwise.
    wrapped = {id(m.base) for m in model.modules()
               if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
    probes = []
    layer_id = 0
    for module in model.modules():
        dispatched = (isinstance(module, ExpertsLoRA)
                      or (isinstance(module, ExpertsNbit) and id(module) not in wrapped))
        if not dispatched:
            continue
        probe = _LayerProbe(layer_id, module)
        probes.append(probe)

        def _pre_hook(mod, args, probe=probe):
            if len(args) >= 2:  # (hidden_states, top_k_index, top_k_weights)
                probe.record_routing(args[1])

        module.register_forward_pre_hook(_pre_hook)

        handle = getattr(module, "_offload", None)
        if handle is not None:
            orig = handle._copy_home_to_device

            def _timed_copy(policy="sync", handle=handle, probe=probe, orig=orig):
                if torch.cuda.is_available():
                    stream = torch.cuda.current_stream(handle.device)
                    start = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    orig(policy)
                    end = torch.cuda.Event(enable_timing=True)
                    end.record(stream)
                    probe.copies.append((start, end, handle._stage_nbytes, policy))
                else:
                    orig(policy)
                    probe.copy_count_no_cuda += 1
                probe.h2d_bytes += handle._stage_nbytes

            handle._copy_home_to_device = _timed_copy
        layer_id += 1

    _STATE = {"probes": probes, "out_path": out_path, "flushed": False}
    atexit.register(flush)
    print(f"[expert-profile] attached to {len(probes)} expert layers -> {out_path}", flush=True)
    return True


def flush() -> None:
    """Reduce events (one synchronize, here only) and write the aggregated JSONL."""
    global _STATE
    if _STATE is None or _STATE["flushed"]:
        return
    _STATE["flushed"] = True
    probes, out_path = _STATE["probes"], _STATE["out_path"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    meta = {
        "row": "meta",
        "phase": os.environ.get("E4B_PROFILE_PHASE", "train"),
        "model": os.environ.get("MODEL", "?"),
        "offload": os.environ.get("OFFLOAD_EXPERTS", "0") == "1",
        "seed": int(os.environ.get("SEED", "0")),
        "steps_env": os.environ.get("STEPS"),
        "pod_id": os.environ.get("POD_ID") or os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "methodology": "CUDA events on the staging stream per copy, reduced at flush with a "
                       "single synchronize; routing via per-forward on-device bincount; "
                       "staging is layer-granular (whole fused stack per visit)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        import bitsandbytes

        meta["bitsandbytes_version"] = bitsandbytes.__version__
    except Exception:
        meta["bitsandbytes_version"] = None

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, sort_keys=True) + "\n")
        for p in probes:
            by_policy = {}
            for start, end, nbytes, policy in p.copies:
                agg = by_policy.setdefault(policy, {"n": 0, "ms": 0.0, "bytes": 0})
                agg["n"] += 1
                agg["ms"] += start.elapsed_time(end)
                agg["bytes"] += nbytes
            f.write(json.dumps({
                "row": "layer",
                "layer_id": p.layer_id,
                "storage_mode": p.storage_mode,
                "num_experts": p.num_experts,
                "forwards": p.forwards,
                "per_expert_bytes": p.per_expert_bytes,
                "h2d_bytes": p.h2d_bytes,
                "stage_copies": sum(a["n"] for a in by_policy.values()) or p.copy_count_no_cuda,
                "h2d_ms_by_policy": {k: round(v["ms"], 3) for k, v in sorted(by_policy.items())},
                "h2d_ms_total": round(sum(a["ms"] for a in by_policy.values()), 3),
            }, sort_keys=True) + "\n")
            hits = p.hits.cpu().tolist() if p.hits is not None else [0] * p.num_experts
            tokens = p.tokens.cpu().tolist() if p.tokens is not None else [0] * p.num_experts
            for e in range(p.num_experts):
                if hits[e] == 0:
                    continue  # cold experts are reconstructed by the summarizer from num_experts
                f.write(json.dumps({
                    "row": "expert",
                    "layer_id": p.layer_id,
                    "expert_id": e,
                    "hits": hits[e],
                    "tokens_routed": tokens[e],
                    "cache_hit": False,  # no cache exists in this pass, by design
                }, sort_keys=True) + "\n")
    print(f"[expert-profile] wrote {out_path}", flush=True)
    _STATE = None


def hot_sets_from_profile(path: str, hot_per_layer: int, *, key: str = "tokens_routed"):
    """Turn a profile into per-layer hot sets: the ``hot_per_layer`` most-routed experts.

    This is the difference between a residency dial that works and one that does not.
    Pinning experts BY INDEX is close to worthless -- measured on DeepSeek-V4-Flash
    (256 experts, top-6), doubling VRAM that way bought 7%, because a routed expert lands
    in a 16-wide index-ordered set about 6% of the time. Frequency is wildly non-uniform,
    so choosing by it is what turns VRAM into speed.

    Args:
        path: JSONL written by :func:`flush` (``E4B_EXPERT_PROFILE``).
        hot_per_layer: how many experts to keep resident per layer.
        key: ``tokens_routed`` (token-slots, the transfer-weighted quantity) or ``hits``
            (forwards that touched the expert at all). ``tokens_routed`` is the right
            default: it is proportional to the reads a hot set actually removes.

    Returns:
        ``list[list[int]]`` indexed by ``layer_id``, each inner list the chosen expert ids
        ordered most-routed first. Layers with no routing rows yield an empty list rather
        than an arbitrary pick -- silently pinning experts 0..H-1 for an unprofiled layer
        would reintroduce exactly the by-index behaviour this function exists to replace.
    """
    n_experts, counts = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row") == "layer":
                n_experts[row["layer_id"]] = row["num_experts"]
            elif row.get("row") == "expert":
                counts.setdefault(row["layer_id"], {})[row["expert_id"]] = row.get(key, 0)
    if not n_experts:
        raise ValueError(
            f"{path}: no layer rows -- the profile is empty. Most likely the model's expert "
            "modules were never dispatched, or E4B_EXPERT_PROFILE was unset at attach time.")

    out = []
    for layer_id in range(max(n_experts) + 1):
        c = counts.get(layer_id, {})
        # cold experts are omitted from the JSONL by design, so absence means zero
        ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        out.append([e for e, n in ranked[:hot_per_layer] if n > 0])
    return out


def coverage_from_profile(path: str, hot_sets, *, key: str = "tokens_routed") -> float:
    """Fraction of routed token-slots that ``hot_sets`` would have served from VRAM.

    The honest predictor of what a hot set buys, computable without re-running the model.
    A by-index set on a 256-expert top-6 layer scores near ``hot_per_layer / num_experts``;
    a frequency-ranked one scores far higher, and the gap is the whole point.
    """
    total = hit = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row") != "expert":
                continue
            n = row.get(key, 0)
            total += n
            lid = row["layer_id"]
            if lid < len(hot_sets) and row["expert_id"] in hot_sets[lid]:
                hit += n
    return (hit / total) if total else 0.0
