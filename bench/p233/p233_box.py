#!/usr/bin/env python3
"""p233_box.py -- lane P233, does serving at the solver placement keep quality? (bench/p233/PREREG-p233.md, MAP 2):
ONE process per phase. P232's box (bench/p232/p232_box.py) by counted edits: this lane's three placements in place of
P232's head knobs, the quality read at group T = 1 and 8 in place of 2, 4 and 8, no speed mode, and the placement proof.

The engine is the shipped server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at grouped-nf4-gemm v0.50.0,
``qwen3_moe`` defaults throughout, built EAGER. Every arm sets ``E4B_PAGED_GRAPHS=0``, ``E4B_INT8_HEAD_BW=1`` and
``E4B_INT8_HEAD_ROWS=8`` (Amendment 1: the head knobs named, never an unset default); the split cap, the fusions and every
other knob stay unset (their e4b default at the launch commit). The arms differ ONLY in where the experts sit:
  A  ``E4B_PAGED_PLACEMENT=all-vram`` (the budgets unset: unused there), the reference;
  S  ``solver`` at ``E4B_PAGED_VRAM_GB=8.0`` / ``E4B_PAGED_DRAM_GB=3.0``: all three tiers, the ruling arm;
  C  ``solver`` at ``8.0`` / ``0``: VRAM and NVMe only, reported.

``--mode quality``: Phase B's instrument (``p115_quality.measure_phase``), called UNMODIFIED and with exactly the keywords
``p115_quality.main()`` passes it (:data:`P115_MAIN_KEYWORDS`: no ``arms``, ``ref_kw`` or ``stand_in``, so the
instrument's own defaults hold), at group T = 1 and 8 in turn in ONE process, 24 wikitext windows x 128 positions after a
512-token prompt, every KV built inside the measurement at the server's 4096 tokens with the server's cap (A2's
``ServedKV``). Arm A's call is ``main --phase off --texts wikitext --windows 24 --group T --ref-dir REF/T<T>``: R, its
``rep`` floor draw and the instrument's other floor arms, R's log-probs saved per T; S and C are ``--phase on``, scored
against A's saved log-probs.

Recorded beside each measurement, never inside it:
- the placement PROOF (:func:`placement_record`): per MoE layer, the experts in VRAM, DRAM and on NVMe as the solver
  resolved them (read off each layer's tier state after the build), their bytes at the arena's bytes per expert, the
  resident stacks' bytes, and the layer's static path (the collapse or the tiered forward);
- per T, the path counts from the engagement (:class:`PathTap`): each layer's calls, the calls the collapse served, and
  a tiered layer's DRAM-row calls on the host's CPU kernels or on the GPU;
- on S and C, each window's teacher-forced argmax and the positions where it differs from A's (:class:`ScoreTap`, which
  observes the instrument's scorer and returns its result unchanged);
- the checkpoint revision, the NF4 arena's sha256, the windows' sha, the resolved levers (``levers.int8_head_bw`` with
  its ``rows`` and ``source``, the split cap, the fusions), the host CPU model, its ISA flags (AVX2, AVX-512, AMX) and the
  CPU kernels' thread count, the card's telemetry beside each T, and the wall times.

    python p233_box.py --mode quality --out quality_A.json --ref-dir work/ref     (P233_ARM=A / S / C)
    python p233_box.py --self-test
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARMS = ("A", "S", "C")
PLACEMENT_KNOB, VRAM_KNOB, DRAM_KNOB = "E4B_PAGED_PLACEMENT", "E4B_PAGED_VRAM_GB", "E4B_PAGED_DRAM_GB"
GRAPHS_KNOB = "E4B_PAGED_GRAPHS"
HEAD_KNOB, ROWS_KNOB = "E4B_INT8_HEAD_BW", "E4B_INT8_HEAD_ROWS"
#: per arm, the value each placement knob must carry (None: unset), the strings the PREREG registers
ARM_VALUE = {"A": {PLACEMENT_KNOB: "all-vram", VRAM_KNOB: None, DRAM_KNOB: None},
             "S": {PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "3.0"},
             "C": {PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "0"}}
BUDGET_GB = {"S": (8.0, 3.0), "C": (8.0, 0.0)}
#: every arm: eager, and the head knobs named (Amendment 1)
EVERY_ARM = {GRAPHS_KNOB: "0", HEAD_KNOB: "1", ROWS_KNOB: "8"}
CAP_KNOB = "E4B_ATTN_SPLIT_CAP"           # unset in every arm: the cap at its qwen3_moe default
DENSE_KNOB = "E4B_DENSE_GEMV_BW"         # unset: the dense attention GEMV off
SPEC_KNOB = "E4B_PAGED_SPEC"              # unset: plain decode
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
#: unset in every arm (their e4b default): the split cap and its fixed form, the fusions and the prefill glue, the
#: dense GEMV, speculative decode, the decode GEMV, and the two levers that refuse at solver (DESIGN-HC.md section 1)
DEFAULT_KNOBS = (CAP_KNOB, "E4B_ATTN_SPLIT_FIXED", DENSE_KNOB, SPEC_KNOB, "E4B_FUSE_PREFILL_GLUE", "E4B_NF4_SMALLM_BW",
                 "E4B_SERVE_EXP_INT4") + FUSION_KNOBS + GEMV_KNOBS
PHASE = {"A": "off", "S": "on", "C": "on"}
GROUPS = (1, 8)                           # T, each its own pass set
WINDOWS, SHAKE_WINDOWS = 24, 8            # 24 (and the shakedown's 8) divide by 1 and 8
#: the keywords p115_quality.main() passes measure_phase, in its order (tests/test_p233.py reads them off main's AST)
P115_MAIN_KEYWORDS = ("phase", "prompt", "cont", "chunk", "floor_chunk", "group", "device", "ref_dir", "counters", "fwd")
#: p115_quality.main()'s argparse defaults for the values it passes (tests/test_p233.py reads them off its AST)
P115_DEFAULTS = {"prompt": 512, "cont": 128, "chunk": 512, "floor_chunk": 256}
#: what measure_phase scores per phase at these keywords: its OFF_ARMS on ``off``, ON_ARMS on ``on``
PHASE_ARMS = {"A": ("R", "rep", "half", "chunk", "mutant_scale"), "S": ("ON",), "C": ("ON",)}
TIERS = ("vram", "dram", "nvme")
ISA_FLAGS = ("avx2", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16", "amx_tile", "amx_bf16", "amx_int8")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
TELEMETRY = ("power.limit", "clocks.max.sm", "temperature.gpu", "clocks.sm", "power.draw")


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the placement knobs carry the arm's values (unset where None), every arm's three knobs theirs, and
    every :data:`DEFAULT_KNOBS` knob is unset."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {}
    for k, want in {**ARM_VALUE[arm], **EVERY_ARM}.items():
        got = (env.get(k) or "").strip()
        if got != (want or ""):
            bad[k] = env.get(k)
    bad.update({k: env.get(k) for k in DEFAULT_KNOBS if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P233_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P233_ARM={arm!r}: {why}")
    return arm


def measure_kwargs(arm, T, a, device, ref_dir, counters, fwd) -> dict:
    """The keywords ``p115_quality.main()`` passes ``measure_phase``, by name and in its order, with this lane's values:
    the phase (A ``off``, S and C ``on``), main's prompt, continuation, chunk and floor chunk, the group T, the server's
    device, the references per T, the counters installed before the build and the forward counter. Nothing else: the
    instrument's own ``arms``, ``ref_kw`` and ``stand_in`` defaults hold."""
    return {"phase": PHASE[arm], "prompt": a.prompt, "cont": a.cont, "chunk": a.chunk, "floor_chunk": a.floor_chunk,
            "group": T, "device": device, "ref_dir": os.path.join(ref_dir, f"T{T}"), "counters": counters, "fwd": fwd}


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def head_calls(head) -> dict:
    """The head's own call counts by row count (``Int8HeadLinear.routed`` / ``.passed``); a bf16 head has none."""
    return {"routed": {str(k): v for k, v in sorted((getattr(head, "routed", None) or {}).items())},
            "passed": {str(k): v for k, v in sorted((getattr(head, "passed", None) or {}).items())}}


def calls_delta(after: dict, before: dict) -> dict:
    """The calls between two :func:`head_calls` reads, zeros dropped."""
    out = {}
    for side in ("routed", "passed"):
        a, b = after.get(side) or {}, before.get(side) or {}
        out[side] = {k: a[k] - b.get(k, 0) for k in a if a[k] - b.get(k, 0)}
    return out


def parse_telemetry(line: str) -> dict:
    """One ``nvidia-smi --query-gpu=<TELEMETRY> --format=csv,noheader,nounits`` line as floats; an unreadable field
    (``[N/A]``, missing) is None."""
    vals = [v.strip() for v in (line or "").split(",")]
    out = {}
    for i, k in enumerate(TELEMETRY):
        try:
            out[k] = float(vals[i])
        except (IndexError, ValueError):
            out[k] = None
    return out


def _telemetry() -> dict:
    """The card now: power limit, max SM clock, temperature, SM clock and draw. Read between timed blocks, never during
    one (a sampler would share the host with the timed loop)."""
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(TELEMETRY), "--format=csv,noheader,nounits"],
                                       text=True, timeout=20).splitlines()[0]
    except Exception:  # noqa: BLE001 -- an unreadable card reads None in every field; the reducer VOIDs on the limit
        line = ""
    return parse_telemetry(line)


def _engagement(parts) -> dict:
    """The knob's engagement now (engines/int8_head_bw.py): per row count, whether the swapped head routed or passed,
    and with ``E4B_INT8_HEAD_ROWS`` above 1 which kernel each routed row count took."""
    from experts4bit_qlora.engines.int8_head_bw import engagement
    return engagement(parts.runner.model)


def _sms():
    import torch
    return torch.cuda.get_device_properties(0).multi_processor_count


def _cc():
    import torch
    c = torch.cuda.get_device_capability(0)
    return f"{c[0]}.{c[1]}"


def moe_states(model) -> list:
    """Each MoE layer's tier state (``_hot_residency``), in the order ``build_engine`` patched them."""
    from experts4bit_qlora.engines.hot_residency import target_modules
    return [m._hot_residency for m in target_modules(model)]


def _nbytes(t) -> int:
    return int(t.numel() * t.element_size()) if hasattr(t, "numel") and hasattr(t, "element_size") else 0


def static_path(st) -> str:
    """The forward a layer takes, read as ``_HotResidency.forward`` gates it: the collapse when ``collapse_resident``,
    no amortization armed and every expert resident in identity order (``_all_hot``); the tiered forward otherwise."""
    collapse = bool(getattr(st, "collapse_resident", False) and getattr(st, "amort", None) is None and st._all_hot())
    return "collapse" if collapse else "tiered"


def placement_record(states, placement, vram_gb, dram_gb, bytes_per_expert) -> dict:
    """The placement as the solver resolved it and the tier holds it, per MoE layer: the experts in VRAM (``hot_ids``),
    DRAM (``dram_ids``) and on NVMe (the rest, ``nvme_set``), the bytes at the arena's bytes per expert, the resident
    stacks' own bytes (``h_*`` on the card, ``d_*`` in host memory), and the static path. Totals beside them."""
    per = []
    for i, st in enumerate(states):
        E = int(st.is_hot.numel())
        vram = int(st.hot_ids.numel())
        dram_ids = getattr(st, "dram_ids", None)
        dram = int(dram_ids.numel()) if dram_ids is not None else 0
        nset = getattr(st, "nvme_set", None)
        nvme = len(nset) if nset is not None else E - vram - dram
        per.append({"layer": i, "experts": E, "vram": vram, "dram": dram, "nvme": nvme, "path": static_path(st),
                    "resident_bytes": {"vram": sum(_nbytes(getattr(st, a, None)) for a in ("h_gu_p", "h_gu_a", "h_dn_p", "h_dn_a")),
                                       "dram": sum(_nbytes(getattr(st, a, None)) for a in ("d_gu_p", "d_gu_a", "d_dn_p", "d_dn_a"))}})
    tot = {t: sum(x[t] for x in per) for t in TIERS}
    return {"placement": placement, "budget_gb": {"vram": vram_gb, "dram": dram_gb},
            "budget_bytes": {"vram": int(vram_gb * 2**30), "dram": int(dram_gb * 2**30)},
            "moe_layers": len(per), "experts_per_layer": sorted({x["experts"] for x in per}),
            "bytes_per_expert": int(bytes_per_expert), "experts": tot,
            "bytes": {t: tot[t] * int(bytes_per_expert) for t in TIERS},
            "resident_bytes": {t: sum(x["resident_bytes"][t] for x in per) for t in ("vram", "dram")},
            "layers": {"collapse": sum(x["path"] == "collapse" for x in per), "tiered": sum(x["path"] == "tiered" for x in per)},
            "per_layer": per}


class PathTap:
    """While open, counts each MoE layer's calls on its tier state: every call (``forward``), the calls the
    all-resident collapse served (``_forward_collapsed``), and a tiered layer's DRAM rows computed on the host's CPU
    kernels (``_dram_contrib``) or copied to the GPU (``_dram_on_gpu``, the prefill's route). The counters are instance
    attributes shadowing the class's methods for the block, removed on exit; the methods run unchanged."""

    WRAPPED = ("forward", "_forward_collapsed", "_dram_contrib", "_dram_on_gpu")

    def __init__(self, states):
        self.states = list(states)
        self.counts = [dict.fromkeys(self.WRAPPED, 0) for _ in self.states]
        self.set = []

    def __enter__(self):
        for st, c in zip(self.states, self.counts):
            for name in self.WRAPPED:
                fn = getattr(st, name, None)
                if fn is None or name in vars(st):
                    continue

                def make(_fn, _c, _n):
                    def counted(*a, **k):
                        _c[_n] += 1
                        return _fn(*a, **k)
                    return counted
                setattr(st, name, make(fn, c, name))
                self.set.append((st, name))
        return self

    def __exit__(self, *exc):
        for st, name in self.set:
            vars(st).pop(name, None)
        self.set = []
        return False

    def report(self) -> dict:
        """Per layer: calls, collapse calls, tiered calls (the rest), DRAM-row calls on the CPU and on the GPU; and the
        layers whose calls were all collapse, all tiered, both (never expected) or none."""
        per = [{"calls": c["forward"], "collapse": c["_forward_collapsed"], "tiered": c["forward"] - c["_forward_collapsed"],
                "dram_cpu": c["_dram_contrib"], "dram_gpu": c["_dram_on_gpu"]} for c in self.counts]
        return {"per_layer": per,
                "totals": {k: sum(x[k] for x in per) for k in ("calls", "collapse", "tiered", "dram_cpu", "dram_gpu")},
                "layers_collapse": sum(1 for x in per if x["collapse"] and not x["tiered"]),
                "layers_tiered": sum(1 for x in per if x["tiered"] and not x["collapse"]),
                "layers_mixed": sum(1 for x in per if x["tiered"] and x["collapse"]),
                "layers_idle": sum(1 for x in per if not x["calls"])}


class ScoreTap:
    """While open, wraps ``p108_box._score`` (the scorer ``measure_phase`` calls once per scored window, looked up on
    the module at each call) and keeps each call's argmax in call order; the scorer's result is returned unchanged. On
    an ``on`` phase the instrument scores its one arm window by window in order, so call k is window k."""

    def __init__(self, mod):
        self.mod, self.argmax, self.orig = mod, [], None

    def __enter__(self):
        self.orig = self.mod._score

        def tap(lp, cont_tokens, _orig=self.orig):
            nll, am = _orig(lp, cont_tokens)
            self.argmax.append(list(am))
            return nll, am
        self.mod._score = tap
        return self

    def __exit__(self, *exc):
        self.mod._score = self.orig
        return False


def diff_positions(am: list, ref_am: list) -> list:
    """The positions where an arm's teacher-forced argmax differs from the reference's."""
    return [p for p, (x, y) in enumerate(zip(am, ref_am)) if x != y] + list(range(min(len(am), len(ref_am)),
                                                                                    max(len(am), len(ref_am))))


def argmax_report(tapped: list, per_window: list, ref_dir: str, n_windows: int, group: int) -> dict:
    """S or C at one T: each window's argmax (from :class:`ScoreTap`), A's (from A's saved log-probs, read as the
    instrument reads them), the positions that differ, and whether their count is measure_phase's own agreement."""
    import torch
    if len(tapped) != n_windows:
        return {"unread": f"the scorer was called {len(tapped)} times for {n_windows} windows"}
    agree = {x.get("window"): x.get("argmax_agree") for x in per_window}
    out = {"argmax": {}, "diff_positions": {}, "consistent": True}
    for g, g0 in enumerate(range(0, n_windows, group)):
        refs = torch.load(os.path.join(ref_dir, f"R_wikitext_g{g}.pt"))
        for i, r in enumerate(refs):
            w = g0 + i
            am, d = tapped[w], diff_positions(tapped[w], r.argmax(-1).tolist())
            out["argmax"][str(w)], out["diff_positions"][str(w)] = am, d
            ag = agree.get(w)
            if not isinstance(ag, (int, float)) or round(ag * len(am)) != len(am) - len(d):
                out["consistent"] = False
        del refs
    return out


def parse_cpuinfo(text: str) -> dict:
    """The first processor's model name and the ISA flags this lane records (:data:`ISA_FLAGS`) from /proc/cpuinfo;
    with no flags line the ISA is unread (None), never False."""
    model, flags, seen = None, set(), False
    for line in (text or "").splitlines():
        k, _, v = line.partition(":")
        k = k.strip()
        if k == "model name" and model is None:
            model = v.strip()
        elif k == "flags" and not seen:
            flags, seen = set(v.split()), True
    isa = ({"avx2": "avx2" in flags, "avx512": "avx512f" in flags, "amx": bool(flags & {"amx_tile", "amx_bf16", "amx_int8"})}
           if seen else None)
    return {"model": model, "isa": isa, "flags": sorted(flags & set(ISA_FLAGS)) if seen else None}


def _pool_size():
    """The CPU kernels' worker count (grouped-nf4-gemm's native pool, started by the hybrid tier), or None."""
    try:
        import gnf4_native
        return int(gnf4_native.load().gnf4_pool_size())
    except Exception:  # noqa: BLE001 -- recorded as unread
        return None


def host_cpu(states=()) -> dict:
    """The host CPU (reported, never ruled): model, ISA flags, logical CPUs, this process's affinity, torch's threads,
    the CPU kernels' pool size and the per-layer thread request (0 = the pool's), and the NUMA nodes."""
    import torch
    try:
        text = open("/proc/cpuinfo", encoding="utf-8", errors="replace").read()
    except OSError:
        text = ""
    rec = parse_cpuinfo(text)
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = None
    try:
        numa = len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()])
    except OSError:
        numa = None
    req = sorted({int(getattr(getattr(st, "mod", None), "_e4b_hybrid_threads", getattr(st, "_threads", 0)) or 0)
                  for st in states})
    rec.update(cpus=os.cpu_count(), affinity=aff, torch_threads=torch.get_num_threads(),
               cpu_kernel_threads=_pool_size(), layer_thread_request=req, numa_nodes=numa)
    return rec


def arena_record(path: str) -> dict:
    """The NF4 arena by content: sha256 of the arena file and of its index, its size, and the time the hash took."""
    t0 = time.time()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    with open(path + ".index.json", "rb") as f:
        idx = hashlib.sha256(f.read()).hexdigest()
    return {"sha256": h.hexdigest(), "bytes": os.path.getsize(path), "index_sha256": idx, "seconds": round(time.time() - t0, 1)}


class ServedKV:
    """Inside the block, every ``Fp8PagedKV`` is built at ``tokens`` per sequence and carries ``cap`` (None = the
    library's plan), as ``build_engine`` builds and sets the server's own. ``built`` keeps one entry per KV: the
    capacity its caller asked for, the one it got, and its ``split_plans`` (the dict itself, which the KV fills in
    place) -- never the KV, whose pool is freed with its pass."""

    def __init__(self, tokens: int, cap):
        self.tokens, self.cap, self.built = int(tokens), cap, []

    def __enter__(self):
        from experts4bit_qlora.engines import fp8_paged_kv as m
        self.cls, self.orig = m.Fp8PagedKV, m.Fp8PagedKV.__init__
        outer = self

        def init(kv, *a, **kw):
            if "max_tokens_per_seq" not in kw:
                raise SystemExit("REFUSED: a KV built without a keyword max_tokens_per_seq; its capacity cannot be served")
            asked = kw["max_tokens_per_seq"]
            if asked > outer.tokens:
                raise SystemExit(f"REFUSED: a KV asked for {asked} tokens, above the server's {outer.tokens}")
            kw["max_tokens_per_seq"] = outer.tokens
            outer.orig(kv, *a, **kw)
            kv.split_cap = outer.cap
            outer.built.append({"asked": asked, "max_tokens_per_seq": outer.tokens, "blocks_per_seq": kv.blocks_per_seq,
                                "split_cap": outer.cap, "plans": kv.split_plans})
        self.cls.__init__ = init
        return self

    def __exit__(self, *exc):
        self.cls.__init__ = self.orig
        return False

    def report(self) -> dict:
        """``(plans, summary)``: the plans merged over every KV (calls summed; plan and split count must agree), and
        the capacities."""
        plans, clash = {}, []
        for b in self.built:
            for B, r in b["plans"].items():
                m = plans.setdefault(str(B), {"calls": 0, "plan": r.get("plan"), "n_split": r.get("n_split")})
                m["calls"] += r.get("calls", 0)
                if (m["plan"], m["n_split"]) != (r.get("plan"), r.get("n_split")):
                    clash.append({"B": B, "first": [m["plan"], m["n_split"]], "then": [r.get("plan"), r.get("n_split")]})
        return {"passes": len(self.built), "asked": sorted({b["asked"] for b in self.built}),
                "max_tokens_per_seq": sorted({b["max_tokens_per_seq"] for b in self.built}),
                "blocks_per_seq": sorted({b["blocks_per_seq"] for b in self.built}),
                "split_cap": self.cap, "plans": plans, "plan_clash": clash}


def _common(cfg, parts, arm, load_s, shake):
    import torch
    info = parts.info
    model = parts.runner.model
    return {"arm": arm, "shakedown": bool(shake), "e4b_sha": os.environ.get("E4B_SHA"),
            "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
            "served_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "knobs": {k: os.environ.get(k) for k in tuple(ARM_VALUE["S"]) + tuple(EVERY_ARM) + DEFAULT_KNOBS},
            "torch": torch.__version__, "sm_count": _sms(), "cc": _cc(), "gpu": torch.cuda.get_device_name(0),
            "load_s": round(load_s, 2),
            "config": {"placement": cfg.placement, "vram_gb": cfg.vram_gb, "dram_gb": cfg.dram_gb, "graphs": cfg.graphs,
                       "max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq, "buckets": list(cfg.buckets),
                       "hot_rows": cfg.hot_rows, "torch_threads": cfg.torch_threads, "device": cfg.device},
            "levers": {"int8_head_bw": info.get("int8_head_bw"), "attn_split_cap": info.get("attn_split_cap"),
                       "fusion_modes": info.get("fusion_modes"), "fusion_sources": info.get("fusion_sources"),
                       "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "prefill_glue": info.get("prefill_glue"),
                       "dense_gemv_bw": info.get("dense_gemv_bw"), "k25_lut": info.get("k25_lut"),
                       "k25_plan": info.get("k25_plan"), "nf4_smallm_bw": info.get("nf4_smallm_bw")},
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "model_type": info.get("model_type"), "moe_layers": info.get("moe_layers"), "experts": info.get("experts"),
            "top_k": info.get("top_k")}


def quality_main(a) -> int:
    t_start = time.time()
    arm = _arm()
    if any(a.windows % T for T in GROUPS):
        raise SystemExit(f"REFUSED: --windows {a.windows} does not divide by every T in {GROUPS}")
    import torch
    import transformers

    import fp8_paged_attn
    import p108_box
    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, _bytes_per_expert, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != ARM_VALUE[arm][PLACEMENT_KNOB] or cfg.max_tokens_per_seq != 4096:
        raise SystemExit(f"REFUSED: arm {arm} builds the server eager at {ARM_VALUE[arm][PLACEMENT_KNOB]} and 4096 tokens "
                         f"per sequence (graphs={cfg.graphs}, placement={cfg.placement}, "
                         f"max_tokens_per_seq={cfg.max_tokens_per_seq})")
    if arm in BUDGET_GB and (cfg.vram_gb, cfg.dram_gb) != BUDGET_GB[arm]:
        raise SystemExit(f"REFUSED: arm {arm}'s budgets are {BUDGET_GB[arm]} GiB, the config read {(cfg.vram_gb, cfg.dram_gb)}")
    arena = arena_record(cfg.arena)
    counters = q.KernelCounters().install()                          # before build_engine, as p115_quality.main
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {"wikitext": q.LOADERS["wikitext"](parts.tokenizer, a.windows, a.prompt, a.cont)}
    common = _common(cfg, parts, arm, load_s, a.shakedown)
    states = moe_states(model)
    placement = placement_record(states, cfg.placement, cfg.vram_gb, cfg.dram_gb, _bytes_per_expert(cfg.arena))
    cap = ((common["levers"]["attn_split_cap"]) or {}).get("cap")
    blocks = -(-cfg.max_tokens_per_seq // 16)
    head = model.get_output_embeddings()
    c_build = _counts()
    phases = {}
    for T in GROUPS:
        kw = measure_kwargs(arm, T, a, cfg.device, a.ref_dir, counters, fwd)
        plan_t = fp8_paged_attn.decode_split_plan(T, 4, blocks, block_tokens=16, ktile=64, compute="fp8", pack_heads=False,
                                                  sms=_sms())
        c1, h1, tel = _counts(), head_calls(head), _telemetry()
        t1 = time.time()
        with ServedKV(cfg.max_tokens_per_seq, cap) as served, PathTap(states) as paths, ScoreTap(p108_box) as score:
            rec = q.measure_phase(model, windows, **kw)
        wall = time.time() - t1
        sk = served.report()
        am = (argmax_report(score.argmax, rec["per_window"]["wikitext"].get("ON") or [], kw["ref_dir"], a.windows, T)
              if PHASE[arm] == "on" else None)
        phases[str(T)] = {**rec, "attn_split_cap": {"cap": cap, "source": (common["levers"]["attn_split_cap"] or {}).get("source"),
                                                    "plans": sk["plans"]},
                          "quality_kv": {**{k: v for k, v in sk.items() if k != "plans"}, "plan_bT": plan_t},
                          "paths": paths.report(), "argmax": am, "head_calls": calls_delta(head_calls(head), h1),
                          "int8_head_bw_measure": _engagement(parts), "dispatch_measure": _delta(_counts(), c1),
                          "telemetry": {"before": tel, "after": _telemetry()}, "wall_s": round(wall, 1)}
        print(f"P233_QT {arm} T={T} windows={rec['windows']} paths={phases[str(T)]['paths']['totals']} "
              f"layers_tiered={phases[str(T)]['paths']['layers_tiered']} quality_kv={phases[str(T)]['quality_kv']} "
              f"{rec['seconds']} s", flush=True)
    out = {"mode": "quality", **common, "transformers": transformers.__version__, "arena": arena,
           "windows_sha256": q.windows_sha(windows["wikitext"]), "placement": placement, "host_cpu": host_cpu(states),
           "dispatch_build": _delta(c_build, c0), "phases": phases,
           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "wall_s": round(time.time() - t_start, 1),
           "status": "ok"}
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(f"P233_QUALITY {arm} experts={placement['experts']} layers={placement['layers']} "
          f"head={(common['levers']['int8_head_bw'] or {}).get('rows')} cpu={out['host_cpu'].get('model')} "
          f"isa={out['host_cpu'].get('isa')} threads={out['host_cpu'].get('cpu_kernel_threads')}", flush=True)
    return 0


# ------------------------------------------------------------------ self-test --
def _self_test_served_kv() -> list:
    """The patch on a stand-in class: the capacity raised to the server's, the cap set, the plans dict held."""
    import types
    from contextlib import contextmanager

    class FakeKV:
        def __init__(self, layers, h, d, *, batch, max_tokens_per_seq, device="cpu", scratch_slots=0):
            self.blocks_per_seq = -(-max_tokens_per_seq // 16)
            self.split_cap = None
            self.split_plans = {}

    @contextmanager
    def fake_module():
        mods = {n: sys.modules.get(n) for n in ("experts4bit_qlora", "experts4bit_qlora.engines",
                                                 "experts4bit_qlora.engines.fp8_paged_kv")}
        root, eng, mod = (types.ModuleType(n) for n in mods)
        mod.Fp8PagedKV = FakeKV
        eng.fp8_paged_kv = mod
        root.engines = eng
        sys.modules.update(dict(zip(mods, (root, eng, mod))))
        try:
            yield
        finally:
            for n, m in mods.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
    out = []
    with fake_module():
        orig = FakeKV.__init__
        with ServedKV(4096, 16) as s:
            kv = FakeKV(48, 4, 128, batch=8, max_tokens_per_seq=656)
            kv.split_plans.setdefault(8, {"calls": 0, "plan": 11, "n_split": 11})["calls"] += 48
            try:
                FakeKV(48, 4, 128, batch=8, max_tokens_per_seq=8192)
                refused = False
            except SystemExit:
                refused = True
        r = s.report()
        out += [("the pass's KV built at the server's capacity, with the cap", (kv.blocks_per_seq, kv.split_cap) == (256, 16)),
                ("the plans held", r["plans"] == {"8": {"calls": 48, "plan": 11, "n_split": 11}} and r["passes"] == 1
                 and r["asked"] == [656] and r["max_tokens_per_seq"] == [4096] and not r["plan_clash"]),
                ("a KV above the server's capacity is refused", refused),
                ("the class is restored on exit", FakeKV.__init__ is orig)]
    return out


class _StandIn:
    """A tier state's forward as ``_HotResidency.forward`` gates it (the collapse, or the tiered forward with its DRAM
    rows on the CPU at decode and on the GPU at prefill), on CPU tensors."""

    collapse_resident = True
    amort = None

    def __init__(self, E, hot, dram, *, order=None):
        import torch
        self.is_hot = torch.zeros(E, dtype=torch.bool)
        self.is_hot[hot] = True
        self.hot_ids = torch.tensor(sorted(hot), dtype=torch.long)
        self.dram_ids = torch.tensor(sorted(dram), dtype=torch.long)
        self.nvme_set = frozenset(range(E)) - set(hot) - set(dram)
        self.g2h = torch.tensor(order if order is not None else list(range(E)))
        self.h_gu_p = torch.zeros(len(hot), 6, 4, dtype=torch.uint8)
        self.h_gu_a = torch.zeros(len(hot), 6, 1, dtype=torch.float32)
        self.d_gu_p = torch.zeros(len(dram), 6, 4, dtype=torch.uint8)
        self._prefill = False

    def _all_hot(self):
        import torch
        return bool(self.is_hot.all()) and bool((self.g2h == torch.arange(self.is_hot.numel())).all())

    def forward(self, x, residual=None):
        if self.collapse_resident and self.amort is None and self._all_hot():
            return self._forward_collapsed(x)
        if self.dram_ids.numel():
            (self._dram_on_gpu if self._prefill else self._dram_contrib)(x)
        return x

    def _forward_collapsed(self, x):
        return x

    def _dram_contrib(self, x):
        return None

    def _dram_on_gpu(self, x):
        return None


def _self_test_paths() -> list:
    """The placement record and the path counts on CPU stand-ins: an all-resident layer, a three-tier layer, a VRAM+NVMe
    layer, and a resident layer whose hot stack is reordered (tiered: the collapse needs identity order)."""
    all_hot = _StandIn(8, list(range(8)), [])
    three = _StandIn(8, [0, 1, 2], [3, 4])
    vn = _StandIn(8, [0, 1, 2], [])
    reordered = _StandIn(4, [0, 1, 2, 3], [], order=[1, 0, 2, 3])
    states = [all_hot, three, vn]
    rec = placement_record(states, "solver", 8.0, 3.0, 10)
    with PathTap(states) as tap:
        for st in states:
            st.forward(1)
            st.forward(2, residual=0)
        three._prefill = True
        three.forward(3)
    rep = tap.report()
    after = all(name not in vars(st) for st in states for name in PathTap.WRAPPED)
    idle = PathTap([_StandIn(4, [0, 1, 2, 3], [])])
    with idle:
        pass
    return [("the experts per tier, as the tier holds them", rec["experts"] == {"vram": 14, "dram": 2, "nvme": 8}
             and [(x["vram"], x["dram"], x["nvme"]) for x in rec["per_layer"]] == [(8, 0, 0), (3, 2, 3), (3, 0, 5)]),
            ("the bytes at the arena's bytes per expert", rec["bytes"] == {"vram": 140, "dram": 20, "nvme": 80}),
            ("the resident stacks' own bytes", rec["resident_bytes"] == {"vram": (8 + 3 + 3) * (24 + 24), "dram": 2 * 24}),
            ("the static paths", [x["path"] for x in rec["per_layer"]] == ["collapse", "tiered", "tiered"]
             and rec["layers"] == {"collapse": 1, "tiered": 2}),
            ("a reordered resident stack is tiered, never the collapse", static_path(reordered) == "tiered"),
            ("the collapse off (amortization armed) is tiered",
             (lambda s: (setattr(s, "amort", {}), static_path(s))[1])(_StandIn(4, [0, 1, 2, 3], [])) == "tiered"),
            ("the budgets recorded", rec["budget_gb"] == {"vram": 8.0, "dram": 3.0} and rec["budget_bytes"]["dram"] == 3 * 2**30),
            ("the path counts per layer", rep["per_layer"] == [
                {"calls": 2, "collapse": 2, "tiered": 0, "dram_cpu": 0, "dram_gpu": 0},
                {"calls": 3, "collapse": 0, "tiered": 3, "dram_cpu": 2, "dram_gpu": 1},
                {"calls": 2, "collapse": 0, "tiered": 2, "dram_cpu": 0, "dram_gpu": 0}]),
            ("the layers by path", (rep["layers_collapse"], rep["layers_tiered"], rep["layers_mixed"], rep["layers_idle"])
             == (1, 2, 0, 0)),
            ("an idle layer is named", idle.report()["layers_idle"] == 1 and idle.report()["layers_collapse"] == 0),
            ("the counters are removed on exit, the methods still run", after and all_hot.forward(5) == 5)]


def _self_test_score() -> list:
    """The scorer tap on a stand-in module, and the argmax differences."""
    import types
    m = types.SimpleNamespace(_score=lambda lp, toks: ([0.5] * len(toks), [t + 1 for t in toks]))
    orig = m._score
    with ScoreTap(m) as tap:
        r1 = m._score(None, [1, 2, 3])
        m._score(None, [7, 8])
    return [("the scorer's result unchanged", r1 == ([0.5, 0.5, 0.5], [2, 3, 4])),
            ("each call's argmax, in order", tap.argmax == [[2, 3, 4], [8, 9]]),
            ("the scorer restored on exit", m._score is orig),
            ("the positions that differ", diff_positions([1, 2, 3, 4], [1, 5, 3, 6]) == [1, 3]),
            ("identical argmax: no position", diff_positions([4, 4], [4, 4]) == []),
            ("a short argmax names its missing positions", diff_positions([1], [1, 2, 3]) == [1, 2]),
            ("a miscount is unread, never scored", "unread" in argmax_report([[1]], [], "", 2, 1))]


def _self_test_cpu() -> list:
    text = ("processor\t: 0\nmodel name\t: AMD EPYC 9554 64-Core Processor\nflags\t\t: fpu sse avx avx2 avx512f "
            "avx512bw avx512vl avx512_bf16 avx512_vnni\nprocessor\t: 1\nmodel name\t: other\nflags\t\t: fpu\n")
    r = parse_cpuinfo(text)
    amx = parse_cpuinfo("model name : Xeon\nflags : avx2 avx512f amx_tile amx_bf16 amx_int8\n")
    return [("the model name", r["model"] == "AMD EPYC 9554 64-Core Processor"),
            ("the ISA flags, first processor", r["isa"] == {"avx2": True, "avx512": True, "amx": False}
             and r["flags"] == ["avx2", "avx512_bf16", "avx512_vnni", "avx512bw", "avx512f", "avx512vl"]),
            ("AMX read", amx["isa"]["amx"] is True),
            ("no flags line: the ISA is unread, never False", parse_cpuinfo("model name : x\n")["isa"] is None
             and parse_cpuinfo("")["flags"] is None)]


def _self_test_kwargs() -> list:
    import types
    a = _parser().parse_args([])
    kA = measure_kwargs("A", 8, a, "cuda", "/r", "K", "F")
    kS = measure_kwargs("S", 1, a, "cuda", "/r", "K", "F")
    return [("the parser's defaults are p115_quality.main's", {k: getattr(a, k) for k in P115_DEFAULTS} == P115_DEFAULTS
             and a.windows == WINDOWS),
            ("A's keywords are main's, in its order", tuple(kA) == P115_MAIN_KEYWORDS),
            ("A at main's values", kA == {"phase": "off", "prompt": 512, "cont": 128, "chunk": 512, "floor_chunk": 256,
                                          "group": 8, "device": "cuda", "ref_dir": os.path.join("/r", "T8"),
                                          "counters": "K", "fwd": "F"}),
            ("S scores on, against A's references at its T", kS["phase"] == "on" and kS["group"] == 1
             and kS["ref_dir"] == os.path.join("/r", "T1") and tuple(kS) == P115_MAIN_KEYWORDS),
            ("C scores on", measure_kwargs("C", 8, a, "cuda", "/r", 0, 0)["phase"] == "on"),
            ("no arms, ref_kw or stand_in", not {"arms", "ref_kw", "stand_in"} & set(kA)),
            ("what each phase scores", PHASE_ARMS == {"A": ("R", "rep", "half", "chunk", "mutant_scale"), "S": ("ON",),
                                                      "C": ("ON",)}),
            ("a shakedown's windows divide by every T", all(SHAKE_WINDOWS % T == 0 and WINDOWS % T == 0 for T in GROUPS)),
            ("the namespace is read, not a copy", measure_kwargs("A", 1, types.SimpleNamespace(
                prompt=1, cont=2, chunk=3, floor_chunk=4), "d", "r", 0, 0)["floor_chunk"] == 4)]


def self_test() -> int:
    base = {**EVERY_ARM}
    S = {**base, PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "3.0"}
    C = {**base, PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "0"}
    A = {**base, PLACEMENT_KNOB: "all-vram"}
    ok = [("arms", ARMS == ("A", "S", "C") and GROUPS == (1, 8)),
          ("A", arm_env_ok("A", A)[0]), ("S", arm_env_ok("S", S)[0]), ("C", arm_env_ok("C", C)[0]),
          ("S padded", arm_env_ok("S", {k: f" {v} " for k, v in S.items()})[0]),
          ("A with a budget named is refused (unused there)", not arm_env_ok("A", {**A, VRAM_KNOB: "8.0"})[0]),
          ("A at solver is refused", not arm_env_ok("A", {**A, PLACEMENT_KNOB: "solver"})[0]),
          ("A with the placement unset is refused", not arm_env_ok("A", base)[0]),
          ("S at another DRAM budget is refused", not arm_env_ok("S", {**S, DRAM_KNOB: "2.0"})[0]),
          ("S at another VRAM budget is refused", not arm_env_ok("S", {**S, VRAM_KNOB: "8"})[0]),
          ("S without its DRAM budget is refused", not arm_env_ok("S", {k: v for k, v in S.items() if k != DRAM_KNOB})[0]),
          ("C with S's DRAM budget is refused", not arm_env_ok("C", {**C, DRAM_KNOB: "3.0"})[0]),
          ("C at all-vram is refused", not arm_env_ok("C", {**C, PLACEMENT_KNOB: "all-vram"})[0]),
          ("graphs unset is refused (eager is named)", not arm_env_ok("S", {k: v for k, v in S.items() if k != GRAPHS_KNOB})[0]),
          ("graphs on is refused", not arm_env_ok("A", {**A, GRAPHS_KNOB: "1"})[0]),
          ("the head unset is refused (Amendment 1)", not arm_env_ok("S", {k: v for k, v in S.items() if k != HEAD_KNOB})[0]),
          ("the head off is refused", not arm_env_ok("A", {**A, HEAD_KNOB: "0"})[0]),
          ("the rows unset is refused (Amendment 1)", not arm_env_ok("C", {k: v for k, v in C.items() if k != ROWS_KNOB})[0]),
          ("the rows at 4 is refused", not arm_env_ok("S", {**S, ROWS_KNOB: "4"})[0]),
          ("the split cap set is refused (its default is every arm's)", not arm_env_ok("S", {**S, CAP_KNOB: "16"})[0]),
          ("a fixed split set is refused", not arm_env_ok("A", {**A, "E4B_ATTN_SPLIT_FIXED": "8"})[0]),
          ("a fusion knob set is refused", not arm_env_ok("S", {**S, "E4B_FUSE_T1_GLUE": "auto"})[0]),
          ("the dense attention GEMV set is refused", not arm_env_ok("A", {**A, DENSE_KNOB: "1"})[0]),
          ("speculative decode set is refused", not arm_env_ok("C", {**C, SPEC_KNOB: "eagle3"})[0]),
          ("the GEMV forced is refused", not arm_env_ok("S", {**S, "GNF4_GEMV_BW": "1"})[0]),
          ("E4B_NF4_SMALLM_BW set is refused", not arm_env_ok("A", {**A, "E4B_NF4_SMALLM_BW": "1"})[0]),
          ("an unknown arm", not arm_env_ok("R", A)[0]),
          ("the budgets the box checks", BUDGET_GB == {"S": (8.0, 3.0), "C": (8.0, 0.0)}),
          ("telemetry parses", parse_telemetry("575.00, 3090, 41, 2842, 80.12") == {
              "power.limit": 575.0, "clocks.max.sm": 3090.0, "temperature.gpu": 41.0, "clocks.sm": 2842.0, "power.draw": 80.12}),
          ("an unreadable field is None", parse_telemetry("[N/A], 3090, 41")["power.limit"] is None),
          ("the head's calls, zeros dropped", calls_delta({"routed": {"1": 5, "8": 2}, "passed": {}},
                                                          {"routed": {"1": 5}, "passed": {}}) == {"routed": {"8": 2}, "passed": {}})]
    ok += _self_test_kwargs()
    ok += _self_test_served_kv()
    ok += _self_test_paths()
    ok += _self_test_score()
    ok += _self_test_cpu()
    bad = [n for n, x in ok if not x]
    print(f"p233_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(ok) - len(bad)}/{len(ok)} cases)")
    return 0 if not bad else 1


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=("quality",))
    p.add_argument("--shakedown", action="store_true")
    p.add_argument("--out")
    p.add_argument("--ref-dir")
    p.add_argument("--windows", type=int, default=WINDOWS)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--floor-chunk", type=int, default=256)
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    if a.self_test:
        return self_test()
    if a.mode == "quality":
        if not (a.out and a.ref_dir):
            raise SystemExit("REFUSED: --out and --ref-dir are required")
        return quality_main(a)
    raise SystemExit("REFUSED: --mode quality or --self-test")


if __name__ == "__main__":
    sys.exit(main())
