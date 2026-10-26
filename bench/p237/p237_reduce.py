#!/usr/bin/env python3
"""p237_reduce.py -- P237's rule (bench/p237/PREREG-p237.md "Engagement", "Measures", "The verdict rule"), over normalized rows.

A ROW is one tc1_arm process from the schedule (a primer, a draw or a UT probe); ``normalize_proc`` builds it from the receipt, the
hook's record (p237_arm.py), the profile and the per-process logs. Primers and probes are recorded and never ruled; the probes are
re-read to re-derive N_t (p237_host.tune), which the UT draws must have run with. Every check FAILS CLOSED: a missing, None or empty
input is a fault, never a pass, and ``--selftest`` removes each piece of evidence in turn to show it.

Draw checks (any fault VOIDs that draw): presence and status; the pins and the release proof; the card and its clock; the cap (the
fraction read back, the reserved peak under it); the step contract and the steps; amendment 81's device-time marker; the engagement
per framework:
- E: E4B_OFFLOAD_STATS on, every layer offloaded, per step exactly 2n - r_i whole-layer stagings in train mode (r_i = 1 when layer
  0's handle is resident at edge i), no routed or speculative staging, one layer_major_step call per step plus the guard's one before
  edge 0, and gnf4's route per call as route_for decides it on sm_89 (dense at 1..16 present groups, fused above);
- UA / UT: the block swap holds the arm's N layers; at every edge its residency agrees with the tensors and at most ``depth`` blocks
  are resident; per step every swapped block has >= M train evictions and >= M - 1 train fetches; the shared counter's structure,
  environment, library, dispatch and unfused checks; the profiled window's per-call decisions equal F15 v2's kernel counts.
Per budget: the step contract equal across every draw; quality (COMPARABLE at 0.05 nats, reported against 0.01); stability (5 %);
the wall and device ratios U*/E. The full card's UA row refused with Unsloth's own not-engaged line reads N_auto = 0.

Standard library only, plus bench/common/unsloth_moe_counter.py (the counter's own checks) and bench/tc1/f15v2.py (its counts).
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "tc1"))
import card_clock  # noqa: E402  -- the per-run clock rule (passes, burn=False), read, never re-typed
import f15v2  # noqa: E402  -- TC1 amendment 80's merged F15 v2 counting, read, never re-typed
import p237_host  # noqa: E402
import unsloth_moe_counter as C  # noqa: E402

N_LAYERS, STEPS, ACCUM, TIMED = 48, 60, 4, (10, 60)    # timed steps 11..60 (0-based 10..59)
PROFILED = (21, 24)                                  # --profile-warm 21 --profile-steps 3: loop steps 21..23, edges 21 and 24 (past
#                                                      the cold step after the eval at 20)
SHAKE_STEPS = 24
FRAMEWORK = {"E": "e4b", "UA": "unsloth", "UT": "unsloth"}
PINS = {"experts4bit-qlora": "0.58.1", "grouped-nf4-gemm": "0.51.0", "unsloth": "2026.10.8", "unsloth_zoo": "2026.10.8"}
TORCH_PREFIX = "2.12.1"
CARD, CAP, MIN_SMS = "4090", [8, 9], 128
CLOCK_FLOOR_MHZ, SWPC_CLEAR = 2500, 0.95
CAP_12G_GIB = 12
RUNNER_UNSLOTH = {"UNSLOTH_MOE_GROUPED_TRITON": "auto", "UNSLOTH_MOE_NF4_FUSED": "0", "UNSLOTH_MOE_NF4_AUTO_MAX_ROWS": "128",
                  "UNSLOTH_MOE_NF4_AUTOTUNE": "0", "UNSLOTH_MOE_FUSED_ROUTER": "0", "UNSLOTH_ENABLE_LOGGING": "1"}
# The box's environment: ONE list for three places, bound by tests/test_p237.py (a mutant widening or narrowing any one fails it): the
# runner's ambient refusal (any key under these prefixes but the registered ones refuses, 78, before STARTED), its env_box.txt grep
# (these prefixes plus the exact names), and check_env_box's registered set. The per-process keys (E4B_OFFLOAD_STATS, TC1_STEP_MEM,
# TC1_ARM_HOOK, TC1_ARM_HOOK_SHA256) are set on each process's own line, after the dump, so the registered set has no TC1_* key. Any
# other PYTORCH_* key, and the image's own environment, is information (env_image.txt), never a fault.
ENV_PREFIXES = ("UNSLOTH", "E4B", "GNF4", "NF4_QLORA", "TC1")
ENV_EXACT = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
ENV_BOX = {**RUNNER_UNSLOTH, "GNF4_TRAIN_GEMM": "auto", **ENV_EXACT, "E4B_SHA": None}   # None: any 40-hex commit (the driver's HEAD)
ENV_REFUSE_GREP = "^(" + "|".join(ENV_PREFIXES) + ")_[A-Za-z0-9_]+"
ENV_BOX_GREP = "^(" + "|".join(p + "_" for p in ENV_PREFIXES) + ")|" + "|".join(f"^{k}=" for k in ENV_EXACT)
UNREAD = "UNREAD"                                    # a measure whose input is missing: never 0, never a guess
WHOLE_LAYER = {"sync", "cold_miss", "prefetch"}      # offload.py's whole-layer staging tags; routed / speculative stage rows
DENSE_AUTO_MAX_GROUPS = 16                           # gnf4 0.51.0 nf4_route: auto is dense at 1..16 present groups, fused above
DEVICE_MS_METHOD = "kernels_only_at_source"         # amendment 81's marker
COMPARABLE, SAME_STACK, UNSTABLE = 0.05, 0.01, 0.05


# -- draw checks ------------------------------------------------------------------------------------------------------------------
def check_present(row):
    if not row:
        return ["no record"]
    why = []
    if row.get("dry"):
        why.append("a dry run")
    if row.get("status") != "ok":
        why.append(f"status {row.get('status')!r}")
    return why


def check_pins(row):
    v = row.get("versions") or {}
    why = [f"{k} {v.get(k)!r} != {want}" for k, want in PINS.items() if v.get(k) != want]
    if not str(v.get("torch") or "").startswith(TORCH_PREFIX):
        why.append(f"torch {v.get('torch')!r} is not {TORCH_PREFIX}")
    if not v.get("release_proof_ok"):
        why.append("the release proof is missing or refused")
    return why


def check_env_box(row):
    """The box's env_box.txt: its key set must EQUAL ENV_BOX at the registered values (E4B_SHA any 40-hex commit). No file, an
    unregistered key (a line that is not KEY=VALUE included) or a missing one VOIDs."""
    env = row.get("env_box")
    if env is None:
        return ["no env_box.txt"]
    why = [f"env_box: an unregistered {k}" for k in sorted(set(env) - set(ENV_BOX))]
    why += [f"env_box: {k} missing" for k in sorted(set(ENV_BOX) - set(env))]
    why += [f"env_box: {k}={env[k]!r} != {v!r}" for k, v in ENV_BOX.items() if v is not None and k in env and env[k] != v]
    if "E4B_SHA" in env and not re.fullmatch(r"[0-9a-f]{40}", env["E4B_SHA"] or ""):
        why.append(f"env_box: E4B_SHA {env['E4B_SHA']!r} is not a commit")
    return why


def check_card(row):
    c, k = row.get("card") or {}, row.get("clock") or {}
    why = []
    if CARD not in str(c.get("name") or ""):
        why.append(f"card {c.get('name')!r}")
    if list(c.get("cap") or []) != CAP:
        why.append(f"cc {c.get('cap')}")
    if (c.get("sm_count") or 0) < MIN_SMS:
        why.append(f"SMs {c.get('sm_count')}")
    try:                                             # the PREREG's per-draw rule is card_clock's run-log mode: the busy median at
        ok, w = card_clock.passes(k, CLOCK_FLOOR_MHZ, SWPC_CLEAR, burn=False)   # the floor, every refusing bit clear on >= 95 %
    except (KeyError, TypeError) as e:
        ok, w = False, f"unreadable clock summary ({type(e).__name__}: {e})"
    if not ok:
        why.append(f"clock: {w}")
    return why


def max_reserved(row):
    vals = [r.get("max_reserved") for r in row.get("step_mem") or []]
    return max(vals) if vals and all(isinstance(v, int) for v in vals) else None


def check_cap(row):
    """12 GiB: every arm's fraction is 12 GiB / total, read back, and the steps' reserved peak is under it. Full card: E runs uncapped
    (no vram_cap); UA / UT set the card's own total (fraction 1.0, read back), since --unsloth-offload-layers needs the cap."""
    vc, budget, arm = row.get("vram_cap"), row.get("budget"), row.get("arm")
    mr = max_reserved(row)
    if mr is None:
        return ["no per-step reserved peak (TC1_STEP_MEM)"]
    if budget == "full" and arm == "E":
        return [] if vc is None else [f"E at the full card carries a cap {vc}"]
    res = (vc or {}).get("resolved") or {}
    frac, read, total = res.get("fraction"), res.get("getter_reads"), res.get("total_bytes")
    if not isinstance(frac, float) or not isinstance(total, int) or read != frac:
        return [f"the cap is unread or not read back: {vc}"]
    want = CAP_12G_GIB * 2 ** 30 / total if budget == "12g" else 1.0
    why = []
    if budget not in ("12g", "full") or abs(frac - want) > 1e-12:
        why.append(f"budget {budget!r}: fraction {frac!r}, not {want!r}")
    if mr > frac * total:
        why.append(f"reserved peak {mr} over the cap {frac * total:.0f}")
    return why


def check_steps(row, steps):
    why = []
    if len(row.get("step_ms") or []) != steps or len(row.get("tokens_per_step") or []) != steps:
        return [f"{len(row.get('step_ms') or [])} steps recorded, not {steps}"]
    sc = row.get("step_contract") or {}
    if sc.get("optimizer_steps") != steps or sc.get("accum") != ACCUM or sc.get("micro_batch") != 2 or not sc.get("tokens_per_step_sha256"):
        why.append(f"step contract {sc}")
    m = ((row.get("profile") or {}).get("summary") or {}).get("device_ms_method")
    if m != DEVICE_MS_METHOD:
        why.append(f"profile device_ms_method {m!r}, not {DEVICE_MS_METHOD!r}")
    return why


def _edges(h, steps):
    edges = h.get("edges") or []
    if [e.get("i") for e in edges] != list(range(steps + 1)):
        return None
    return edges


def route_for(n_groups):
    return "dense" if 1 <= n_groups <= DENSE_AUTO_MAX_GROUPS else "fused"


def check_e4b(row, steps):
    h = row.get("hook") or {}
    facts = h.get("facts") or {}
    edges = _edges(h, steps)
    if edges is None:
        return [f"step edges are not 0..{steps}"]
    why = []
    if (facts.get("env") or {}).get("E4B_OFFLOAD_STATS") != "1" or facts.get("stats_enabled") is not True:
        why.append("E4B_OFFLOAD_STATS is not on")
    if facts.get("offload_handles") != N_LAYERS:
        why.append(f"{facts.get('offload_handles')} offloaded layers, not {N_LAYERS}")
    if (row.get("layer_major") or {}).get("given") != 1:
        why.append(f"layer_major {row.get('layer_major')}")
    stage, lm = h.get("stage") or {}, h.get("layer_major_calls") or {}
    if lm.get("pre") != 1:
        why.append(f"{lm.get('pre')} layer_major_step calls before edge 0, not the guard's 1")
    for i in range(steps):
        st = stage.get(str(i)) or {}
        train = {k.split("|", 1)[1]: v for k, v in st.items() if k.startswith("train|")}
        n = sum(v[0] for v in train.values())
        r = 1 if (edges[i].get("staging") or {}).get("resident_handle") == 0 else 0
        if n != 2 * N_LAYERS - r:
            why.append(f"step {i}: {n} train stagings, not {2 * N_LAYERS - r} (layer 0 resident at the edge: {bool(r)})")
        bad = sorted(set(train) - WHOLE_LAYER)
        if bad:
            why.append(f"step {i}: staged by {bad}")
        if lm.get(str(i)) != 1:
            why.append(f"step {i}: {lm.get(str(i))} layer_major_step calls, not 1")
    for e, log in sorted((h.get("route_log") or {}).items()):
        for key, n in log.items():
            m = re.fullmatch(r"(\d+):(\w+)", key)
            if not m:
                why.append(f"interval {e}: unreadable route-log key {key!r}")
            elif m.group(2) != route_for(int(m.group(1))):
                why.append(f"interval {e}: {n} calls at {m.group(1)} groups took {m.group(2)}; route_for says {route_for(int(m.group(1)))}")
    if not any((h.get("route_log") or {}).get(str(i)) for i in range(steps)):
        why.append("no gnf4 route was logged in any step")
    return why[:12]


def _safe(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return False, f"{getattr(fn, '__name__', fn)}: unreadable snapshot ({type(e).__name__}: {e})"


def window_decisions(edges):
    """The per-call GEMM decisions inside the profiled window (edge 20 -> edge 23): (Triton-decided, torch-decided)."""
    a, b = (edges[PROFILED[0]].get("unsloth") or {}).get("dispatch"), (edges[PROFILED[1]].get("unsloth") or {}).get("dispatch")
    if a is None or b is None:
        return None
    d = {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b) if k.startswith("gemm:")}
    return (sum(v for k, v in d.items() if k.endswith(":triton")), sum(v for k, v in d.items() if k.endswith(":torch")))


def check_unsloth(row, steps, n_want):
    h = row.get("hook") or {}
    facts = h.get("facts") or {}
    edges = _edges(h, steps)
    if edges is None:
        return [f"step edges are not 0..{steps}"]
    why = []
    if row.get("ckpt_mode") != "unsloth":
        why.append(f"ckpt_mode {row.get('ckpt_mode')!r}")
    bs = facts.get("block_swap")
    resolved = ((row.get("unsloth_offload_layers") or {}).get("resolved") or {}).get("n_layers")
    if not bs or not bs.get("indices"):
        return why + ["no block swap"]
    idx, depth = [str(i) for i in bs["indices"]], bs.get("depth")
    if len(idx) != resolved or (n_want is not None and len(idx) != n_want):
        why.append(f"the swap holds {len(idx)} layers; resolved {resolved}, the arm's N {n_want}")
    if not isinstance(depth, int) or depth < 1:
        why.append(f"prefetch depth {depth!r}")
    for e in edges:
        s = e.get("swap")
        if not s:
            why.append(f"edge {e.get('i')}: no swap state")
            continue
        if s.get("evicted_nonempty") or s.get("resident_empty"):
            why.append(f"edge {e['i']}: residency disagrees with the tensors {s}")
        if isinstance(depth, int) and len(s.get("resident") or []) > depth:
            why.append(f"edge {e['i']}: {len(s['resident'])} blocks resident > depth {depth}")
    fetch, evict = h.get("fetch") or {}, h.get("evict") or {}
    for i in range(steps):
        f = (fetch.get(str(i)) or {}).get("train") or {}
        v = (evict.get(str(i)) or {}).get("train") or {}
        short = [L for L in idx if (v.get(L) or 0) < ACCUM or ((f.get(L) or [0])[0]) < ACCUM - 1]
        if short:
            why.append(f"step {i}: blocks {short[:6]} under {ACCUM} evictions / {ACCUM - 1} fetches")
    snap = edges[-1].get("unsloth")
    st = h.get("structural")
    if not snap:
        return why + ["no counter snapshot"]
    if not st or st.get("steps") != steps:
        return why + [f"no structural count over {steps} steps ({(st or {}).get('steps')})"]
    why += [f"structure: {f}" for f in st.get("faults") or []]
    if not row.get("unsloth_env"):
        why.append("no UNSLOTH_* environment record (the arm's env dump)")
    registered = {**RUNNER_UNSLOTH, **((row.get("unsloth_knobs") or {}).get("env_set") or {})}
    for ok, w in (_safe(C.check_env, snap, dict(row.get("unsloth_env") or {}), registered), _safe(C.check_library, snap),
                  _safe(C.check_dispatch, snap, st, base_must_be_triton=False), _safe(C.check_unfused, snap)):
        if not ok:
            why.append(w)
    if steps >= PROFILED[1]:
        wd, prof = window_decisions(edges), row.get("profile")
        try:
            c = f15v2.counts(prof)
        except (KeyError, TypeError, ZeroDivisionError) as e:
            c = None
            why.append(f"F15 v2 counts: unreadable profile ({type(e).__name__})")
        if wd is None:
            why.append("no dispatch snapshot at the profiled window's edges")
        elif c is not None:
            n = (prof.get("summary") or {}).get("profiled_steps") or 0
            if round(c["triton_gemm"] * n) != wd[0] or round(c["torch_grouped"] * n) != wd[1]:
                why.append(f"profiled window: decisions Triton {wd[0]} / torch {wd[1]} != kernels {c['triton_gemm'] * n:g} / {c['torch_grouped'] * n:g}")
    return why[:12]


def draw_faults(row, steps=STEPS, n_want=None):
    why = check_present(row)
    if why:
        return why
    why += check_pins(row) + check_env_box(row) + check_card(row) + check_cap(row) + check_steps(row, steps)
    why += check_e4b(row, steps) if row.get("framework") == "e4b" else check_unsloth(row, steps, n_want)
    return why


# -- the readings --------------------------------------------------------------------------------------------------------------------
def s_per_step(row):
    ms = (row.get("step_ms") or [])[TIMED[0]:TIMED[1]]
    return statistics.median(ms) / 1e3 if ms else None


def device_ms(row):
    return ((row.get("profile") or {}).get("summary") or {}).get("device_ms")


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _or(x):
    return UNREAD if x is None else x


def measures(row):
    """Every per-draw measure the PREREG registers ("Measures (recorded per draw)"), computed here and never assembled by hand: each
    reads UNREAD, never 0, when its input is missing. Setup partitions the time before the first timed step:
    load_s + placement_s + warmup_s == prologue_s + step 0, where placement = prologue - load - E's guard (eval0 and the harness setup
    included) and warm-up = step 0 + E's guard."""
    e4b = row.get("framework") == "e4b"
    h, box, clk = row.get("hook") or {}, row.get("card") or {}, row.get("clock") or {}
    lm_guard = ((row.get("layer_major") or {}).get("guard") or {}) if e4b else {}
    load, pro = _num(row.get("load_s")), _num(row.get("prologue_s"))
    step_ms = row.get("step_ms") or []
    s0 = _num(step_ms[0]) if step_ms else None
    guard = _num(lm_guard.get("s")) if e4b else 0.0
    m = {"heldout_loss": _or(_num(row.get("heldout_loss"))), "s_per_step": _or(s_per_step(row)), "device_ms": _or(_num(device_ms(row))),
         "setup_load_s": _or(load),
         "setup_placement_s": UNREAD if None in (load, pro, guard) else round(pro - load - guard, 3),
         "setup_warmup_s": UNREAD if None in (s0, guard) else round(s0 / 1e3 + guard, 3),
         "peak_vram_gb": _or(_num(row.get("peak_vram_gb"))),
         "guard_peak_vram_gb": _or(_num(lm_guard.get("peak_vram_gb"))) if e4b else "n/a (no guard)",
         "smi_peak_used_mib": _or(_num(row.get("smi_peak_used_mib"))),
         "dmon_rx_mbps": _or(row.get("dmon_rx")), "dmon_tx_mbps": _or(row.get("dmon_tx")),
         "pcie_gen_max": _or(box.get("pcie_gen_max") or None), "pcie_width_max": _or(box.get("pcie_width_max") or None),
         "sm_busy_median_mhz": _or(_num(clk.get("sm_median"))), "clock_reasons_seen": clk["reasons_seen"] if "reasons_seen" in clk else UNREAD,
         "cpu_class": _or(box.get("cpu_class") or None)}
    hr = row.get("host_ram") or {}
    for k in ("rss_hwm_gb", "ru_maxrss_gb", "cgroup_peak_gb"):
        m[f"host_ram_{k}"] = _or(_num(hr.get(k)))
    steps = [str(i) for i in range(*TIMED) if i < len(step_ms)] or None
    per, by = [], {}
    for i in steps or []:
        if e4b:
            st = (h.get("stage") or {}).get(i)
            train = {k.split("|", 1)[1]: v for k, v in (st or {}).items() if k.startswith("train|")}
            per.append(sum(v[1] for v in train.values()) if st is not None else None)
            for pol, v in train.items():
                by[pol] = by.get(pol, 0) + v[1]
        else:
            f = ((h.get("fetch") or {}).get(i) or {}).get("train")
            per.append(sum(v[1] for v in f.values()) if f is not None else None)
    m["h2d_bytes_per_step"] = UNREAD if not per or any(v is None for v in per) else statistics.median(per)
    if e4b:
        m["h2d_bytes_by_policy"] = by if by and m["h2d_bytes_per_step"] != UNREAD else UNREAD
    return m


def parse_dmon(text):
    """``nvidia-smi dmon -s ut`` rows: the median rxpci and txpci (MB/s) over the samples, by the header's own column names."""
    cols, rx, tx = None, [], []
    for line in (text or "").splitlines():
        parts = line.lstrip("#").split()
        if line.startswith("#") and "rxpci" in parts:
            cols = parts
            continue
        if line.startswith("#") or not parts or cols is None or len(parts) != len(cols):
            continue
        for name, out in (("rxpci", rx), ("txpci", tx)):
            try:
                out.append(float(parts[cols.index(name)]))
            except ValueError:
                pass
    return (statistics.median(rx) if rx else None), (statistics.median(tx) if tx else None)


def parse_smi_used(text):
    """The peak ``memory.used`` (MiB) of a ``timestamp, memory.used`` log."""
    vals = []
    for line in (text or "").splitlines():
        try:
            vals.append(float(line.rsplit(",", 1)[1]))
        except (IndexError, ValueError):
            pass
    return max(vals) if vals else None


def unstable(d1, d2):
    why = []
    for name, get in (("wall", s_per_step), ("device", device_ms)):
        x, y = get(d1), get(d2)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            why.append(f"{name}: unread")
        elif abs(x - y) / ((x + y) / 2) > UNSTABLE:
            why.append(f"{name}: draws {x:.4g} / {y:.4g} apart by more than {UNSTABLE:.0%}")
    return why


def _mean(rows, key):
    vals = [r.get(key) for r in rows]
    return statistics.mean(vals) if vals and all(isinstance(v, (int, float)) for v in vals) else None


def compare(u, e):
    """One U*/E comparison at one budget: quality (COMPARABLE at 0.05 nats; 0.01 reported), the wall ratio (the verdict) and the device
    ratio, each the median over the cross-draw ratios, quoted only when both arms are VALID, COMPARABLE and stable."""
    out = {"u": u["arm"], "e": "E"}
    hu, he = _mean(u["draws"], "heldout_loss"), _mean(e["draws"], "heldout_loss")
    if hu is None or he is None:
        return dict(out, quality="UNREAD")
    d = abs(hu - he)
    out.update(heldout_u=round(hu, 5), heldout_e=round(he, 5), delta=round(d, 5), within_same_stack=d <= SAME_STACK,
               quality="COMPARABLE" if d <= COMPARABLE else "QUALITY_FAIL")
    if out["quality"] != "COMPARABLE" or u["unstable"] or e["unstable"]:
        return dict(out, ratio=None, why="not COMPARABLE or UNSTABLE: no ratio quoted")
    for name, get in (("wall", s_per_step), ("device", device_ms)):
        rs = [get(x) / get(y) for x in u["draws"] for y in e["draws"]]
        out[f"{name}_ratio"] = round(statistics.median(rs), 4)
        out[f"{name}_ratio_range"] = [round(min(rs), 4), round(max(rs), 4)]
    return out


def reduce_budget(rows, budget, n_auto, n_t):
    """One budget's verdict from its draws (primers and probes excluded)."""
    draws = [r for r in rows if r.get("budget") == budget and r.get("kind") == "draw"]
    out = {"budget": budget, "n_auto": n_auto, "n_t": n_t, "arms": {}}
    contracts = set()
    for arm in ("E", "UA", "UT"):
        ds = sorted((r for r in draws if r.get("arm") == arm), key=lambda r: r.get("draw"))
        if arm != "E" and n_auto == 0:
            out["arms"][arm] = {"state": "N_AUTO_ZERO", "draws": len(ds)}
            continue
        statuses = sorted(str(r.get("status")) for r in ds)
        if len(ds) == 2 and statuses == ["oom", "oom"]:  # a REPRODUCED OOM is a reading: "does not fit at this budget" (ruled before
            faults = [check_pins(r) + check_env_box(r) for r in ds]   # data); the stack must still be the registered one
            out["arms"][arm] = {"state": "VOID" if any(faults) else "OOM", "faults": faults, "draws": ds, "unstable": [],
                                "words": "does not fit at this budget"}
            continue
        if len(ds) == 2 and statuses == ["ok", "oom"]:
            out["arms"][arm] = {"state": "VOID", "faults": [["one draw OOM and the other not: an unstable memory reading"]],
                                "draws": ds, "unstable": ["one draw OOM"]}
            continue
        if arm == "UT" and not ds and (out["arms"].get("UA") or {}).get("state") == "OOM":
            out["arms"][arm] = {"state": "NOT_RUN", "draws": 0, "words": "UA did not fit, so N_auto is unread and there is nothing to tune"}
            continue
        n_want = n_auto if arm == "UA" else n_t
        faults = [draw_faults(r, n_want=n_want) for r in ds]
        valid = len(ds) == 2 and not any(faults)
        for r in ds:
            contracts.add(json.dumps(r.get("step_contract"), sort_keys=True))
        out["arms"][arm] = {"state": "VALID" if valid else "VOID", "faults": faults, "draws": ds,
                            "unstable": unstable(*ds) if len(ds) == 2 else ["not two draws"]}
    if len(contracts) > 1:
        out["contract_fault"] = f"{len(contracts)} different step contracts across the budget's draws"
    e = out["arms"]["E"]
    out["comparisons"] = []
    for arm in ("UA", "UT"):
        u = out["arms"][arm]
        if u["state"] == "VALID" and e["state"] == "VALID" and "contract_fault" not in out:
            out["comparisons"].append(compare(dict(u, arm=arm), e))
    void = [a for a, v in out["arms"].items() if v["state"] == "VOID"] + (["contract"] if "contract_fault" in out else [])
    out["verdict"] = "VOID" if void else "READ"
    out["void"] = void
    if n_auto == 0:
        out["words"] = "Unsloth needed no offload at the full card; E's offload was not required to fit."
    for a in out["arms"].values():                      # the rows stay in the receipts; the verdict carries their scalars
        if "draws" in a and isinstance(a["draws"], list):
            a["draws"] = [{"index": r.get("index"), "draw": r.get("draw"), **measures(r)} for r in a["draws"]]
    return out


# -- the receipts on disk -> rows ------------------------------------------------------------------------------------------------
RELEASES = ("experts4bit-qlora", "grouped-nf4-gemm", "unsloth", "unsloth_zoo", "transformers")


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def release_proof_ok(path):
    seen = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                seen[d.get("dist")] = bool(d.get("record_equal"))
    except (OSError, ValueError):
        return False
    return all(seen.get(d) is True for d in RELEASES)


def clock_facts(path):
    """A process's gpuclk log through card_clock.py's own summary (the burn's parser and busy threshold)."""
    try:
        return card_clock.summarize_log(path)
    except OSError:
        return {}


def tag_of(idx, kind, budget, arm, draw, off=None):
    return f"p237_{idx}_{kind}_{budget}_{arm}_d{draw}" + (f"_n{off}" if off is not None else "")


def _text(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None


def read_env(path):
    """``KEY=VALUE`` lines into a dict; a line that is not one is kept under its own text as a key, so it reads as unregistered."""
    text = _text(path)
    if text is None:
        return None
    env = {}
    for line in text.splitlines():
        if line.strip():
            k, sep, v = line.partition("=")
            env[k if sep and k else line] = v
    return env


def normalize_proc(work, idx, kind, budget, arm, draw, box, proof_ok, off=None, env_box=None):
    fw = FRAMEWORK[arm]
    base = os.path.join(work, "loads", str(idx), f"qwen3_{fw}_{tag_of(idx, kind, budget, arm, draw, off)}")
    cell = _json(base + ".json") or {}
    rec = ((cell.get("arm_hook") or {}).get("record")) or {}
    env = cell.get("env") or {}
    rx, tx = parse_dmon(_text(os.path.join(work, "loads", f"dmon_{idx}.txt")))
    return {"index": idx, "kind": kind, "budget": budget, "arm": arm, "draw": draw, "framework": fw, "offload_given": off,
            "status": cell.get("status"), "reason": cell.get("reason"), "dry": False, "cell": cell,
            "versions": {**{k: env.get(k) for k in PINS}, "torch": env.get("torch"), "release_proof_ok": proof_ok},
            "card": box, "clock": clock_facts(os.path.join(work, "loads", f"gpuclk_{idx}.txt")),
            "vram_cap": cell.get("vram_cap"), "step_mem": cell.get("step_mem"), "step_contract": cell.get("step_contract"),
            "layer_major": cell.get("layer_major"), "unsloth_offload_layers": cell.get("unsloth_offload_layers"),
            "unsloth_knobs": cell.get("unsloth_knobs"), "unsloth_env": env.get("unsloth_env"), "ckpt_mode": cell.get("ckpt_mode"),
            "hook": rec, "step_ms": cell.get("step_ms"), "tokens_per_step": cell.get("tokens_per_step"),
            "profile": _json(base + "_profile.json"), "heldout_loss": cell.get("eval_loss_final"),
            "peak_vram_gb": cell.get("peak_vram_gb"), "trainable_params": cell.get("trainable_params"), "env_box": env_box,
            "load_s": cell.get("load_s"), "prologue_s": cell.get("prologue_s"), "host_ram": cell.get("host_ram"),
            "smi_peak_used_mib": parse_smi_used(_text(os.path.join(work, "loads", f"vram_{idx}.txt"))), "dmon_rx": rx, "dmon_tx": tx}


def _log(work, idx):
    try:
        return open(os.path.join(work, "logs", f"load_{idx}.log"), encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def reduce_work(work, mode="read"):
    """The box: the schedule it ran, every row, N_auto and N_t re-derived from the receipts, and the verdict per budget. READ only
    when every budget READs (each fault is reported on its row; a VOID draw is never dropped to rescue the rest). A shakedown reports
    the same checks at its own step count and is never a reading."""
    import p237_schedule
    box = _json(os.path.join(work, "box.json")) or {}
    proof = release_proof_ok(os.path.join(work, "release_proof.json"))
    env_box = read_env(os.path.join(work, "env_box.txt"))
    sched = p237_schedule.schedule(shake=mode == "shake")
    steps_of = {"primer": 10, "draw": SHAKE_STEPS if mode == "shake" else STEPS}
    rows, n_auto, n_t, probes = [], {}, {}, {}
    for s in sched:
        idx, kind, budget, arm, draw = s["index"], s["kind"], s["budget"], s["arm"], s["draw"]
        if kind == "probes":
            na = n_auto.get(budget)
            outs = []
            for k in range(1, 5):
                if na is None or na - k < 1:
                    break
                r = normalize_proc(work, f"{idx}.{k}", "probe", budget, "UT", 0, box, proof, off=na - k, env_box=env_box)
                o = p237_host.probe_outcome(r["cell"], CAP_12G_GIB if budget == "12g" else box.get("card_gib"))
                outs.append({"n": na - k, "outcome": o})
                if o.split()[0] in ("oom", "fault"):
                    break
            probes[budget] = outs
            n_t[budget] = None if na is None else (None if na == 0 else p237_host.tune(na, [o["outcome"] for o in outs]))
            continue
        off = None if arm == "E" else ("auto" if arm == "UA" else n_t.get(budget))
        if arm == "UT" and off is None:
            continue
        r = normalize_proc(work, idx, kind, budget, arm, draw, box, proof, off=off, env_box=env_box)
        rows.append(r)
        if arm == "UA" and kind == "draw" and draw == 1:
            n_auto[budget] = p237_host.n_auto(r["cell"], _log(work, idx))
    out = {"mode": mode, "n_auto": n_auto, "n_t": n_t, "probes": probes, "budgets": [], "image_env": read_env(os.path.join(work, "env_image.txt")),
           "primers": [{"index": r["index"], "budget": r["budget"], "arm": r["arm"], "cold_cache": True, **measures(r)}
                       for r in rows if r["kind"] == "primer"]}
    steps = steps_of["draw"]
    for budget in sorted({s["budget"] for s in sched}, key=["12g", "full"].index):
        if mode == "shake":
            b = {"budget": budget, "rows": [{"index": r["index"], "arm": r["arm"], "kind": r["kind"],
                                             "faults": draw_faults(r, steps=steps, n_want=n_auto.get(budget) if r["arm"] == "UA" else None),
                                             "measures": measures(r)}
                                            for r in rows if r["budget"] == budget]}
            b["verdict"] = "VOID" if any(x["faults"] for x in b["rows"]) else "READ"
        else:
            b = reduce_budget(rows, budget, n_auto.get(budget), n_t.get(budget))
        out["budgets"].append(b)
    out["verdict"] = "READ" if out["budgets"] and all(b["verdict"] == "READ" for b in out["budgets"]) else "VOID"
    return out


# -- the self-test ---------------------------------------------------------------------------------------------------------------
def _good_e(steps=STEPS):
    edges = [{"i": i, "mode": "train", "staging": {"resident_handle": None if i == 0 else 0}} for i in range(steps + 1)]
    stage = {str(i): {"train|sync": [2 * N_LAYERS - (0 if i == 0 else 1), (2 * N_LAYERS - (0 if i == 0 else 1)) * 10 ** 6, 0]}
             for i in range(steps)}
    stage["pre"] = {"load|sync": [3, 0, 0], "eval|sync": [N_LAYERS, 0, 0], "train|sync": [2 * N_LAYERS, 0, 0]}
    return {"status": "ok", "dry": False, "framework": "e4b", "arm": "E", "budget": "12g", "kind": "draw", "draw": 1,
            "versions": {**PINS, "torch": "2.12.1+cu130", "release_proof_ok": True},
            "card": {"name": "NVIDIA GeForce RTX 4090", "cap": [8, 9], "sm_count": 128, "pcie_gen_max": "4", "pcie_width_max": "16",
                     "cpu_class": "ryzen-7000-9000-desktop"},
            "env_box": {**{k: v for k, v in ENV_BOX.items() if v is not None}, "E4B_SHA": "a" * 40},
            "load_s": 300.0, "prologue_s": 420.0, "peak_vram_gb": 11.5, "smi_peak_used_mib": 12100.0, "dmon_rx": 9000.0, "dmon_tx": 200.0,
            "host_ram": {"rss_hwm_gb": 110.0, "ru_maxrss_gb": 110.0, "cgroup_peak_gb": 115.0},
            "clock": {"busy_samples": 400, "sm_median": 2600, "busy_clean_fraction": 1.0, "busy_soft_clean_fraction": 1.0,
                      "hard_reasons_any_sample": [], "reasons_seen": []},
            "vram_cap": {"given_gib": 12.0, "resolved": {"fraction": 12 * 2 ** 30 / 25_757_220_864, "total_bytes": 25_757_220_864,
                                                        "getter_reads": 12 * 2 ** 30 / 25_757_220_864}},
            "step_mem": [{"step": i + 1, "max_reserved": 11 * 2 ** 30} for i in range(steps)],
            "step_contract": {"optimizer_steps": steps, "accum": ACCUM, "micro_batch": 2, "tokens_per_step_sha256": "ab"},
            "layer_major": {"given": 1, "guard": {"s": 30.0, "peak_vram_gb": 14.0}}, "step_ms": [3000.0] * steps, "tokens_per_step": [16000] * steps,
            "profile": {"summary": {"device_ms_method": DEVICE_MS_METHOD, "device_ms": 2000.0, "profiled_steps": 3},
                        "top_device": [], "top_cpu": []},
            "hook": {"facts": {"env": {"E4B_OFFLOAD_STATS": "1"}, "stats_enabled": True, "offload_handles": N_LAYERS},
                     "edges": edges, "stage": stage, "layer_major_calls": {"pre": 1, **{str(i): 1 for i in range(steps)}},
                     "route_log": {str(i): {"128:fused": 2 * N_LAYERS, "12:dense": 4} for i in range(steps)}},
            "heldout_loss": 1.20}


def _good_u(arm="UA", n=24, steps=STEPS):
    idx = list(range(N_LAYERS - n, N_LAYERS))
    triton, torch_ = 300, 60                               # per step, Triton- and torch-decided GEMMs
    snaps = []
    for i in range(steps + 1):
        snaps.append({"dispatch": {"gemm:lora:triton": 200 * i, "gemm:many:triton": (triton - 200) * i, "gemm:many:torch": torch_ * i}})
    edges = [{"i": i, "mode": "train", "swap": {"resident": idx[:2], "evicted_nonempty": [], "resident_empty": []},
              "unsloth": snaps[i]} for i in range(steps + 1)]
    fetch = {str(i): {"train": {str(L): [ACCUM, 1000 * ACCUM] for L in idx}} for i in range(steps)}
    evict = {str(i): {"train": {str(L): ACCUM for L in idx}} for i in range(steps)}
    prof = {"summary": {"device_ms_method": DEVICE_MS_METHOD, "device_ms": 2500.0, "profiled_steps": 3},
            "top_device": [{"name": f15v2.TRITON_GEMM, "count": triton * 3}], "top_cpu": [{"name": f15v2.CPU_GROUPED, "count": torch_ * 3}]}
    row = dict(_good_e(steps), framework="unsloth", arm=arm, ckpt_mode="unsloth", profile=prof, heldout_loss=1.22,
               unsloth_offload_layers={"given": "auto" if arm == "UA" else n, "resolved": {"n_layers": n, "indices": idx}},
               unsloth_knobs={"env_set": {"UNSLOTH_MOE_BACKEND": "grouped_mm"}},
               unsloth_env={**RUNNER_UNSLOTH, "UNSLOTH_MOE_BACKEND": "grouped_mm"})
    row["hook"] = {"facts": {"block_swap": {"indices": idx, "blocks": n, "depth": 2}}, "edges": edges, "fetch": fetch, "evict": evict,
                   "structural": {"steps": steps, "faults": []}}
    return row


def selftest():
    import copy
    cases = 0
    # the counter's own checks are the library's, exercised by its own tests; here they are stood in by passing fakes, so this test
    # binds THIS reducer's wiring (which check runs on which arm, and that a failing one VOIDs)
    real = (C.check_env, C.check_library, C.check_dispatch, C.check_unfused)
    calls = []

    def ok(name):
        def f(*a, **k):
            calls.append(name)
            return True, ""
        return f
    C.check_env, C.check_library, C.check_dispatch, C.check_unfused = (ok("env"), ok("library"), ok("dispatch"), ok("unfused"))
    try:
        e, u = _good_e(), _good_u()
        assert draw_faults(e) == [], draw_faults(e)
        assert draw_faults(u, n_want=24) == [], draw_faults(u, n_want=24)
        assert sorted(set(calls)) == ["dispatch", "env", "library", "unfused"], calls
        cases += 3
        for row, key in [(e, k) for k in ("versions", "card", "clock", "vram_cap", "step_mem", "step_contract", "layer_major", "hook",
                                          "profile", "step_ms")] + [(u, k) for k in ("hook", "unsloth_offload_layers", "ckpt_mode",
                                                                                  "profile", "unsloth_env")]:
            r = copy.deepcopy(row)                      # RULE 7: every piece of evidence removed -> a fault
            r.pop(key)
            assert draw_faults(r, n_want=24), f"{row['arm']} without {key} still reads clean"
            cases += 1
        for mutate, label in [
            (lambda r: r.update(status="oom"), "an OOM row"),
            (lambda r: r.update(dry=True), "a dry row"),
            (lambda r: r["versions"].update(unsloth="2026.10.9"), "another Unsloth"),
            (lambda r: r["clock"].update(sm_median=2400), "a capped card"),
            (lambda r: r["clock"].update(busy_clean_fraction=0.94), "refusing bits on 6 % of busy samples"),
            (lambda r: r["clock"].update(busy_samples=9), "a card not read under load"),
            (lambda r: r["clock"].pop("busy_clean_fraction"), "an unreadable clock summary"),
            (lambda r: r["card"].update(name="NVIDIA GeForce RTX 5090"), "another card"),
            (lambda r: r["step_mem"][7].update(max_reserved=13 * 2 ** 30), "a reserved peak over the cap"),
            (lambda r: r["vram_cap"]["resolved"].update(getter_reads=1.0), "a cap not read back"),
            (lambda r: r.update(budget="full"), "E carrying a cap at the full card"),
            (lambda r: r["hook"]["stage"]["30"].update({"train|sync": [2 * N_LAYERS, 0, 0]}), "one staging too many"),
            (lambda r: r["hook"]["stage"]["0"].update({"train|sync": [2 * N_LAYERS - 1, 0, 0]}), "step 0 one short (cold)"),
            (lambda r: r["hook"]["stage"]["5"].update({"train|routed": [1, 0, 0]}), "a routed staging"),
            (lambda r: r["hook"]["stage"].pop("41"), "a step with no stagings"),
            (lambda r: r["hook"]["layer_major_calls"].update(pre=0), "no guard call"),
            (lambda r: r["hook"]["layer_major_calls"].update({"9": 0}), "a step on the stock loop"),
            (lambda r: r["hook"]["route_log"]["3"].update({"12:fused": 1}), "a route route_for would not take"),
            (lambda r: r["hook"]["route_log"]["3"].update({"x": 1}), "an unreadable route key"),
            (lambda r: r["hook"].update(route_log={}), "no route logged"),
            (lambda r: r["hook"]["facts"]["env"].update(E4B_OFFLOAD_STATS=None), "stats off"),
            (lambda r: r["hook"]["facts"].update(offload_handles=47), "a layer not offloaded"),
            (lambda r: r["hook"]["edges"].pop(), "a missing closing edge"),
            (lambda r: r["step_contract"].update(accum=8), "another accumulation"),
            (lambda r: r["profile"]["summary"].update(device_ms_method="legacy"), "another device-time method"),
            (lambda r: r["tokens_per_step"].pop(), "a step short"),
        ]:
            r = copy.deepcopy(e)
            mutate(r)
            assert draw_faults(r), f"E: {label} still reads clean"
            cases += 1
        # E: a step after an eval starts cold (no layer resident at its edge): 96 is right there, 95 is not
        r = copy.deepcopy(e)
        r["hook"]["edges"][21]["staging"]["resident_handle"] = None
        assert draw_faults(r), "a cold step reading 95 still reads clean"
        r["hook"]["stage"]["21"]["train|sync"][0] = 2 * N_LAYERS
        assert draw_faults(r) == [], draw_faults(r)
        cases += 2
        for mutate, label in [
            (lambda r: r.update(ckpt_mode="hf"), "another checkpointer"),
            (lambda r: r["hook"]["facts"].update(block_swap=None), "no swap"),
            (lambda r: r["hook"]["facts"]["block_swap"].update(indices=r["hook"]["facts"]["block_swap"]["indices"][1:]), "a swap one short"),
            (lambda r: r["hook"]["edges"][12]["swap"].update(evicted_nonempty=[30]), "an evicted block holding weights"),
            (lambda r: r["hook"]["edges"][12]["swap"].update(resident_empty=[46]), "a resident block holding an empty tensor"),
            (lambda r: r["hook"]["edges"][12]["swap"].update(resident=[24, 25, 26]), "more blocks resident than the depth"),
            (lambda r: r["hook"]["edges"][12].pop("swap"), "an edge with no swap state"),
            (lambda r: r["hook"]["evict"]["17"]["train"].update({"30": ACCUM - 1}), "a block evicted once too few"),
            (lambda r: r["hook"]["fetch"]["17"]["train"].update({"30": [ACCUM - 2, 0]}), "a block fetched twice too few"),
            (lambda r: r["hook"]["fetch"].pop("17"), "a step with no fetches"),
            (lambda r: r["hook"]["structural"].update(faults=["step 3: expert module 7 ran no training pass"]), "a structural fault"),
            (lambda r: r["hook"]["structural"].update(steps=59), "a structure over the wrong steps"),
            (lambda r: r["hook"]["edges"][-1].pop("unsloth"), "no closing snapshot"),
            (lambda r: r["hook"]["edges"][24]["unsloth"]["dispatch"].update({"gemm:many:torch": 60 * 24 + 1}), "a window decision more than the kernels"),
            (lambda r: r["profile"]["top_cpu"][0].update(count=179), "a window kernel short"),
            (lambda r: r["unsloth_offload_layers"]["resolved"].update(n_layers=23), "a resolved count that is not the swap's"),
        ]:
            r = copy.deepcopy(u)
            mutate(r)
            assert draw_faults(r, n_want=24), f"UA: {label} still reads clean"
            cases += 1
        assert draw_faults(u, n_want=20), "UT run at another N than N_t still reads clean"
        cases += 1
        for name in ("env", "library", "dispatch", "unfused"):     # each counter check, failing, VOIDs the draw
            saved = getattr(C, f"check_{name}")
            setattr(C, f"check_{name}", lambda *a, **k: (False, "failing on purpose"))
            assert draw_faults(u, n_want=24), f"a failing check_{name} still reads clean"
            setattr(C, f"check_{name}", saved)
            setattr(C, f"check_{name}", lambda *a, **k: (_ for _ in ()).throw(KeyError("counts")))
            assert any("unreadable snapshot" in w for w in draw_faults(u, n_want=24)), f"a raising check_{name} is not a fault"
            setattr(C, f"check_{name}", saved)
            cases += 2
        # the box environment: one registered set, at its values
        for mutate, label in [
            (lambda r: r["env_box"].update(E4B_TRAIN_DETERMINISTIC="1"), "an ambient e4b knob"),
            (lambda r: r["env_box"].update(TC1_QKV_FLOOR="1"), "an ambient TC1 key"),
            (lambda r: r["env_box"].pop("GNF4_TRAIN_GEMM"), "a registered key missing"),
            (lambda r: r["env_box"].update(UNSLOTH_MOE_NF4_FUSED="auto"), "a registered key at another value"),
            (lambda r: r["env_box"].update(E4B_SHA="main"), "a launch commit that is not a sha"),
            (lambda r: r.update(env_box=None), "no env_box.txt"),
        ]:
            r = copy.deepcopy(e)
            mutate(r)
            assert any("env_box" in w for w in draw_faults(r)), f"{label} still reads clean"
            cases += 1
        # the measures: every registered one computed per draw, UNREAD (never 0) when its input is removed
        m = measures(e)
        assert m["setup_load_s"] == 300.0 and m["setup_placement_s"] == 90.0 and m["setup_warmup_s"] == 33.0, m
        assert m["h2d_bytes_per_step"] == 95 * 10 ** 6 and m["h2d_bytes_by_policy"] == {"sync": 50 * 95 * 10 ** 6}, m
        assert m["guard_peak_vram_gb"] == 14.0 and m["host_ram_cgroup_peak_gb"] == 115.0 and m["pcie_gen_max"] == "4", m
        assert measures(u)["h2d_bytes_per_step"] == 24 * 1000 * ACCUM and measures(u)["guard_peak_vram_gb"].startswith("n/a")
        assert measures(u)["setup_placement_s"] == 120.0
        assert UNREAD not in m.values() and UNREAD not in measures(u).values(), (m, measures(u))
        cases += 5
        for label, mutate, keys in [
            ("load_s", lambda r: r.pop("load_s"), ("setup_load_s", "setup_placement_s")),
            ("prologue_s", lambda r: r.pop("prologue_s"), ("setup_placement_s",)),
            ("the guard", lambda r: r["layer_major"].pop("guard"), ("setup_placement_s", "setup_warmup_s", "guard_peak_vram_gb")),
            ("step_ms", lambda r: r.pop("step_ms"), ("setup_warmup_s", "s_per_step", "h2d_bytes_per_step")),
            ("host_ram", lambda r: r.pop("host_ram"), ("host_ram_rss_hwm_gb", "host_ram_ru_maxrss_gb", "host_ram_cgroup_peak_gb")),
            ("a step's stagings", lambda r: r["hook"]["stage"].pop("33"), ("h2d_bytes_per_step", "h2d_bytes_by_policy")),
            ("the vram log", lambda r: r.update(smi_peak_used_mib=None), ("smi_peak_used_mib",)),
            ("the dmon log", lambda r: r.update(dmon_rx=None, dmon_tx=None), ("dmon_rx_mbps", "dmon_tx_mbps")),
            ("the box facts", lambda r: r.update(card={}), ("pcie_gen_max", "pcie_width_max", "cpu_class")),
            ("the clock", lambda r: r.update(clock={}), ("sm_busy_median_mhz", "clock_reasons_seen")),
            ("the profile", lambda r: r.pop("profile"), ("device_ms",)),
            ("the held-out loss", lambda r: r.pop("heldout_loss"), ("heldout_loss",)),
            ("peak_vram_gb", lambda r: r.pop("peak_vram_gb"), ("peak_vram_gb",)),
        ]:
            r = copy.deepcopy(e)
            mutate(r)
            got = measures(r)
            assert all(got[k] == UNREAD for k in keys), (label, {k: got[k] for k in keys})
            cases += 1
        r = copy.deepcopy(u)
        r["hook"]["fetch"].pop("40")
        assert measures(r)["h2d_bytes_per_step"] == UNREAD
        cases += 1
        dmon = "\n".join(["# gpu sm mem enc dec jpg ofa rxpci txpci", "# Idx % % % % % % MB/s MB/s", " 0 90 40 0 0 0 0 9000 200",
                          " 0 95 41 0 0 0 0 11000 300", " 0 - - - - - - - -"])
        assert parse_dmon(dmon) == (10000.0, 250.0), parse_dmon(dmon)
        assert parse_dmon("") == (None, None) and parse_smi_used("\n".join(["t, 100", "t, 12100", "bad"])) == 12100.0
        assert parse_smi_used(None) is None
        cases += 2
        # the full card: E uncapped; UA / UT at fraction 1.0
        f = copy.deepcopy(u)
        f["budget"] = "full"
        f["vram_cap"]["resolved"].update(fraction=1.0, getter_reads=1.0)
        assert draw_faults(f, n_want=24) == [], draw_faults(f, n_want=24)
        ef = dict(copy.deepcopy(e), budget="full", vram_cap=None)
        assert draw_faults(ef) == [], draw_faults(ef)
        cases += 2
        # budgets: two VALID draws per arm read; ratios and quality
        rows = []
        for arm, mk in (("E", _good_e), ("UA", _good_u), ("UT", lambda: _good_u("UT", 20))):
            for d in (1, 2):
                r = mk()
                r.update(arm=arm, draw=d, index=len(rows) + 1)
                rows.append(r)
        b = reduce_budget(rows, "12g", 24, 20)
        assert b["verdict"] == "READ" and len(b["comparisons"]) == 2, b
        c = b["comparisons"][0]
        assert c["quality"] == "COMPARABLE" and c["wall_ratio"] == 1.0 and c["device_ratio"] == 1.25 and c["within_same_stack"] is False, c
        rows2 = copy.deepcopy(rows)
        rows2[3]["heldout_loss"] = 1.40                     # UA's mean moves 0.09 from E's: QUALITY_FAIL, no ratio
        assert reduce_budget(rows2, "12g", 24, 20)["comparisons"][0]["quality"] == "QUALITY_FAIL"
        rows3 = copy.deepcopy(rows)
        rows3[1]["step_ms"] = [3300.0] * STEPS              # E's draws 10 % apart: UNSTABLE, no ratio
        assert reduce_budget(rows3, "12g", 24, 20)["comparisons"][0].get("ratio", 0) is None
        rows4 = copy.deepcopy(rows)
        rows4[5]["step_contract"]["tokens_per_step_sha256"] = "cd"
        assert reduce_budget(rows4, "12g", 24, 20)["verdict"] == "VOID"
        assert reduce_budget(rows[:-1], "12g", 24, 20)["verdict"] == "VOID"          # a draw missing
        z = reduce_budget([dict(copy.deepcopy(r), budget="full", vram_cap=None) for r in rows if r["arm"] == "E"], "full", 0, None)
        assert z["verdict"] == "READ" and z["arms"]["UA"]["state"] == "N_AUTO_ZERO" and "no offload" in z["words"], z
        cases += 6
        # OOM on a draw (ruled before data): both draws OOM = a READ outcome "does not fit", out of every ratio, never VOID; one draw
        # OOM VOIDs the pair; a hook fault (harness_error, exit 10) or any other status stays VOID; the other budget is unchanged
        oom = lambda r: {k: r[k] for k in ("arm", "draw", "index", "kind", "budget", "framework", "versions", "env_box")} | {"status": "oom"}   # noqa: E731
        rows_oom = [oom(r) if r["arm"] == "E" else r for r in rows]
        b = reduce_budget(rows_oom, "12g", 24, 20)
        assert b["verdict"] == "READ" and b["arms"]["E"]["state"] == "OOM" and b["comparisons"] == [], b
        assert b["arms"]["UA"]["state"] == "VALID" and b["arms"]["E"]["words"] == "does not fit at this budget"
        assert reduce_budget([dict(copy.deepcopy(r), budget="full", vram_cap=None) for r in rows if r["arm"] == "E"], "full", 0, None)["verdict"] == "READ"
        one = [oom(r) if r["arm"] == "E" and r["draw"] == 1 else r for r in rows]
        assert reduce_budget(one, "12g", 24, 20)["arms"]["E"]["state"] == "VOID"
        assert reduce_budget(one, "12g", 24, 20)["verdict"] == "VOID"
        hook = [dict(oom(r), status="harness_error") if r["arm"] == "E" else r for r in rows]
        assert reduce_budget(hook, "12g", 24, 20)["verdict"] == "VOID"
        bad_env = [dict(oom(r), env_box=None) if r["arm"] == "E" else r for r in rows]
        assert reduce_budget(bad_env, "12g", 24, 20)["arms"]["E"]["state"] == "VOID"      # an OOM on an unregistered stack reads nothing
        ua_oom = [oom(r) if r["arm"] == "UA" else r for r in rows if r["arm"] != "UT"]
        bu = reduce_budget(ua_oom, "12g", None, None)
        assert bu["verdict"] == "READ" and bu["arms"]["UA"]["state"] == "OOM" and bu["arms"]["UT"]["state"] == "NOT_RUN", bu
        cases += 7
    finally:
        C.check_env, C.check_library, C.check_dispatch, C.check_unfused = real
    # the host decisions the runner and this reducer share
    assert p237_host.tune(24, ["ok", "ok", "oom"]) == 22 and p237_host.tune(24, ["oom"]) == 24 and p237_host.tune(24, ["over"] * 4) == 24
    assert p237_host.tune(24, ["ok", "fault"]) is None
    cases += 2
    return cases


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--work", help="a box's run directory (/root/p237 on the box, or a fetched copy)")
    ap.add_argument("--mode", choices=("read", "shake"), default="read")
    ap.add_argument("--out", help="where the verdict JSON goes")
    a = ap.parse_args(argv)
    if a.selftest:
        print(f"P237 REDUCE SELFTEST OK cases={selftest()}")
        return 0
    if not a.work or not a.out:
        ap.error("--work and --out, or --selftest")
    out = reduce_work(a.work, a.mode)
    with open(a.out, "wb") as f:
        f.write((json.dumps(out, indent=1, sort_keys=True, default=str) + "\n").encode())
    print(f"P237 {out['verdict']} ({a.mode}): " + "; ".join(f"{b['budget']} {b['verdict']}" for b in out["budgets"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
