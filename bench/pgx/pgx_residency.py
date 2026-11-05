"""PGX residency instrument: VRAM hit rate and page-ins per token for the paged baselines, from what each engine exposes.

SGLang Paged Experts keeps no counters (D section 4), so its residency is replayed exactly from routing:
- an UNTIMED instrument pass, run with the same prompts and token ids as the timed pass, dumps every MoE layer's
  ``topk_ids`` per forward call (``pgx_sglang_hook.py``), one JSON line per call;
- :func:`sglang_residency` replays those calls, per layer and in order, through ``replay.SeededLRU`` (SGLang's device
  rule when a call's routed ENTRIES fit K, its host wave plan otherwise; verified request by request against
  PE/residency.py and paged_experts.cuh at 297e54fd by the per-layer differential, e4b#2168).
The pager counts hits and misses per tier on the device, including graph replays, but logs them only on eager calls.
With ``VLLM_EXPERT_PAGER_LOG_INTERVAL=1`` and one eager probe request at each pass boundary, :func:`pager_pass_counts`
reads the per-layer cumulative counters from the log and differences them.

Pure stdlib plus ``replay.py``; no torch, no GPU. Self-tested by ``tests/test_pgx_residency.py``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPLAY = os.path.join(_HERE, "..", "train-routing-trace", "replay.py")


def _replay_module():
    mod = sys.modules.get("pgx_replay")
    if mod is None:
        spec = importlib.util.spec_from_file_location("pgx_replay", _REPLAY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pgx_replay"] = mod
        spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the routing dump ---------------------------------
# One JSON object per MoE forward call, in call order:
#   {"pass": "<pass id>", "call": <int, per process>, "layer": <int>, "tokens": <int>, "ids": [[e, ...], ...],
#    "decode": <bool>, "host_loads": <int or null>}
# ids is topk_ids as nested lists (tokens x top_k). decode marks a call whose rows are all decode tokens (from the
# engine's forward mode). host_loads is SGLang's own LRUPolicy loads for the call when the host path ran it (graphs
# off), the dump's self-check; null when the call ran on the device path.

def read_dump(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for key in ("pass", "call", "layer", "ids"):
                if key not in r:
                    raise ValueError(f"{path}:{n}: dump row lacks {key!r}")
            if not isinstance(r["ids"], list) or any(not isinstance(t, list) for t in r["ids"]):
                raise ValueError(f"{path}:{n}: ids must be tokens x top_k lists")
            rows.append(r)
    return rows


def _entries(ids) -> list[int]:
    """The call's routed entries, flattened in token-major order as topk_ids.flatten() hands them to the engine,
    padding ids (< 0) dropped as PE's executor drops them (executor.py:64)."""
    return [int(e) for tok in ids for e in tok if int(e) >= 0]


def sglang_residency(rows, k: int, scored_pass: str, decode_only: bool = True) -> dict:
    """Replay every call of every pass, in order, through one SeededLRU of K slots per layer (warm state carries
    across passes, as it does in the server), and tally ``scored_pass``.

    Hits and misses count DISTINCT experts per call (the engine pages a distinct expert once per call); page-ins are
    the slot loads the rule issues. ``decode_only`` restricts the tally to decode calls (prefill chunks still update
    the state). Per token = per decode token of the scored pass."""
    rp = _replay_module()
    if not rows:
        raise ValueError("empty routing dump")
    first_layer = min(int(r["layer"]) for r in rows)
    tiers = {}
    c = defaultdict(int)
    by_layer = defaultdict(lambda: defaultdict(int))
    for r in rows:
        lay = int(r["layer"])
        t = tiers.get(lay)
        if t is None:
            t = tiers[lay] = rp.SeededLRU(k)
        h, m, p = t.request(_entries(r["ids"]))
        if r["pass"] != scored_pass or (decode_only and not r.get("decode", True)):
            continue
        c["calls"] += 1
        c["hits"] += h
        c["misses"] += m
        c["page_ins"] += p
        by_layer[lay]["hits"] += h
        by_layer[lay]["misses"] += m
        by_layer[lay]["page_ins"] += p
        if lay == first_layer:
            c["tokens"] += len(r["ids"])
    return _summary(c, by_layer)


def host_path_check(rows, k: int) -> dict:
    """The dump's self-check on a graphs-off instrument pass: every call ran SGLang's host path, so its own
    LRUPolicy loads (``host_loads``) must equal plan_waves replayed from the dumped ids. Returns the mismatch count
    and the first mismatch. Calls without host_loads are skipped and counted."""
    pol = {}
    bad, first, skipped, checked = 0, None, 0, 0
    for r in rows:
        if r.get("host_loads") is None:
            skipped += 1
            continue
        lay = int(r["layer"])
        t = pol.get(lay)
        if t is None:
            t = pol[lay] = _HostLRU(k)
        loads = t.request(sorted(set(_entries(r["ids"]))))
        checked += 1
        if loads != int(r["host_loads"]):
            bad += 1
            first = first or {"pass": r["pass"], "call": r["call"], "layer": lay, "replay": loads,
                              "engine": int(r["host_loads"])}
    return {"checked": checked, "mismatched": bad, "first": first, "skipped_device_calls": skipped}


class _HostLRU:
    """SGLang's host path alone (PE/residency.py: LRUPolicy seeded 0..K-1, plan_waves), the path every call takes
    with graphs off. Kept separate from SeededLRU, whose device rule breaks ties differently (finding 2 of the
    per-layer replay note)."""

    def __init__(self, k: int):
        from collections import OrderedDict
        self.k = k
        self.resident = OrderedDict((e, e) for e in range(k))

    def request(self, distinct) -> int:
        ordered = [e for e in distinct if e in self.resident] + [e for e in distinct if e not in self.resident]
        loads = 0
        for wave in [ordered[i:i + self.k] for i in range(0, len(ordered), self.k)] or [[]]:
            for e in wave:
                if e in self.resident:
                    self.resident.move_to_end(e)
            for e in wave:
                if e not in self.resident:
                    _, slot = self.resident.popitem(last=False)
                    self.resident[e] = slot
                    loads += 1
        return loads


def _summary(c, by_layer) -> dict:
    routed = c["hits"] + c["misses"]
    tok = c["tokens"]
    return {"calls": c["calls"], "tokens": tok, "hits": c["hits"], "misses": c["misses"], "page_ins": c["page_ins"],
            "hit_rate": c["hits"] / routed if routed else None,
            "page_ins_per_token": c["page_ins"] / tok if tok else None,
            "misses_per_token": c["misses"] / tok if tok else None,
            "layers": {str(k): dict(v) for k, v in sorted(by_layer.items())}}


# ------------------------------------------------------------------ the pager's log ----------------------------------
# experts.py _expert_pager_log at e9264fda:
#   "vllm-expert-pager %s: vram hit %.1f%% (%d/%d), ram hit %.1f%% (%d/%d)"  -> layer_name, vh, vh+vm, rh, rh+rm
_PAGER_RE = re.compile(r"vllm-expert-pager (?P<layer>\S+): vram hit [0-9.]+% \((?P<vh>\d+)/(?P<vt>\d+)\), "
                       r"ram hit [0-9.]+% \((?P<rh>\d+)/(?P<rt>\d+)\)")
_GIVEUP_RE = re.compile(r"vllm-expert-pager ssd: .*?(?P<giveups>\d+) give-ups")


def pager_snapshots(text: str) -> list[dict]:
    """Every counter line in log order: {layer, vh, vm, rh, rm}. The counters are cumulative since the server
    started (they live on the device and include graph replays)."""
    out = []
    for m in _PAGER_RE.finditer(text):
        vh, vt, rh, rt = (int(m[g]) for g in ("vh", "vt", "rh", "rt"))
        out.append({"layer": m["layer"], "vh": vh, "vm": vt - vh, "rh": rh, "rm": rt - rh})
    return out


def pager_giveups(text: str) -> int:
    """The SSD tier's give-up count, latest line (0 when the SSD tier is off, as in stage 1). Any give-up flags the
    cell (gather.py:44-47)."""
    found = [int(m["giveups"]) for m in _GIVEUP_RE.finditer(text)]
    return found[-1] if found else 0


def pager_pass_counts(before: str, after: str) -> dict:
    """Per-pass VRAM-tier counts from two log excerpts: ``before`` ends with the probe that opened the pass, ``after``
    with the probe that closed it. Each layer's LAST snapshot in each excerpt is its cumulative value at that probe;
    the difference is the pass plus one probe's own lookups (a few per layer, reported as ``probe_included``).
    Fails closed: a layer present in one excerpt and not the other, or a counter that went DOWN, raises."""
    a = {s["layer"]: s for s in pager_snapshots(before)}
    b = {s["layer"]: s for s in pager_snapshots(after)}
    if set(a) != set(b) or not a:
        raise ValueError(f"pager layers differ across the pass ({len(a)} before, {len(b)} after)")
    tot = defaultdict(int)
    layers = {}
    for lay in sorted(a):
        d = {k: b[lay][k] - a[lay][k] for k in ("vh", "vm", "rh", "rm")}
        if min(d.values()) < 0:
            raise ValueError(f"pager counter went down on {lay}: {d} (a restarted server is a new pass)")
        layers[lay] = d
        for k, v in d.items():
            tot[k] += v
    routed = tot["vh"] + tot["vm"]
    return {"vram_hits": tot["vh"], "vram_misses": tot["vm"], "ram_hits": tot["rh"], "ram_misses": tot["rm"],
            "hit_rate": tot["vh"] / routed if routed else None, "layers": layers, "probe_included": True}
