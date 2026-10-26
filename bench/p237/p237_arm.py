#!/usr/bin/env python3
"""p237_arm.py -- P237's TC1_ARM_HOOK module (``TC1_ARM_HOOK=p237_arm:make`` with ``TC1_ARM_HOOK_SHA256`` its staged sha256;
tc1_arm.py's load_arm_hook calls the factory with no arguments and refuses any other bytes).

OBSERVE ONLY. Every wrapper it installs calls the wrapped function with the same arguments and returns its own output, unchanged
(tests/test_p237.py shows it per wrapper), and nothing here sets a knob, a dispatch, a cap or an offload: those come only from
tc1_arm's registered options (--vram-cap-gib, --unsloth-offload-layers, --layer-major). No phase launches device work: the reads
are Python counters, tensor ``numel()`` / ``data_ptr()`` and attributes, never a kernel, a copy or a synchronize.

The phases tc1_arm calls:
- ``post_import(args)``: Unsloth arms install ``UnslothMoeCounter`` (bench/common/unsloth_moe_counter.py, e4b#2132, imported
  unchanged) and wrap unsloth_zoo's ``block_swap._Block.prefetch`` / ``evict`` (class attributes, looked up per call, so the blocks
  the load builds are counted); e4b arms wrap ``nf4_route.train_gemm_route`` (gnf4's route, read at call time), the offload's
  ``_OffloadStats.record_copy`` (one call per real host-to-device staging, ``E4B_OFFLOAD_STATS=1``; ``stage()`` returns before it
  when the layer is already staged) and, under ``--layer-major 1``, ``layer_major.layer_major_step`` (tc1_arm's
  ``layer_major_engine`` imports it at call time).
- ``post_load(args, model)``: after the load and every harness setup, before eval0. Unsloth arms hand the loaded ``*MoeExperts``
  modules to ``patch_loaded`` and record the block swap (indices, blocks, depth, host and pinned bytes); e4b arms record the
  offload handles. Both keep the model, to read ``model.training`` (eval_loss switches it, tc1_arm.py:1067/1073) per counted event.
- ``step_edge(i)``: the edge before optimizer step i, and once after the last (i = N). From here until the next edge, every
  counted event belongs to interval i. Before edge 0 the interval is ``pre``: the load, eval0 and, under --layer-major, the drift
  guard (its stock forwards and its first layer-major call run in train mode BEFORE edge 0, tc1_arm.py:4506-4524). Counts are
  kept PER INTERVAL, never a total minus something. A snapshot of the swap / staging state is taken at each edge.
- ``record()``: all of it, for the receipt's ``arm_hook.record``.

What the record does NOT decide: engagement, the bars and VOID are this lane's reducer's (p237_reduce.py), against the PREREG.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = "pre"                                          # the interval before edge 0


def find_block_swap(model):
    """Unsloth 2026.10.8's ``model._unsloth_block_swap`` (unsloth/models/_utils.py:9101, :9776), on the model or under a PEFT wrapper's
    ``base_model.model``: the same lookup as tc1_arm.py's ``unsloth_offload_resolved``. None when absent."""
    bs = getattr(model, "_unsloth_block_swap", None)
    if bs is None:
        inner = getattr(model, "base_model", None)
        bs = getattr(getattr(inner, "model", None), "_unsloth_block_swap", None) if inner is not None else None
    return bs


class P237Arm:
    def __init__(self):
        self.fw, self.micro_batches, self.layer_major = None, 1, 0   # read from tc1_arm's args at post_import
        self.model = None
        self.counter = None
        self.edge = PRE                      # the current interval's key: "pre", then str(i) from step_edge(i)
        self.edges = []                      # one snapshot per step_edge call
        self.facts = {}
        self.route_log = {}                  # interval -> {"n_groups:route": calls}
        self.stage = {}                      # interval -> {"mode|policy": [stages, bytes, copies]}   (e4b)
        self.lm_calls = {}                   # interval -> layer_major_step calls                    (e4b)
        self.fetch = {}                      # interval -> {"mode": {block_id: [fetches, bytes]}}     (Unsloth)
        self.evict = {}                      # interval -> {"mode": {block_id: evictions}}           (Unsloth)
        self._block_layer = {}               # id(block) -> layer index (post_load, from the swap's own indices)
        self._saved = []                     # (owner, attr, original): what uninstall restores

    # -- helpers --------------------------------------------------------------------------------------------------------------
    def _mode(self):
        """``train`` / ``eval`` by the kept model's own flag; ``load`` before post_load has handed the model over."""
        if self.model is None:
            return "load"
        return "train" if bool(getattr(self.model, "training", False)) else "eval"

    def _patch(self, owner, attr, wrapper, original):
        wrapper.__wrapped__ = original
        self._saved.append((owner, attr, original))
        setattr(owner, attr, wrapper)

    def uninstall(self):
        """Restore every wrapped attribute (the tests' processes run several hooks; tc1_arm's own process ends after one)."""
        for owner, attr, original in reversed(self._saved):
            setattr(owner, attr, original)
        self._saved = []
        if self.counter is not None:
            self.counter.uninstall()

    # -- post_import ----------------------------------------------------------------------------------------------------------
    def post_import(self, args):
        self.fw = getattr(args, "framework", None)
        self.micro_batches = int(getattr(args, "accum", 1) or 1)   # tc1_arm: one optimizer step = accum micro-batches
        self.layer_major = int(getattr(args, "layer_major", 0) or 0)
        self.facts["env"] = {k: os.environ.get(k) for k in ("E4B_OFFLOAD_STATS", "E4B_OFFLOAD_ARENA", "E4B_OFFLOAD_PIN_SLAB",
                                                             "UNSLOTH_DISABLE_PINNED_MEMORY")}
        if self.fw == "unsloth":
            sys.path.insert(0, os.path.join(HERE, "..", "common"))
            from unsloth_moe_counter import UnslothMoeCounter
            self.counter = UnslothMoeCounter().install()
            self._install_block_swap()
        elif self.fw == "e4b":
            self._install_route()
            self._install_staging()
            if self.layer_major:
                self._install_layer_major()

    def _install_block_swap(self):
        from unsloth_zoo import block_swap
        cls = block_swap._Block
        prefetch, evict, hook = cls.prefetch, cls.evict, self

        def counted_prefetch(blk, slot):
            was = blk.resident                # _Block.prefetch returns at once when resident (block_swap.py:214-216): not a fetch
            out = prefetch(blk, slot)
            if not was and blk.resident:
                row = hook.fetch.setdefault(hook.edge, {}).setdefault(hook._mode(), {}).setdefault(id(blk), [0, 0])
                row[0] += 1
                row[1] += int(blk.nbytes())   # the aligned per-device sizes one fetch copies (bufs[d].copy_(host_buf[d]))
            return out

        def counted_evict(blk):
            was = blk.resident                # evict returns at once when not resident (block_swap.py:204-206)
            out = evict(blk)
            if was and not blk.resident:
                m = hook.evict.setdefault(hook.edge, {}).setdefault(hook._mode(), {})
                m[id(blk)] = m.get(id(blk), 0) + 1
            return out

        self._patch(cls, "prefetch", counted_prefetch, prefetch)
        self._patch(cls, "evict", counted_evict, evict)

    def _install_route(self):
        import nf4_route
        fn, hook = nf4_route.train_gemm_route, self

        def train_gemm_route(dev=None, n_groups=None):
            out = fn(dev, n_groups)
            log = hook.route_log.setdefault(hook.edge, {})
            key = f"{n_groups}:{out}"
            log[key] = log.get(key, 0) + 1
            return out

        self._patch(nf4_route, "train_gemm_route", train_gemm_route, fn)

    def _install_staging(self):
        from experts4bit_qlora.engines import offload
        cls = offload._OffloadStats
        fn, hook = cls.record_copy, self

        def record_copy(stats, start, end, nbytes, ncopies, policy):
            out = fn(stats, start, end, nbytes, ncopies, policy)
            row = hook.stage.setdefault(hook.edge, {}).setdefault(f"{hook._mode()}|{policy}", [0, 0, 0])
            row[0] += 1
            row[1] += int(nbytes)
            row[2] += int(ncopies)
            return out

        self._patch(cls, "record_copy", record_copy, fn)

    def _install_layer_major(self):
        from experts4bit_qlora.engines import layer_major
        fn, hook = layer_major.layer_major_step, self

        def layer_major_step(*args, **kwargs):
            hook.lm_calls[hook.edge] = hook.lm_calls.get(hook.edge, 0) + 1
            return fn(*args, **kwargs)

        self._patch(layer_major, "layer_major_step", layer_major_step, fn)

    # -- post_load ------------------------------------------------------------------------------------------------------------
    def post_load(self, args, model):
        self.model = model
        cfg = getattr(model, "config", None)
        cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        self.facts["num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
        if self.fw == "unsloth":
            inner = [m for _, m in model.named_modules() if type(m).__name__.endswith("MoeExperts")]
            self.counter.patch_loaded(inner)
            self.facts["expert_modules"] = len(inner)
            bs = find_block_swap(model)
            if bs is None:
                self.facts["block_swap"] = None
            else:
                idx = [int(i) for i in (getattr(bs, "indices", None) or [])]
                blocks = list(getattr(bs, "blocks", None) or [])
                self._block_layer = {id(b): idx[k] for k, b in enumerate(blocks) if k < len(idx)}
                self.facts["block_swap"] = {"indices": idx, "blocks": len(blocks), "depth": int(getattr(bs, "depth", 0)),
                                            "host_bytes": int(bs.host_bytes()), "pinned_bytes": int(getattr(bs, "pinned_bytes", 0)),
                                            "block_bytes": [int(b.nbytes()) for b in blocks],
                                            "params_per_block": [len(b.params) for b in blocks]}
        elif self.fw == "e4b":
            from experts4bit_qlora.engines import offload
            self._handles = offload.offload_handles(model)
            self.facts["offload_handles"] = len(self._handles)
            self.facts["stats_enabled"] = bool(offload._stats_enabled())

    # -- step_edge ------------------------------------------------------------------------------------------------------------
    def _swap_state(self):
        """Host reads only: which blocks the swap holds resident, and whether every frozen parameter of each block agrees with it
        (an evicted block's parameters point at 0-element tensors, block_swap.py:207-208; a resident one's at its slot views)."""
        bs = find_block_swap(self.model) if self.model is not None else None
        if bs is None:
            return None
        resident, evicted_nonempty, resident_empty = [], [], []
        for b in list(getattr(bs, "blocks", None) or []):
            L = self._block_layer.get(id(b))
            sizes = [int(p.numel()) for p in b.params]
            if b.resident:
                resident.append(L)
                if any(s == 0 for s in sizes):
                    resident_empty.append(L)
            elif any(s != 0 for s in sizes):
                evicted_nonempty.append(L)
        return {"resident": resident, "evicted_nonempty": evicted_nonempty, "resident_empty": resident_empty}

    def _staging_state(self):
        from experts4bit_qlora.engines import offload
        st = offload._STATS
        res = offload._ExpertOffload._resident
        handles = getattr(self, "_handles", [])
        return {"copies_total": None if st is None else len(st.copies),
                "bytes_total": None if st is None else sum(int(c[2]) for c in st.copies),
                "cold_misses": None if st is None else int(st.cold_misses),
                "resident_handle": next((k for k, h in enumerate(handles) if h is res), None)}

    def step_edge(self, i):
        self.edge = str(int(i))
        if self.counter is not None:
            self.counter.mark_step(i)        # passes after this edge belong to step i (a bookkeeping index)
        snap = {"i": int(i), "mode": self._mode()}
        if self.fw == "unsloth":
            snap["swap"] = self._swap_state()
            snap["unsloth"] = self.counter.snapshot()
        elif self.fw == "e4b":
            snap["staging"] = self._staging_state()
        self.edges.append(snap)

    # -- record ---------------------------------------------------------------------------------------------------------------
    def record(self):
        n_steps = max(len(self.edges) - 1, 0)          # edges 0..N close N steps
        rec = {"p237_arm": 1, "framework": self.fw, "micro_batches": self.micro_batches, "layer_major": self.layer_major,
               "facts": self.facts, "edges": self.edges, "n_steps": n_steps}
        if self.fw == "unsloth":
            name = lambda bid: str(self._block_layer[bid]) if bid in self._block_layer else f"unmapped:{bid}"   # noqa: E731
            rec["fetch"] = {e: {m: {name(b): v for b, v in d.items()} for m, d in by.items()} for e, by in self.fetch.items()}
            rec["evict"] = {e: {m: {name(b): v for b, v in d.items()} for m, d in by.items()} for e, by in self.evict.items()}
            rec["structural"] = self.counter.structural(self.micro_batches, n_steps)
        elif self.fw == "e4b":
            rec["stage"] = self.stage
            rec["route_log"] = self.route_log
            if self.layer_major:
                rec["layer_major_calls"] = self.lm_calls
        return rec


def make():
    """The factory tc1_arm calls, with no arguments (TC1_ARM_HOOK=p237_arm:make)."""
    return P237Arm()


__all__ = ["P237Arm", "find_block_swap", "make"]
