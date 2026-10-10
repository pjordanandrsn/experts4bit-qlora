# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Speculative greedy decode for the ONE request decoding alone (lane SD2, ``bench/sd2/PREREG-sd2.md`` B1, B2, B5).

:class:`SpecDecoder` runs the cycle for a request that :class:`~.paged_runner.PagedModelRunner` finds decoding alone.
The target's side is injected (``verify``): on the server, a decode bucket whose rows share the request's slot
through the pool's alias slots. A test can inject a toy target instead.

**One step,** for a request whose last emitted token ``x[base]`` sits at position ``base`` in neither KV:
1. ``k_eff = min(k, budget - 1, capacity - 1 - base)``: a verify never runs past the request's length or its slot's last
   position (the maintainer's aliasing condition 3). Below 1 the request drops to T == 1 for the rest of its life;
2. ``verify(slot, n, ids, pos)`` runs rows ``[x[base], d_1 .. d_k_eff]`` at positions ``base .. base + k_eff`` and
   returns the target's argmax per row. The auxiliary states land in ``aux.dec``;
3. on the device: ``a`` = the leading matches ``d_{i+1} == g_i`` (``engines/speculative.py``'s strict rule); the slot's
   length becomes ``base + a + 1``, the step's last length write (B2); the draft extends over every verified row and
   chains the next drafts from row ``a``;
4. ONE host read: ``a`` and the rows' argmax. The tokens emitted are ``g_0 .. g_a``.

A request that is not alone, or is freed, drops its state and decodes at T == 1 for the rest of its life (B5). The
draft would lack the auxiliary states of the positions it did not see.
"""
from __future__ import annotations

import torch


class SpecDecoder:
    """``drafter``: an :class:`~.eagle3_draft.Eagle3Drafter` (or anything with its ``prefill`` / ``extend_and_draft``);
    ``aux``: an :class:`~.eagle3_draft.AuxStates`; ``kv``: the pool (``set_len_device``, ``note_len``);
    ``capacity``: positions a slot can hold; ``verify(slot, n, ids, pos) -> argmax [n]``, set by the runner."""

    def __init__(self, *, drafter, aux, kv, k: int, capacity: int, device="cuda", verify=None):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.drafter, self.aux, self.kv = drafter, aux, kv
        self.k, self.capacity = int(k), int(capacity)
        self.device = torch.device(device)
        self.verify = verify
        self.state: dict[int, dict] = {}                 # rid -> {"slot", "base"}: at most one, the request alone
        self.drafts = torch.zeros(self.k, dtype=torch.long, device=self.device)
        self.last = torch.zeros(1, dtype=torch.long, device=self.device)
        # static inputs and outputs of the post-verify step (a CUDA graph per row count, captured by the runner)
        self.tok_in = torch.zeros(self.k + 1, dtype=torch.long, device=self.device)
        self.pos_in = torch.zeros(self.k + 1, dtype=torch.long, device=self.device)
        self.a_out = torch.zeros(1, dtype=torch.long, device=self.device)
        self.len_out = torch.zeros(1, dtype=torch.long, device=self.device)
        self._post_graphs: dict[int, object] = {}
        self.stats = {"prefills": 0, "steps": 0, "drafted": 0, "accepted": 0, "emitted": 0, "short_steps": 0,
                      "dropped_batched": 0, "dropped_end": 0, "post_replays": 0, "post_eager": 0}

    # ---- the request's life -------------------------------------------------------------------------------------
    def eligible(self, rid: int) -> bool:
        return rid in self.state

    def start(self, rid: int, slot: int, tokens) -> None:
        """A prompt completed while its request was the only one bound: ``tokens`` is the prompt plus the first
        generated token, ``aux.pre[:P]`` holds the prompt's states. Writes the draft's context and its first drafts."""
        P = len(tokens) - 1
        if P < 1:
            raise ValueError("a speculative start needs a prompt of at least one token")
        self.state = {rid: {"slot": int(slot), "base": P}}
        tok = torch.tensor(list(tokens[1:]), dtype=torch.long, device=self.device)
        self.drafts.copy_(self.drafter.prefill(tok, self.aux.pre[:P]))
        self.last.fill_(int(tokens[-1]))
        self.stats["prefills"] += 1

    def drop(self, rid: int, why: str) -> None:
        if self.state.pop(rid, None) is not None and why in ("batched", "end"):
            self.stats[f"dropped_{why}"] += 1

    # ---- one step -----------------------------------------------------------------------------------------------
    def k_eff(self, base: int, budget: int) -> int:
        return min(self.k, int(budget) - 1, self.capacity - 1 - int(base))

    def step(self, rid: int, budget: int):
        """One speculative step for ``rid``: the tokens it emits, or None when it must decode at T == 1 (its state is
        dropped). ``budget`` is how many tokens the request may still emit."""
        st = self.state[rid]
        base, slot = st["base"], st["slot"]
        ke = self.k_eff(base, budget)
        if ke < 1:
            self.drop(rid, "end")
            return None
        n = ke + 1
        ids = torch.cat([self.last, self.drafts[:ke]])
        torch.arange(base, base + n, out=self.pos_in[:n])
        tok = self.verify(slot, n, ids, self.pos_in[:n])
        self.tok_in[:n].copy_(tok)
        g = self._post_graphs.get(n)
        if g is not None:
            g.replay()
            self.stats["post_replays"] += 1
        else:
            self._post(n)
            self.stats["post_eager"] += 1
        self.kv.set_len_device(slot, self.len_out)       # the step's last length write (B2)
        host = torch.cat([self.a_out, self.tok_in[:n]]).tolist()   # the step's one host read
        a = int(host[0])
        emitted = host[1:a + 2]
        st["base"] = base + a + 1
        self.kv.note_len(slot, st["base"])
        s = self.stats
        s["steps"] += 1
        s["drafted"] += ke
        s["accepted"] += a
        s["emitted"] += a + 1
        s["short_steps"] += int(ke < self.k)
        return emitted

    def _post(self, n: int) -> None:
        """On the device, from ``tok_in[:n]`` and ``pos_in[:n]``: the accept, the new length, the last token and the
        next drafts. Every input and output is a static buffer, so the runner can capture it per row count."""
        ke = n - 1
        tok = self.tok_in[:n]
        match = (tok[:ke] == self.drafts[:ke]).long()
        a = match.cumprod(0).sum().reshape(1)
        self.a_out.copy_(a)
        self.len_out.copy_(self.pos_in[:1] + a + 1)
        self.last.copy_(tok.index_select(0, a))
        self.drafts.copy_(self.drafter.extend_and_draft(tok, self.aux.dec[:n], self.pos_in[0], a[0]))

    def capture_post(self, n: int, warmup: int = 2) -> None:
        """Capture :meth:`_post` for ``n`` verified rows as a CUDA graph (on scratch state: the inputs are whatever the
        buffers hold, and the draft cache's entries it writes are overwritten by the next real step)."""
        if self.device.type != "cuda":
            return
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(max(1, warmup)):
                self._post(n)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._post(n)
        torch.cuda.synchronize(self.device)
        self._post_graphs[n] = g

    def census(self) -> dict:
        s = dict(self.stats)
        s["k"] = self.k
        s["tau_live"] = (s["emitted"] / s["steps"]) if s["steps"] else None
        s["acceptance"] = (s["accepted"] / s["drafted"]) if s["drafted"] else None
        s["post_graphs"] = sorted(self._post_graphs)
        return s
