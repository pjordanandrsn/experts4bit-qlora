#!/usr/bin/env python3
"""sd2_box.py -- lane SD2 (e4b#1313), the box side of the CUDA proof `sd2-prove-N` (bench/sd2/PREREG-sd2.md, Amendment 1).

**`--prove`** builds the engine as the shipped server builds it (`PagedServeConfig.from_env()` + `build_engine`), with
`E4B_PAGED_SPEC=eagle3`, `E4B_PAGED_SPEC_K=3` and the pinned head, graphs on. The build under test is the stacked
integration commit. Then, in this order, it writes `prove.json`:
- **census:** every decode and verify bucket (1, 2, 3, 4, 8, 16) and every post-verify graph (2, 3, 4) captured; the
  head's three hooks installed;
- **capture, bitwise:** each verify bucket's replay equals its padded eager step, logits bit for bit, at one position;
- **addressing (V0):** S2-lite's construction at 16 positions on R row 0 and on C-think row 0, for k = 1, 2, 3:
  - k + 1 + 4 plain T == 1 steps (the oracle), a rewind, one verify fed the oracle's tokens as drafts;
  - the 4 T == 1 steps after it.
  The rows' argmax agreement with the oracle, the continuations' agreement, and each verify row's mean |Δ log p| of the
  oracle's token are recorded;
- **the mutants:** the addressing check again under each of (a) the stagger one low (rows 1..k read one short),
  (b) the alias append one position low (rows 1..k write one low, read right) and (c) the verify rows' RoPE positions
  +1. Each must fail the 0.90 bar;
- **the draft:** the in-engine drafter's chains against `sd1_eagle3.Eagle3Draft.chain_at` on the same captured states, at
  8 truncation points of each of 16 R and 16 C-think prompts (768 ids);
- **the transition:** R row 1 speculates alone; R row 2 is admitted mid-flight. Row 1's draft is dropped, the census
  counts it, both finish at their lengths, and after every plain step the device lengths equal the host mirror.

Nothing here is timed: the proof is correctness only, and its numbers are never quoted as speed.

**`--self-test`** checks the agreement and Δ log p arithmetic and the prompt digests on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

K_SERVER = 3                      # the proof's server: verify buckets 2, 3, 4 and post graphs for each
POSITIONS = 16                    # addressing positions per gate row and k (PREREG V0)
MUTANT_POSITIONS = 8
CONT = 4                          # T == 1 steps after each verify
TRUNC_POINTS = 8                  # draft-gate truncation points per prompt
BAR_ADDRESS, BAR_DRAFT = 0.90, 0.99
BUCKETS_WANTED = (1, 2, 3, 4, 8, 16)


def digest(obj) -> str:
    """P109's digest (compact JSON, no key sorting), so its prompts files verify here."""
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def load_rows(path):
    pf = json.load(open(path))
    if digest(pf["rows"]) != pf["prompts_sha256"]:
        raise SystemExit(f"REFUSED: {path} does not match its own digest")
    return [list(map(int, r)) for r in pf["rows"]]


def agreement(a, b) -> float:
    pairs = list(zip(a, b))
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else 0.0


def dlogp(logits_a, logits_b, toks):
    """Per row: |log p_a(tok) - log p_b(tok)|, fp32 log-softmax."""
    import torch
    la = torch.log_softmax(logits_a.float(), -1)
    lb = torch.log_softmax(logits_b.float(), -1)
    idx = torch.as_tensor(toks, device=la.device).view(-1, 1)
    return (la.gather(1, idx) - lb.gather(1, idx)).abs().view(-1).tolist()


class LogitsTap:
    """A forward hook on the runner's model, registered before the decode graphs are captured: for a decode-shaped
    forward ([b, 1, V] logits) it keeps a reference to logits[:, -1], one per (graph|eager, b). A captured replay
    rewrites its graph's tensor in place, so the reference reads that replay's logits once the step has synchronized."""

    def __init__(self):
        self.refs = {"graph": {}, "eager": {}}

    def hook(self, _mod, _args, out):
        import torch
        lg = getattr(out, "logits", None)
        if lg is not None and lg.dim() == 3 and lg.shape[1] == 1:
            capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            kind = "graph" if capturing else "eager"
            self.refs[kind][int(lg.shape[0])] = lg[:, -1]

    def get(self, b, graphed=True):
        return self.refs["graph" if graphed else "eager"][b]


def install_tap(pr, tap):
    """Wrap enable_decode_graphs so the hook is on the model before any capture."""
    real = pr.PagedModelRunner.enable_decode_graphs

    def wrapped(self, *a, **kw):
        self.model.register_forward_hook(tap.hook)
        return real(self, *a, **kw)
    pr.PagedModelRunner.enable_decode_graphs = wrapped


# ------------------------------------------------------------------------------------------------- the gates --

class Gate:
    def __init__(self, parts, tap):
        import torch
        self.torch = torch
        self.parts, self.tap = parts, tap
        self.runner = parts.runner
        self.kv = self.runner.kv
        self.spec = self.runner.spec
        self.rid = 10_000

    def begin(self, prompt):
        """Bind and prefill one request alone; returns (rid, slot). The prompt completes alone, so the draft starts."""
        r, cfg_chunk = self.runner, self.parts.scheduler.chunk_tokens   # the server's own chunking
        self.rid += 1
        rid, slot = self.rid, 0
        r.bind(rid, slot, list(prompt))
        start = 0
        while start < len(prompt):
            take = min(cfg_chunk, len(prompt) - start)
            r.run_prefill([(rid, start, take)])
            start += take
        return rid, slot

    def end(self, rid):
        self.runner.free_slot(rid)

    def plain(self, rid):
        """One T == 1 step through the plain bucket (bypassing the speculative dispatch): (token, logits row)."""
        got = self.runner._run_decode_bucketed([rid])[rid]
        return got, self.tap.get(1).clone()

    def rewind(self, rid, slot, base):
        r = self.runner
        self.kv.rewind(slot, base)
        r.tokens[rid] = r.tokens[rid][:base + 1]
        r.pos_of[rid] = base + 1

    def verify(self, rid, slot, base, drafts, graphed=True):
        """One verify of len(drafts) + 1 rows fed ``drafts``; returns (argmax rows, logits rows). Leaves the slot's
        device length at base + 1 (the publish), as a real step does before its rollback."""
        torch, r = self.torch, self.runner
        n = len(drafts) + 1
        ids = torch.tensor([r.tokens[rid][base]] + list(drafts), dtype=torch.long, device="cuda")
        pos = torch.arange(base, base + n, device="cuda")
        saved = r._graphs[n]
        if not graphed:
            r._graphs[n] = None
        try:
            tok = self.spec.verify(slot, n, ids, pos)
        finally:
            r._graphs[n] = saved
        torch.cuda.synchronize()
        return tok.tolist(), self.tap.get(n, graphed).clone()

    def settle(self, rid, slot, base, drafts, g_last):
        """After a verify that accepted every draft: the slot holds rows base .. base + n - 1, the next input is the
        verify's last argmax."""
        torch, r = self.torch, self.runner
        n = len(drafts) + 1
        self.kv.set_len_device(slot, torch.tensor(base + n, device="cuda"))
        self.kv.note_len(slot, base + n)
        r.tokens[rid] = r.tokens[rid][:base + 1] + list(drafts) + [int(g_last)]
        r.pos_of[rid] = base + n + 1

    def addressing(self, prompt, k, positions):
        """S2-lite's construction at ``positions`` positions: (row agreement, continuation agreement, mean |Δ log p|
        per verify row index)."""
        rid, slot = self.begin(prompt)
        rows_a, rows_b, cont_a, cont_b = [], [], [], []
        dl = [[] for _ in range(k + 1)]
        try:
            for _ in range(positions):
                base = self.runner.pos_of[rid] - 1
                oracle, ologits = [], []
                for _ in range(k + 1 + CONT):
                    t, lg = self.plain(rid)
                    oracle.append(t)
                    ologits.append(lg)
                self.rewind(rid, slot, base)
                g, vlogits = self.verify(rid, slot, base, oracle[:k])
                rows_a += g
                rows_b += oracle[:k + 1]
                d = dlogp(vlogits, self.torch.stack([x[0] for x in ologits[:k + 1]]), oracle[:k + 1])
                for i, v in enumerate(d):
                    dl[i].append(v)
                self.settle(rid, slot, base, oracle[:k], g[k])
                for j in range(CONT):
                    t, _ = self.plain(rid)
                    cont_a.append(t)
                    cont_b.append(oracle[k + 1 + j])
        finally:
            self.end(rid)
        return {"rows_agree": agreement(rows_a, rows_b), "rows": len(rows_a),
                "cont_agree": agreement(cont_a, cont_b), "cont": len(cont_a),
                "mean_abs_dlogp": [round(sum(v) / len(v), 6) if v else None for v in dl]}

    def capture_bitwise(self, prompt):
        """Each verify bucket's replay against its padded eager step at the same position, logits bit for bit."""
        out = {}
        rid, slot = self.begin(prompt)
        try:
            for _ in range(4):
                self.plain(rid)
            base = self.runner.pos_of[rid] - 1
            drafts = [self.runner.tokens[rid][base]] * K_SERVER
            for n in range(2, K_SERVER + 2):
                g1, l1 = self.verify(rid, slot, base, drafts[:n - 1], graphed=True)
                self.rewind(rid, slot, base)
                g2, l2 = self.verify(rid, slot, base, drafts[:n - 1], graphed=False)
                self.rewind(rid, slot, base)
                out[str(n)] = bool(self.torch.equal(l1, l2)) and g1 == g2
        finally:
            self.end(rid)
        return out

    def draft_gate(self, prompts, ref):
        """In-engine drafts against SD1's chain_at at TRUNC_POINTS truncations of each prompt, on the states the served
        prefill captured."""
        torch = self.torch
        same = total = 0
        for prompt in prompts:
            rid, slot = self.begin(prompt)
            try:
                P = len(prompt)
                aux = self.spec.aux.pre[:P].clone()
                first = self.runner.tokens[rid][P]
                toks = list(prompt) + [first]
                pts = sorted({max(2, (P * (j + 1)) // TRUNC_POINTS) for j in range(TRUNC_POINTS)})
                for Pj in pts:
                    got = self.spec.drafter.prefill(torch.tensor(toks[1:Pj + 1], device="cuda"), aux[:Pj]).tolist()
                    want = ref.chain_at(torch.tensor(toks[:Pj + 1]), aux[:Pj], Pj - 1, K=K_SERVER)
                    same += sum(a == b for a, b in zip(got, want))
                    total += K_SERVER
            finally:
                self.end(rid)
        return {"agree": same / total if total else 0.0, "same": same, "of": total}

    def transition(self, row_a, row_b, n_a=64, n_b=16):
        """Row A speculates alone; row B is admitted mid-flight; both finish; lengths equal the mirror after every
        plain step; the census counts one drop."""
        from experts4bit_qlora.engines import paged_runner as pr
        sched, r, kv = self.parts.scheduler, self.runner, self.kv
        before = dict(self.spec.census())
        mismatches = []
        real = pr.PagedModelRunner._run_decode_bucketed

        def checked(this, rids):
            got = real(this, rids)
            self.torch.cuda.synchronize()
            for q in rids:
                s_ = this.slot_of[q]
                for layer in range(kv.L):
                    if int(kv.seq_lens[layer, s_]) != kv._seen[layer][s_]:
                        mismatches.append((q, layer))
            return got
        pr.PagedModelRunner._run_decode_bucketed = checked
        try:
            a = sched.add_request(list(row_a), max_new_tokens=n_a)
            for _ in range(200):
                sched.step()
                if self.spec.census()["steps"] - before["steps"] >= 4:
                    break
            spec_steps_alone = self.spec.census()["steps"] - before["steps"]
            b = sched.add_request(list(row_b), max_new_tokens=n_b)
            sched.run_until_idle()
        finally:
            pr.PagedModelRunner._run_decode_bucketed = real
        done = {q.rid: len(q.out) for q in sched.done}
        after = self.spec.census()
        return {"spec_steps_alone": spec_steps_alone, "dropped_batched": after["dropped_batched"] - before["dropped_batched"],
                "len_a": done.get(a), "len_b": done.get(b), "want": [n_a, n_b], "mirror_mismatches": len(mismatches),
                "state_left": len(self.spec.state)}


# ----------------------------------------------------------------------------------------------- the mutants --

def mutant(name, gate):
    """Monkeypatches for the three addressing mutants; returns the undo."""
    kv, spec = gate.kv, gate.spec
    if name == "a":                                   # the stagger one low: rows 1..k read one short
        real = kv.graph_bucket_load

        def load(st, slots, staging=None):
            real(st, slots, staging)
            if len(slots) > 1 and set(slots[1:]) <= set(kv.alias) and "lens" in st:
                st["lens"][:, 1:] -= 1
        kv.graph_bucket_load = load
        return lambda: setattr(kv, "graph_bucket_load", real)
    if name == "b":                                   # the alias append one low; reads restored to right
        real_bind, real_load = kv.alias_bind, kv.graph_bucket_load

        def bind(slot, n):
            got = real_bind(slot, n)
            if got:
                kv.seq_lens[:, kv._alias_idx[:n]] -= 1
            return got

        def load(st, slots, staging=None):
            real_load(st, slots, staging)
            if len(slots) > 1 and set(slots[1:]) <= set(kv.alias) and "lens" in st:
                st["lens"][:, 1:] += 1
        kv.alias_bind, kv.graph_bucket_load = bind, load

        def undo():
            kv.alias_bind, kv.graph_bucket_load = real_bind, real_load
        return undo
    if name == "c":                                   # the verify rows' RoPE positions +1
        real_verify = spec.verify

        def verify(slot, n, ids, pos):
            return real_verify(slot, n, ids, pos + 1)
        spec.verify = verify
        return lambda: setattr(spec, "verify", real_verify)
    raise ValueError(name)


def prove_main(a) -> int:
    import torch
    from safetensors.torch import load_file

    import sd1_eagle3 as ref_mod
    from experts4bit_qlora.engines import paged_runner as pr
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    R = load_rows(a.prompts_r)
    C = load_rows(a.prompts_c)
    cfg = PagedServeConfig.from_env()
    if cfg.spec != "eagle3" or cfg.spec_k != K_SERVER or not cfg.graphs:
        raise SystemExit(f"REFUSED: the proof needs E4B_PAGED_SPEC=eagle3, K={K_SERVER} and graphs (got {cfg.spec}, "
                         f"{cfg.spec_k}, graphs={cfg.graphs})")
    tap = LogitsTap()
    install_tap(pr, tap)
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    rec = {"load_s": round(time.perf_counter() - t0, 2), "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "config": {k: (list(v) if isinstance(v, tuple) else v)
                                                              for k, v in vars(cfg).items() if k != "token"}}
    runner = parts.runner
    gs = parts.info.get("graph_status") or {}
    rec["census"] = {"graph_status": {str(b): gs.get(b) for b in BUCKETS_WANTED},
                     "post_graphs": {str(n): v for n, v in (getattr(runner, "spec_graph_status", None) or {}).items()},
                     "spec_build": parts.info.get("spec")}
    gate = Gate(parts, tap)
    rec["capture_bitwise"] = gate.capture_bitwise(R[0])
    print("SD2_PROVE capture_bitwise " + json.dumps(rec["capture_bitwise"]), flush=True)
    rec["addressing"] = {}
    for name, row in (("R0", R[0]), ("C0", C[0])):
        for k in (1, 2, 3):
            res = gate.addressing(row, k, POSITIONS)
            rec["addressing"][f"{name}_k{k}"] = res
            print(f"SD2_PROVE addressing {name} k={k} " + json.dumps(res), flush=True)
    rec["mutants"] = {}
    for m in ("a", "b", "c"):
        undo = mutant(m, gate)
        try:
            res = gate.addressing(R[0], 3, MUTANT_POSITIONS)
        finally:
            undo()
        rec["mutants"][m] = res
        print(f"SD2_PROVE mutant {m} " + json.dumps(res), flush=True)
    ref = ref_mod.Eagle3Draft(load_file(os.path.join(a.head_dir, "model.safetensors")), device="cuda")
    rec["draft"] = gate.draft_gate(R + C, ref)
    print("SD2_PROVE draft " + json.dumps(rec["draft"]), flush=True)
    rec["transition"] = gate.transition(R[1], R[2])
    print("SD2_PROVE transition " + json.dumps(rec["transition"]), flush=True)
    rec["census"]["spec"] = runner.spec.census()
    rec["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    json.dump(rec, open(a.out, "w"), indent=1)
    print("SD2_PROVE done", flush=True)
    return 0


def self_test() -> int:
    import torch
    bad = []
    if agreement([1, 2, 3], [1, 2, 4]) != 2 / 3 or agreement([], []) != 0.0:
        bad.append("agreement")
    a = torch.tensor([[0.0, 1.0, 2.0]])
    b = torch.tensor([[0.0, 1.0, 3.0]])
    d = dlogp(a, b, [2])[0]
    want = abs(float(torch.log_softmax(a, -1)[0, 2] - torch.log_softmax(b, -1)[0, 2]))
    if abs(d - want) > 1e-6 or dlogp(a, a, [1])[0] != 0.0:
        bad.append("dlogp")
    rows = [[1, 2], [3]]
    if digest(rows) != hashlib.sha256(b"[[1,2],[3]]").hexdigest():
        bad.append("digest")
    tap = LogitsTap()
    out = type("O", (), {"logits": torch.zeros(3, 1, 5)})()
    tap.hook(None, None, out)
    if tuple(tap.get(3, graphed=False).shape) != (3, 5):
        bad.append("tap")
    if bad:
        print("sd2_box self-test FAILED:", bad)
        return 1
    print("sd2_box self-test OK (agreement, dlogp, digest, logits tap)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prove", action="store_true")
    ap.add_argument("--prompts-r")
    ap.add_argument("--prompts-c")
    ap.add_argument("--head-dir")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prove:
        return prove_main(a)
    ap.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
