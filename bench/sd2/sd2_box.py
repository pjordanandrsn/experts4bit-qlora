#!/usr/bin/env python3
"""sd2_box.py -- lane SD2 (e4b#1313), the box side of the CUDA proof `sd2-prove-N` (bench/sd2/PREREG-sd2.md, Amendment 1)
and of the read `sd2-5090-N` (Amendment 3).

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

Nothing in the proof is timed: it is correctness only, and its numbers are never quoted as speed.

**The read** (`sd2-5090-N`, Amendment 3), one process per stage, each the shipped server from the environment:
- **`--read-v`** (k = 3, graphs on): V0 on rows outside calibration (capture bitwise on R1; the logit gate on R1..R3 and
  C1..C3 at k = 1..3; mutants a/b/c on R1 and C1; the draft on R4..R15 and C4..C15; the transition R1 then R2). It is
  written and judged (`sd2_reduce.v0_fails`) before any timing; a failure stops the stage. Then V1 on R row 0's live
  slot: the T == 1 step's wall twice (an A/A that brackets the k terms) and, per k, its own drafter and spec decoder on
  the same weights: the verify and the post-verify graph by CUDA events, the whole step by wall; each the median of 64.
- **`--read-e ARM`** (OFF, ON1 or ON2; graphs on): each workload's 16 rows one request at a time, a warm pass at SHORT,
  then 3 rounds of SHORT and LONG; P109's slope; the speculative census and the verify buckets' replays per workload.
- **`--read-q`** (graphs off, spec off): P115 Phase B's `measure_phase` at its bytes (R, rep, chunk, mutant_scale; one
  window a pass), then ON1 and ON2 by `paged_verify_pass`: teacher-forced verify steps of k + 1 rows through the
  runner's own `_spec_verify`, scored against R's saved log-probs; then the reported w16 draw.

**`--self-test`** checks the agreement and Δ log p arithmetic, the prompt digests, the verify plan, and the verify
assembly against a T == 1 pass on a toy causal target (with its stagger and plan mutants), on CPU.
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
        sched, kv = self.parts.scheduler, self.kv
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


# ===================================================================================== the read (Amendment 3) ==

GATE_ROWS = (1, 2, 3)              # V0's gate rows, R and C-think, outside sd2-prove-2/3's calibration rows (R0, C0)
DRAFT_ROWS = tuple(range(4, 16))   # V0's draft-gate prompts, R and C-think
TIMED, WARM = 64, 8                # V1: steps timed per term, and warm-up steps before them
E_WORKLOADS = {"R": 160, "C-think": 256, "C-nothink": 256}   # LONG per workload (SD1's lengths)
SHORT, E_REPS = 32, 3
Q_BUCKETS = (1, 2, 4, 8, 16)       # P110's BUCKETS (a test pins them equal)
W16 = 16                           # Q's reported w16 draw: windows a pass


def median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])) if s else None


def verify_plan(P, C, k):
    """The verify steps of a teacher-forced pass scoring positions P .. P + C - 1 of a window. The prefill's last row
    scores P. Each step (base, use) feeds k + 1 tokens at positions base .. base + k and keeps its first ``use`` rows,
    which score base + 1 .. base + use; only the last step keeps fewer than k + 1. So every scored position after P
    comes from a verify of k + 1 rows."""
    plan, base, end = [], P, P + C - 1
    while base < end:
        plan.append((base, min(k + 1, end - base)))
        base += k + 1
    return plan


def step_ids(w, base, k):
    """The k + 1 tokens a step at ``base`` feeds: w[base .. base + k], the last repeated past the window's end (those
    rows are causal after every kept row, and are discarded)."""
    ids = [int(t) for t in w[base:base + k + 1]]
    return ids + [ids[-1]] * (k + 1 - len(ids))


def assemble_verify(first_lp, step, w, P, C, k):
    """``first_lp`` [V]: the prefill's log-probs for position P. ``step(ids, base) -> [k + 1, V]``: the log-probs for
    positions base + 1 .. base + k + 1, given ``ids`` fed at positions base .. base + k, every row accepted. Returns
    [C, V], row i scoring w[P + i]."""
    import torch
    rows = [first_lp.reshape(1, -1)]
    for base, use in verify_plan(P, C, k):
        rows.append(step(step_ids(w, base, k), base)[:use])
    out = torch.cat(rows, 0)
    if out.shape[0] != C:
        raise AssertionError(f"the verify pass scored {out.shape[0]} positions, expected {C}")
    return out


class _VerifyOnly:
    """What ``enable_speculation`` and the prefill need from a spec, for a runner that only verifies (Q's ON passes):
    k, an aux whose mode the prefill sets (no hooks are installed), and no draft state."""

    def __init__(self, k):
        import types
        self.k, self.aux, self.state = k, types.SimpleNamespace(mode=None), {}

    def start(self, rid, slot, tokens):
        pass

    def drop(self, rid, why):
        pass


def paged_verify_pass(model, w, P, C, k, device, chunk=512):
    """Q's ON pass for ONE window, P110's ``paged_pass`` construction (padded eager buckets, device grouping) with k
    alias slots: prefill w[:P], then the verify steps of :func:`verify_plan` through the runner's own ``_spec_verify``,
    each fed the true tokens, every row accepted (the length moves to base + k + 1). Returns fp32 log-probs [C, V]
    and the runner's graph stats. The caller runs ``p108_box._release`` after it, as ``measure_phase`` does."""
    import torch

    import p108_box
    from experts4bit_qlora.engines import hot_residency as hr
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry
    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    kv = Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=1, max_tokens_per_seq=P + C + 16,
                    device=device, scratch_slots=max(Q_BUCKETS), alias_slots=k)
    runner = PagedModelRunner(model, kv, device=device)
    runner.enable_speculation(_VerifyOnly(k))
    saved = (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0])
    last, inner = {}, model.forward

    def keep(*a, **kw):
        o = inner(*a, **kw)
        last["logits"] = o.logits
        return o
    try:
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False
        status = runner.enable_decode_graphs(Q_BUCKETS, capture=False, verbose=False)
        if k + 1 not in status:
            raise RuntimeError(f"no verify bucket of {k + 1} rows: {sorted(status)}")
        model.forward = keep
        with p108_box.Attention(False), torch.no_grad():
            runner.bind(0, 0, [int(t) for t in w[:P]])
            for s in range(0, P, chunk):
                runner.run_prefill([(0, s, min(chunk, P - s))])
            first = last["logits"][0, -1].float().log_softmax(-1)

            def step(ids, base):
                n = len(ids)
                runner._spec_verify(0, n, torch.tensor(ids, device=device), torch.arange(base, base + n, device=device))
                lp = last["logits"][:n, -1].float().log_softmax(-1)
                kv.set_len_device(0, torch.tensor(base + n, device=device))
                kv.note_len(0, base + n)
                return lp
            out = assemble_verify(first, step, w, P, C, k).cpu()
        stats = {str(b): dict(v) for b, v in runner.graph_stats.items()}
        return out, stats
    finally:
        model.forward = inner
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = saved


def _modules():
    """The package modules this process loaded: the audit's evidence that a lazily imported file stayed off the path."""
    return sorted(m for m in sys.modules if m.startswith("experts4bit_qlora"))


def _build(env_ok, tap=None):
    """The shipped server's build (``PagedServeConfig.from_env()`` + ``build_engine``), refused unless ``env_ok(cfg)``
    returns None. V0 needs the logits tap; E and Q build without it."""
    import torch

    from experts4bit_qlora.engines import paged_runner as pr
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    why = env_ok(cfg)
    if why:
        raise SystemExit(f"REFUSED: {why}")
    if tap is not None:
        install_tap(pr, tap)
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    torch.cuda.synchronize()
    conf = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items() if k != "token"}
    return cfg, parts, conf, round(time.perf_counter() - t0, 2)


def read_v_main(a) -> int:
    """Stage V. V0 on rows outside calibration, written before any timing (the reducer VOIDs the read on any failure),
    then V1's timed terms on R row 0's live slot."""
    import torch
    from safetensors.torch import load_file

    import sd1_eagle3 as ref_mod
    from experts4bit_qlora.engines.eagle3_draft import Eagle3Drafter
    from experts4bit_qlora.engines.spec_decode import SpecDecoder

    R, C = load_rows(a.prompts_r), load_rows(a.prompts_c)
    tap = LogitsTap()
    cfg, parts, conf, load_s = _build(lambda c: None if (c.spec == "eagle3" and c.spec_k == K_SERVER and c.graphs)
                                      else f"stage V needs E4B_PAGED_SPEC=eagle3, K={K_SERVER} and graphs (got "
                                           f"{c.spec}, {c.spec_k}, graphs={c.graphs})", tap)
    runner = parts.runner
    gs = parts.info.get("graph_status") or {}
    rec = {"stage": "V", "load_s": load_s, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "config": conf,
           "census": {"graph_status": {str(b): gs.get(b) for b in BUCKETS_WANTED},
                      "post_graphs": {str(n): v for n, v in (getattr(runner, "spec_graph_status", None) or {}).items()},
                      "spec_build": parts.info.get("spec")}}
    gate = Gate(parts, tap)
    v0 = {"gate_rows": list(GATE_ROWS), "draft_rows": list(DRAFT_ROWS),
          "capture_bitwise": gate.capture_bitwise(R[GATE_ROWS[0]]), "addressing": {}, "mutants": {}}
    print("SD2_V0 capture_bitwise " + json.dumps(v0["capture_bitwise"]), flush=True)
    for name, rows in (("R", R), ("C", C)):
        for i in GATE_ROWS:
            for k in (1, 2, 3):
                res = gate.addressing(rows[i], k, POSITIONS)
                v0["addressing"][f"{name}{i}_k{k}"] = res
                print(f"SD2_V0 addressing {name}{i} k={k} " + json.dumps(res), flush=True)
    for m in ("a", "b", "c"):
        for name, rows in (("R", R), ("C", C)):
            undo = mutant(m, gate)
            try:
                res = gate.addressing(rows[GATE_ROWS[0]], 3, MUTANT_POSITIONS)
            finally:
                undo()
            v0["mutants"][f"{m}_{name}{GATE_ROWS[0]}"] = res
            print(f"SD2_V0 mutant {m} {name}{GATE_ROWS[0]} " + json.dumps(res), flush=True)
    ref = ref_mod.Eagle3Draft(load_file(os.path.join(a.head_dir, "model.safetensors")), device="cuda")
    v0["draft"] = gate.draft_gate([R[i] for i in DRAFT_ROWS] + [C[i] for i in DRAFT_ROWS], ref)
    print("SD2_V0 draft " + json.dumps(v0["draft"]), flush=True)
    v0["transition"] = gate.transition(R[GATE_ROWS[0]], R[GATE_ROWS[1]])
    print("SD2_V0 transition " + json.dumps(v0["transition"]), flush=True)
    rec["v0"] = v0
    import sd2_reduce
    rec["v0_fails"] = sd2_reduce.v0_fails(rec)                  # rule a2 on these rows; the GPU tests are the runner's
    json.dump(rec, open(a.out, "w"), indent=1, default=str)     # V0 on disk before any timing
    if rec["v0_fails"]:
        print("SD2_V VOID (V0 failed: " + ", ".join(rec["v0_fails"]) + "); no timing", flush=True)
        return 0

    # ---- V1: the timed terms, on R row 0's live slot at moving positions (timing, not addressing)
    spec3 = runner.spec
    d3 = spec3.drafter

    def anchor():
        rid, _slot = gate.begin(R[0])
        try:
            for _ in range(WARM):
                runner._run_decode_bucketed([rid])
            walls = []
            for _ in range(TIMED):
                t0 = time.perf_counter()
                runner._run_decode_bucketed([rid])      # its host read synchronizes
                walls.append(time.perf_counter() - t0)
        finally:
            gate.end(rid)
        return round(median(walls) * 1e3, 4)

    class _Timed:
        """A post-verify graph whose replays are bracketed by CUDA events."""

        def __init__(self, g, ev):
            self.g, self.ev = g, ev

        def replay(self):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            self.g.replay()
            e.record()
            self.ev.append((s, e))

    v1 = {"timed": TIMED, "warm": WARM, "anchor_a_ms": anchor()}
    for k in (1, 2, 3):
        drafter = Eagle3Drafter(d3.w, k=k, max_positions=d3.max_positions, n_heads=d3.nh, n_kv=d3.nkv, head_dim=d3.hd,
                                rope_theta=d3.theta, eps=d3.eps, device="cuda")      # the same weights, its own cache
        spec = SpecDecoder(drafter=drafter, aux=spec3.aux, kv=runner.kv, k=k, capacity=spec3.capacity, device="cuda",
                           verify=runner._spec_verify)
        try:
            spec.capture_post(k + 1)
            post_status = "graph"
        except Exception as e:                  # noqa: BLE001 -- recorded; the reducer VOIDs an eager post
            torch.cuda.synchronize()
            post_status = f"eager: {type(e).__name__}: {str(e)[:160]}"
        ev_v, ev_p = [], []

        def timed_verify(slot, n, ids, pos, _real=runner._spec_verify, _ev=ev_v):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            tok = _real(slot, n, ids, pos)
            e.record()
            _ev.append((s, e))
            return tok
        spec.verify = timed_verify
        if post_status == "graph":
            spec._post_graphs[k + 1] = _Timed(spec._post_graphs[k + 1], ev_p)
        runner.spec = spec
        try:
            rid, _slot = gate.begin(R[0])               # the prompt completes alone: this spec starts
            try:
                for _ in range(WARM):
                    runner._run_decode_spec(rid)
                del ev_v[:], ev_p[:]
                walls = []
                for _ in range(TIMED):
                    t0 = time.perf_counter()
                    runner._run_decode_spec(rid)        # its one host read synchronizes
                    walls.append(time.perf_counter() - t0)
                torch.cuda.synchronize()
            finally:
                gate.end(rid)
        finally:
            runner.spec = spec3
        full = median(walls) * 1e3
        ver = median([s.elapsed_time(e) for s, e in ev_v])
        post = median([s.elapsed_time(e) for s, e in ev_p]) if ev_p else None
        v1[f"k{k}"] = {"full_ms": round(full, 4), "verify_ms": round(ver, 4),
                       "post_ms": None if post is None else round(post, 4),
                       "loop_ms": None if post is None else round(full - ver - post, 4),
                       "verify_events": len(ev_v), "post_events": len(ev_p), "post_status": post_status,
                       "census": spec.census()}
        print(f"SD2_V1 k={k} " + json.dumps(v1[f"k{k}"], default=str), flush=True)
        del spec, drafter
    v1["anchor_b_ms"] = anchor()                        # the A/A brackets the k terms
    print(f"SD2_V1 anchors {v1['anchor_a_ms']} {v1['anchor_b_ms']} ms", flush=True)
    rec["v1"] = v1
    rec["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    rec["modules_loaded"] = _modules()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("SD2_V done", flush=True)
    return 0


def read_e_main(a) -> int:
    """Stage E, one arm in its own process: each workload's 16 rows one request at a time (W1's shape) on the 16-slot
    server; one warm pass at SHORT, then E_REPS rounds of SHORT and LONG; decode tok/s is P109's slope. The identity
    tokens are the timed LONG passes' own (whether all three agree is recorded)."""
    import torch

    import p109_box
    arm = a.arm.split("-")[0]
    want = {"OFF": ("off", 0), "ON1": ("eagle3", 1), "ON2": ("eagle3", 2)}[arm]
    cfg, parts, conf, load_s = _build(
        lambda c: None if ((c.spec, int(c.spec_k or 0)) == want and c.graphs and c.max_seqs == 16)
        else f"arm {a.arm} needs spec {want}, graphs and max_seqs 16 (got {c.spec}, {c.spec_k}, graphs={c.graphs}, "
             f"max_seqs={c.max_seqs})")
    runner = parts.runner
    spec = getattr(runner, "spec", None)
    rows_of = {"R": load_rows(a.prompts_r), "C-think": load_rows(a.prompts_c), "C-nothink": load_rows(a.prompts_cn)}

    def one_pass(rows, n_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        toks = [p109_box.run_pass(parts, torch, [row], n_new)[2][0] for row in rows]   # alone; exact lengths asserted
        return time.perf_counter() - t0, toks

    def graph_stats():
        return {str(b): dict(v) for b, v in (getattr(runner, "graph_stats", None) or {}).items()}

    rec = {"stage": "E", "arm": a.arm, "spec": cfg.spec, "k": cfg.spec_k, "load_s": load_s, "config": conf,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "graph_status": {str(b): s for b, s in (parts.info.get("graph_status") or {}).items()},
           "post_graphs": {str(n): v for n, v in (getattr(runner, "spec_graph_status", None) or {}).items()},
           "spec_build": parts.info.get("spec"), "workloads": {}}
    for w, long_ in E_WORKLOADS.items():
        rows = rows_of[w]
        c0 = dict(spec.census()) if spec is not None else None
        g0 = graph_stats()
        one_pass(rows, SHORT)                                       # warm, untimed
        walls, longs = {str(SHORT): [], str(long_): []}, []
        for _ in range(E_REPS):
            walls[str(SHORT)].append(one_pass(rows, SHORT)[0])
            wall, toks = one_pass(rows, long_)
            walls[str(long_)].append(wall)
            longs.append(toks)
        s = p109_box.slope(walls[str(SHORT)], walls[str(long_)], len(rows), SHORT, long_)
        g1 = graph_stats()
        wrec = {"walls": walls, **s, "short": SHORT, "long": long_, "rows": len(rows),
                "graph_stats_delta": {b: {kk: g1[b][kk] - g0.get(b, {}).get(kk, 0) for kk in g1[b]} for b in g1},
                "identity_tokens": longs[0], "long_reps_identical": all(t == longs[0] for t in longs[1:])}
        if spec is not None:
            c1 = dict(spec.census())
            d = {kk: c1[kk] - c0[kk] for kk in ("prefills", "steps", "drafted", "accepted", "emitted", "short_steps",
                                                  "dropped_batched", "dropped_end", "post_replays", "post_eager")}
            d["tau_live"] = d["emitted"] / d["steps"] if d["steps"] else None
            d["acceptance"] = d["accepted"] / d["drafted"] if d["drafted"] else None
            wrec["spec_census"] = d
        rec["workloads"][w] = wrec
        print(f"SD2_E {a.arm} {w} decode_tok_s={s.get('decode_tok_s')}"
              + (f" tau_live={wrec['spec_census']['tau_live']}" if spec is not None else ""), flush=True)
    rec["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    rec["max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
    rec["modules_loaded"] = _modules()
    json.dump(rec, open(a.out, "w"), default=str)
    print(f"SD2_E done {a.arm}", flush=True)
    return 0


def read_q_main(a) -> int:
    """Stage Q: P115 Phase B's instrument at its bytes (R, rep, chunk, mutant_scale; one window a pass), then ON1 and
    ON2 at the verify shape against R's saved log-probs, then the reported w16 draw."""
    import torch

    import p108_box
    import p115_quality as pq
    cfg, parts, conf, load_s = _build(
        lambda c: None if (not c.graphs and c.spec == "off" and c.placement == "all-vram")
        else f"stage Q builds the default server eager, spec off, all-vram (got graphs={c.graphs}, {c.spec}, "
             f"{c.placement})")
    model = parts.runner.model
    model.eval()
    P, C = a.prompt, a.cont
    windows = {t: pq.LOADERS[t](parts.tokenizer, a.windows, P, C) for t in pq.TEXTS}
    rec = pq.measure_phase(model, windows, phase="off", prompt=P, cont=C, chunk=512, floor_chunk=256, group=1,
                           device=cfg.device, ref_dir=a.ref_dir, arms=("R", "rep", "chunk", "mutant_scale"))
    on, on_eng = {}, {}
    for k in (1, 2):
        arm = f"ON{k}"
        on[arm], on_eng[arm] = {t: [] for t in windows}, {t: [] for t in windows}
        for t, wins in windows.items():
            for i, w in enumerate(wins):
                ref = torch.load(os.path.join(a.ref_dir, f"R_{t}_g{i}.pt"))[0]
                lp, stats = paged_verify_pass(model, w, P, C, k, cfg.device)
                p108_box._release(cfg.device)          # the pass's pool, once its frame is gone (P108 Amendment 3)
                nll, am = p108_box._score(lp, w[P:P + C])
                kl = p108_box._kl(ref, lp)
                on[arm][t].append({"window": i, "nll": sum(nll) / len(nll),
                                   "argmax_agree": sum(int(x == y) for x, y in zip(am, ref.argmax(-1).tolist())) / len(am),
                                   "kl": sum(kl) / len(kl)})
                on_eng[arm][t].append(stats)
            print(f"SD2_Q {arm} {t} done ({len(wins)} windows)", flush=True)
    rec["on"] = {"per_window": on, "graph_stats": on_eng, "plan": {f"ON{k}": verify_plan(P, C, k) for k in (1, 2)},
                 "windows_sha256": {t: pq.windows_sha(w) for t, w in windows.items()}}
    w16 = pq.measure_phase(model, windows, phase="off", prompt=P, cont=C, chunk=512, floor_chunk=256, group=W16,
                           device=cfg.device, ref_dir=a.ref_dir + "-w16", arms=("R",))
    rec["w16"] = {"per_window": w16["per_window"], "windows_sha256": w16["windows_sha256"], "group": W16}
    rec.update({"stage": "Q", "load_s": load_s, "config": conf, "e4b_sha": os.environ.get("E4B_SHA"),
                "gnf4_sha": os.environ.get("GNF4_SHA"), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "modules_loaded": _modules()})
    json.dump(rec, open(a.out, "w"), default=str)
    print("SD2_Q done", flush=True)
    return 0


def _toy_target(V=11):
    """A causal toy target for the CPU checks: the log-probs at position p are a fixed function of w[:p]. Returns
    (lp_at(prefix), make_step(cache, stagger)): a verify step over a position-indexed cache that the step writes
    before its rows read it, as the alias rows do; ``stagger`` -1 reads one short (mutant a's shape)."""
    import torch

    def lp_at(prefix):
        h = torch.zeros(V)
        for i, t in enumerate(prefix):
            h[(t * 7 + i * 3) % V] += 1.0 + 0.1 * i
            h = h.roll(1)
        return h.log_softmax(-1)

    def make_step(cache, stagger=0):
        def step(ids, base):
            for i, t in enumerate(ids):
                cache[base + i] = t
            return torch.stack([lp_at([cache[p] for p in range(base + i + 1 + stagger)]) for i in range(len(ids))])
        return step
    return lp_at, make_step


def verify_assembly_check(P=9, C=13, k=2, stagger=0, plan_shift=0):
    """The toy check the maintainer asked for: the verify pass's per-position log-probs against a T == 1
    teacher-forced pass, on a target whose arithmetic is row-count independent. True when they are equal."""
    import torch
    lp_at, make_step = _toy_target()
    w = [(5 * i + 3) % 11 for i in range(P + C)]
    t1 = torch.stack([lp_at(w[:p]) for p in range(P, P + C)])
    cache = {p: w[p] for p in range(P)}
    step = make_step(cache, stagger)
    shifted = (lambda ids, base: step(step_ids(w, base + plan_shift, k), base)) if plan_shift else step
    got = assemble_verify(lp_at(w[:P]), shifted, w, P, C, k)
    return bool(torch.equal(got, t1))


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
    if not verify_assembly_check() or verify_assembly_check(stagger=-1) or verify_assembly_check(plan_shift=1):
        bad.append("verify assembly")
    if [use for _b, use in verify_plan(512, 128, 2)][-1:] != [1] or sum(u for _b, u in verify_plan(512, 128, 2)) != 127:
        bad.append("verify plan")
    if median([3, 1, 2]) != 2 or median([4, 1, 2, 3]) != 2.5:
        bad.append("median")
    if bad:
        print("sd2_box self-test FAILED:", bad)
        return 1
    print("sd2_box self-test OK (agreement, dlogp, digest, logits tap, verify plan and assembly, median)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prove", action="store_true")
    ap.add_argument("--read-v", action="store_true")
    ap.add_argument("--read-e", dest="arm")
    ap.add_argument("--read-q", action="store_true")
    ap.add_argument("--prompts-r")
    ap.add_argument("--prompts-c")
    ap.add_argument("--prompts-cn")
    ap.add_argument("--head-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prove:
        return prove_main(a)
    if a.read_v:
        return read_v_main(a)
    if a.arm:
        return read_e_main(a)
    if a.read_q:
        return read_q_main(a)
    ap.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
