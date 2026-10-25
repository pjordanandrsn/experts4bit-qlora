#!/usr/bin/env python3
"""g3check.py -- the G3 GPU check (bench/kv-pool-g/g3-gpu-check/CHECK.md): with ``E4B_PAGED_KV_POOL=shared`` at 64
slots on Qwen3-30B-A3B int4, the server CAPTURES the prefill graph and replays it, the case sc2j-5090-1 refused for
headroom and e4b#2087 fixed. A correctness check, not a reading.

  drive --port P --model M [--n 6] [--max-tokens 8]   send N prompts long enough that each first chunk is the graph's
                                                      512 tokens (so it must replay); every response must complete
  verdict --health FILE [--out FILE]                  PASS or FAIL from the server's OWN /health; exit 0 on PASS
  --self-test

PASS needs every one of these, read from the server's state, never from the environment it was given. A missing field
is a FAIL (fail closed):
  - the configuration SC2j's shared arm ran, from the live server: top-level model == "Qwen/Qwen3-30B-A3B",
    engine.max_tokens_per_seq == 4096, and int4 experts AND int4 attention by ``levers``' census of the model as served
    (``int4_expert_layers`` counts the expert layers carrying an int4 store and must equal ``moe_layers`` > 0, a full
    stack). Attention binds exactly too, derived from the served model's own layer count ``L`` = ``moe_layers`` (every
    Qwen3-30B-A3B layer is an MoE layer): the int4 enable swapped q, k, v and o on every layer
    (``attn_int4_rtn_projections`` == 4 L), the QKV fusion merged q, k and v into one projection on every layer
    (``fuse_qkv_n`` == L), and the ``Int4Linear`` modules counted on the model as served are what that leaves
    (``int4_attn_projections`` == 4 L - 2 L = 2 L). sc2j-5090-1's own record reads 192, 48 and 96 at L = 48. Never
    the launch env;
  - engine.max_seqs == 64 and engine.kv_pool == "shared";
  - engine.kv_pool_sizing.pool_rows < claim_all_rows (the shared pool's own sizing ran, not the claim-all fallback),
    and the derivation's fitting 64-slot candidate carries prefill_graph_headroom_bytes > 0 (#2087's hold-back took
    part in the bisection);
  - prefill_graph.requested == "auto" and prefill_graph.status == "on". Under ``auto`` the runtime captures with
    ``require_headroom`` (``serve_paged`` calls ``runner.enable_prefill_graph(cfg.chunk_tokens, require_headroom=(cfg.prefill_graph == "auto"))``),
    and ``PagedModelRunner.enable_prefill_graph`` assigns ``self._prefill_graph`` only after the capture passed the
    headroom gate, ``free_after_bytes >= prefill_graph_headroom_bytes(pool_bytes, bulk_flush_bytes)``, on unrounded
    bytes; ``prefill_graph_stats`` reports "on" only while that attribute is set, and "refused" with the reason
    otherwise. So "on" under "auto" is the capture evidence;
  - prefill_graph.T == 512 and prefill_graph.replays > 0 after the prompts: the captured graph served first chunks.
A refused graph (sc2j-5090-1's record) is a FAIL. No server state at all (no /health record: the server never answered)
is UNREAD, exit 3: the check did not read, which is a harness fault and never a FAIL of the fix under test
(g3-gpu-check-1: the server died at import). The MiB fields /health reports are rounded, so the verdict records
free_after_mib - (pool_mib + bulk_flush_mib) as information only and never binds on it (the gate already compared the
unrounded bytes).
"""
import argparse
import json
import sys
import urllib.request

SLOTS, T = 64, 512
MODEL, TOKENS = "Qwen/Qwen3-30B-A3B", 4096
PARAGRAPH = ("The shared KV pool hands each running sequence the cache rows it needs, step by step, instead of "
             "reserving a full sequence for every slot up front. ")


def drive(port: int, model: str, n: int, max_tokens: int) -> int:
    prompt = PARAGRAPH * 40                     # ~1,000 tokens: the first chunk is T, the rest a later chunk
    bad = 0
    for i in range(n):
        body = json.dumps({"model": model, "prompt": prompt + f" Request {i}.", "max_tokens": max_tokens,
                           "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            ptoks = (out.get("usage") or {}).get("prompt_tokens")
            ok = bool(out.get("choices")) and out["choices"][0].get("finish_reason") in ("length", "stop") \
                and isinstance(ptoks, int) and ptoks > T          # longer than T: the first chunk is the graph's
            print(f"G3CHECK request {i} prompt_tokens={ptoks} finish={(out.get('choices') or [{}])[0].get('finish_reason')}"
                  f" {'ok' if ok else 'BAD'}", flush=True)
        except Exception as e:                  # noqa: BLE001 -- any failure is a failed request, recorded
            ok = False
            print(f"G3CHECK request {i} failed: {e}", flush=True)
        bad += not ok
    print(f"G3CHECK drive n={n} failed={bad}", flush=True)
    return 0 if bad == 0 else 1


def verdict(h: dict) -> dict:
    if not isinstance(h, dict) or not h:
        return {"verdict": "UNREAD", "why": ["no server state: no /health record was read"], "prefill_graph": None,
                "max_seqs": None, "kv_pool": None}
    why = []
    e = (h or {}).get("engine") or {}
    lv = (h or {}).get("levers") or {}
    if (h or {}).get("model") != MODEL:
        why.append(f"model {(h or {}).get('model', '<missing>')!r}, not {MODEL!r}")
    if e.get("max_tokens_per_seq") != TOKENS:
        why.append(f"engine.max_tokens_per_seq {e.get('max_tokens_per_seq', '<missing>')!r}, not {TOKENS}")
    il, ml = lv.get("int4_expert_layers"), lv.get("moe_layers")
    if not (isinstance(il, int) and isinstance(ml, int) and not isinstance(il, bool) and ml > 0 and il == ml):
        why.append(f"levers.int4_expert_layers {il!r} of moe_layers {ml!r}: the experts are not int4 on every MoE layer")
    ia, rtn, fq = lv.get("int4_attn_projections"), lv.get("attn_int4_rtn_projections"), lv.get("fuse_qkv_n")
    ints = all(isinstance(x, int) and not isinstance(x, bool) for x in (ia, rtn, fq, ml))
    if not ints or ml <= 0 or rtn != 4 * ml or fq != ml or ia != rtn - 2 * fq:
        why.append(f"int4 attention not on every projection of {ml!r} layers: attn_int4_rtn_projections {rtn!r} "
                   f"(want 4 L), fuse_qkv_n {fq!r} (want L), int4_attn_projections {ia!r} (want 2 L, as served)")
    if e.get("max_seqs") != SLOTS:
        why.append(f"engine.max_seqs {e.get('max_seqs', '<missing>')!r}, not {SLOTS}")
    if e.get("kv_pool") != "shared":
        why.append(f"engine.kv_pool {e.get('kv_pool', '<missing>')!r}, not 'shared'")
    sz = e.get("kv_pool_sizing")
    if not isinstance(sz, dict):
        why.append("no engine.kv_pool_sizing record")
    else:
        rows, full = sz.get("pool_rows"), sz.get("claim_all_rows")
        if not isinstance(rows, int) or not isinstance(full, int) or rows >= full:
            why.append(f"kv_pool_sizing pool_rows {rows!r} not under claim_all_rows {full!r}: the shared sizing did "
                       f"not size this pool")
        der = sz.get("derivation") if isinstance(sz.get("derivation"), dict) else {}
        cands = [c for c in der.get("candidates") or [] if c.get("max_seqs") == SLOTS and c.get("fits") is True]
        hb = cands[0].get("prefill_graph_headroom_bytes") if cands else None
        if not isinstance(hb, int) or hb <= 0:
            why.append(f"no fitting {SLOTS}-slot candidate with prefill_graph_headroom_bytes > 0 (got {hb!r}): the "
                       f"hold-back did not take part")
    pg = (h or {}).get("prefill_graph")
    margin = None
    if not isinstance(pg, dict):
        why.append("no prefill_graph record")
    else:
        if pg.get("requested") != "auto":
            why.append(f"prefill_graph.requested {pg.get('requested', '<missing>')!r}, not 'auto' (only auto captures "
                       f"under the headroom gate)")
        if pg.get("status") != "on":
            why.append(f"prefill_graph.status {pg.get('status', '<missing>')!r}, not 'on'"
                       + (f": {pg.get('why')}" if pg.get("why") else ""))
        if pg.get("T") != T:
            why.append(f"prefill_graph.T {pg.get('T', '<missing>')!r}, not {T}")
        if not isinstance(pg.get("replays"), int) or isinstance(pg.get("replays"), bool) or pg["replays"] <= 0:
            why.append(f"prefill_graph.replays {pg.get('replays', '<missing>')!r}: the graph served no first chunk")
        vals = [pg.get(k) for k in ("free_after_mib", "pool_mib", "bulk_flush_mib")]
        if all(isinstance(v, (int, float)) for v in vals):
            margin = vals[0] - vals[1] - vals[2]
    return {"verdict": "PASS" if not why else "FAIL", "why": why, "info_margin_mib_rounded": margin,
            "prefill_graph": pg, "max_seqs": e.get("max_seqs"), "kv_pool": e.get("kv_pool"),
            "model": (h or {}).get("model"), "max_tokens_per_seq": e.get("max_tokens_per_seq"),
            "int4_expert_layers": il, "moe_layers": ml, "int4_attn_projections": ia,
            "attn_int4_rtn_projections": rtn, "fuse_qkv_n": fq}


def _health(**over):
    h = {"model": MODEL, "levers": {"moe_layers": 48, "int4_expert_layers": 48, "int4_attn_projections": 96,
                                    "attn_int4_rtn_projections": 192, "fuse_qkv_n": 48},
         "engine": {"max_seqs": 64, "max_tokens_per_seq": 4096, "kv_pool": "shared",
                    "kv_pool_sizing": {"pool_rows": 9768, "claim_all_rows": 16448, "derivation": {"candidates": [
                        {"max_seqs": 64, "fits": True, "pool_rows": 9768,
                         "prefill_graph_headroom_bytes": 1139802112}]}}},
         "prefill_graph": {"requested": "auto", "status": "on", "T": 512, "replays": 6, "eager_chunks": 12, "pool_mib": 426,
                           "free_after_mib": 1800, "bulk_flush_mib": 319}}
    for path, val in over.items():
        node = h
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        if val is KeyError:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = val
    return h


def self_test() -> int:
    refused = {"status": "refused", "why": "memory: the graph's private pool holds 426 MiB for its life and 724 MiB "
                                           "is free after capture"}
    cases = [
        ("captured, replayed, headroom held", _health(), "PASS"),
        ("refused for headroom (sc2j-5090-1)", _health(prefill_graph=dict(refused, requested="auto")), "FAIL"),
        ("on but never replayed", _health(prefill_graph__replays=0), "FAIL"),
        ("replays missing", _health(prefill_graph__replays=KeyError), "FAIL"),
        ("T missing", _health(prefill_graph__T=KeyError), "FAIL"),
        ("T not 512", _health(prefill_graph__T=256), "FAIL"),
        ("status off", _health(prefill_graph={"status": "off"}), "FAIL"),
        ("no prefill_graph record", _health(prefill_graph=KeyError), "FAIL"),
        ("32 slots", _health(engine__max_seqs=32), "FAIL"),
        ("claim-all", _health(engine__kv_pool="claim-all"), "FAIL"),
        ("no sizing record", _health(engine__kv_pool_sizing=KeyError), "FAIL"),
        ("pool at the claim-all size", _health(engine__kv_pool_sizing__pool_rows=16448), "FAIL"),
        ("pool_rows None", _health(engine__kv_pool_sizing__pool_rows=None), "FAIL"),
        ("claim_all_rows missing", _health(engine__kv_pool_sizing__claim_all_rows=KeyError), "FAIL"),
        ("no hold-back field", _health(engine__kv_pool_sizing__derivation={"candidates": [
            {"max_seqs": 64, "fits": True}]}), "FAIL"),
        ("hold-back zero", _health(engine__kv_pool_sizing__derivation={"candidates": [
            {"max_seqs": 64, "fits": True, "prefill_graph_headroom_bytes": 0}]}), "FAIL"),
        ("64-slot candidate did not fit", _health(engine__kv_pool_sizing__derivation={"candidates": [
            {"max_seqs": 64, "fits": False, "prefill_graph_headroom_bytes": 1}]}), "FAIL"),
        ("no derivation", _health(engine__kv_pool_sizing__derivation=None), "FAIL"),
        ("forced on (requested 1, no headroom gate)", _health(prefill_graph__requested="1"), "FAIL"),
        ("requested missing", _health(prefill_graph__requested=KeyError), "FAIL"),
        ("replays a bool", _health(prefill_graph__replays=True), "FAIL"),
        ("rounded margin short is information, not a verdict", _health(prefill_graph__free_after_mib=744), "PASS"),
        ("MiB fields missing is information, not a verdict", _health(prefill_graph__free_after_mib=KeyError), "PASS"),
        ("wrong model", _health(model="ibm-granite/granite-3.1-3b-a800m-instruct"), "FAIL"),
        ("model missing", _health(model=KeyError), "FAIL"),
        ("2048 tokens a slot", _health(engine__max_tokens_per_seq=2048), "FAIL"),
        ("max_tokens_per_seq missing", _health(engine__max_tokens_per_seq=KeyError), "FAIL"),
        ("bf16 experts (no int4 store)", _health(levers__int4_expert_layers=0), "FAIL"),
        ("int4 experts on part of the stack", _health(levers__int4_expert_layers=47), "FAIL"),
        ("moe_layers missing", _health(levers__moe_layers=KeyError), "FAIL"),
        ("bf16 attention (no Int4Linear)", _health(levers__int4_attn_projections=0), "FAIL"),
        ("int4 attention on 95 of 96 served projections", _health(levers__int4_attn_projections=95), "FAIL"),
        ("int4 enable swapped 191 of 192", _health(levers__attn_int4_rtn_projections=191), "FAIL"),
        ("QKV fused on 47 of 48 layers", _health(levers__fuse_qkv_n=47, levers__int4_attn_projections=98), "FAIL"),
        ("int4 attention count missing", _health(levers__int4_attn_projections=KeyError), "FAIL"),
        ("rtn count missing", _health(levers__attn_int4_rtn_projections=KeyError), "FAIL"),
        ("fuse_qkv_n missing", _health(levers__fuse_qkv_n=KeyError), "FAIL"),
        ("no levers record", _health(levers=KeyError), "FAIL"),
        ("empty health: the server never answered", {}, "UNREAD"),
        ("no health at all", None, "UNREAD"),
        ("a server record with nothing in it but an error", {"status": "error", "error": "OOM"}, "FAIL"),
    ]
    fails = 0
    for name, h, want in cases:
        got = verdict(h)["verdict"]
        fails += got != want
        print(f"G3CHECK self-test {'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"G3CHECK self-test {'PASS' if not fails else 'FAIL'} ({len(cases) - fails}/{len(cases)})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("drive", "verdict"))
    ap.add_argument("--port", type=int)
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--health")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "drive":
        if not a.port or not a.model:
            ap.error("drive needs --port and --model")
        return drive(a.port, a.model, a.n, a.max_tokens)
    if a.cmd == "verdict":
        if not a.health:
            ap.error("verdict needs --health")
        try:
            h = json.load(open(a.health))
        except (OSError, ValueError) as e:
            h = {}
            print(f"G3CHECK unreadable health: {e}", flush=True)
        v = verdict(h)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(v, f, indent=1)
        print("G3CHECK_VERDICT " + json.dumps({k: v[k] for k in ("verdict", "why", "max_seqs", "kv_pool")}), flush=True)
        return {"PASS": 0, "UNREAD": 3}.get(v["verdict"], 1)
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
