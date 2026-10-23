#!/usr/bin/env python3
"""p225_gate.py -- lane P225's proof that the run produced a genuinely trained attention adapter, served it whole, and
scored disjoint data (bench/p225/PREREG-p225.md, "The trained adapter is proven" and "Engagement and identity").

It reads only the real files the run wrote, never a summary:
- the trainer's own receipt (``run/receipts_REF/*.json``: ``status``, ``workload.steps``, ``measured.timed_steps``);
- the adapter Loggetta wrote (``run/adapter_REF/``: ``adapter_manifest.json``, schema ``backend-adapter/1``, and
  ``adapter.safetensors``, whose B tensors are read byte by byte);
- the reference fingerprint (``run/fingerprint_REF.json``);
- the data files (``run/data/train.jsonl``, ``run/data/heldout.jsonl``);
- the paged arm's slot records (``run/slot_*.json`` with ``arm == "PA"``).

Each failure is NAMED: ``INCOMPLETE_TRAIN``, ``ADAPTER_REVISION``, ``ADAPTER_SHA``, ``ADAPTER_TARGETS``,
``ADAPTER_ZERO_B``, ``ADAPTER_FINGERPRINT``, ``LEAK``, ``PARTIAL_ENGAGEMENT``, ``IDENTITY``. The reducer turns any of
them into VOID with that reason. Pure Python (the B tensors are bf16 read from the file's bytes).

    python p225_gate.py --run RUN_DIR --revision <sha> --out gates_p225.json
    python p225_gate.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import struct
import sys
import tempfile

STEPS = 80                                   # the registered train (P224's canary A, --steps 80)
LAYERS, PROJ = 48, ("q_proj", "k_proj", "v_proj", "o_proj")
TARGETS = LAYERS * len(PROJ)                 # 192 attention projections (Qwen3-30B-A3B)
FINGERPRINT = {"attention": 192, "experts": 96}
TARGET_RE = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_(A|B)")
_BF16_ZERO = {0x0000, 0x8000}                # +0.0 and -0.0


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_complete(run: str) -> list:
    """The trainer's own record: status OK, the registered steps, and timed steps ending at the last one."""
    recs = sorted(glob.glob(os.path.join(run, "receipts_REF", "*.json")))
    if not recs:
        return [f"INCOMPLETE_TRAIN: no trainer receipt in {os.path.join(run, 'receipts_REF')}"]
    r = json.load(open(recs[-1], encoding="utf-8"))
    why = []
    if r.get("status") != "OK":
        why.append(f"INCOMPLETE_TRAIN: trainer status {r.get('status')!r}")
    steps = (r.get("workload") or {}).get("steps")
    if steps != STEPS:
        why.append(f"INCOMPLETE_TRAIN: workload.steps {steps!r}, registered {STEPS}")
    timed = str((r.get("measured") or {}).get("timed_steps") or "")
    if not re.fullmatch(rf"\d+\.\.{STEPS}", timed):
        why.append(f"INCOMPLETE_TRAIN: measured.timed_steps {timed!r} does not end at step {STEPS}")
    return why


def _safetensors(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    return header, base


def adapter_proof(run: str, revision: str) -> tuple[list, dict]:
    """The adapter Loggetta wrote: revision, file sha, every target present, every B nonzero, the fingerprint."""
    ad = os.path.join(run, "adapter_REF")
    why, facts = [], {}
    try:
        man = json.load(open(os.path.join(ad, "adapter_manifest.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"ADAPTER_TARGETS: no readable adapter manifest ({type(e).__name__})"], facts
    if man.get("resolved_model_revision") != revision:
        why.append(f"ADAPTER_REVISION: manifest revision {man.get('resolved_model_revision')!r}, registered {revision!r}")
    wpath = os.path.join(ad, man.get("weights") or "adapter.safetensors")
    want = ((man.get("files") or {}).get(os.path.basename(wpath)) or {}).get("sha256")
    got = _sha256(wpath) if os.path.exists(wpath) else None
    facts["adapter_sha256"] = got
    if got is None or got != want:
        why.append(f"ADAPTER_SHA: file sha256 {got} against the manifest's {want}")
        return why, facts
    header, base = _safetensors(wpath)
    seen = {"A": set(), "B": set()}
    zero_b = []
    with open(wpath, "rb") as f:
        for name, meta in header.items():
            m = TARGET_RE.fullmatch(name)
            if not m:
                continue
            key = (int(m.group(1)), m.group(2))
            seen[m.group(3)].add(key)
            if m.group(3) == "B":
                if meta.get("dtype") != "BF16":
                    why.append(f"ADAPTER_TARGETS: {name} is {meta.get('dtype')}, registered BF16")
                    continue
                a, b = meta["data_offsets"]
                f.seek(base + a)
                raw = f.read(b - a)
                words = struct.unpack(f"<{len(raw) // 2}H", raw)
                if all(w in _BF16_ZERO for w in words):
                    zero_b.append(name)
    want_keys = {(L, p) for L in range(LAYERS) for p in PROJ}
    facts["targets"] = {"A": len(seen["A"]), "B": len(seen["B"]), "zero_B": len(zero_b)}
    if seen["A"] != want_keys or seen["B"] != want_keys:
        why.append(f"ADAPTER_TARGETS: A {len(seen['A'])} / B {len(seen['B'])} of the registered {TARGETS} targets")
    if zero_b:
        why.append(f"ADAPTER_ZERO_B: {len(zero_b)} B tensors are all zero (first {zero_b[0]})")
    try:
        fp = json.load(open(os.path.join(run, "fingerprint_REF.json"), encoding="utf-8")).get("fingerprint") or {}
        mods = {k: (fp.get(k) or {}).get("modules") for k in FINGERPRINT}
    except (OSError, ValueError):
        mods = {}
    facts["fingerprint_modules"] = mods
    if mods != FINGERPRINT:
        why.append(f"ADAPTER_FINGERPRINT: modules {mods}, registered {FINGERPRINT}")
    return why, facts


def _row_sha(line: str) -> str:
    return hashlib.sha256(json.dumps(json.loads(line)["messages"], sort_keys=True).encode("utf-8")).hexdigest()


def disjoint(run: str) -> list:
    """No held-out row equals a training row, by the full row text's sha256, read from the real data files."""
    try:
        train = {_row_sha(x) for x in open(os.path.join(run, "data", "train.jsonl"), encoding="utf-8") if x.strip()}
        held = [_row_sha(x) for x in open(os.path.join(run, "data", "heldout.jsonl"), encoding="utf-8") if x.strip()]
    except (OSError, ValueError, KeyError) as e:
        return [f"LEAK: the data files are unreadable ({type(e).__name__}); disjointness is unproven"]
    n = sum(h in train for h in held)
    return [f"LEAK: {n} held-out rows equal a training row"] if n else []


def engagement(run: str, adapter_sha: str | None) -> list:
    """Every PA slot: the server merged all registered targets, served the adapter by its sha, refused the base name,
    and ran the paged (graphed) path."""
    why, pa = [], []
    for f in sorted(glob.glob(os.path.join(run, "slot_*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("arm") == "PA":
            pa.append((os.path.basename(f), d))
    if not pa:
        return ["PARTIAL_ENGAGEMENT: no PA slot record"]
    for name, d in pa:
        ident = d.get("identity") or {}
        merged = (ident.get("merge") or {}).get("merged")
        if merged != TARGETS:
            why.append(f"PARTIAL_ENGAGEMENT: {name} merged {merged!r} of {TARGETS} targets")
        if d.get("graphs") is not True:
            why.append(f"PARTIAL_ENGAGEMENT: {name} did not run the graphed paged path (graphs={d.get('graphs')!r})")
        if adapter_sha is not None and ident.get("adapter_id") != adapter_sha:
            why.append(f"IDENTITY: {name} served adapter {ident.get('adapter_id')!r}, not the trained file {adapter_sha}")
        if ident.get("base_name_status") != 404:
            why.append(f"IDENTITY: {name} answered the base name with {ident.get('base_name_status')!r}, not 404")
    return why


def gates(run: str, revision: str) -> dict:
    proof, facts = adapter_proof(run, revision)
    why = train_complete(run) + proof + disjoint(run) + engagement(run, facts.get("adapter_sha256"))
    return {"ok": not why, "failures": why, "facts": facts}


# ---------------------------------------------------------------------------------------------------- the self-test

def _bf16(x: float) -> bytes:
    return struct.pack("<I", struct.unpack("<I", struct.pack("<f", x))[0])[2:]


def _fixture(d: str, revision: str, zero_layer: int | None = None, drop: str | None = None, steps: int = STEPS) -> str:
    """A run directory in the real records' shape (field names as the trainer, Loggetta and P224's box write them)."""
    run = os.path.join(d, "run")
    for sub in ("receipts_REF", "adapter_REF", "data"):
        os.makedirs(os.path.join(run, sub), exist_ok=True)
    json.dump({"status": "OK", "workload": {"steps": steps}, "measured": {"timed_steps": f"3..{steps}"}},
              open(os.path.join(run, "receipts_REF", "t.json"), "w"))
    header, blob = {}, b""
    for L in range(LAYERS):
        for p in PROJ:
            for ab, shape in (("A", [2, 4]), ("B", [4, 2])):
                name = f"model.layers.{L}.self_attn.{p}.lora_{ab}"
                if name == drop:
                    continue
                v = 0.0 if (ab == "B" and L == zero_layer) else 0.5
                data = b"".join(_bf16(v) for _ in range(shape[0] * shape[1]))
                header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
                blob += data
    h = json.dumps(header).encode()
    w = os.path.join(run, "adapter_REF", "adapter.safetensors")
    open(w, "wb").write(struct.pack("<Q", len(h)) + h + blob)
    json.dump({"resolved_model_revision": revision, "weights": "adapter.safetensors",
               "files": {"adapter.safetensors": {"sha256": _sha256(w)}}},
              open(os.path.join(run, "adapter_REF", "adapter_manifest.json"), "w"))
    json.dump({"ok": True, "fingerprint": {"attention": {"modules": 192}, "experts": {"modules": 96}}},
              open(os.path.join(run, "fingerprint_REF.json"), "w"))
    rows = lambda ts: "".join(json.dumps({"messages": [{"role": "user", "content": t}]}) + "\n" for t in ts)  # noqa: E731
    open(os.path.join(run, "data", "train.jsonl"), "w").write(rows(["a", "b"]))
    open(os.path.join(run, "data", "heldout.jsonl"), "w").write(rows(["c", "d"]))
    for slot in ("b0B", "b1B"):
        json.dump({"arm": "PA", "slot": slot, "graphs": True,
                   "identity": {"adapter_id": _sha256(w), "base_name_status": 404, "merge": {"merged": 192}}},
                  open(os.path.join(run, f"slot_{slot}.json"), "w"))
    json.dump({"arm": "P0", "slot": "b0A1", "graphs": True, "identity": {"merge": {}}},
              open(os.path.join(run, "slot_b0A1.json"), "w"))
    return run


def self_test() -> int:
    n = 0
    rev = "r" * 40

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1

    def names(g):
        return {w.split(":")[0] for w in g["failures"]}
    with tempfile.TemporaryDirectory() as d:
        g = gates(_fixture(d, rev), rev)
        ok(g["ok"] and g["facts"]["targets"] == {"A": 192, "B": 192, "zero_B": 0}, f"a whole adapter passes: {g}")
    with tempfile.TemporaryDirectory() as d:
        ok(names(gates(_fixture(d, rev, steps=65), rev)) == {"INCOMPLETE_TRAIN"}, "65 of 80 steps is INCOMPLETE_TRAIN")
    with tempfile.TemporaryDirectory() as d:
        run = _fixture(d, rev)
        os.remove(os.path.join(run, "receipts_REF", "t.json"))
        ok(names(gates(run, rev)) == {"INCOMPLETE_TRAIN"}, "no trainer receipt is INCOMPLETE_TRAIN")
    with tempfile.TemporaryDirectory() as d:
        ok(names(gates(_fixture(d, rev, zero_layer=7), rev)) == {"ADAPTER_ZERO_B"}, "an all-zero B is named")
    with tempfile.TemporaryDirectory() as d:
        g = gates(_fixture(d, rev, drop="model.layers.47.self_attn.o_proj.lora_B"), rev)
        ok(names(g) == {"ADAPTER_TARGETS"}, f"a missing target is named: {g['failures']}")
    with tempfile.TemporaryDirectory() as d:
        ok(names(gates(_fixture(d, rev), "s" * 40)) == {"ADAPTER_REVISION"}, "another base revision is named")
    with tempfile.TemporaryDirectory() as d:
        run = _fixture(d, rev)
        with open(os.path.join(run, "adapter_REF", "adapter.safetensors"), "ab") as f:
            f.write(b"\0")
        ok("ADAPTER_SHA" in names(gates(run, rev)), "a file that is not the manifest's is named")
    with tempfile.TemporaryDirectory() as d:
        run = _fixture(d, rev)
        open(os.path.join(run, "data", "heldout.jsonl"), "a").write(json.dumps({"messages": [{"role": "user", "content": "a"}]}) + "\n")
        ok(names(gates(run, rev)) == {"LEAK"}, "a held-out row equal to a training row is LEAK")
    with tempfile.TemporaryDirectory() as d:
        run = _fixture(d, rev)
        s = json.load(open(os.path.join(run, "slot_b1B.json")))
        s["identity"]["merge"]["merged"] = 188
        json.dump(s, open(os.path.join(run, "slot_b1B.json"), "w"))
        ok(names(gates(run, rev)) == {"PARTIAL_ENGAGEMENT"}, "188 of 192 merged is PARTIAL_ENGAGEMENT")
    with tempfile.TemporaryDirectory() as d:
        run = _fixture(d, rev)
        s = json.load(open(os.path.join(run, "slot_b0B.json")))
        s["identity"]["base_name_status"] = 200
        json.dump(s, open(os.path.join(run, "slot_b0B.json"), "w"))
        ok(names(gates(run, rev)) == {"IDENTITY"}, "a base name served while the adapter is up is IDENTITY")
    with tempfile.TemporaryDirectory() as d:
        run = _fixture(d, rev)
        fp = json.load(open(os.path.join(run, "fingerprint_REF.json")))
        fp["fingerprint"]["attention"]["modules"] = 191
        json.dump(fp, open(os.path.join(run, "fingerprint_REF.json"), "w"))
        ok(names(gates(run, rev)) == {"ADAPTER_FINGERPRINT"}, "a fingerprint off the registration is named")
    print(f"p225_gate self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run")
    ap.add_argument("--revision")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="apply the argument checks a real run applies, then stop")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.run and a.revision and a.out):
        ap.error("--run, --revision and --out (or --self-test)")
    if a.check_args:
        print("ARGS OK")
        return 0
    g = gates(a.run, a.revision)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(g, f, indent=1)
    print("P225_GATES " + json.dumps({"ok": g["ok"], "failures": g["failures"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
