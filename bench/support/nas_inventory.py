#!/usr/bin/env python3
"""Inventory the LAN model store, and say which families it can actually answer.

Run: python bench/support/nas_inventory.py > bench/support/NAS-INVENTORY.md

The plan for this lane assumed checkpoints had to come over the WAN, which made
it download-bound and pointed it at rented boxes chosen for measured bandwidth.
That was never checked against the fleet. It is wrong: /share/models holds real
checkpoints for 8 of the 9 claimed families. The constraint is the probe host's
RAM, not download cost.

So this is the document that should have existed before the plan did. It is
generated rather than hand-written for the same reason the support table now is
-- a hand-kept inventory of a store someone else fills goes stale silently.

Sizes come from summing ``*.safetensors`` via one ``ls``, NOT from ``du -sh`` per
directory: recursive du across 63 directories on that HDD pool timed out twice
while writing this. The safetensors sum is the number that matters anyway -- it
is what a load has to read.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone

NAS = "admin@10.0.0.68"
# BOTH stores. An earlier version scanned only /share/models and therefore
# reported kimi_k3 as "not on the LAN at all" while a complete 1.4 TB Kimi-K3
# snapshot sat in the other one. An inventory that under-reports is worse than
# no inventory: it argues for renting a box to fetch something already here.
SRCS = ["/share/models", "/share/ZFS532_DATA/hf-models"]

# Run one remote pass. The QNAP has Entware python3; per-directory ssh round
# trips for 63 entries would dominate the runtime.
REMOTE = r"""
import json, os, sys


def describe(root, name):
    d = os.path.join(root, name)
    cfg = os.path.join(d, "config.json")
    mt = experts = hidden = topk = layers = inter = None
    if os.path.exists(cfg):
        try:
            c = json.load(open(cfg))
            t = c.get("text_config", c)
            mt = c.get("model_type")
            experts = (t.get("num_experts") or t.get("num_local_experts")
                       or t.get("n_routed_experts"))
            hidden = t.get("hidden_size")
            topk = t.get("num_experts_per_tok") or t.get("moe_topk")
            layers = t.get("num_hidden_layers")
            inter = t.get("moe_intermediate_size") or t.get("intermediate_size")
        except Exception:
            mt = "unreadable-config"
    total = shards = 0
    try:
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                shards += 1
                try:
                    total += os.path.getsize(os.path.join(d, f))
                except OSError:
                    pass
    except OSError:
        pass
    # Is a fetch still running or half-done? nas_fetch.sh writes one .done per
    # file plus a .filelist, so the pair says so. Without this the inventory
    # cannot tell a partial download from a complete checkpoint, and would
    # happily argue for probing a truncated one.
    want = have = None
    fl = os.path.join(d, ".filelist")
    if os.path.exists(fl):
        try:
            want = sum(1 for ln in open(fl) if ln.strip())
        except OSError:
            want = None
        have = 0
        for dirpath, _dirnames, files in os.walk(d):
            have += sum(1 for f in files if f.endswith(".done"))
    return {"name": name, "root": root, "model_type": mt, "experts": experts,
            "hidden": hidden, "top_k": topk, "layers": layers, "inter": inter,
            "bytes": total, "shards": shards, "want": want, "have": have}


out = []
for root in sys.argv[1:]:
    if not os.path.isdir(root):
        continue
    for name in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, name)):
            out.append(describe(root, name))
print(json.dumps(out))
"""


def fetch() -> list[dict]:
    roots = " ".join(shlex.quote(s) for s in SRCS)
    cmd = f"python3 -c {shlex.quote(REMOTE)} {roots}"
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", NAS, cmd],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"remote inventory failed: {r.stderr.strip()[:400]}")
    return json.loads(r.stdout)


HOST_GB = 64
USABLE_GB = 48        # leave macOS and the page cache room; a load that swaps is not a measurement

# nf4 stores 4 bits per weight plus an fp32 absmax per 64-element block, so
# ~0.5 + 4/64 bytes/param. Non-expert weights stay in the requested dtype.
NF4_BYTES_PER_PARAM = 0.5 + 4 / 64
BF16_BYTES_PER_PARAM = 2


def loaded_gb(r: dict) -> float | None:
    """Estimate the RESIDENT footprint after the loader quantizes the experts.

    The first version of this judged a checkpoint by its bf16 size on disk,
    which is the wrong quantity and overstates what cannot be loaded -- the
    whole point of the loader is that the expert stacks land at roughly a
    quarter. Getting this wrong argues for renting a box to run something the
    probe host could have handled, which is the same error the download-bound
    premise made one level up ([[finding_measure_the_right_quantity]] territory).

    Expert parameters come from the config geometry: layers x experts x 3
    projections x hidden x moe_intermediate (gate and up are hidden->inter, down
    is inter->hidden, so all three are hidden*inter). Everything else --
    attention, embeddings, router, norms, any dense MLP -- stays bf16. Returns
    None when the config does not carry enough to compute it, rather than
    guessing.

    **This is a planning estimate, NOT a measurement, and it is unvalidated.**
    The obvious way to check it is to watch a probe's RSS, and that does not
    work: RSS counts mmap'd safetensors pages, so it conflates the checkpoint
    being READ with memory actually HELD. A live Qwen3-30B-A3B probe sat at
    32.8 GB RSS against this function's ~18 GB, and that discrepancy is not
    evidence either way -- 57 GB of shards were being streamed through mmap at
    the time. Peak anonymous memory would be the right quantity; RSS is not it.

    So treat the verdict as a triage hint. The only real test is a load attempt,
    which is cheap and fails safely: the probe records the outcome as a row
    either way. Do not let a "fits" here become the reason a rental was skipped,
    or a "needs a bigger host" become the reason one was booked.
    """
    total_params = r["bytes"] / BF16_BYTES_PER_PARAM
    if not total_params:
        return None
    e, h, L, i = r.get("experts"), r.get("hidden"), r.get("layers"), r.get("inter")
    if not all(isinstance(v, int) and v > 0 for v in (e, h, L, i)):
        return None                     # dense, or a config we cannot read
    expert_params = L * e * 3 * h * i
    expert_params = min(expert_params, total_params)     # never exceed the whole model
    other = total_params - expert_params
    return (expert_params * NF4_BYTES_PER_PARAM + other * BF16_BYTES_PER_PARAM) / 2**30


def verdict(r: dict) -> str:
    gb = r["bytes"] / 2**30
    if gb == 0:
        return "config only"
    est = loaded_gb(r)
    if est is None:
        # No expert geometry: nothing gets quantized, so the bf16 size IS the
        # footprint.
        return f"~{gb:,.0f} GB bf16, dense — {'fits' if gb <= USABLE_GB else 'too big'}"
    if est <= USABLE_GB:
        return f"**fits {HOST_GB} GB** (~{est:,.0f} GB loaded)"
    return f"needs a bigger host (~{est:,.0f} GB loaded)"


def main() -> int:
    rows = fetch()
    sys.path.insert(0, ".")
    try:
        from experts4bit_qlora.loader import SUPPORTED_ARCHITECTURES as CLAIMED
    except Exception:
        CLAIMED = {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"# LAN model stores — `{NAS}`\n")
    print("Roots scanned: " + ", ".join(f"`{s}`" for s in SRCS) + "\n")
    print(f"Generated {now} by `bench/support/nas_inventory.py`. Sizes are the sum of")
    print("`*.safetensors` (what a load must read), not `du` of the directory — several of")
    print("these carry duplicate formats (`metal/`, `original/`) a load never touches.\n")

    claimed_hits: dict[str, list] = {k: [] for k in CLAIMED}
    print("| directory | model_type | experts | hidden | shards | safetensors | fetch | verdict |")
    print("|---|---|---|---:|---:|---:|---|---|")
    for r in rows:
        gb = r["bytes"] / 2**30
        want, have = r.get("want"), r.get("have")
        if want:
            fetch_state = "complete" if (have or 0) >= want else f"**partial {have}/{want}**"
        else:
            fetch_state = "—"
        # A partial fetch is NOT coverage. Counting one would put a family in the
        # "reachable without renting" column on the strength of a download that
        # has not finished.
        if r["model_type"] in claimed_hits and gb > 0 and not fetch_state.startswith("**"):
            claimed_hits[r["model_type"]].append((r["name"], gb, loaded_gb(r)))
        print(f"| `{r['name']}` | `{r['model_type'] or '—'}` | {r['experts'] or '—'} "
              f"| {r['hidden'] or '—'} | {r['shards']} | {gb:,.1f} GB | {fetch_state} "
              f"| {verdict(r)} |")

    print(f"\n## Coverage of the {len(CLAIMED)} claimed families\n")
    print("| claimed model_type | checkpoints here | smallest | reachable without renting |")
    print("|---|---:|---|---|")
    for mt in CLAIMED:
        hits = sorted(claimed_hits.get(mt, []), key=lambda h: h[1])
        if not hits:
            print(f"| `{mt}` | 0 | — | **no — not on the LAN at all** |")
            continue
        name, gb, est = hits[0]
        ok = ("yes" if (est is not None and est <= USABLE_GB) or (est is None and gb <= USABLE_GB)
              else "no — needs a bigger host")
        est_txt = f", ~{est:,.0f} GB loaded" if est is not None else ""
        print(f"| `{mt}` | {len(hits)} | `{name}` ({gb:,.1f} GB{est_txt}) | {ok} |")

    absent = [mt for mt in CLAIMED if not claimed_hits.get(mt)]
    def _needs_big(h):
        return (h[2] if h[2] is not None else h[1]) > USABLE_GB
    big = [mt for mt in CLAIMED
           if claimed_hits.get(mt) and all(_needs_big(h) for h in claimed_hits[mt])]
    print(f"\n**{len(CLAIMED) - len(absent)} of {len(CLAIMED)}** claimed families have a real "
          f"checkpoint on the LAN. Absent: {', '.join(f'`{m}`' for m in absent) or 'none'}. "
          f"Present but too large for a 64 GB probe host: "
          f"{', '.join(f'`{m}`' for m in big) or 'none'}.\n")
    print("So the rentals this lane actually needs are the large families, not bandwidth.\n")
    print("**The loaded-size figures are estimates from config geometry, not measurements.**")
    print("They are computed by `loaded_gb()`; RSS cannot validate them because it counts")
    print("mmap'd checkpoint pages alongside memory actually held. Use them to order the")
    print("queue, not to decide that a rental is unnecessary -- a load attempt is cheap and")
    print("records its own outcome.\n")
    print("One caveat on the table above: **`gemma4_text` has no separate release** and never")
    print("will. It is the text tower of a `gemma4` multimodal config -- what a text-only")
    print("QLoRA loads -- so a `gemma-4-26B-A4B` checkpoint is the evidence for both rows,")
    print("reached by a different config path rather than a different download. Read its")
    print("zero as \"no separate artifact\", not as \"nothing to test\".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
