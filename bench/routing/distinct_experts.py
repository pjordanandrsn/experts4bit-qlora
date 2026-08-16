"""How many DISTINCT experts does a real batch route to?

The question decides whether a *compacted* staged stack (`[R, ...]` plus an id
remap) is worth the kernel-contract change `engines/nvme_train.py` declines to
make. Compaction only saves memory when a batch touches far fewer than `E`
experts; if a few hundred tokens touch nearly all of them, the full `[E, ...]`
shape costs a training batch nothing.

Measured **exactly**, with no model forward and no 149 GB download, by exploiting
a property of DeepSeek-V4: its first `num_hash_layers` MoE layers use
`DeepseekV4HashRouter`, where *which* experts a token selects is a frozen
`tid2eid[input_ids]` lookup. Only ~19 MB of tensors are needed.

Scope, stated because it bounds the claim: this covers the hash-routed layers
only (3 of 43 on V4-Flash). The remaining layers use the learned top-k router and
cannot be measured without a real forward.

Beware the quantity: this is NOT the routing skew that informed hot sets exploit.
Those care which experts are hit *often*; this cares which are hit *at all*. Heavy
frequency-skew still leaves the tail touched.

    python bench/routing/distinct_experts.py
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

REPO = "deepseek-ai/DeepSeek-V4-Flash"
CORPUS = "https://www.gutenberg.org/files/2701/2701-0.txt"      # Moby-Dick
BYTES_PER_EXPERT = 13_369_344          # measured from the real baked arena
MIN_UNIQUE_IDS = 2_000
BATCHES = (("1 x 128", 128), ("1 x 256", 256), ("2 x 256", 512),
           ("4 x 256", 1024), ("decode", 1))


def load_tid2eid(layer: int = 0) -> np.ndarray:
    """The frozen token-id -> expert-id table for one hash-routed layer."""
    index_path = hf_hub_download(REPO, "model.safetensors.index.json")
    key = f"layers.{layer}.ffn.gate.tid2eid"
    shard = json.load(open(index_path))["weight_map"][key]
    with safe_open(hf_hub_download(REPO, shard), framework="np") as f:
        return f.get_tensor(key)


def real_token_ids() -> np.ndarray:
    """Token ids from REAL prose.

    Token diversity is the entire input to this measurement, so a synthetic
    corpus invalidates it. The first version of this probe used hand-written
    repeated sentences — 75 unique ids in 3,640 tokens — and reported "76% saved"
    at 128 tokens, which described the text and not the router. Hence the refusal
    below rather than a quiet fallback.
    """
    raw = urllib.request.urlopen(CORPUS, timeout=60).read().decode("utf-8", "replace")
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    ids = np.array(tok(raw[20_000:400_000])["input_ids"], dtype=np.int64)
    unique = len(np.unique(ids))
    print(f"corpus: {len(ids):,} tokens, {unique:,} unique ids "
          f"({100 * unique / len(ids):.1f}% type/token)")
    if unique < MIN_UNIQUE_IDS:
        raise SystemExit(
            f"corpus has only {unique} unique token ids; that biases the distinct "
            "count downward, which IS the result. Refusing to report it.")
    return ids


def main() -> None:
    tid2eid = load_tid2eid()
    experts = int(tid2eid.max()) + 1
    ids = real_token_ids()
    full_gib = experts * BYTES_PER_EXPERT / 2 ** 30
    print(f"\ntid2eid {tid2eid.shape}, {experts} experts, top-{tid2eid.shape[1]}")
    print(f"{'batch':<10}{'tokens':>7}{'uniq ids':>10}{'distinct':>10}"
          f"{'% of E':>9}{'GiB/layer':>11}{'saved':>8}")

    for label, tokens in BATCHES:
        # Several offsets so one passage cannot dominate the answer.
        offsets = [0, len(ids) // 4, len(ids) // 2, 3 * len(ids) // 4]
        distinct, uniq = [], []
        for off in offsets:
            seg = ids[off:off + tokens]
            if len(seg) < tokens:
                seg = ids[:tokens]
            distinct.append(len(np.unique(tid2eid[seg])))
            uniq.append(len(np.unique(seg)))
        d, u = int(np.mean(distinct)), int(np.mean(uniq))
        gib = d * BYTES_PER_EXPERT / 2 ** 30
        print(f"{label:<10}{tokens:>7}{u:>10}{d:>10}{100 * d / experts:>8.0f}%"
              f"{gib:>11.2f}{100 * (1 - d / experts):>7.0f}%")

    print(f"\nfull [E, ...] blocks-only stack: {full_gib:.2f} GiB/layer")
    print("Hash-routed layers only (num_hash_layers=3 of 43); the learned top-k "
          "layers need a real forward.")


if __name__ == "__main__":
    main()
