"""Build DQ11 canonical random adapters and token rows on CPU, before readings.

Downloads only public config/tokenizer/data metadata and files, never base weights.
Use the recorded builder environment. It writes a new directory, never overwrites
an earlier seal. Actual training workers copy these bytes, not a reseeded guess.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

MODEL = "mistralai/Mistral-7B-v0.1"
REV = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
ALPACA_REV = "12567cabf869d7c92e573c7c783905fc160e9639"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
PROMPT = ("Below is an instruction that describes a task, paired with an input that provides further context. "
          "Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packed(tokenizer, texts, count, *, separator=""):
    ids = tokenizer.encode(separator.join(texts), add_special_tokens=False).ids
    if len(ids) < count * 2048:
        raise ValueError("registered split has insufficient tokens; no repeat/truncation fallback")
    return [ids[i * 2048:(i + 1) * 2048] for i in range(count)]


def build(directory):
    import importlib.metadata as md
    from peft.tuners.lora.layer import LoraLayer
    import pyarrow.parquet as pq
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    import torch

    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    sources = directory / "source"
    sources.mkdir()
    files = {}

    def fetch(url, name):
        path = sources / name
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
        files[name] = {"url": url, "sha256": sha(path), "bytes": path.stat().st_size}
        return path

    config = json.loads(fetch(f"https://huggingface.co/{MODEL}/resolve/{REV}/config.json", "config.json").read_text())
    tokenizer_file = fetch(f"https://huggingface.co/{MODEL}/resolve/{REV}/tokenizer.json", "tokenizer.json")
    fetch(f"https://huggingface.co/{MODEL}/resolve/{REV}/tokenizer_config.json", "tokenizer_config.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    assert tokenizer.token_to_id("</s>") == 2
    rows = json.loads(fetch(f"https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/{ALPACA_REV}/alpaca_data_cleaned.json",
                            "alpaca_data_cleaned.json").read_text())
    wiki_file = fetch(f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REV}/wikitext-2-raw-v1/test-00000-of-00001.parquet",
                      "wikitext-test.parquet")
    wiki = pq.read_table(wiki_file)["text"].to_pylist()
    def render(rows):
        return [PROMPT.format(row["instruction"], row["input"], row["output"]) + "</s>" for row in rows]
    blocks = {"train": packed(tokenizer, render(rows[:-512]), 40),
              "alpaca-heldout": packed(tokenizer, render(rows[-512:]), 8),
              "wikitext-test": packed(tokenizer, wiki, 8, separator="\n")}
    tokens = directory / "tokens.json"
    tokens.write_text(json.dumps(blocks, separators=(",", ":")) + "\n")
    torch.manual_seed(3407)
    tensors = {}
    h, inter = config["hidden_size"], config["intermediate_size"]
    kv = config["num_key_value_heads"] * (h // config["num_attention_heads"])
    projections = (("self_attn", "q_proj", h, h), ("self_attn", "k_proj", h, kv),
                   ("self_attn", "v_proj", h, kv), ("self_attn", "o_proj", h, h),
                   ("mlp", "gate_proj", h, inter), ("mlp", "up_proj", h, inter),
                   ("mlp", "down_proj", inter, h))
    for layer in range(config["num_hidden_layers"]):
        for group, name, dim_in, dim_out in projections:
            a, b = torch.empty((16, dim_in)), torch.empty((dim_out, 16))
            holder = SimpleNamespace(lora_A={"default": SimpleNamespace(weight=a)},
                                     lora_B={"default": SimpleNamespace(weight=b)},
                                     lora_bias={"default": False}, lora_embedding_A={}, lora_embedding_B={})
            LoraLayer.reset_lora_parameters(holder, "default", True)
            key = f"layers.{layer}.{group}.{name}"
            tensors[key + ".A"] = a
            tensors[key + ".B"] = b
    adapter = directory / "adapter_init.safetensors"
    save_file(tensors, str(adapter))
    manifest = {"schema": "dq11-inputs/1", "model": MODEL, "revision": REV,
                "builder": {name: md.version(name) for name in ("torch", "peft", "transformers", "tokenizers", "safetensors", "pyarrow")},
                "initializer_source_sha256": sha(inspect.getsourcefile(LoraLayer.reset_lora_parameters)),
                "sources": files, "alpaca_rows": len(rows), "holdout_rows": 512,
                "adapter_tensor_count": len(tensors), "adapter_elements": sum(t.numel() for t in tensors.values()),
                "assets": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in (tokens, adapter)},
                "tokens": {name: {"blocks": len(value), "targets": len(value) * 2047,
                                  "sha256": hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()}
                           for name, value in blocks.items()}}
    (directory / "locked_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.out), indent=2))
