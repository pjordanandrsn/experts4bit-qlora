"""Deterministic synthetic safetensors subjects, bounded to one tensor; never pretrained weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SUBJECTS = ("qwen3_14b", "llama31_8b", "qwen3_32b")
CONFIG_HASHES = {
    "qwen3_14b": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a",
    "qwen3_32b": "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb",
    "llama31_8b": "54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd",
}


def config_for(subject, config_dir=None):
    from transformers import AutoConfig, Qwen3Config

    if subject == "tiny":
        # MLP codes exceed e4b's 1 MiB threshold; attention codes do not. Both paths must execute on CUDA.
        return Qwen3Config(hidden_size=1024, intermediate_size=4096, num_hidden_layers=2,
                           num_attention_heads=8, num_key_value_heads=4, head_dim=128, vocab_size=512,
                           max_position_embeddings=4096, tie_word_embeddings=False)
    if subject not in SUBJECTS:
        raise ValueError(f"unregistered subject {subject}")
    directory = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
    file = directory / (subject + ".json")
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_HASHES[subject]:
        raise ValueError("registered configuration checksum differs")
    cfg = json.loads(raw)
    return AutoConfig.for_model(cfg.pop("model_type"), **cfg)


def build(subject, directory, config_dir=None):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    config = config_for(subject, config_dir)
    if subject == "tiny":
        config.save_pretrained(directory)
    else:
        source = (Path(config_dir) if config_dir else Path(__file__).parent / "configs") / (subject + ".json")
        (directory / "config.json").write_bytes(source.read_bytes())
    with init_empty_weights():
        tree = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    total, files = 0, {}
    for i, (name, meta) in enumerate(tree.state_dict().items()):
        seed = int.from_bytes(hashlib.sha256((subject + ":" + name).encode()).digest()[:8], "little") % (2**63 - 1)
        if meta.ndim == 1:
            tensor = torch.ones(meta.shape, dtype=torch.bfloat16) if name.endswith("weight") else torch.zeros(meta.shape, dtype=torch.bfloat16)
        else:
            tensor = torch.randn(meta.shape, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(seed))
            tensor.mul_(config.initializer_range)
        file = directory / f"tensor-{i:05d}.safetensors"
        save_file({name: tensor}, str(file))
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[file.name] = {"tensor": name, "shape": list(meta.shape), "seed": seed, "sha256": digest}
        total += tensor.numel() * tensor.element_size()
        del tensor
    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, **{f"w{i}": i for i in range(3, config.vocab_size)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.save_pretrained(directory)
    manifest = {"schema": "dq7-subject/1", "subject": subject, "synthetic": True, "pretrained": False,
                "checkpoint_bytes": total, "config_source_sha256": CONFIG_HASHES.get(subject), "files": files}
    (directory / "dq7-subject.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=(*SUBJECTS, "tiny"))
    parser.add_argument("directory")
    parser.add_argument("--config-dir")
    args = parser.parse_args()
    print(json.dumps(build(args.subject, args.directory, args.config_dir)))
