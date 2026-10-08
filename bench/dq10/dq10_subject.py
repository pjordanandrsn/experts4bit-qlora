"""New-family synthetic subjects, one tensor at a time, preserving tied aliases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CONFIG_HASHES = {
    "mistral7b_v03": "affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99",
    "smollm3_3b": "c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1",
}


def config_for(subject, config_dir=None):
    from transformers import AutoConfig, MistralConfig, SmolLM3Config

    if subject == "tiny-smollm3":
        return SmolLM3Config(hidden_size=1024, intermediate_size=4096, num_hidden_layers=2,
                             num_attention_heads=8, num_key_value_heads=4, vocab_size=512,
                             max_position_embeddings=4096, tie_word_embeddings=True,
                             bos_token_id=1, eos_token_id=1, pad_token_id=2,
                             no_rope_layers=[1, 1], layer_types=["full_attention"] * 2)
    if subject == "tiny-mistral":
        return MistralConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=2,
                             num_attention_heads=8, num_key_value_heads=4, vocab_size=512,
                             max_position_embeddings=4096, tie_word_embeddings=False,
                             bos_token_id=1, eos_token_id=1, pad_token_id=2)
    if subject not in CONFIG_HASHES:
        raise ValueError(f"unregistered subject {subject}")
    root = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
    raw = (root / (subject + ".json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_HASHES[subject]:
        raise ValueError("registered configuration checksum differs")
    cfg = json.loads(raw)
    return AutoConfig.for_model(cfg.pop("model_type"), **cfg)


def write_tensors(config, subject, directory):
    """Parameter object identity on meta determines seed aliases; meta data_ptr is not an identity."""
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        tree = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    # Accelerate's parameter registration can break aliases inside its context. Re-tie after leaving it.
    tree.tie_weights()
    canonical, aliases = {}, {}
    for name, parameter in tree.named_parameters(remove_duplicate=False):
        source = canonical.setdefault(id(parameter), name)
        if name != source:
            aliases[name] = source
    total, files = 0, {}
    for i, (name, meta) in enumerate(tree.state_dict().items()):
        source = aliases.get(name, name)
        seed = int.from_bytes(hashlib.sha256((subject + ":" + source).encode()).digest()[:8], "little") % (2**63 - 1)
        if meta.ndim == 1:
            tensor = (torch.ones if source.endswith("weight") else torch.zeros)(meta.shape, dtype=torch.bfloat16)
        else:
            tensor = torch.randn(meta.shape, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(seed))
            tensor.mul_(config.initializer_range)
        file = Path(directory) / f"tensor-{i:05d}.safetensors"
        save_file({name: tensor}, str(file))
        sha = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                sha.update(chunk)
        files[file.name] = {"tensor": name, "shape": list(meta.shape), "seed": seed,
                            "seed_source": source, "sha256": sha.hexdigest()}
        total += tensor.numel() * tensor.element_size()
        del tensor
    return {"schema": "dq10-subject/1", "subject": subject, "synthetic": True, "pretrained": False,
            "checkpoint_bytes": total, "config_source_sha256": CONFIG_HASHES.get(subject),
            "tied_aliases": aliases, "files": files}


def build(subject, directory, config_dir=None):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    config = config_for(subject, config_dir)
    if subject.startswith("tiny-"):
        config.save_pretrained(directory)
    else:
        source = (Path(config_dir) if config_dir else Path(__file__).parent / "configs") / (subject + ".json")
        (directory / "config.json").write_bytes(source.read_bytes())
    manifest = write_tensors(config, subject, directory)
    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, **{f"w{i}": i for i in range(3, config.vocab_size)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.save_pretrained(directory)
    (directory / "dq10-subject.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=(*CONFIG_HASHES, "tiny-smollm3", "tiny-mistral"))
    parser.add_argument("directory")
    parser.add_argument("--config-dir")
    args = parser.parse_args()
    print(json.dumps(build(args.subject, args.directory, args.config_dir)))
