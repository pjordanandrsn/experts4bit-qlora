# Checkpoint repository binding

`ra_checkpoint.py` runs under selected `python -I -S -B`, without release imports.
Its schema-1 manifest names the input spec, staged instruments, reviewed input
lock and exact lock SHA-256. Model/revision come from the fixed adjacent RA pins.

Before and after hashing the complete materialized checkpoint tree, it retains
bounded, clock-read Hugging Face model-info and recursive-tree JSON. Both lists
must agree on every path, Git blob address, size and LFS identity. Pagination,
redirects, proxies, duplicate JSON fields and unknown LFS pointer conventions
refuse. Ordinary files bind to Git blob addresses; LFS payloads bind to SHA-256
and size, and their canonical pointers are independently Git-addressed. The
local inventory must include **every** repository file, including documentation,
with no cache files, symlinks or extras. The safetensors index must name every
materialized weight shard. Input, stage, helper and authority drift refuse.

This establishes equality with the Hub's reported pinned repository index. It
is not publisher signature verification, dataset/calibration authority,
tokenizer execution, loader consumption, GPU evidence or launch clearance. No
model download or installation is performed. ABBA wiring remains a separate
component; the fixed worker tool set does not yet include this proposed gate.

API shape: [Hugging Face model-info and repository-tree reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api).
