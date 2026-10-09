### RA pinned checkpoint repository binding

Add a standalone no-site checkpoint gate that reconciles the Hub's complete
model-info/tree indexes with the reviewed materialized input inventory, checks
Git/LFS addresses and safetensors shard coverage, and retains failed authority
records. Repository-index equality does not establish runtime consumption or
proof clearance. Implemented by the Codex desktop release-anchor executor.
