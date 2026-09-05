# Writing release notes

The first paragraph of a release note is for the person deciding whether to
upgrade. It says, in ordinary language and in this order:

1. **which problem changed** — what a user can now do, or what stopped being
   wrong (not which function was added);
2. **who is affected** — the model families, hardware and environments the
   change reaches, and the ones it does not;
3. **whether to upgrade** — "upgrade if …", "no action if …", and any floor
   on the related package.

Everything the project already does well follows unchanged after that
paragraph: the mechanism, the measurements with their receipts and tiers,
the corrections and retractions, the caveats and refused arms. A number in a
release note is a quote of an entry in `docs/claims.json`; the entry, not the
note, decides whether it is still current.

Historical release notes are not rewritten to this shape. Releases are cut
from `main` by the maintainer; do not tag or publish from a branch.

The README is written against `main` and links only `main`; the one place
it names a release is the generated block between
`<!-- release-block:start -->` and `<!-- release-block:end -->`. The release
recipe is therefore: add the `## <version> — <date>` section here in
`CHANGELOG.md` and bump `pyproject.toml` (and `__version__`) in the same
change, then run `python scripts/check_readme_claims.py --write-release-block`
— the block's version is derived from that heading and cross-checked
against `pyproject.toml`, never typed; `scripts/check_readme_claims.py` in CI
fails a README whose block is stale or hand-edited, and
`scripts/check_readme_links.py` holds the tag it pins to the same version.

Example opening:

> **0.34.0.** Single-stream decode of Qwen3-30B-A3B on the calibrated int4
> stack is faster because the round-2 norm+rotary fold now engages on the
> separate-projection attention that stack actually runs; nothing changes for
> the fused-qkv path or for training. Affects serving on Qwen3-MoE-shaped
> attention (q/k/v/o with per-head norms) on sm_80+ NVIDIA GPUs under Linux;
> Granite and Mixtral get the norm-less variant of the same fold. Upgrade if
> you serve with `E4B_FUSE_T1_GLUE_R2=1`; requires grouped-nf4-gemm ≥ 0.28.0.
