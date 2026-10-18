### Behaviour change (#1732): the B=1 fusion knobs default to `auto` on Mixtral

An unset `E4B_PAGED_FUSE_QKV`, `E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2` or `E4B_FUSE_ROUTER_EPI` now resolves to `auto`
on Mixtral (`mixtral`), as it already does on Qwen3-MoE and Qwen3.5/3.6-MoE. Three folds engage there: the glue fold's
RMSNorms, round 2's decoder layers and attentions, and the router epilogue (no fused q/k/v). Two reads license it:
quality at T == 1 against the family's own floor on every knob (`e4b.serve.fam.fused-stack-t1.mixtral.5090.2026-10-09`),
and one request decoding in 0.9393 and 0.9277 of the step on one RTX 5090
(`e4b.serve.p135.fold-speed.mixtral.5090.2026-10-18`, P135, `bench/p135/RESULTS-p135.md`). Both reads ran on
Mixtral-8x7B. The default keys on `model_type`, so it covers every `mixtral` checkpoint, Mixtral-8x22B included, which
neither read covers. The folds change arithmetic, so greedy text can change on Mixtral. The way back is `0` on each
knob. The prefill glue knobs are unchanged: unset, `E4B_FUSE_PREFILL_GLUE` stays `0` on Mixtral.
