# Common input gate (draft)

`ra_inputs.py` inventories every regular file in three materialized input
roots: `checkpoint` (including tokenizer files), `datasets`, and
`calibration_inputs`. Roots cannot overlap or traverse symlinks. Prepared
files live separately. Nothing is silently ignored. Arenas and caches are
release outputs, outside these roots; each release bakes separately.

The absolute-path spec has exactly `schema: 1`, `battery: proof|reading`,
`trees` (the three role-to-directory paths), and `files` with five paths:
`train_data`, `train_tokens`, `corpus_tokens`, `decode_prompts`,
`quality_windows`. It uses the adjacent registered source pins and a verified
instrument stage. Materialize Hugging Face snapshot links before inventory.

The extra retained corpus file has `model`, `revision`,
`dataset: wikitext-2-raw-v1:test`, and flat nonnegative integer `ids`.
It supplies the token stream against which P109's 16 slices and P115's 12
windows are checked. TC1 keeps its whole token-file hash separate from its
compact train/eval hash. Decode uses native compact JSON hashes; quality
uses native default JSON spacing. The registered SC2 function derives all
188 warm/burst/point arrivals, prompt indices and output lengths.

`candidate --spec SPEC --stage STAGE --lock LOCK --out LOCK` writes a new
candidate inventory. Review its completeness and source evidence, then bind
the exact lock SHA in the launch manifest. `verify` requires that SHA and
recomputes every inventory and projection, checking again after parsing.
Paths are excluded from identity, so identical files at distinct release
roots compare equal. A resealed replacement lock cannot pass the old SHA.
Neither candidate creation nor a successful check authorizes a launch.

After verification, `bind_native` compares each retained native receipt to
the projection, alongside the component's existing checks. SC2 has no native
revision field: its retained server evidence `config.model/revision` is
required as well as the driver's model, prompts and plan. Keep raw records.
Supervisor wiring, loader-open evidence and pre/post-process verification
remain to be implemented.

This gate proves declared inventory bytes and fixture projections only.
An inventory does not establish authoritative checkpoint completeness,
dataset origin, tokenizer regeneration or runtime consumption. The proof
must independently establish those facts, actual installed imports/defaults,
GPU engagement and cleanup. Synthetic CPU controls grant no proof clearance.
