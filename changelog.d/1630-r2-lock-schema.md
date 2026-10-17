### Fixed

- R2-Mixtral's checkpoint lock (`bench/fam235/model_files.json`, #1630) now names its format, `q235-model-files/1`. Q235's fetch,
  which R2 runs unchanged, refused the R2-named schema, and the first A2000 leg VOIDed at its fetch (0 USD). A test now runs
  Q235's own fetch check on R2's lock.
