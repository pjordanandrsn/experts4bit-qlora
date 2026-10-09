# Fixed tokenizer recipe foundation

`ra_tokenizer_recipe.py` reproduces the registered unpacked TC1 Alpaca
preparation and full-text WikiText tokenization followed by RA's fixed decode
and quality windows. It accepts a tokenizer object and returns complete
projections; it never loads a tokenizer, dataset, package or file.

Training renders all 1,200 train and 48 eval examples with the registered
Alpaca prompt and tokenizer EOS, uses truncation at 2,048 tokens, drops rows
shorter than eight tokens, and takes the first eight retained eval rows.
The returned metadata and compact train/eval payload hash match TC1's
unpacked preparation. WikiText filters rows with `strip()`, joins the original
retained text with two newlines, tokenizes the complete text with
`return_tensors="pt"`, and returns the full IDs plus every fixed window.

Focused CPU controls use explicit tokenizer and tensor doubles. They compare
against selected functions from complete registered source fixtures, checked
offline against the registered SHA256 and pinned Git blob IDs. The tests need
neither Git history nor network access and do not import the complete
instruments or any release. Those controls do not
authenticate a caller tokenizer, model revision, dataset label, or tensor.

This foundation is an unreviewed proposal. It adds no CLI, worker inventory
entry, ABBA probe, import permission or launch path. A future separately
reviewed caller must bind the source recipe, complete input bytes, checkpoint
tokenizer assets/configuration, selected interpreter and installed runtime,
then retain guarded regeneration and independent native consumption evidence.
The TC1 weight-loader revision gap, placement calibration reads and optional
calibration no-call evidence remain separate requirements. No dependency or
proof-image installation is authorized by this helper.
