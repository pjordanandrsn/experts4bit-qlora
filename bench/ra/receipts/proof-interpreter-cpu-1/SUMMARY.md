# RA selected-interpreter CPU controls

The official image manifest, config, base layer and complete compressed
interpreter layer were verified by SHA256. The selected CPython 3.11.13
binary and 2,148 retained runtime files/links were checked after transfer.
Running those exact interpreter/stdlib bytes on a native Linux CPU host matched
all 16 canonical AST bindings and rejected all 16 body mutations. The host
uses a different libc from the image; this is interpreter compatibility, not
a full-image boot/install or on-card premise. No GPU battery or rental ran.

The AST input retains the reviewed helper and source bodies. Registries record
the Git sources; result digests identify each match. To reproduce, copy these
inputs into an empty scratch directory before running the extraction helper,
then run the AST check with the selected Linux interpreter and the input JSON
as its argument. The runtime extraction inventory is bound to the official OCI
layer, and the helper's complete compressed-stream digest must pass first.

The wheel check reports 87 retained archives and 85 common dependencies. It
proves neither installation nor dependency closure; the setuptools startup
hook remains a reviewed-adapter gate. No release clearance is claimed.
