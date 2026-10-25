### Claims register: `evidence_private` names a private receipt the way a public reader may see it

`scripts/check_claims_register.py` (shared with grouped-nf4-gemm, byte-identical) now refuses an `evidence_private`
string that carries an absolute path, a URI, a receipts path under a checkout directory, a 7+ digit id, a UUID or a
public IPv4 address (`docs/claims-schema.md`, Private runs). The house forms pass: `receipts/<repo>/<date>/<run>/...`,
an `INT4B16/` file, or prose, each with an annotation that may name paths relative to the receipt. All 579 values in
this register already pass. `tests/test_check_claims_register.py` carries a passing case for each form and a refusal
mutant for each shape.
