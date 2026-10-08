# Python 3.11 native AST fields, CPU only

Serving CI found a test assumption: Python 3.11 FunctionDef._fields omits
type_params, so an attached attribute correctly leaves the native AST digest
unchanged. The test now asserts that behavior explicitly; the serializer is
unchanged. The exact selected OCI-extracted CPython 3.11.13 on native Linux
matches all 16 registered fingerprints, rejects all 16 native-body mutants
and passes the empty-optional and non-field contracts. The native host libc
differs from the pinned image; this is no full-image installation or GPU proof.
