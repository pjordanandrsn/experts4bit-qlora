# Security policy

## Reporting

**Do not open a public issue.** Either private channel:

- [GitHub private advisory](https://github.com/pjordanandrsn/experts4bit-qlora/security/advisories/new)
- `security@cerinamroth.com` — encrypt to the
  [published key](https://cerinamroth.com/.well-known/cerinamroth-pubkey.asc)
  (`gpg --locate-keys security@cerinamroth.com`)

Canonical contact: [`security.txt`](https://cerinamroth.com/.well-known/security.txt).
The [disclosure policy](https://cerinamroth.com/policy/) applies: good-faith
research, coordinated disclosure, no publication before a reasonable remediation
window.

## In scope

This package loads other people's checkpoints and reads on-disk arenas, so the
surface is **what it parses and what it trusts** — it has no network boundary.

- **Checkpoint and arena parsing.** Safetensors headers, manifests, offsets,
  lengths, keymaps. A crafted file causing an out-of-bounds read, a wild offset,
  or memory disclosure is in scope.
- **Provenance.** The package asserts that loaded expert bytes are the released
  bytes. Making that check pass on bytes that do not match is the
  highest-severity class here — it defeats the guarantee rather than crashing.
- **Placement and residency.** A path that reads one expert's rows while
  reporting another's is a correctness bug with security consequences for anyone
  relying on provenance.
- **Deserialization** of anything that becomes Python objects or tensor metadata.

## Not in scope

- Quantization changing numbers. That is a documented tradeoff; fidelity
  disputes belong in an issue.
- Resource exhaustion from parameters you chose (`hot_rows` above your RAM, an
  arena larger than your disk). These raise deliberately.
- Vulnerabilities in torch, triton, bitsandbytes or NVIDIA drivers — report
  upstream, though a heads-up is welcome.

## What to expect

Acknowledgement when seen — one maintainer, best-effort, not an SLA. Then
confirmation or a reasoned disagreement, a fix, a release, and credit unless you
prefer otherwise. **Honest severity applies in both directions**: a
well-described medium is more useful than an inflated critical, and disagreement
about severity is welcome.
