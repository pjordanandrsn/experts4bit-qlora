# `p131-5090-1`: a pre-flight stall (NOT_RUN, not a strike)

Vast instance redacted-f34fa66c on machine 154241 (an RTX 5090 at $0.8444/h), 2026-10-19 05:55Z to 06:05Z. Cost $0.027. The
instance stayed `created` (stuck loading) for 600 s, so the controller's pre-flight failed, the command never ran,
and the box was destroyed (`teardown-proof.json`, reason `preflight-failed`). The maintainer ruled it a host-side
stall, the same class as DQ11's machine 147454: not a strike, and not the reading's one created run. The machine is
curated CAUTION in the host registry and was excluded from the reading's later pins. The reading ran as
[`p131-5090-2`](../p131-5090-2/README.md).

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here.
