# `p148-5090-1`: refused at create ($0)

The pinned offer, 56417893 on machine 23497 ($0.6111/h), passed the screen (3 of 3 reads over a 40 s hold), the ceiling
check, `--validate` and the same-minute quote at 2026-10-19T06:15:15Z. The provider then refused the create: the offer
had been taken between search and create (`no_such_ask`). No instance existed and nothing was billed. A refusal before
create is not a strike. The launcher's $0 receipt and its ledger row are in the private receipt store. The read ran
under the registered pre-create spare, [`p148-5090-2`](../p148-5090-2/README.md). `SHA256SUMS` covers this file.
