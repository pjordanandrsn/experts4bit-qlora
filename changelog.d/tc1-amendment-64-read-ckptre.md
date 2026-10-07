### TC1 amendment 64 read: the reentrant checkpoint becomes e4b's default checkpoint (P178, P179, P181-P183 HELD; P180 FALSIFIED)

- Packed rows, torch 2.12 (`tc1-5090-129`, Ryzen 9 5900XT, $1.00): the reentrant checkpoint steps 0.980 of Hugging Face's with an
  identical training peak; the offload's copies cost 1.023 of the reentrant step (P180 FALSIFIED, <= 1.01).
- Field recipe, torch 2.8 (`tc1-5090-128`, EPYC 7K62, $1.47, host-bound at device busy 0.404): the reentrant checkpoint steps 0.838.
- Held-out within 0.004 on both. Register row `e4b.train.ckpt-flavour.default.5090.2026-10-07`, a README section, STATUS, receipts.
