### Read: TC1 amendment 44 — `E4B_CHUNKED_LM_LOSS=auto` costs the field recipe nothing and its gate never fired (P99–P103 HELD); it becomes e4b's default

- `tc1-5090-97` ($0.79, Threadripper PRO 3955WX, a quiet host): `auto` against the stock loss at the field recipe. Shipped **0.992**
  [0.978, 1.006], matched **0.999** [0.997, 1.001], peak unchanged, held-out within 0.0001; every `auto` arm ran its 240 training forwards
  stock (P99).
- With amendment 43's P98 HELD, amendment 44's rule makes `auto` e4b's default (row
  `e4b.train.chunked-lm-loss.auto.default-decision.5090.2026-10-05`); STATUS says so. The default flip is its own PR.
