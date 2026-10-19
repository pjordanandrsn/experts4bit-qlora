# `p151-5090-3`: a host refusal before the install (rc 17), not a strike

One RTX 5090 (cc 12.0, driver 595.71.05) on an AMD host (16 CPUs): Vast instance 56708659 on machine 138291,
2026-10-19T09:59:31Z to 10:02:28Z. Cost $0.020. Code: e4b `309cd671` (the registration's merge, #1816), grouped-nf4-gemm
`9b824155` (K35's merge).

The box's power floor refused it before anything was installed: the card's power limit read **520 W** (enforced 520 W)
against the registered 575 W floor. No arm ran and no data was seen. Machine 138291 had no recorded power cap before
this run. After it, P151's launches excluded every host without a recorded cap of at least 575 W
(`vast_power_limit_unknown: exclude`), and Amendment 1 (#1819) added spares `-4` to `-6` for refusals before the
install.

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here.
