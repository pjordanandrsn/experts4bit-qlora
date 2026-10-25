# `p151-5090-1`: a host refusal before the install (rc 17), not a strike

One RTX 5090 (cc 12.0, driver 570.133.20) on an AMD host (32 CPUs): Vast instance redacted-c6926ac6 on machine 43578,
2026-10-19T09:42:36Z to 09:44:47Z. Cost $0.015. Code: e4b `309cd671` (the registration's merge, #1816), grouped-nf4-gemm
`9b824155` (K35's merge).

The box's power floor refused it before anything was installed: the card's power limit read **520 W** (enforced 520 W)
against the registered 575 W floor (`REFUSAL`; `summary.txt`, `POWER_LIMIT_W 520 (floor 575)`). No arm ran and no data
was seen. Machine 43578 had no recorded power cap before this run, so the launcher's registry screen could not exclude
it; the run's own receipt now records the cap there.

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here.
