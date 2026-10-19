# `p152-5090-1`: refused at the power floor (rc 17, not a strike)

One RTX 5090 (cc 12.0, driver 580.119.02) on an AMD EPYC 9655 host (48 CPUs, 108 GiB RAM), 2026-10-19T16:25Z. Cost
$0.015.

The host registry's latest record for this machine was a 575 W power cap read on 2026-09-24. On the box the card read
**475 W**, below the registered 575 W floor, so the runner refused before installing anything (`REFUSAL`:
`refused: power.limit 475 W`, exit code 17). By the registration this is a power-floor refusal, not a strike. It spent
one of the lane's three run ids.

What changed for the next pick: the pin screen was meant to take only hosts whose latest recorded cap of at least 575 W
was read within the last three days, freshest first, which only narrows the registered host set. For the host
`p152-5090-2` then took, that "fresh" record was a misread: a 575 W value from the kit's frozen calibration file, which
ships in every run directory and is not a reading from any box. The misread was removed from the screen the same
afternoon. The reading is unaffected: `p152-5090-2` read 575 W on its own box before the install.

The launcher's receipt and ledger row are in the private receipt store, and the teardown is proven there. `SHA256SUMS`
covers every file here. The run's instance file and the launcher's teardown proof name the provider's instance, so they
stay only in that store, byte-exact, listed in `STORE-ONLY.sha256` by their path within the run's receipt.
