# HP3 (R): e4b against Unsloth 2026.9.14, held-out parity on byte-identical bases, NOT_READ (host-limited)

Pre-registration: [`HP3-PREREG.md`](HP3-PREREG.md) (experts4bit-qlora#1932), with Amendments 1-4 (#2073, #2100, #2111, #2112).
The (R) read is the registered fallback: e4b K00 against Unsloth 2026.9.14 on torch 2.8.0+cu128; axolotl waits for an amendment.
It launched from Amendment 4's merge (`8be4af31`) under the 6.0 h guard. Lane spend: USD 2.369 actual, against USD 6.20 estimated.

- **`hp3r-5090-1` (NOT_READ, host-limited; no strike):** the box's record, its byte-identical files and the list of the files
  kept only in the private receipt store are in [`receipts/hp3r-5090-1/`](receipts/hp3r-5090-1/README.md).

## The verdict

**Every pair reads NOT_READ: host-limited, fewer than 3 seeds fit.** This is the registered outcome of the seed plan
(Amendment 1 part 7): "Fewer than 3 seeds fit: n = 0. The box stops after seed 3410, with no evaluator pass. Every pair reads
NOT_READ, host-limited, with a host refusal note. It is no strike."

- **The instrument gate passed.** G1' reads PASS for the e4b-Unsloth pair: the two bases have the same digest, and every
  zero-adapter row agrees within 1e-05. That makes the pair readable. It is not a parity reading.
- **The seed plan read n = 0.** It reads wall time and the deadline only, never a loss. Seed 3410 took 4,814 s of wall, and three
  seeds need 12,688 s against the 9,072 s left above the reserve.
- **No figure from seed 3410 is read here**, and no evaluator pass ran. There is no parity number, no ordering and no reduced-power
  read.

## What is next

A re-run on the registered host-attributable spare, `hp3r-5090-2`, needs a host rule that this box would have failed. The card
held its SM clock flat near 1,490 MHz with no reason bit, and the host was slow single-threaded. That rule is a reviewed kit edit,
or an amendment if the PREREG's host rule must change. Until a re-run reads, HP3 has no result: "not read: host-limited (fewer
than 3 seeds fit)".
