### SC2h registered (#846): P118's decode lookahead at 64 slots (bench and tests only; no default moves)

- **Why.** The 64-slot decode-step census (#1639) ranked the exposed host gap first of four levers: up to 2.47 ms of
  SC2e's 18.4 ms bucket-64 step, most of it the per-stream token dispatch. P118's opt-in lookahead hides that gap and was
  read only at one request, where the gap is about a twelfth as large.
- **The box** (`bench/sc2/sc2h_box_o.sh`, box O): eight cold-start s64a servers on 0.54.0's defaults, `la`
  (`E4B_PAGED_DECODE_LOOKAHEAD=1`) against `base`, two same-seed replicates per arm per draw.
  - SC2f's ladder plus a knee block that SC2f's reading places, by a rule registered now with every branch defined.
  - A LOOKAHEAD gate on each server's start `/health` and step trace.
- **The rule** (`bench/sc2/sc2h_reduce.py`, 38 self-test cases).
  - The bucket-64 step's saving is read against `base`'s own host gap (the ceiling) and the same-arm replicate spread
    (the null).
  - The knee gain is read as SC2f reads it.
  - Outcomes: FASTER (step and knee in both draws; the default-on flip then comes to the maintainer as its own PR),
    STEP_ONLY (licenses nothing), KNEE_SATURATED, FLAT, REGRESSED, and TOKENS_DIFFER (serial texts must be identical).
  - The fallback, not bundled: one event-loop wakeup per step for the token dispatch.
- **Harness.** `sc1_run.sh` / `sc1_drive.sh` gain box O. The driver runs the knee rule on the controller before any
  rental and stages SC2f's verdict. `staged.sha256` pins the three new files.
