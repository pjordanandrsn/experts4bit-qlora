### SC1g A7 read (#846): FRAGILE_POSITIONS -- e4b's worst positions are router-flip positions in both arms, so the rule cannot attribute (a)'s excess to them (bench and tests only)

- **The run.** `sc1g-diag-a7-4` (box J, $0.701, Vast 145701) captured the decode GEMV's expert ids per position for
  (a) `KEEP_NF4=0` and (b), on all four windows. The perturbation control is bit-identical, every capture passes its
  gates, and layer 0's flip rate is exactly 0, as it must be.
- **Reading (conv2).** (a)'s top-1 % KL positions all flip (21/21 against 0.706, p 6.4e-4), but so do (b)'s own. By
  precedence that is FRAGILE_POSITIONS, and router-flip attribution is set aside.
- **Descriptive.** conv2's flips sit in the first quarter of decode, where (a)'s excess is (5.8 flipped layers per
  position, against about 1.3–2.4 later). (a)'s and (b)'s worst positions overlap on only 3 of 21.
- **Post-data, flagged.** The control compares (a) with (b), not each arm with the reference, so it cannot separate
  "positions fragile in both arms" from "each arm's tail sits where its own routing left the reference's". The next
  instrument is the reference's per-position routing.
- **Correction.** The A6 continuation read blamed busy neighbours for its slow conv1 arms. This box ran on the same host
  at higher load at normal speed, so the cause is unknown.
- **Launch history:** one $0.007 pre-flight NOT_RUN and two $0 refusals over my exclusion entry. **Spend:** the lane is
  at $10.843.
