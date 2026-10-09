### P125 Amendment 2 (#1313): the gate record keeps the box's own window count

- **What happened.** The second proof, `p125-prove-2`, VOIDed on a record-shape bug: the quality instrument's own
  `windows` field, merged last, overwrote each gate's count.
- **The fix.** The box now builds every gate record through `gate_record`, with its own keys on top. A pin test composes
  a record from the instrument's real return keys and fails on the old merge order.
- **What the records showed.** Re-reduced with the count restored, the proof read READ: the int4 head builds with #1426,
  the calibration is deterministic, and the gate numbers reproduced identically on two boxes.
- **Next.** `p125-prove-3` proves the fixed box end to end.
