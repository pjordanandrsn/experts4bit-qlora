### Read: TC1c amendment 9 — the same-stack H100 box reads UNTESTED (e4b's draws 18 % and 29 % apart); 1.061 stays the H100 position

- `tc1c-h100-22`, the first complete lane on RunPod Secure (H100 NVL, Xeon 8452Y, 18 vCPUs allotted, $3.19/h against a registered $2.80
  ceiling, on the owner's go). e4b 2.968 / 3.557 and 3.428 / 2.568 s/step, Unsloth 3.127 / 3.021: P27 and P28 UNTESTED, P29 HELD.
  Nothing is quoted.
- Candidate cause, recorded, not read: the lane ran 72 OpenMP threads (the host's physical cores) in an 18-vCPU pod, and its load gate
  read the host's load (7.5–42). The launcher booked $3.48 from partial billing; the box cost about $6.78 (adertha-agents #174).
