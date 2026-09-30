# P84 amendment 1 rehearsed on the home A2000 (2026-09-30, not a reading)

The amended runner (the CPU-vendor refusal), `P84_PROVE=1`, in throwaway `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`
containers on the QNAP's RTX A2000. That host's CPU is an Intel Xeon W-1250 (`GenuineIntel`), which makes it a natural
test of the refusal.

| arm | `P84_CPU_VENDOR` | rc | result |
|---|---|---:|---|
| V0 | unset (the registered `AuthenticAMD`) | **16** | `REFUSED: host CPU vendor is 'GenuineIntel', the lane registers AuthenticAMD (amendment 1)`. The refusal comes after the staged-pin check and the card refusals, and before the install and the fetch. |
| V1 | `GenuineIntel` (the knob lifted) | **0** | PROVED. The summary says REHEARSAL. Both stacks installed and tripwired, and both stamps are fp32. |
