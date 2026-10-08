### P123 Amendment 2 (#1313): the census class map names torch.cat

- **What happened.** The first proof, `p123-prove-1`, VOIDed: 2.2 % of Granite's B = 1 step went unnamed, against
  SC1b's 2 % gate. All of it was `torch.cat` (`CatArrayBatchedCopy`, two per layer from the unfused `rotate_half`).
- **Why.** SC1b's map lists `cat`, but its name match is case-sensitive and Nsight names the kernel in CamelCase.
- **The fix.** Class map v1.1 names `CatArrayBatchedCopy` beside `cat`; SC1b's census code is unchanged.
- **The tests.** A committed name inventory (the A2000's and the proof's) lets the pin test check that the map names
  every kernel it will meet.
- **Also from that proof.** Amendment 1's router check licensed 32 of 32 on a 5090, confirming #1398 on a real card.
