### P131 Amendment 2: a warm-up process 0, and the GPU state per process (#846)

The proof `p131-prove-1` PROVED, but its first process was about 4.7 % faster on every arm, which alone trips the NOISY
span guard. Before any reading data, a warm-up process 0 now runs Phase B and a speed pass whose timing is reported only.
It never enters the span guard, the gains, the default's interval or any bar. Processes 1–8 are speed-only. Every
integrity check (FUNCTION, DETERMINISM, engagement, refusals, the decode check) still reads process 0. Every process
records nvidia-smi's SM and memory clocks, power, temperature and clock event reasons at the start and end of its speed
pass, reported only. The reducer's self-test grows to 44 cases. A reading guard of 2.5 h (worst about $2.80) is proposed
from the proof's measured 7-minute process load.
