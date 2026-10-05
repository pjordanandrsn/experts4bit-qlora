### SC2c amendment A1 (#846): box H pins torch 2.8.0 like boxes C–I (bench and tests only)

- When box I merged beside box H (#1140, #1132), box H was left out of `sc1_run.sh`'s `torch==2.8.0` pin.
  `sc2c-prove-2` installed torch 2.14.1+cu130 / triton 3.8.0 instead of SC2b's and SC2g's 2.8.0+cu128 / 3.4.0.
- A1 adds H to the pin; a test asserts every python3 box (C–I) carries it. The reading runs after a proof on the fixed
  harness. Design, rule, predictions and guards unchanged.
