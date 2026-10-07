### P115 and K33 runners: a GPU the image's torch cannot use is the host floor (exit 18), not a harness error

- `bench/p115/p115_run.sh` and `bench/k33/k33_run.sh` refused a CUDA-unusable host with exit 10 ("DUD BOX"). That code names no
  machine, so the launcher recorded HARNESS_ERROR, and a relaunch could buy the same host. `p115-5090-1` drew Vast machine 34887
  ($0.044), the host that broke TC1's `tc1-5090-119`.
- Both runners now probe as TC1 amendment 61 does:
  - unusable CUDA exits **18** with a REFUSAL line, so the launcher names the machine;
  - a torch that will not import, which is the image's fault, stays 10.
- P115 records this as PREREG amendment 1. No rule, prediction or budget changes. Staged pins are updated, and the
  staged-pin tests pin exactly one 18, in the no-cuda branch.
