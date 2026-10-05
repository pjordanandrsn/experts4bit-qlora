### DQ3 (#1083): the box refuses a host that will not hand out the subject's memory, at rc 18 before any install (#1171)

- `dq3-5090-1` ($0.058) died on the subject's first allocation. A 2.90 GiB embedding raised CUDA OOM with 30.85 GiB
  free and 0 bytes allocated by PyTorch, on host 564677 (Ryzen 9 9950X, driver 595.84). The same allocation succeeded
  on DQ2's 5090 and in the A2000 rehearsal.
- An arm failure (rc 11) cannot name the machine, so `bench/dq3/dq3_vram_probe.py` now runs first. It asks for
  3.5 GiB, then 2 GiB blocks to 28 GiB, each written. Only its `torch.OutOfMemoryError` exit (3) becomes the
  registered host-floor refusal (rc 18), so a relaunch can exclude that machine. Every other probe failure goes to the
  harness rc 9 and excludes nothing: no CUDA device, an image torch without kernels for the card, or an exception.
- `tests/test_dq3_lane.py` drives the real runner with a fake `nvidia-smi` and `python`:
  - an OOM refusal exits 18 and never reaches the install;
  - a non-OOM probe error exits 9 and names nothing;
  - the link check still comes first;
  - a passing probe proceeds.
