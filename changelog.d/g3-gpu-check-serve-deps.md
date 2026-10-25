### G3 GPU check: e4b with its serve extra, a web-stack tripwire, and one exit code per cause (after g3-gpu-check-1)

- g3-gpu-check-1 died at the server's start with `No module named 'uvicorn'`. The kit's install came from the
  in-process g3-smoke kit, so it lacked the server's web stack. That run read nothing; it is HARNESS_ERROR, not a G3
  verdict.
- The box now installs `experts4bit-qlora[serve]` at the commit under test, with fastapi and uvicorn pinned at
  `bench/sc2/sc2_box_e.sh`'s versions (inside the extra's ranges). A second tripwire imports the web stack by name, at
  its pins, before the 61 GB fetch, and the main tripwire imports `create_app` and `main`.
- Exit codes are one per cause: FAIL 1, UNREAD 31 (no server state, or never ready), the prompts failed 32. They are
  listed in `CHECK.md`, and none is a launcher host-excluding code.
- `tests/test_g3_gpu_check.py` covers the `[serve]` install and its pins, the web-stack tripwire failing without
  uvicorn or off its pins, each exit code from the run script's own block, and a bench-wide guard: every bench script
  that starts `serve_paged` installs uvicorn, or runs SC2's `install_sc2_client`.
