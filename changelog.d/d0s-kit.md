### D0s kit: the served read's box runner, its controller driver, the REF argmax tool and the reducer

- `bench/d0s/d0s_ref.py` (CPU, USD 0, run before any rental) computes REF's and Q's per-position argmax on D0's 64
  held-out rows. It uses D0's own pinned forward, checked against its SHA256SUMS before import. Every row's NLL must
  reproduce D0's `arms.csv` to a relative 1e-6, or the output says VOID and the read does not rent.
- `bench/d0s/d0s_box.py` and `d0s_run.sh` run on the box from a clean checkout of the launch commit, its kit files
  checked against `staged.sha256`:
  - the refusals and the version tripwire come first, then the premise (`tests/test_merged_arena_gpu.py`, 2 passed
    with no skips);
  - each arm is one subprocess: the base arena baked on the card, the attention-only adapter derived from each
    fixture's /2, and the merged-arena writer for MERGED-Q and MERGED-W (a refusal is recorded as WRITER_REFUSED);
  - each arm's pass is a teacher-forced pass on the paged runner, then W1/W16 cost.
  `D0S_SHAKE=1` is the shakedown: a tiny model, no checkpoint, no adapters.
- `bench/d0s/d0s_drive.sh` is the launcher's `--command`. It refuses unless every staged pin equals the tree and, for
  the reading, unless both derived adapters are present and hold no links. It starts the lane under a fresh nonce and
  fetches with `rsync --no-links` (rule 8), never the checkpoint, the arenas or the adapters' weights.
- `bench/d0s/d0s_reduce.py` (CPU, standard library) applies the registration's rule to the arm records: retention CI,
  the registered absolute margin, the pooled canary and the cost band. It names WRITER_REFUSED, NOT_READ and VOID.
- `tests/test_d0s_kit.py` (CPU) covers:
  - the reducer at each gate's boundary;
  - the REF walk on a tiny OLMoE-shaped checkpoint (REF equals Q without an expert delta, and differs with one);
  - the reproduction check;
  - the attention-only derivation;
  - the staged pins;
  - the driver's refusal and dry run.
