# The W8A16 quality premise (#1313): does int8 weight-only rounding keep P125's quality gate?

**A premise, not a registered reading.** It decides only whether the int8 kernel work starts. Its numbers are disclosed
verbatim as seen data in any later registration, and that registration's box re-measures quality itself. They are never
quoted as a result.

## Why

At B = 1 on one RTX 5090, the dense GEMM class is the largest single-stream class: 1.81 ms of the step (P123's census;
P138's census n = 1 reproduces it). On the default it is bf16 on cuBLAS: the attention projections (fused q/k/v and
o) and the lm_head, plus the router. P125 tried int4 for the attention projections and the head. It was 1.15× and
1.24× faster at one request, but **not licensed**: argmax agreement was 0.929 and 0.918 against the 0.95 gate. No lane
has tried int8 weight-only (W8A16) for them. It halves the bytes these layers stream per token, and int8's rounding
step is 16× finer than int4's. Scaled from P125's measured int4 savings, the speed ceiling is about 1.12× at one
request; the bandwidth bound is about 1.23×. Quality is the cheapest way this lever can die, so it is tested first.

## The premise

**The subject is P125's, exactly:**
- e4b `4934aba9` (0.50.0, the commit P125's box installed) with grouped-nf4-gemm 0.43.0 (`6ee2e104`);
- transformers 5.17.0 and bitsandbytes 0.50.2;
- Qwen3-30B-A3B at `ad44e777`;
- the default server built eager at all-VRAM, with every int4 lever and calibration knob unset.

**The instrument is P125's, verbatim.** `p125_box.py`, `p125_reduce.py`, Phase D's `p115_quality.py` and the boxes it
imports are staged at P125's pinned bytes (tested equal to `staged-p125.sha256`). It has the same gates and windows:
- `t1`: wikitext at one window per pass, 108 windows;
- `k16`: wikitext in 16-row pieces, 112 windows;
- `c4`: c4val1, 16 windows, reported only.

**The arms** (`i8_box.py`, one process each):
- **A:** the default, exactly P125's A. It writes R's log-probs.
- **I8:** A's build, then `i8_fakequant.py` rounds in place every attention projection (48 fused q/k/v and 48 o) and
  the lm_head to its per-output-row symmetric int8 round trip: scale = row max / 127, round, clamp to ±127, dequantize.
  It is scored ON against A.
- The modules stay bf16 `nn.Linear` on the same cuBLAS path, so only the weights' values change. Storing the round trip
  in bf16 adds bf16's own rounding (at most 2^-8 of each value), so the premise slightly overstates int8's error.
- The router gate, embeddings, norms and experts are untouched.

**The rule** (`i8_reduce.py`, P125's own `gate` function at `t1` and `k16`): argmax agreement ≥ 0.95, mean d − 2 SE ≤
bound, SE ≤ bound / 3, mean d ≤ bound, with bound = ln(1 + 0.05 / ppl_A) over that gate's windows.
- **GO:** both gates PASS. The W8A16 kernel work may start (gnf4, with correctness against fp32 at the real shapes),
  then a 5090 microbench against cuBLAS bf16 at the real shapes, then a registration.
- **PARK:** either gate FAILs. No kernel; the lane parks with the number.
- **UNDERPOWERED:** neither gate failed, but one lacks power. It is reported with the windows it would have needed.
- **VOID:** a record is missing or off its arm; the round trip does not cover exactly 97 modules; a gate's windows are
  off P125's counts. VOID, or a box that stops before both arms, reruns once on another host under the spare id.

**Reporting:** A's ppl per gate beside P125's prior (ppl_R 8.762), each module's round-trip error, and c4's ppl for
both arms.

## The box

- **Run ids:** `w8a16-premise-5090-1`, with `-2` as a pre-create spare.
- **Host:** one RTX 5090 on Vast, verified-secure with direct SSH. The market-rule pin needs:
  - reliability2 ≥ 0.98 and inet_down ≥ 1000 Mbps;
  - a container disk ordered ≥ 170 GB (the runner's 150 GB floor + 20) and host RAM ≥ 60 GiB;
  - no failed most-recent host outcome in the receipts;
  - presence in 6 of 6 machine-filtered reads within the 2 minutes before the pin;
  - the GPU plus the ordered storage at ≤ $0.85/h, and download at ≤ $0.005/GB.
- **Guard:** 1.25 h. The download budget is 70 GB (the 61 GB checkpoint plus C4 validation).
- **Worst case:** 1.25 h × $0.85/h + 70 GB × $0.005/GB = **$1.41** per run, within the $1.50 cap.
- **Order on the box** (`i8_run.sh`, P125's runner with the speed arms and prompts dropped):
  1. refusals; P125's install of the subject and its tripwire; the self-tests; P125's GPU premise (39 passed);
  2. the fetch; the NF4 bake; the windows check;
  3. quality arm A, then quality arm I8;
  4. the reduce.
- **Before launch:** the lane's own guards (`--validate` at the current launcher, the ceiling check at the ledger's
  remote tip, the pin above), and the receipt committed within minutes of teardown.
