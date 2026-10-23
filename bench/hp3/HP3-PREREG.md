# HP3: e4b vs Unsloth vs axolotl held-out parity, on one build with byte-identical bases

**Status: registered 2026-10-21, before any data**, on #1932: one review outside the HP lane, then the maintainer's pass. Ruling
5 was added on the harness PR, also before any data. Nothing launches without the maintainer's exec ACK with the kit hash, from
main.

**Scope label, in every verdict line:** *parity of the exported adapters under one evaluator on byte-identical bases*. HP3 says
nothing about any framework's own in-loop loss.

**Labels, in every verdict line (build ruling):**
- **(i)** Unsloth 2026.10.7 + unsloth_zoo 2026.10.7 is a NEW version, not the 2026.9.14 that HP1 and HP2 read.
- **(ii)** axolotl 0.20.0 runs with kernels OFF: eager experts, no kernels plugin, no `lora_*_kernel`. That is not AXO1767's
  characterised configuration. AXO1767 Amendment 1's four engagement fields assert it, and a mismatch VOIDs the arm.
- **(iii)** The build is torch 2.14.0+cu130 with bitsandbytes 0.50.2, not HP2's or HPN's cu128. HBD found a one-slot difference
  across builds at L2, so **no HP3 number is compared with an HP2 or HPN number.**

## Why the gate changed (HP2's rule and record are not amended)

HP2 gated its instrument on δ0 = mean_r [(E_e4b(r) − I_e4b(r)) − (E_uns(r) − I_uns(r))]:
- E is the evaluator's zero-adapter read on that framework's exported base;
- I is that framework's own in-loop step-0 loss.

The gate was built when the two bases differed across builds (HBD: torch 2.8.0+cu128 vs 2.12.1+cu130 differ in one attention
slot). On one build the bases are byte-identical after dequantization (HBD, HPN). The evaluator is deterministic (HPN's N0), so
E_e4b(r) = E_uns(r) exactly, and δ0 = −mean_r (I_e4b(r) − I_uns(r)). That is the difference between the frameworks' training
forwards. It is part of the treatment, not the instrument.

HPN read 8 measured exactly that on one build, with a base term of 0 and an in-loop term of −δ0:

| depth | δ0 (SE) | \|δ0\| + 2 SE | m_need |
|---|---|---|---|
| L4 | 0.004872 (0.000335) | 0.00554 | 0.0111 |
| L8 | 0.0052 (0.000647) | 0.0065 | 0.0130 |

HP2's STOP bar is m_need ≤ 0.005, so its G1 would STOP on a matched build by construction, already at L4. The headline Δ never
reads I: it is the evaluator on each arm's exported adapter, over identical bases. HP3 therefore gates what Δ depends on: that the
bases are identical and that the evaluator reads them identically. That is a new instrument rule, so this is a new registration
(HP1's recorded lesson). HP2's G1, its record and its scores stand as written.

## The question

On Qwen3-30B-A3B @ `ad44e777`, with all three frameworks on one torch/CUDA/bitsandbytes build, does each framework's exported
adapter score at **parity** on held-out loss, read by one evaluator, or **better** or **worse**? The three pairs are
e4b vs Unsloth, e4b vs axolotl and Unsloth vs axolotl.

- **Recipe:** HP2's (HP1's field recipe): 200 steps, r = alpha = 16, the registered Alpaca rows, HP1's 48 held-out rows, seeds
  3410–3412 paired by `order_sha` *(superseded by Amendment 1: seeds 3410–3414)*.
- **e4b:** K00, the shipped settings, as HP2.
- **Unsloth:** its own default path on the build, as a user gets it. The MoE backend it selects is recorded (HPN's t28 build ran
  native_torch and HP2's cu130 build ran grouped_mm).
- **axolotl 0.20.0, kernels OFF:** eager experts, no kernels plugin, every `lora_{mlp,qkv,o,embedding}_kernel` false. This is
  **not axolotl's as-shipped path**, and the record says so. The instrument forces it: AXO1767 read KERNELS, so an adapter trained
  under axolotl's kernels gives its in-loop number only under those kernels. The eager evaluator read the export +0.0345 above its
  in-loop number (`hp1k-5090-8`), which would VOID every kernels-on run at 6(b)'s 0.01. A kernels-on axolotl arm would need an
  evaluator that runs axolotl's kernels, and that is a different instrument.

## The matched build (ruled: (P))

From PyPI `requires_dist`, read 2026-10-21, there is no shared torch at the registered versions:
- axolotl 0.20.0 needs torch ≥2.13.0, ≤2.14.0; transformers ==5.17.0; peft ==0.21.0; bitsandbytes ==0.50.2.
- Unsloth 2026.9.14 needs torch <2.13.0 and transformers ≤5.5.0.
- axolotl 0.18.0 and 0.19.0 reach older torch, but every axolotl pins transformers ≥5.14.1, and 0.18.0 pins bitsandbytes 0.49.1.
- Unsloth 2026.10.7 (2026-10-19) allows torch <2.15 and transformers ≤5.17.0.

**(P), ruled 2026-10-21 12:03Z:** one build for all three.

| component | pin |
|---|---|
| torch | 2.14.0+cu130 (the newest the axolotl 0.20.0 cap allows; cu130 is the only Linux route for torch ≥2.13) |
| CUDA | the cu130 wheel's runtime; the host driver must be ≥580, as every cu130 venv already requires |
| bitsandbytes | 0.50.2, the same `libbitsandbytes_cuda130.so` sha256 in all three venvs (`hbd_libraries`) |
| triton | the version torch 2.14.0's wheel pins, the same in all three venvs |
| Unsloth | 2026.10.7 + unsloth_zoo 2026.10.7 (**not** HP1/HP2's 2026.9.14; labelled as a new version) |
| axolotl | 0.20.0, the release AXO1767 characterised |
| transformers / peft | each framework's own resolution, recorded per venv, never gated: e4b 5.18.0 / 0.21.2, axolotl ==5.17.0 / ==0.21.0, Unsloth's resolve ≤5.17.0 |

**Why transformers and peft are not part of "matched build".** HBD and HPN found byte-identical dequantized bases across e4b and
Unsloth under one torch and bitsandbytes build, while their transformers versions differed. The base bytes come from the
quantize kernel and torch, and G1' checks the identity on the box in any case.

**Resolver check, in the shakedown:** each venv's installed torch, CUDA, bitsandbytes (and lib sha) and triton are recorded. A venv
whose resolver lands on anything other than the pins above fails the shakedown, and the read box does not launch. tc1_run.sh's
comment says axolotl 0.20.0 resolved torch 2.14.1+cu130 on an earlier box, against the metadata's ≤2.14.0. The shakedown records
what is actually installed, and the pin is forced with an explicit `torch==2.14.0` constraint in every venv.

**Install routes.**
- Unsloth 2026.10.7's install extras stop at `cu130-torch2121`; there is no `cu130-torch2140` extra. Unsloth goes in as the plain
  `unsloth==2026.10.7` + `unsloth_zoo==2026.10.7` under the explicit `torch==2.14.0` constraint, from the cu130 index. Its own
  `requires_dist` allows torch <2.15.
- axolotl 0.20.0 also needs xformers. Recent xformers releases need torch ≥2.10, so neither framework forces a different torch.
- e4b and grouped-nf4-gemm go into their own venv under the same constraint.
- Whether these routes install and import is a shakedown question. A failure there triggers the (R) fallback.

**e4b on torch 2.14.0 is untested.** The shakedown is a correctness check only, never data. HP3 makes **no e4b compatibility
claim** for torch 2.14.

**The fallback, decided before any data.** HP3 runs **(R)** instead if the cheap-rental shakedown either:
- cannot resolve, install or import all three frameworks on the matched build (the resolver check above), whatever the host; or
- fails G1' at L2, meaning the bases are not byte-identical.

A failure is **host-attributable** only if it is one of:
- a pre-install host check: driver, compute capability, RAM, disk or power;
- a provider or boot error before any pip resolve.

A host-attributable failure uses the spare and is not the (R) trigger. Every resolve, install or import failure of the three
frameworks is the trigger, so a bad host cannot turn the choice of build into a choice made by result.

**An arm that installs and imports but fails to RUN (ruling 5)** is also the (R) trigger by default. (P)'s premise is that all
three frameworks work on the matched build, and e4b on torch 2.14 is exactly the untested part. The only exceptions are three
pre-named host classes, which use the spare instead:
- CUDA out-of-memory on a card whose VRAM is under the registered floor (16 GB);
- a driver, Xid or ECC error;
- provider preemption or teardown mid-run.

Anything else is (R), including a Python exception, a kernel compile error or an unsupported op.
- The class is read from the TERMINAL error only, by `hp3_build.py classify`: the receipt's reason plus the log's final
  exception line, never by hand. A host class must match that error's own signature, so an earlier incidental mention (an NVML
  warning, an `import pynvml` frame, a config dump's `ECC errors=0`) never makes a failure host-attributable.
- The summary line names the class and quotes the error line.
- If the spare also fails host-attributably, the lane stops and reports: no (R) and no read. (R) is for a framework/build
  incompatibility, not a bad host.

(R) is e4b vs Unsloth **2026.9.14** on HPN's build, torch 2.8.0+cu128, with label (i) dropped and label (iii) naming that build.
axolotl waits for an amendment. **Only the shakedown can trigger the fallback, never a Δ.** Once the read box has launched, the
build is fixed.

(Q), two matched pairs on two builds, was not chosen.

## The instrument gate: G1' (deterministic, before any confirmation arm)

Phase B runs one calibration step per framework on the 300 calibration rows (held-out pool rows 97–396, as HP2), exports each base
and runs the evaluator's zero-adapter pass on each. G1' then checks two things for each framework f against e4b's base:
1. **Dequantized digests equal:** `hp1_base_digest` of f's exported base equals e4b's.
2. **The evaluator reads them identically:** the zero-adapter per-row losses on f's base equal e4b's to 1e-5 on every one of the
   300 rows (HPN's N0 band), and so the row means are equal.

- **A mismatch on either check is a CLOSE** for f's pairs, worded "bases differ: outside this registration". f runs no
  confirmation arm. Build-vs-framework stays one variable at a time.
- G1' is checked per pair against e4b's base. Two bases that each equal e4b's equal each other, so the Unsloth–axolotl pair
  passes when both of its frameworks pass.
- **The CLOSE is per framework (ruling 4).** A CLOSE drops only f's pairs, so a surviving e4b–Unsloth pair is still read if
  axolotl's base differs. Only if every pair closes does the driver end the draw, with the box torn down before phase C.
- Any closed pair makes the three-way read NOT_READ. The record names the pair and both digests.
- G1' is deterministic: no interval and no SE. It is computed by the registered reducer (`hp3_reduce.py g1`), which writes
  `hp3/g1.json` and exits 4 on a whole-box CLOSE. A CLOSE is a reading, not a strike.

## The evaluator

HP2's one evaluator (`bench/hp1/hp1_eval.py`, as merged) reads every base and every exported adapter. HP3 runs it in the e4b venv on
the matched build, so the evaluator's process is on that build too.

## Per run: the 6(b) round-trip VOID (as HP2)

For run s of framework f, a_f(s) = (evaluator − in-loop step N) − (evaluator zero adapter − in-loop step 0), on the 48 scored
rows. Run s VOIDs iff |a_f(s)| > 0.01. That is `ROUNDTRIP_MATCH`, AXO1767's MATCH band, read from `hp3/g1.json` as HP2 does.
- An axolotl run also VOIDs if its **engagement check** fails. The check uses AXO1767 Amendment 1's four fields, read from the
  arm's own record (never from a label): mode = off; experts implementation = eager; plugins = none; effective LoRA-kernel set =
  empty.
- A silent fallback onto the kernel path fails the check and cannot pass as data.
- **No redraw.** HP2 redrew a run-level VOID once, but HP3 does not. A VOID run makes its pairs NOT_READ, and that is reported as
  it stands.

## The verdict, per pair: G2'

For each pair (A, B) whose G1' passed, Δ_AB = mean over seeds of (A − B) evaluator held-out row means on the 48 scored rows,
paired by seed and `order_sha`.
- Its interval is the 90 % t-interval, df = seeds − 1.
- The run VOIDs if the evaluator recorded no in-loop step-N number for it (`evaluator_minus_inloop_final`), with that reason.
- The margin is m = min(max(σ_seed, 0.005), 0.02), with σ_seed as HP2 defines it (`hp1_reduce.pooled_sd` over the pair's seeds).
- **G2'** is HP2's realized-margin TOST on Δ alone. HP2's δ0 term is dropped: on identical bases Δ_base = 0 exactly, which G1'
  has already checked.
- A pair needs all three seeds *(superseded by Amendment 1: all five)* of both frameworks non-VOID to be read. Otherwise it reads NOT_READ, naming the VOID runs.

**Bands and wording (HP2's, unchanged):**
- **BEATS / BEHIND:** the 90 % interval excludes 0 on that side and |Δ̄| > σ_seed.
- **PARITY:** "no measurable difference at margin m: Δ̄ with its 90 % interval [lo, hi] inside ±m". Never "equal", "matches" or
  "the same".
- **INCONCLUSIVE:** "no measurable effect, with its interval: Δ̄, [lo, hi]".
- **CLOSE (G1'):** "bases differ: outside this registration".
- Every verdict line carries the scope label and labels (i)–(iii).

## The three-way read

The three pair verdicts are reported as a table, each with its own interval, and there is no family-wise correction. Each pair is
its own registered question. The three-way summary is a statement about that table:
- **THREE-WAY PARITY:** all three pairs read PARITY at their own m. That is an intersection of TOSTs, so it is conservative and
  needs no correction.
- **ORDERED:** all three pairs are BEATS/BEHIND, and those verdicts are consistent with one order. The RESULTS line says three
  pairs were read, so a reader can apply their own correction.
- **MIXED:** all three pairs were read, and the verdicts are neither of the above. Worded pair by pair, with no summary claim.
- **NOT_READ:** any pair is CLOSE or NOT_READ. The record names each closed pair, with both frameworks' base digests, and each
  unread pair with its VOID runs. The read pairs are still reported, each with its own verdict, but there is no three-way summary.

## Reported, never gated (the in-loop term named in each)

- **δ0 and δN, per pair, as HP2 defines them.** On identical bases these are the in-loop term: the difference between the two
  frameworks' training forwards at step 0 and at step N.
- **b_f = mean_r (E_f(r) − I_f(r)) for each framework:** the evaluator's offset from that framework's own training forward.
  - **The home-field direction:** the framework whose |b_f| is smallest is the one whose training forward the evaluator sits
    closest to. A second-order train/eval mismatch could favour that side, so the reader is shown it beside every verdict.
- **Per-row |gap| statistics** (mean, sd and max) for each framework; z_f on the 48 scored rows; σ_seed.
- **The installed versions per venv,** the Unsloth MoE backend, and axolotl's four engagement fields.
- **Context, no new instrument (ruling 6):** HPN read 8's N0, where the in-loop reader I repeated exactly per row for e4b and for
  Unsloth at L4 and L8 (torch 2.8.0+cu128, one build). HP3 does not re-gate it.

## Predictions (scored HELD / FALSIFIED / UNTESTED)

- **H1:** G1' passes for all three frameworks. The reason is HBD's result: the one slot that differed did so across builds, and
  within each build both frameworks' dequantized bases were identical. HPN found the same at L4 and L8. Axolotl's base has not been
  read under HBD, so for axolotl H1 is a prediction from the shared quantize kernel, not from a measurement.
- **H2:** e4b vs Unsloth reads PARITY or INCONCLUSIVE.
- **H3:** each axolotl pair (kernels off) reads PARITY or INCONCLUSIVE. Axolotl's as-shipped kernel path is not read, and H3 says
  nothing about it.

## Hosts and budget (no A2000: owner directive 2026-10-21)

**1. Shakedown (never data):** the cheapest rented card that runs the 2-layer cut with all three venvs on the matched build.
- Requirements: host driver ≥580 (cu130), compute capability ≥7.5 (cu130 does not build for older cards), ≥16 GB VRAM, disk ≥120 GB, PyPI ≥10 MB/s (AXO1767's probe). It must not be an RTX 5090,
  the read card's class.
- It proves the kit end to end:
  - the three installs and the resolver check;
  - the 2-layer fetch (`TC1_FETCH_ALLOW`, the first shards only);
  - one calibration step and an export per framework;
  - G1' on real records;
  - one 4-step confirmation run per framework, with 6(b) and the engagement check;
  - the reducer's `g1` and `verdict` on those real records.
- Guard 2.5 h at a cap of $0.40/h, plus about 20 GB downloaded at $0.011/GB, gives a worst case of **$1.22**
  *(superseded by Amendment 1: $2.10, with the 100 GB download term)*. One attempt, and one spare after a harness fix.

**2. The read:** one RTX 5090 (HP2's class; the build-sensitive quantize kernel is also arch-sensitive), at the full 48 layers,
under the registered host rule (driver ≥580, RAM ≥98 GB, disk ≥220 GB).

| Phase | Work | Estimate |
|---|---|---|
| A | three installs (venv-e4b, venv-unsloth, venv-axolotl, all on the matched build), fetch, the resolver check | ~1.0 h |
| B | calibration: one step per framework on the 300 rows, three exports, three zero-adapter passes → **G1'** | ~1.2 h |
| C | confirmation: seeds 3410–3412 per framework that passed G1' (≤9 runs, with axolotl's eager path the slowest) | ~3.4 h *(superseded by Amendment 1: 3410–3414, ≤15 runs, ~6.4 h)* |
| D | the evaluator's pass on the confirmation runs → 6(b), the pair verdicts, the three-way read | ~0.9 h *(superseded by Amendment 1: ~1.5 h)* |

- That totals about 6.5 h *(superseded by Amendment 1: about 10.1 h)*. A whole-box CLOSE at G1' tears down at about 2.2 h.
- **The deciding-depth rule (HPN's floor rule, applied to one depth):** the registered depth is the full 48 layers. An arm that
  cannot run at 48 on the read card is recorded as a depth limit, and its pairs read NOT_READ. No shallower depth substitutes for
  it.
- Guard 8 h at $0.85/h, plus 90 GB at $0.011/GB, gives a worst case of **$7.79** *(superseded by Amendment 1: guard 11 h, $10.45)*. One box-started attempt and one pre-create spare.

**Lane worst case:** 2 × $1.22 (shakedown and spare) + $7.79 = **$10.23**, under the lane's $15 *(superseded by Amendment 1:
$14.65)*.

## Rulings taken (maintainer, before any HP3 data)

1. 2026-10-21 11:36Z: YES to one prereg covering e4b, Unsloth and axolotl on one build. The shakedown goes on a cheap rental, not
   the A2000, and is never data. The read is on one RTX 5090. Items (1)–(4) are covered above.
2. 2026-10-21 11:51Z: option (b) for the gate, with these conditions:
   - the scope label;
   - deterministic G1' (digests and zero-adapter means), where a mismatch is a CLOSE and a teardown before confirmation;
   - δ0, δN and b_f reported with the in-loop term named, and the home-field direction shown;
   - 6(b) as HP2, with G2' as the TOST on Δ alone;
   - this section's explanation of why the gate changed, with HP2 not amended;
   - the I-reader's repeatability not gated, HPN's N0 cited as context;
   - review by a reviewer outside the HP lane, then the maintainer's pass.
3. 2026-10-21 12:03Z: the build is (P), with labels (i)–(iii) in every verdict line and the (R) fallback above, triggered
   only by the shakedown. Each framework's transformers and peft are recorded and reported, never gated, and so is what each
   resolver actually installs. The shakedown is a correctness check on e4b at torch 2.14, with no compatibility claim. Then: this
   PREREG, a review outside the HP lane, the maintainer's pass, and an exec ACK with the kit hash.
4. 2026-10-21 13:23Z, on the review: these refine ruling 2's teardown, which was written for a two-way read.
   - **(A)** The CLOSE is per framework: G1' is checked per pair against e4b's base, and the box tears down early only if every
     pair closes. Any closed pair makes the three-way read NOT_READ, and the record names the pair and its digests.
   - **(B)** The three-way read takes the reducer's NOT_READ rule, and ORDERED needs all three pairs.
   - **(C)** The shakedown host needs compute capability ≥7.5. Only a pre-install host check or a provider/boot error before any
     pip resolve is host-attributable and uses the spare. Any resolve, install or import failure on the build is the (R)
     trigger.
   - A run missing `evaluator_minus_inloop_final` VOIDs, with its reason.
5. 2026-10-21 14:23Z, on the harness review, before any data: a shakedown arm that installs and imports but fails to run is the
   (R) trigger, unless its captured error text is one of the three pre-named host classes ("The fallback" above), which use the
   spare. If the spare also fails host-attributably, the lane stops with no (R) and no read.

## Amendment 1 (2026-10-21, before any HP3 data): five seeds per framework

**Why.** A power pre-read ($0, from prior records, never an HP3 number) found that 3 seeds read PARITY about as often as a coin flip
at plausible noise. Every number in this section is an ESTIMATE.
- **No prior record measures σ_seed or the seed-level ρ.** All 33 committed 200-step Qwen3-30B-A3B arms ran seed 3407 (21 e4b, 12
  axolotl, 0 Unsloth), and the private receipt store adds only 60-step arms at seed 0.
- **What prior records do give is a fixed-seed floor.** Six TC1 RTX 5090 boxes (`tc1-5090-35/36/40/46/47/48`; seed 3407; 200
  steps; 16 in-loop rows):
  - across boxes, e4b's sd is 0.00078 (n = 7) and axolotl's 0.00070 (n = 6);
  - same-box redraws give sd/√2 of 0.00098–0.00124;
  - corr(e4b, axolotl) across boxes is −0.16, and their paired difference has sd 0.00115.

  So σ_seed ≥ ~0.001. These are in-loop numbers on 16 rows; HP3 reads the evaluator on 48.
- **The table.** P(PARITY | true Δ = 0) for one pair under this registration's own verdict rule, with BEATS / BEHIND tested before
  PARITY, m = min(max(σ_pooled, 0.005), 0.02) and a 90 % t-interval on the paired difference. It is the evaluator's independent
  re-implementation (4000 reps, seed 20261021), reproduced to the digit by `bench/hp3/hp3_power.py`:

| σ_seed | ρ | 3 seeds | 5 seeds | 8 seeds |
|---|---|---|---|---|
| 0.001 | 0 | 0.86 | 0.92 | 0.94 |
| 0.002 | 0 | 0.40 | 0.81 | 0.94 |
| 0.002 | 0.5 | 0.70 | 0.96 | 0.98 |
| 0.003 | 0 | 0.17 | 0.44 | 0.81 |
| 0.003 | 0.5 | 0.39 | 0.82 | 0.98 |
| 0.005 | 0 | 0.05 | 0.10 | 0.27 |
| 0.005 | 0.82 | 0.48 | 0.88 | 0.99 |

- **REPORTED context, a property of the HP2/HP3 rule and not a defect to fix.** Under true parity at small σ, the BEATS / BEHIND
  step makes a false directional call 5.4–9.0 % of the time at ρ = 0 (`hp3_power.py`'s last column). That caps P(PARITY) near
  0.92–0.94 even when the noise is low.
- **Cost.** 8 seeds would run about 14.3 h, so a ~15.5 h guard: 15.5 × $0.85 + 100 GB × $0.011 + 2 × $2.10 ≈ $18, over the lane's
  $15. 5 is the largest count that fits.

**The change.**
1. **Seeds:** 3410–3414 per framework that passes G1', each seed paired by `order_sha`. A pair needs all five seeds of both
   frameworks non-VOID to be read. The verdict rule, its order, the margin and the 90 % t-interval (now df = 4) are unchanged.
   `hp3_reduce.paired_reading` is `hp1_reduce.paired_reading`'s rule over HP3's seeds, and a test checks they are equal at HP1's
   three seeds.
2. **ρ is REPORTED only:** each pair's verdict records `rho_seed_reported`, the correlation over seeds between the two frameworks'
   row means. It gates nothing.
3. **The read box:** a guard of 11 h at $0.85/h, plus the launcher's 100 GB × $0.011 download reservation, gives a worst case of
   **$10.45**. The phases are estimated at ~10.1 h: A ~1.0, B ~1.2, C ~6.4 for ≤15 runs with axolotl's eager path the slowest, and
   D ~1.5.
4. **The shakedown:** unchanged (one seed, 4 steps). Each attempt's worst case is 2.5 h × $0.40 + 100 GB × $0.011 = **$2.10**.
5. **Lane worst case:** 2 × $2.10 + $10.45 = **$14.65**, under the lane's $15.
   - The approval at 15:59Z quoted $12.78. That figure carried 20 GB (shakedown) and 90 GB (read) download terms rather than the
     launcher's 100 GB reservation. This amendment states the reservation.

6. **Run ids (bench/common rule 2's form).**
   - The shakedown is `hp3-shake-1`, with `hp3-shake-2` as its one spare.
   - The read is `hp3-5090-1`, with `hp3-5090-2` as its one pre-create spare.
   - At most one shakedown box and one read box start, beyond a host-attributable spare (ruling 5).
   - A refusal at create uses no id: the launcher receipts it as `<id>-refused-*`, and the next attempt takes the same id.
   - No code parses an id's number.

7. **A slow host shortens the read; it never voids it.** The 11 h guard leaves ~9 % headroom over the ~10.1 h of phases.
   - **Decided by timing alone.** The seed count reads wall time and the deadline, never a loss, a receipt's result or a verdict,
     so it is not an outcome-dependent stopping rule.
   - **The plan.** Phase C is seed-major: at each seed, e4b, then Unsloth, then axolotl, for every framework that passed G1'. So
     the seeds a box completes are always a PREFIX of 3410–3414, and none can be dropped selectively. After seed 3410 completes
     for every framework, the runner measures that seed's wall time W, across all of its frameworks (e4b runs first and axolotl's
     eager path is the slowest, so one arm would underestimate a seed).
     - It sets n to the largest k in {5, 4, 3} with (k − 1) × W + 360 s × (frameworks) × k + 900 s ≤ the time left − 900 s. The
       evaluator's pass is estimated at ~6 min a run.
     - It writes `hp3/seed_plan.json` (n, the seeds, W, the time left, each k's need, `basis: timing`, the reason,
       `written_at_utc`) BEFORE any later seed or the evaluator's pass.
   - **The read.** The box runs exactly n seeds, and `hp3_reduce.py verdict --seed-plan` reads the first n (df = n − 1).
     - The verdict records `seed_count`, `seeds_read` and `seed_plan_reason` (e.g. "fit: 4 seeds at seed 3410's wall time 2400 s").
     - A read on 3 or 4 seeds is a **reduced-power read**, stated as such in the record and every RESULTS line. It is never
       upgraded later.
     - A VOID run among the planned seeds still makes its pairs NOT_READ.
   - **The refusals.** The reducer refuses a plan whose n is not 0, 3, 4 or 5, whose seeds are not that prefix, or whose basis is
     not timing.
   - **The content binding (the maintainer, 20:54Z, replacing an mtime check that a re-derivation cannot confirm).**
     - The runner hands the plan to the evaluator explicitly (`hp1_eval.py --seed-plan`). At its start, before any model or base
       is touched, the evaluator records `plan_sha256`, the sha256 of the plan's raw bytes, in `hp3/eval_P.json`.
     - The reducer, given `--seed-plan`, refuses unless every evaluator output's `plan_sha256` equals that file's hash. A missing
       or null field also refuses.
     - Given no plan, it refuses any evaluator output that carries one: the runner made a plan the reducer was not handed.
     - What this binds: the plan was fixed before the evaluator's pass. That is the pre-data property, and it survives the
       store, which keeps content and not mtimes. The runner writes the plan before seed 3411 by its code order. That is
       recorded (`summary.txt`'s `HP3 SEED_PLAN` line, the plan's `written_at_utc`, a clock read kept for the record only), but
       the binding does not claim it.
     - HP1, HP2, HPN and the shakedown never pass `--seed-plan`, so their evaluator output is unchanged.
   - **Fewer than 3 seeds fit:** n = 0. The box stops after seed 3410, with no evaluator pass. Every pair reads NOT_READ,
     host-limited, with a host refusal note. It is no strike, and the spend is minimal.
   - **A plan that cannot be written** (a harness fault, never the host's): the box prints `HP3 SEED_PLAN_UNWRITABLE` and stops
     after seed 3410, with no evaluator pass. Every pair reads NOT_READ, and the record names the harness, never HOST_LIMITED.
   - **The power each count can read.** ESTIMATE, P(PARITY | true Δ = 0) under the registered verdict order, from
     `bench/hp3/hp3_power.py`. The 3 and 5 columns are the table above; the 4 column is `--counts 4` on its own stream, at the
     same RNG seed:

     | σ_seed | ρ | 3 seeds | 4 seeds | 5 seeds |
     |---|---|---|---|---|
     | 0.001 | 0 | 0.86 | 0.91 | 0.92 |
     | 0.002 | 0 | 0.40 | 0.66 | 0.81 |
     | 0.002 | 0.5 | 0.70 | 0.91 | 0.96 |
     | 0.003 | 0 | 0.17 | 0.28 | 0.44 |
     | 0.003 | 0.5 | 0.39 | 0.64 | 0.82 |
     | 0.005 | 0 | 0.05 | 0.07 | 0.10 |
     | 0.005 | 0.82 | 0.48 | 0.73 | 0.88 |

   - **Cost:** unchanged. The guard stays at 11 h: read $10.45, lane $14.65.

8. **The card's memory, not only its name** (the reviewer, 2026-10-23, after a name-only screen elsewhere pinned a 48 GB modded RTX
   4090).
   - The box checks `memory.total` right after the card-class check, for both HP3 tokens, against the class's band in MiB:
     RTX A4000 15000–17000, L4 22000–24576, RTX 4090 23000–25000, L40S 44000–49152, RTX 5090 31000–33000.
   - Outside its band, or a class with no band, the box refuses with rc 14 before any staging. That is a pre-install host check,
     so it is host-attributable and uses the spare (ruling 4(C)), never the (R) trigger.
   - The shakedown's screen applies the same bands to each offer's `gpu_ram`.

**Unchanged:** everything else in this registration, including G1', 6(b), G2', the three-way read, the labels and the (R) fallback.

**Rulings:** part 7 was ruled by the maintainer at 2026-10-23 19:56Z and 19:57Z ((b) plus (c), before data), with the reviewer's
two details (project from the first complete seed; record the count and its timing-only reason). The rest was approved by the
maintainer at 15:59Z on 2026-10-21, conditional on the evaluator's re-run of the simulation. The evaluator's checks:
(a) the margin formula PASS, (b) the paired t quantile PASS, (c) a discrepancy (0.40, not 0.45) that came from the pre-read's
PARITY-only test and strengthens the case. This amendment merges before the read box is created.

## Harness

- `bench/hp3/hp3_reduce.py` (in this PR): `g1` (G1', exit 4 on a whole-box CLOSE) and `verdict` (6(b), G2', the three-way read). It reuses
  `hp2_reduce` and `hp1_reduce`'s `paired_reading`, `pooled_sd`, `order_problems` and `manifest_problems`.
- `bench/tc1/tc1_run.sh`: the `qwen3hp3` token, with three venvs on the matched build (the explicit torch constraint), the
  resolver check, and `tc1_hp3_family` (phases A–D). The shakedown variant uses the 2-layer cut.
- `bench/tc1/tc1_arm.py`: nothing new for axolotl. The training arm already records its `axolotl` block (`kernels`,
  `config_experts_implementation`, `plugins`, `cfg_lora_kernels`), and the HP3 arm passes `--axolotl-kernels off --axolotl-best 0`.
  The older TC1 "matched" axolotl rows ran `grouped_mm` with no kernels mode recorded, so they are not this configuration.
- `hp3_reduce.py verdict` takes each axolotl arm's receipt (`--arm-receipt`) and applies `axo1767_reduce.engagement` in its
  `off` mode.
- `tests/test_hp3.py` (in this PR) covers four areas:
  - G1' on HPN read 8's real L8 base records (public projections), where the reported δ0 equals HPN's 0.0052, the in-loop term;
  - CLOSE on one row outside 1e-5, on a moved row mean, and on a different digest, with the whole-box exit 4 and its record;
  - the 6(b) and engagement VOIDs;
  - the three-way PARITY / ORDERED / MIXED reads.
- To build before the shakedown (a later PR, before the exec ACK): the `qwen3hp3` token and family in `tc1_run.sh`. That covers
  the three venvs with the explicit `torch==2.14.0` constraint, the resolver check, the 2-layer shakedown variant and the (R)
  fallback switch.
