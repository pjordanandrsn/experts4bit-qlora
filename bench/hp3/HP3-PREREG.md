# HP3: e4b vs Unsloth vs axolotl held-out parity, on one build with byte-identical bases

**Status: DRAFT, not registered. No HP3 data exists.** The maintainer has ruled on the gate design and on the build, (P), with
a fallback named before any data (below, "Rulings taken"). The draft is reviewed by a reviewer outside the HP lane, then by the maintainer.
Nothing launches before the maintainer's ruling and an exec ACK with the kit hash.

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
  3410–3412 paired by `order_sha`.
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
- cannot install all three frameworks on the matched build (the resolver check above); or
- fails G1' at L2, meaning the bases are not byte-identical.

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
- If both other frameworks CLOSE, the driver ends the draw and the box tears down before phase C.
- **One choice for the reviewer:** a CLOSE drops only f's pairs rather than the whole box, so a surviving e4b–Unsloth pair is
  still read if axolotl's base differs. The alternative is one CLOSE for the whole box.
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
- The margin is m = min(max(σ_seed, 0.005), 0.02), with σ_seed as HP2 defines it (`hp1_reduce.pooled_sd` over the pair's seeds).
- **G2'** is HP2's realized-margin TOST on Δ alone. HP2's δ0 term is dropped: on identical bases Δ_base = 0 exactly, which G1'
  has already checked.
- A pair needs all three seeds of both frameworks non-VOID to be read. Otherwise it reads NOT_READ, naming the VOID runs.

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
- **ORDERED:** every read pair is BEATS/BEHIND, and those verdicts are consistent with one order. The RESULTS line says three pairs
  were read, so a reader can apply their own correction.
- **MIXED:** anything else, worded pair by pair with no summary claim.

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
- Requirements: host driver ≥580 (cu130), ≥16 GB VRAM, disk ≥120 GB, PyPI ≥10 MB/s (AXO1767's probe). It must not be an RTX 5090,
  the read card's class.
- It proves the kit end to end:
  - the three installs and the resolver check;
  - the 2-layer fetch (`TC1_FETCH_ALLOW`, the first shards only);
  - one calibration step and an export per framework;
  - G1' on real records;
  - one 4-step confirmation run per framework, with 6(b) and the engagement check;
  - the reducer's `g1` and `verdict` on those real records.
- Guard 2.5 h at a cap of $0.40/h, plus about 20 GB downloaded at $0.011/GB, gives a worst case of **$1.22**. One attempt, and one
  spare after a harness fix.

**2. The read:** one RTX 5090 (HP2's class; the build-sensitive quantize kernel is also arch-sensitive), at the full 48 layers,
under the registered host rule (driver ≥580, RAM ≥98 GB, disk ≥220 GB).

| Phase | Work | Estimate |
|---|---|---|
| A | three installs (venv-e4b, venv-unsloth, venv-axolotl, all on the matched build), fetch, the resolver check | ~1.0 h |
| B | calibration: one step per framework on the 300 rows, three exports, three zero-adapter passes → **G1'** | ~1.2 h |
| C | confirmation: seeds 3410–3412 per framework that passed G1' (≤9 runs, with axolotl's eager path the slowest) | ~3.4 h |
| D | the evaluator's pass on the confirmation runs → 6(b), the pair verdicts, the three-way read | ~0.9 h |

- That totals about 6.5 h. A whole-box CLOSE at G1' tears down at about 2.2 h.
- **The deciding-depth rule (HPN's floor rule, applied to one depth):** the registered depth is the full 48 layers. An arm that
  cannot run at 48 on the read card is recorded as a depth limit, and its pairs read NOT_READ. No shallower depth substitutes for
  it.
- Guard 8 h at $0.85/h, plus 90 GB at $0.011/GB, gives a worst case of **$7.79**. One box-started attempt and one pre-create spare.

**Lane worst case:** 2 × $1.22 (shakedown and spare) + $7.79 = **$10.23**, under the lane's $15.

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
