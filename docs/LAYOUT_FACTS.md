# LAYOUT_FACTS — combine order, determinism, and RNG wiring (Addendum 3 §3 + T1.0b)

```
status:  source facts + measurement precedence, filed BEFORE any routed-stream
         Phase 0 code (Addendum 3 clock). Rule R9 both ways: source reads are facts
         about code; measured behavior takes precedence where it exists.
```

## Combine order (source fact)

The resident MoE combine (`_vendor/experts.py:486–513`, mirrored in
`lora.py::ExpertsLoRA.forward`) is a Python loop over `expert_hit` (ascending expert
id) with `final_hidden_states.index_add_(0, token_idx, …)` in fp32 — contribution
order follows expert-hit order; `index_add_` is *nominally* non-deterministic on GPU.
The single-token decode fast-path (`_forward_decode`) sums in routing order instead —
an ulp-level ordering difference by design, eval/training never take it (multi-row).

## Determinism (measured — takes precedence over the nominal fact)

- **Eval forward: bitwise-repeatable** on the RunPod A5000 stack (torch 2.8.0+cu128):
  the n=64 determinism repeat reproduced 64/64 example losses bit-for-bit, and
  resident-vs-offload was bitwise-identical 384/384 (D2). The nominal `index_add_`
  non-determinism did not manifest on these shapes/dtypes/kernels.
- **One training step: bitwise-repeatable** (null a≡b of the D3 certificate, all five
  configurations, default kernels included) — forward, backward-with-recompute,
  clip, and AdamW step reproduce exactly; placement adds nothing (a≡c bitwise).
- **Full-run (150-step) training determinism: UNKNOWN.** Two same-config grid legs
  differed (the bf16 0.0108 pair), so either determinism breaks at some later
  step/shape or an unidentified run-level difference existed. The divergence-onset
  probe (gated) is the resolver.
- Determinism is claimed for **this host class only**; T5(c) showed a different
  architecture (4090) is deterministic per-arch but offset from the A5000.

## Rule for routed-stream implementations (v3 §3.1 discipline)

Implementation must match the MEASURED behavior of its target host: run the bitwise
eval-repeat on that host first; require bitwise match if it holds there,
within-measured-null otherwise. Do not design to the nominal non-determinism.

## RNG wiring (T1.0b, enumerated from installed packages on the pod)

- torch 2.8.0 `checkpoint()`: `preserve_rng_state` is a kwargs option whose docstring
  states that on the **non-reentrant path the flag "doesn't take effect and we always
  preserve RNG state"** — preservation is unconditional for `use_reentrant=False`;
  the implementation (`_checkpoint_without_reentrant_generator`) stashes/restores
  CPU + device states via `get_device_states`.
- transformers 5.13.0 `gradient_checkpointing_enable` wires
  `functools.partial(checkpoint, **gradient_checkpointing_kwargs)` — our
  `{"use_reentrant": False}` passes through unmodified; nothing overrides RNG
  handling.
- `experts4bit_qlora/offload.py` contains **zero RNG calls** (no manual_seed / rand* /
  generator access) — staging and eviction cannot move any RNG stream.
- The training path itself consumes no RNG after LoRA-A init (no dropout anywhere —
  T1.0), so the preserved-state machinery has nothing to preserve in production
  config; the D3 dropout-ON trio verified it works when dropout is forced (bitwise
  through recompute with 16 modules at p=0.1).

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T20:03:53Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `edd9ce5ec493d9c7d3749ba5a4804298ded542af802c42553944b774512f41a6` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[?!!#&?O?&o#~!#&=]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
| ...+*o+.++=     |
|.  .o=o =o+ o    |
|. ..o.+o.Eo. .. +|
| . .. .o o ..+ B*|
|        S . . B+=|
|         . o . .o|
|          o . .  |
|           o .   |
|           .+    |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info LAYOUT_FACTS.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify LAYOUT_FACTS.md.ots LAYOUT_FACTS.md` succeeds against the on-disk bytes.
- Anchor file: `LAYOUT_FACTS.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
