# TC1 amendment 81: where the moved figures are quoted

16 register rows; 34 document lines.

| register row | status | field | figure (old -> new) | run | method | where |
|---|---|---|---|---|---|---|
| `e4b.train.ckpt-flavour.default.5090.2026-10-07` | measured | claim | 0.404 -> 0.389 (as `0.404`) | tc1-5090-128 | reducer | P182 |
| `e4b.train.ckpt-offload.field-torch28.5090.2026-10-07` | measured | claim | 0.775 -> 0.764 (as `0.775`) | tc1-5090-125 | reducer | P174 |
| `e4b.train.ckpt-offload.field-torch28.5090.2026-10-07` | measured | claim | 0.795 -> 0.782 (as `0.795`) | tc1-5090-125 | reducer | P175 |
| `e4b.train.env-gap.torch28.profile.qwen3.5090.2026-10-06` | measured | claim | 0.850 -> 0.846 (as `0.850`) | tc1-5090-110 | reducer | P136 |
| `e4b.train.env-gap.torch28.profile.qwen3.5090.2026-10-06` | measured | claim | 0.965 -> 0.961 (as `0.965`) | tc1-5090-110 | reducer | P136 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02` | measured | notes | 0.234 -> 0.232 (as `0.234`) | tc1-5090-16 | reducer | P8 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02` | measured | notes | 0.234 -> 0.232 (as `0.234`) | tc1-5090-16 | cell | qwen3_unsloth_ckpt_unsloth_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02` | measured | notes | 0.480 -> 0.473 (as `0.480`) | tc1-5090-16 | cell | qwen3_e4b_fused_attn4_m_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.host-variance` | measured | claim | 0.234 -> 0.232 (as `0.234`) | tc1-5090-16 | reducer | P8 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.host-variance` | measured | claim | 0.234 -> 0.232 (as `0.234`) | tc1-5090-16 | cell | qwen3_unsloth_ckpt_unsloth_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.host-variance` | measured | claim | 0.480 -> 0.473 (as `0.480`) | tc1-5090-16 | cell | qwen3_e4b_fused_attn4_m_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio` | measured | claim | 0.333 -> 0.328 (as `0.333`) | tc1-5090-137 | reducer | P208 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio` | measured | claim | 0.486 -> 0.470 (as `0.486`) | tc1-5090-137 | reducer | P208 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio` | measured | claim | 1.457 -> 1.435 (as `1.457`) | tc1-5090-137 | reducer | P208 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio` | measured | claim | 1.924 -> 1.953 (as `1.924`) | tc1-5090-137 | reducer | P205 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio` | measured | claim | 1693.9 -> 1639.9 (as `1693.9`) | tc1-5090-137 | reducer | P205 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio` | measured | claim | 3259.1 -> 3203.2 (as `3259.1`) | tc1-5090-137 | reducer | P205 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio` | measured | value | 1.924 -> 1.953 (as `1.924`) | tc1-5090-137 | reducer | P205 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio` | measured | claim | 0.783 -> 0.780 (as `0.783`) | tc1-5090-136 | reducer | unsloth/ckpt_unsloth_m_pv_d2 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio` | measured | claim | 0.993 -> 0.990 (as `0.993`) | tc1-5090-136 | reducer | P201 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio` | measured | claim | 1.160 -> 1.161 (as `1.160`) | tc1-5090-136 | reducer | P200 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio` | measured | claim | 10042.8 -> 10006.7 (as `10042.8`) | tc1-5090-136 | reducer | P200 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio` | measured | claim | 11649.4 -> 11613.2 (as `11649.4`) | tc1-5090-136 | reducer | P200 |
| `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio` | measured | value | 1.160 -> 1.161 (as `1.16`) | tc1-5090-136 | reducer | P200 |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile` | measured | claim | 0.234 -> 0.232 (as `0.234`) | tc1-5090-16 | reducer | P8 |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile` | measured | claim | 0.234 -> 0.232 (as `0.234`) | tc1-5090-16 | cell | qwen3_unsloth_ckpt_unsloth_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile` | measured | claim | 0.480 -> 0.473 (as `0.480`) | tc1-5090-16 | cell | qwen3_e4b_fused_attn4_m_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile` | measured | claim | 0.308 -> 0.297 (as `0.308`) | tc1c-h100-2 | reducer | P8 |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile` | measured | claim | 0.308 -> 0.297 (as `0.308`) | tc1c-h100-2 | cell | qwen3_unsloth_ckpt_unsloth_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile` | measured | claim | 0.562 -> 0.553 (as `0.562`) | tc1c-h100-2 | cell | qwen3_e4b_fused_attn4_m_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04` | measured | notes | 0.305 -> 0.294 (as `0.305`) | tc1c-h100-3 | reducer | P8 |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04` | measured | notes | 0.305 -> 0.294 (as `0.305`) | tc1c-h100-3 | cell | qwen3_unsloth_ckpt_unsloth_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04` | measured | notes | 0.665 -> 0.653 (as `0.665`) | tc1c-h100-3 | cell | qwen3_e4b_fused_attn4_m_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.moe-keep` | measured | notes | 0.306 -> 0.294 (as `0.306`) | tc1c-h100-4 | reducer | P8 |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.moe-keep` | measured | notes | 0.306 -> 0.294 (as `0.306`) | tc1c-h100-4 | cell | qwen3_unsloth_ckpt_unsloth_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.moe-keep` | measured | notes | 0.659 -> 0.643 (as `0.659`) | tc1c-h100-4 | cell | qwen3_e4b_fused_attn4_m_prof device_busy_fraction |
| `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.route-v2` | measured | claim | 0.454 -> 0.442 (as `0.454`) | tc1c-h100-9 | cell | qwen3_e4b_fused_attn4_m_prof device_busy_fraction |
| `e4b.train.p129.fused-qkv.qwen3.5090.2026-10-10` | measured | claim | 0.987 -> 0.990 (as `0.987`) | tc1-5090-146 | reducer | DEVICE |
| `e4b.train.p129.fused-qkv.qwen3.5090.2026-10-10` | measured | claim | 0.464 -> 0.446 (as `0.464`) | tc1-5090-147 | reducer | PREMISE |
| `e4b.train.p129.fused-qkv.qwen3.5090.2026-10-10` | measured | claim | 0.987 -> 0.988 (as `0.987`) | tc1-5090-147 | reducer | DEVICE |
| `e4b.train.p129.fused-qkv.qwen3.5090.2026-10-10` | measured | claim | 0.987 -> 0.984 (as `0.987`) | tc1-5090-147 | reducer | DEVICE |
| `e4b.train.p129.fused-qkv.second-host.5090.2026-10-10` | measured | claim | 0.464 -> 0.446 (as `0.464`) | tc1-5090-147 | reducer | PREMISE |
| `e4b.train.p129.fused-qkv.second-host.5090.2026-10-10` | measured | claim | 0.539 -> 0.521 (as `0.539`) | tc1-5090-150 | reducer | e4b/fused_attn4_m_q0_d2 |
| `e4b.train.p129.fused-qkv.second-host.5090.2026-10-10` | measured | claim | 0.988 -> 0.984 (as `0.988`) | tc1-5090-150 | reducer | DEVICE |
| `e4b.train.single-ladder-auto.field.5090.2026-10-08` | measured | claim | 1.043 -> 1.045 (as `1.043`) | tc1-5090-139 | reducer | P218 |
| `e4b.train.single-ladder-auto.field.5090.2026-10-08` | measured | notes | 0.475 -> 0.460 (as `0.475`) | tc1-5090-139 | reducer | P215 |
| `e4b.train.single-ladder-auto.gpu-bound.5090.2026-10-08` | measured | claim | 0.884 -> 0.871 (as `0.884`) | tc1-5090-141 | reducer | P215 |
| `e4b.train.single-ladder-auto.gpu-bound.5090.2026-10-08` | measured | claim | 0.999 -> 1.000 (as `0.999`) | tc1-5090-141 | reducer | P218 |
| `e4b.train.single-ladder-auto.gpu-bound.5090.2026-10-08` | measured | claim | 1.055 -> 1.056 (as `1.055`) | tc1-5090-141 | reducer | P218 |
| `e4b.train.single-ladder.field.5090.2026-10-08` | measured | claim | 0.330 -> 0.317 (as `0.33`) | tc1-5090-138 | cell | qwen3sladder_e4b_fused_attn4_m_l0 device_busy_fraction |
| `e4b.train.single-ladder.field.5090.2026-10-08` | measured | claim | 0.417 -> 0.400 (as `0.417`) | tc1-5090-138 | reducer | e4b/fused_attn4_shipped_l0 |
| `e4b.train.single-ladder.field.5090.2026-10-08` | measured | claim | 0.417 -> 0.402 (as `0.417`) | tc1-5090-138 | reducer | P210 |

| document line | figure (old -> new) | run | method | text |
|---|---|---|---|---|
| README.md:95 | 1.924 -> 1.953 (as `1.92`) | tc1-5090-137 | reducer | / Qwen3-30B-A3B QLoRA · RTX 5090 · TC1's field recipe, both frameworks on one stack, e4b at every default (`e4b.train.h2 |
| README.md:96 | 1.160 -> 1.161 (as `1.160`) | tc1-5090-136 | reducer | / Packed 4,096-token Qwen3 training · RTX 5090 · e4b at its defaults (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packe |
| docs/CHOOSING.md:228 | 1.924 -> 1.953 (as `1.924`) | tc1-5090-137 | reducer | / RTX 5090, 32 GB / Qwen3-30B-A3B / train / Resident, `enable_fast_train(model, dgrad=True)` at its defaults / Unsloth's |
| docs/COMPARATORS.md:69 | 0.454 -> 0.442 (as `0.45`) | tc1c-h100-9 | cell | / axolotl / `e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0` / 2026-10-04 / |
| docs/REPRODUCE.md:21 | 0.486 -> 0.470 (as `0.49`) | tc1-5090-137 | reducer | / Training, TC1's field recipe (Qwen3-30B-A3B, seq 2048, micro-batch 2 x accum 4) / Unsloth 2026.10.8 / not read at 0.58 |
| docs/REPRODUCE.md:21 | 0.493 -> 0.478 (as `0.49`) | tc1-5090-137 | reducer | / Training, TC1's field recipe (Qwen3-30B-A3B, seq 2048, micro-batch 2 x accum 4) / Unsloth 2026.10.8 / not read at 0.58 |
| docs/REPRODUCE.md:21 | 1.924 -> 1.953 (as `1.924`) | tc1-5090-137 | reducer | / Training, TC1's field recipe (Qwen3-30B-A3B, seq 2048, micro-batch 2 x accum 4) / Unsloth 2026.10.8 / not read at 0.58 |
| docs/REPRODUCE.md:22 | 0.486 -> 0.470 (as `0.49`) | tc1-5090-137 | reducer | / Training, packed 4,096-token rows / Unsloth / not read today (no lane registered it) / Unsloth spends **1.160x** e4b's |
| docs/REPRODUCE.md:22 | 0.493 -> 0.478 (as `0.49`) | tc1-5090-137 | reducer | / Training, packed 4,096-token rows / Unsloth / not read today (no lane registered it) / Unsloth spends **1.160x** e4b's |
| docs/REPRODUCE.md:22 | 1.160 -> 1.161 (as `1.160`) | tc1-5090-136 | reducer | / Training, packed 4,096-token rows / Unsloth / not read today (no lane registered it) / Unsloth spends **1.160x** e4b's |
| docs/REPRODUCE.md:25 | 0.486 -> 0.470 (as `0.49`) | tc1-5090-137 | reducer | / Training Mixtral-8x7B, both frameworks on one stack / Unsloth / not read today (its lane didn't register it) / Unsloth |
| docs/REPRODUCE.md:25 | 0.493 -> 0.478 (as `0.49`) | tc1-5090-137 | reducer | / Training Mixtral-8x7B, both frameworks on one stack / Unsloth / not read today (its lane didn't register it) / Unsloth |
| docs/REPRODUCE.md:82 | 1.160 -> 1.161 (as `1.160`) | tc1-5090-136 | reducer | - **Claim:** `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio`. Unsloth spends **1.160×** e4b's GPU |
| docs/REPRODUCE.md:102 | 1.924 -> 1.953 (as `1.924`) | tc1-5090-137 | reducer | - **The other recipe:** at TC1's field recipe (seq 2048, micro-batch 2 × accum 4) the device ratio read 1.924 (medians |
| docs/STATUS-RECORD.md:176 | 0.480 -> 0.473 (as `0.48`) | tc1-5090-16 | cell | (device-busy 0.48 e4b vs 0.23 Unsloth; the fused path records 4.3 × fewer |
| docs/STATUS-RECORD.md:194 | 1.924 -> 1.953 (as `1.92`) | tc1-5090-137 | reducer | every current default, both frameworks profiled on one stack): **Unsloth spends 1.92× e4b's GPU time per step; wall-cloc |
| docs/STATUS-RECORD.md:196 | 0.486 -> 0.470 (as `0.49`) | tc1-5090-137 | reducer | against e4b's 0.49, so a slower host stretches Unsloth's step more. The GPU-time ratio is what remains as host overhead  |
| docs/STATUS-RECORD.md:196 | 0.493 -> 0.478 (as `0.49`) | tc1-5090-137 | reducer | against e4b's 0.49, so a slower host stretches Unsloth's step more. The GPU-time ratio is what remains as host overhead  |
| docs/STATUS-RECORD.md:221 | 0.330 -> 0.317 (as `0.33`) | tc1-5090-138 | cell | **0.797** of its time. `aten::bmm`'s CPU time per call fell from 305 µs to 24.5 µs, for 4.1 % more device time and 0.33  |
| docs/STATUS-RECORD.md:221 | 0.331 -> 0.320 (as `0.33`) | tc1-5090-138 | cell | **0.797** of its time. `aten::bmm`'s CPU time per call fell from 305 µs to 24.5 µs, for 4.1 % more device time and 0.33  |
| docs/STATUS-RECORD.md:229 | 1.055 -> 1.056 (as `1.05`) | tc1-5090-141 | reducer | with `auto`, inside the registered 1.05, and the bf16 arm 1.002. `bmm`'s CPU time per call fell to 0.092 of `l0`'s, devi |
| docs/STATUS-RECORD.md:299 | 1.160 -> 1.161 (as `1.160`) | tc1-5090-136 | reducer | amendment 68, every arm profiled): Unsloth spends **1.160×** e4b's GPU time per step, against 1.141 on amendment 66's ho |
| docs/STATUS-RECORD.md:370 | 0.665 -> 0.653 (as `0.665`) | tc1c-h100-3 | cell | EQUIVALENT. e4b's own step fell ×0.768 since the first H100 box; on this card it is now mostly device-bound (device-busy |
| docs/STATUS-RECORD.md:374 | 0.454 -> 0.442 (as `0.45`) | tc1c-h100-9 | cell | peak VRAM (34.08 vs 24.27 GB) and about equal energy — quoted beside the default-settings 0.817, never in place of it. g |
| docs/STATUS.md:92 | 1.924 -> 1.953 (as `1.92`) | tc1-5090-137 | reducer | **Qwen3-30B-A3B against Unsloth 2026.9.14: Unsloth spends 1.92× e4b's GPU time per step (medians of two draws); |
| docs/STATUS.md:95 | 0.486 -> 0.470 (as `0.49`) | tc1-5090-137 | reducer | the host: Unsloth runs 14× e4b's CPU ops per step and keeps its GPU busy for 0.33 of its step against e4b's 0.49, so a |
| docs/STATUS.md:95 | 0.493 -> 0.478 (as `0.49`) | tc1-5090-137 | reducer | the host: Unsloth runs 14× e4b's CPU ops per step and keeps its GPU busy for 0.33 of its step against e4b's 0.49, so a |
| docs/STATUS.md:119 | 1.160 -> 1.161 (as `1.160`) | tc1-5090-136 | reducer | **On packed 4,096-token rows, Unsloth 2026.9.14 spends 1.160× e4b's GPU time per step (medians of two draws).** The |
| docs/capabilities.json:237 | 0.454 -> 0.442 (as `0.45`) | tc1c-h100-9 | cell | "e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0", |
| docs/solutions/qlora-fused-moe-experts.md:182 | 0.454 -> 0.442 (as `0.45`) | tc1c-h100-9 | cell | - `e4b.train.h2h.unsloth.qwen3.5090.2026-10-03` — TC1 amendment 19, the matched position after amendments 10-15: Unsloth |
| llms-full.txt:666 | 0.454 -> 0.442 (as `0.45`) | tc1c-h100-9 | cell | - `e4b.train.h2h.unsloth.qwen3.5090.2026-10-03` — TC1 amendment 19, the matched position after amendments 10-15: Unsloth |
| llms-full.txt:1142 | 0.454 -> 0.442 (as `0.45`) | tc1c-h100-9 | cell | "e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0", |
| llms-full.txt:2199 | 1.924 -> 1.953 (as `1.92`) | tc1-5090-137 | reducer | **Qwen3-30B-A3B against Unsloth 2026.9.14: Unsloth spends 1.92× e4b's GPU time per step (medians of two draws); |
| llms-full.txt:2202 | 0.486 -> 0.470 (as `0.49`) | tc1-5090-137 | reducer | the host: Unsloth runs 14× e4b's CPU ops per step and keeps its GPU busy for 0.33 of its step against e4b's 0.49, so a |
| llms-full.txt:2202 | 0.493 -> 0.478 (as `0.49`) | tc1-5090-137 | reducer | the host: Unsloth runs 14× e4b's CPU ops per step and keeps its GPU busy for 0.33 of its step against e4b's 0.49, so a |
| llms-full.txt:2226 | 1.160 -> 1.161 (as `1.160`) | tc1-5090-136 | reducer | **On packed 4,096-token rows, Unsloth 2026.9.14 spends 1.160× e4b's GPU time per step (medians of two draws).** The |
| llms-full.txt:2962 | 1.924 -> 1.953 (as `1.924`) | tc1-5090-137 | reducer | / RTX 5090, 32 GB / Qwen3-30B-A3B / train / Resident, `enable_fast_train(model, dgrad=True)` at its defaults / Unsloth's |
| llms-full.txt:3489 | 0.480 -> 0.473 (as `0.480`) | tc1-5090-16 | cell | - e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile [measured]: Lane TC1c with TC1, 2026-10-02: the profiled  |
| llms-full.txt:3489 | 0.562 -> 0.553 (as `0.562`) | tc1c-h100-2 | cell | - e4b.train.h2h.unsloth.qwen3.h100.2026-10-02.dispatch-profile [measured]: Lane TC1c with TC1, 2026-10-02: the profiled  |
| llms-full.txt:3541 | 0.454 -> 0.442 (as `0.45`) | tc1c-h100-9 | cell | - e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0 [measured]: TC1c amendment 8, one rented H100 NVL (tc1c-h100-15): the  |
| llms-full.txt:3618 | 0.486 -> 0.470 (as `0.49`) | tc1-5090-137 | reducer | - e4b.train.padded-delta-switches.field.5090.2026-10-22 [measured]: TC1 amendment 74 (#835), descriptive with one fixed  |
| llms-full.txt:3618 | 0.492 -> 0.477 (as `0.49`) | tc1-5090-139 | reducer | - e4b.train.padded-delta-switches.field.5090.2026-10-22 [measured]: TC1 amendment 74 (#835), descriptive with one fixed  |
| llms-full.txt:3618 | 0.493 -> 0.478 (as `0.49`) | tc1-5090-137 | reducer | - e4b.train.padded-delta-switches.field.5090.2026-10-22 [measured]: TC1 amendment 74 (#835), descriptive with one fixed  |
| llms-full.txt:3625 | 0.884 -> 0.871 (as `0.884`) | tc1-5090-141 | reducer | - e4b.train.single-ladder-auto.gpu-bound.5090.2026-10-08 [measured]: TC1 amendment 72 (#835), amendment 71's box unchang |
| llms-full.txt:3626 | 0.427 -> 0.411 (as `0.43`) | tc1-5090-138 | reducer | - e4b.serve.p123.default-decode-census.qwen3.5090.2026-10-09 [measured]: Lane P123, one RTX 5090: a kernel-class census  |
| llms-full.txt:3635 | 0.464 -> 0.446 (as `0.464`) | tc1-5090-147 | reducer | - e4b.train.p129.fused-qkv.qwen3.5090.2026-10-10 [measured]: Lane P129 Amendment 3 (#835), one RTX 5090 (tc1-5090-147, A |
