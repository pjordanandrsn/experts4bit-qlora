# LAN model stores — `admin@10.0.0.68`

Roots scanned: `/share/models`, `/share/ZFS532_DATA/hf-models`

Generated 2026-09-09 by `bench/support/nas_inventory.py`. Sizes are the sum of
`*.safetensors` (what a load must read), not `du` of the directory — several of
these carry duplicate formats (`metal/`, `original/`) a load never touches.

| directory | model_type | experts | hidden | shards | safetensors | fetch | verdict |
|---|---|---|---:|---:|---:|---|---|
| `.hf-home` | `—` | — | — | 0 | 0.0 GB | — | config only |
| `AI21-Jamba2-Mini` | `jamba` | 16 | 4096 | 21 | 96.1 GB | — | **fits 64 GB** (~27 GB loaded) |
| `DeepSeek-V4-Flash` | `deepseek_v4` | 256 | 4096 | 46 | 148.7 GB | — | **fits 64 GB** (~42 GB loaded) |
| `DeepSeek-V4-Flash-0731` | `deepseek_v4` | 256 | 4096 | 48 | 155.4 GB | — | **fits 64 GB** (~44 GB loaded) |
| `DeepSeek-V4-Flash-Base` | `deepseek_v4` | 256 | 4096 | 46 | 274.4 GB | — | needs a bigger host (~77 GB loaded) |
| `DeepSeek-V4-Flash-DSpark` | `deepseek_v4` | 256 | 4096 | 48 | 155.4 GB | — | **fits 64 GB** (~44 GB loaded) |
| `DeepSeek-V4-Flash-Vision-Exp` | `deepseek_v4` | 256 | 4096 | 48 | 156.3 GB | — | **fits 64 GB** (~44 GB loaded) |
| `DeepSeek-V4-Pro` | `deepseek_v4` | 384 | 7168 | 64 | 805.3 GB | — | needs a bigger host (~227 GB loaded) |
| `DeepSeek-V4-Pro-0813` | `deepseek_v4` | 384 | 7168 | 66 | 831.4 GB | — | needs a bigger host (~234 GB loaded) |
| `DeepSeek-V4-Pro-Base` | `deepseek_v4` | 384 | 7168 | 64 | 1,495.7 GB | — | needs a bigger host (~421 GB loaded) |
| `DeepSeek-V4-Pro-DSpark` | `deepseek_v4` | 384 | 7168 | 66 | 831.4 GB | — | needs a bigger host (~234 GB loaded) |
| `ERNIE-4.5-21B-A3B-PT` | `ernie4_5_moe` | — | 2560 | 9 | 40.9 GB | — | ~41 GB bf16, dense — fits |
| `EXAONE-4.0.1-32B` | `exaone4` | — | 5120 | 14 | 59.6 GB | — | ~60 GB bf16, dense — too big |
| `EXAONE-4.5-33B` | `exaone4_5` | — | 5120 | 2 | 64.0 GB | — | ~64 GB bf16, dense — too big |
| `GLM-4.7-Flash` | `glm4_moe_lite` | 64 | 2048 | 48 | 58.2 GB | — | **fits 64 GB** (~20 GB loaded) |
| `GLM-5.3` | `glm_moe_dsa` | 256 | 6144 | 141 | 703.7 GB | — | needs a bigger host (~198 GB loaded) |
| `Hunyuan-A13B-Instruct` | `hunyuan_v1_moe` | 64 | 4096 | 33 | 149.8 GB | — | ~150 GB bf16, dense — too big |
| `Hy3` | `hy_v3` | 192 | 4096 | 99 | 556.5 GB | — | needs a bigger host (~168 GB loaded) |
| `K-EXAONE-236B-A23B` | `exaone_moe` | 128 | 6144 | 96 | 441.6 GB | — | needs a bigger host (~131 GB loaded) |
| `Kimi-K2.6-DFlash` | `qwen3` | — | 7168 | 1 | 6.5 GB | — | ~6 GB bf16, dense — fits |
| `Kimi-K2.7-Code-DFlash` | `qwen3` | — | 7168 | 1 | 6.5 GB | — | ~6 GB bf16, dense — fits |
| `Kimi-Linear-48B-A3B-Instruct` | `kimi_linear` | 256 | 2304 | 20 | 91.5 GB | — | **fits 64 GB** (~26 GB loaded) |
| `Ling-3.0-flash` | `bailing_hybrid` | 512 | 2560 | 24 | 237.5 GB | — | needs a bigger host (~68 GB loaded) |
| `Ling-3.0-tiny` | `bailing_hybrid` | 128 | 1536 | 32 | 14.7 GB | — | **fits 64 GB** (~5 GB loaded) |
| `Llama-3.1-405B-Instruct` | `llama` | — | 16384 | 191 | 756.0 GB | — | ~756 GB bf16, dense — too big |
| `Llama-4-Maverick-17B-128E-Instruct` | `llama4` | 128 | 5120 | 55 | 748.0 GB | — | needs a bigger host (~210 GB loaded) |
| `MiniMax-H3` | `—` | — | — | 0 | 0.0 GB | — | config only |
| `MiniMax-M2` | `minimax_m2` | 256 | 3072 | 130 | 214.3 GB | — | needs a bigger host (~60 GB loaded) |
| `MiniMax-M2.7` | `minimax_m2` | 256 | 3072 | 125 | 214.3 GB | — | needs a bigger host (~60 GB loaded) |
| `MiniMax-M2.7-DFlash` | `qwen3` | — | 3072 | 1 | 2.4 GB | — | ~2 GB bf16, dense — fits |
| `MiniMax-M3` | `minimax_m3_vl` | 128 | 6144 | 59 | 795.5 GB | — | needs a bigger host (~224 GB loaded) |
| `MiniMax-M3-DSpark` | `qwen3` | — | 6144 | 1 | 10.0 GB | — | ~10 GB bf16, dense — fits |
| `Mixtral-8x22B-Instruct-v0.1` | `mixtral` | 8 | 6144 | 59 | 261.9 GB | — | needs a bigger host (~81 GB loaded) |
| `Moonlight-16B-A3B-Instruct` | `deepseek_v3` | 64 | 2048 | 27 | 29.7 GB | — | **fits 64 GB** (~10 GB loaded) |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nemotron_h` | 128 | 2688 | 14 | 61.3 GB | — | **fits 64 GB** (~17 GB loaded) |
| `OLMoE-1B-7B-0924` | `olmoe` | 64 | 2048 | 3 | 12.9 GB | — | **fits 64 GB** (~4 GB loaded) |
| `Olmo-3-1125-32B` | `olmo3` | — | 5120 | 14 | 60.0 GB | — | ~60 GB bf16, dense — too big |
| `Phi-3.5-MoE-instruct` | `phimoe` | 16 | 4096 | 17 | 78.0 GB | — | **fits 64 GB** (~24 GB loaded) |
| `Qwen-AgentWorld-35B-A3B` | `qwen3_5_moe` | 256 | 2048 | 21 | 64.6 GB | — | **fits 64 GB** (~21 GB loaded) |
| `Qwen3-30B-A3B` | `qwen3_moe` | 128 | 2048 | 16 | 56.9 GB | — | **fits 64 GB** (~18 GB loaded) |
| `Qwen3-32B` | `qwen3` | — | 5120 | 17 | 61.0 GB | — | ~61 GB bf16, dense — too big |
| `Qwen3-4B-Instruct-2507` | `qwen3` | — | 2560 | 3 | 7.5 GB | — | ~7 GB bf16, dense — fits |
| `Qwen3-8B` | `qwen3` | — | 4096 | 5 | 15.3 GB | — | ~15 GB bf16, dense — fits |
| `Qwen3-Coder-480B-A35B-Instruct` | `qwen3_moe` | 160 | 6144 | 241 | 894.4 GB | — | needs a bigger host (~268 GB loaded) |
| `Qwen3.5-4B` | `qwen3_5` | — | 2560 | 2 | 8.7 GB | — | ~9 GB bf16, dense — fits |
| `Qwen3.5-9B` | `qwen3_5` | — | 4096 | 4 | 18.0 GB | — | ~18 GB bf16, dense — fits |
| `Qwen3.6-35B-A3B` | `qwen3_5_moe` | 256 | 2048 | 26 | 67.0 GB | — | **fits 64 GB** (~24 GB loaded) |
| `Qwen3.8-2.4T-A95B` | `qwen3_5_moe_text` | 512 | 8192 | 213 | 4,556.4 GB | — | needs a bigger host (~1,382 GB loaded) |
| `Qwen3.8-27B` | `qwen3_5` | — | 5120 | 18 | 51.7 GB | — | ~52 GB bf16, dense — too big |
| `Ring-2.6-1T` | `bailing_hybrid` | 256 | 8192 | 175 | 970.3 GB | — | needs a bigger host (~273 GB loaded) |
| `Seed-OSS-36B-Instruct` | `seed_oss` | — | 5120 | 15 | 67.3 GB | — | ~67 GB bf16, dense — too big |
| `Solar-Open2-250B` | `solar_open2` | 320 | 4096 | 94 | 466.2 GB | — | needs a bigger host (~143 GB loaded) |
| `_manifests` | `—` | — | — | 0 | 0.0 GB | — | config only |
| `bge-m3` | `xlm-roberta` | — | 1024 | 0 | 0.0 GB | — | config only |
| `e4b-serve` | `—` | — | — | 0 | 0.0 GB | — | config only |
| `gemma-4-26B-A4B` | `gemma4` | 128 | 2816 | 4 | 48.1 GB | — | **fits 64 GB** (~17 GB loaded) |
| `gemma-4-26B-A4B-it` | `gemma4` | 128 | 2816 | 2 | 48.1 GB | complete | **fits 64 GB** (~17 GB loaded) |
| `gemma-4-31B` | `gemma4` | — | 5376 | 2 | 58.3 GB | complete | ~58 GB bf16, dense — too big |
| `gemma-4-31B-it` | `gemma4` | — | 5376 | 2 | 58.3 GB | complete | ~58 GB bf16, dense — too big |
| `gnf4-bench` | `—` | — | — | 0 | 0.0 GB | — | config only |
| `gpt-oss-20b` | `gpt_oss` | 32 | 2880 | 3 | 12.8 GB | — | **fits 64 GB** (~4 GB loaded) |
| `granite-3.0-1b-a400m-instruct` | `granitemoe` | 32 | 1024 | 1 | 2.5 GB | — | **fits 64 GB** (~1 GB loaded) |
| `granite-4.2-3b` | `granite` | — | 2560 | 2 | 6.8 GB | — | ~7 GB bf16, dense — fits |
| `hf-120b` | `—` | — | — | 0 | 0.0 GB | — | config only |
| `hf-235b` | `—` | — | — | 0 | 0.0 GB | — | config only |
| `orpheus-3b-0.1-ft` | `llama` | — | 3072 | 4 | 14.1 GB | — | ~14 GB bf16, dense — fits |
| `whisper-large-v3` | `whisper` | — | — | 3 | 8.6 GB | — | ~9 GB bf16, dense — fits |
| `whisper-large-v3-turbo` | `whisper` | — | — | 1 | 1.5 GB | — | ~2 GB bf16, dense — fits |
| `moonshotai_Kimi-K3` | `kimi_k3` | 896 | 7168 | 96 | 1,453.7 GB | — | needs a bigger host (~409 GB loaded) |

## Coverage of the 9 claimed families

| claimed model_type | checkpoints here | smallest | reachable without renting |
|---|---:|---|---|
| `olmoe` | 1 | `OLMoE-1B-7B-0924` (12.9 GB, ~4 GB loaded) | yes |
| `qwen3_moe` | 2 | `Qwen3-30B-A3B` (56.9 GB, ~18 GB loaded) | yes |
| `qwen3_5_moe` | 2 | `Qwen-AgentWorld-35B-A3B` (64.6 GB, ~21 GB loaded) | yes |
| `gpt_oss` | 1 | `gpt-oss-20b` (12.8 GB, ~4 GB loaded) | yes |
| `gemma4` | 4 | `gemma-4-26B-A4B-it` (48.1 GB, ~17 GB loaded) | yes |
| `gemma4_text` | 0 | — | **no — not on the LAN at all** |
| `granitemoe` | 1 | `granite-3.0-1b-a400m-instruct` (2.5 GB, ~1 GB loaded) | yes |
| `kimi_k3` | 1 | `moonshotai_Kimi-K3` (1,453.7 GB, ~409 GB loaded) | no — needs a bigger host |
| `deepseek_v4` | 9 | `DeepSeek-V4-Flash` (148.7 GB, ~42 GB loaded) | yes |

**8 of 9** claimed families have a real checkpoint on the LAN. Absent: `gemma4_text`. Present but too large for a 64 GB probe host: `kimi_k3`.

So the rentals this lane actually needs are the large families, not bandwidth.

**The loaded-size figures are estimates from config geometry, not measurements.**
They are computed by `loaded_gb()`; RSS cannot validate them because it counts
mmap'd checkpoint pages alongside memory actually held. Use them to order the
queue, not to decide that a rental is unnecessary -- a load attempt is cheap and
records its own outcome.

One caveat on the table above: **`gemma4_text` has no separate release** and never
will. It is the text tower of a `gemma4` multimodal config -- what a text-only
QLoRA loads -- so a `gemma-4-26B-A4B` checkpoint is the evidence for both rows,
reached by a different config path rather than a different download. Read its
zero as "no separate artifact", not as "nothing to test".
