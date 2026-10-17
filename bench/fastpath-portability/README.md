# Qwen3-30B-A3B's fast paths on other families: a portability matrix (Track B1, 2026-10-17)

An assessment from records and code only: no runs, no rentals, no default changed. It asks which of the levers behind
"the Qwen 30B speed" carry to the other families e4b serves, and why the rest don't.

- **Code read at:** e4b `ac6b894e`, grouped-nf4-gemm `5c3d3099` (v0.46.0 plus two CI commits; `kernel/nf4_grouped.py` is
  unchanged since v0.46.0).
- **Families:** the 13 committed fixtures in `tests/fixtures/topology_controls/` (revision and sha256 in its
  `SOURCES.json`).
- **Inputs from the other Track B tables:** B3's (K16, K19) is folded in. B2's dispatch audit (per family and component, W1
  and W16) is **pending**, and the cells marked "B2" wait for it.

## 1. The receipt behind "the Qwen 30B speed"

`e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07` (`bench/p116/`), on one RTX 5090 (170 SMs):

- **Setup:** Qwen3-30B-A3B NF4 on the default `serve_paged` server, with bucketed decode graphs.
- **Lever:** grouped-nf4-gemm K33's bandwidth-targeted NF4 decode GEMV (`GNF4_GEMV_BW`, `prmt32` at K33's plans).
- **Result:** one request decodes at 97.89 → 121.70 tok/s, g1 1.2417 (the lower of the two ABBA pairs; geomean 1.243).
  Sixteen requests: 1.0003, because a step with T > 1 never reaches it. Quality is within P110's bar.
- **Default since:** grouped-nf4-gemm#509 (0.43.0) made it `GNF4_GEMV_BW=auto`.

It is one lever in a stack. The other Qwen3-30B defaults with their own receipts, all on the 5090:

| lever | read | claim |
|---|---|---|
| bucketed decode graphs | 9.02x at W1, 5.60x at W16, against eager | `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03` |
| fused stack: q/k/v, two glue rounds, router epilogue | 1.43x at W1 | `e4b.serve.p115.fused-stack-speed.qwen3.5090.2026-10-07` |
| K25 NF4 grouped small-M | 1.57x at W16 | `e4b.serve.p121.k25-w16.qwen3.5090.2026-10-08` |
| decode folds on prefill rows | 1.12x prefill | `e4b.serve.p130.prefill-glue-speed.qwen3.5090.2026-10-10` |

## 2. How each lever is gated

| lever | gate | narrow because | where |
|---|---|---|---|
| K33 BW GEMV (auto) | exact (N, K) in `_BW_SHAPES` **and** ≥ 160 SMs | the shapes are Qwen3-30B-A3B's two: (1536, 2048), (2048, 768) | gnf4 `kernel/nf4_grouped.py:190`, `:221-225` |
| K33 plans | `_BW_PLANS`: Qwen3-30B, Granite-3.1 and OLMoE shapes; (16, 256, 4, 1) for any other shape | auto uses only Qwen's two | `nf4_grouped.py:199-201` |
| dot-pad (K6-B), K1 baked plan | the same two shapes, ≥ 160 SMs | shadowed by BW at the defaults | `nf4_grouped.py:171`, `:1649-1652` |
| fused stack defaults | `model_type` in `FUSION_DEFAULT_FAMILIES = {qwen3_moe, qwen3_5_moe_text, qwen3_5_moe}` | family allow-list; the code itself engages by structure | e4b `serve_paged.py:156-160`, `:372` |
| fused q/k/v (serving and training) | class name `== "Qwen3MoeAttention"` | class-name gate | `qkv_fuse.py:105`; `train_qkv_fuse.py:58` |
| prefill glue default | `model_type in {"qwen3_moe"}` | family allow-list | `serve_paged.py:173-176`, `:416` |
| decode graphs | CUDA, all-VRAM placement, fused fp8 KV append (cc ≥ 8.9, not AMD) | hardware and placement only | `serve_paged.py:186-204`; `fp8_paged_kv.py:48-65` |
| K25 NF4 grouped small-M | NF4 store, device grouping, ≤ 256 routed rows | structure only | `hot_residency.py:410-484` |
| PDL | cc ≥ 9.0, ≤ 8 rows | hardware feature | gnf4 `int4_b32.py:100-123` |
| int4 expert store (opt-in) | a calibrated pack per family; `gpt_oss` takes its native MXFP4 store instead | per-family calibration | e4b `int4_experts.py:568` |
| K16 int4 attention (opt-in) | `nn.Linear` children of a class named `*Attention`; K % 32 | structure only | `int4_attn.py:391-400`; gnf4 `int4_smallm.py:100-112` |
| K19 split tiles (opt-in) | int4 store, > 256 routed rows (route), > 512 (split), `E4B_K19_PREFILL_BIG_M` | structure only | `hot_residency.py:201-207`, `:749`, `:1049-1056` |

Only K33's auto gate in grouped-nf4-gemm, two serving allow-lists and one class name tie a fast path to Qwen. Everything
else is gated on structure, shape class, rows or hardware.

## 3. The matrix: served families on the 5090

Classes (the maintainer's): **1** works through shape-general code; **2** blocked by narrow auto-dispatch; **3** needs
tuning; **4** needs a kernel or code port; **5** cannot safely apply. "Engages" without a receipt on that card means the gain
is inferred. A train-only or unserved family is N/A.

Served (B3: `serve_paged`'s `paged_state_refusal` refuses none of these): Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3.6-35B-A3B,
OLMoE, Granite-3.1, gpt-oss-20b, Gemma-4-26B, Mixtral-8x7B, ERNIE-4.5. Not served (conv, mamba or latent MoE layers): LFM2,
Granite-4.0-h, both Nemotrons.

| lever (default?) | Qwen3-30B | Qwen3-235B | Qwen3.6 | OLMoE | Granite-3.1 | gpt-oss | Gemma-4 | Mixtral | ERNIE |
|---|---|---|---|---|---|---|---|---|---|
| decode graphs (default) | **1, record** (P109) | **M** | 1, engages | 1, engages | 1, engages | 1, engages | 1, engages | 1, engages | 1, engages |
| fused stack (default for Qwen) | **1, record** (P115) | 1, engages (inferred, solver placement, W1) | **1 for the router epilogue** (only it engages: quality PASS, step 0.973); q/k/v and folds 4 | 2, allow-list (quality unread) | **5**: FAM FAIL | **5**: FAM FAIL | **5**: not licensable with the instrument | **2**: quality PASS, blocked by the allow-list, no speed read | 2, allow-list; round-2 rotary refuses ERNIE (4) |
| fused q/k/v | **1, record** | 1, engages (inferred, solver placement, W1) | 4 (gated q, a different class) | 2 (class name) | 2 (class name; 0 engaged, P115 C) | 2 (class name); its biased q/k/v would also refuse: 4 | 2 (class name) | 2 (class name; 0 engaged, FAM) | 2 (class name) |
| prefill glue (default for qwen3_moe) | **1, record** (P130) | 1, engages (inferred, solver placement) | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| K33 BW GEMV, NF4, W1 | **1, record** (P116) | 3: no plan for (3072, 4096), (4096, 1536) | 3: no plan for (1024, 2048), (2048, 512) | **2: plans exist, auto excludes** | **2: plans exist, auto excludes** | N/A (native MXFP4) | 3 | 3 | 3 |
| K25 NF4 grouped small-M, W16 (default) | **1, record** (P121) | **M** | 1, engages | **1, quality licensed** (P96) | **1, quality licensed** (P96) | N/A (MXFP4: K21, `e4b.serve.p90…`) | 1, engages | 1, engages | 1, engages |
| int4 expert store + K19/K23/tiles (opt-in) | **1, record** (int4 lanes P88/P89/P122/P126) | 3 + **M** | 3: calibration and quality per family | 3 | 3 | N/A (MXFP4 store) | 3 | 3 | 3 |
| K16 int4 attention, 17–64 rows (opt-in; B3) | **1, record** (P124) | 1, engages (inferred, solver placement) | 1, engages (10 full-attention layers only) | 1, engages | 1, engages | 1, engages (bias carried) | 1, engages (audio tower: B2) | 1, engages | 1, engages |
| K19 split tiles, prefill (opt-in; B3) | 1, engages (bitwise on the A2000; gain unread) | 1 under the store; **M** | 1, possibly 3 (16 rows per expert) | 1, engages | 1, engages | **4** (MXFP4 route is K21) | 1, possibly 3 (down K = 704) | 1, engages | 1, engages |
| attention: fp8 compute, split-KV, fused KV append | B2 | B2 | B2 | B2 | B2 | B2 | B2 | B2 | B2 |

**The two int4 rows are bounded by their stores.** Every K19 cell sits under the int4 expert store's row: outside
Qwen3-30B that store is class 3, so read "1, engages" as "1 under the store (store: 3)". K16 likewise sits under the RTN int4
attention store (`E4B_SERVE_ATTN_INT4`, opt-in), whose quality is per family and unread.

**M**: Qwen3-235B-A22B's NF4 experts (about 128 GB) do not fit one 32 GB or 96 GB card. `serve_paged` serves it through
the solver placement (`E4B_PAGED_PLACEMENT=solver`, `serve_paged.py:466`, `:572`, `:1396-1431`). M marks the levers whose
gate needs the all-VRAM placement there:
- decode graphs: batched graphs refuse anything else (`serve_paged.py:186-204`, `:1305-1308`);
- K25: only that path turns on device grouping, which defaults off (`hot_residency.py:48`);
- the int4 expert store, which needs every layer all-hot (`serve_paged.py:1128-1152`), with K19 under it.

The fused stack, q/k/v, prefill glue and K16 don't depend on the placement in the code, so their 235B cells read "engages
(inferred, solver placement)": no served 235B read exists.

Every "engages" cell outside Qwen3-30B is gain inferred. Quality is per family and unread unless a claim is named.

## 4. Other cards

- **K33 (auto):** its only hardware gate is ≥ 160 SMs.
  - **RTX 4090** (sm_89, 128 SMs): **2, possibly 3** for every family, Qwen3-30B included. The plans were picked on sm_120.
  - **RTX PRO 6000** (sm_120, 188 SMs): **1, engages**, gain inferred.
  - **A2000** (sm_86, 26 SMs): 2, and correctness only.
- **PDL** needs cc ≥ 9.0, a hardware feature: **N/A (hardware)** on the 4090 and the A2000. That is not unsafe, and no code
  port makes it apply.
- **Decode graphs** need the fused fp8 KV append (cc ≥ 8.9). The A2000 (sm_86) runs eager: **4**, a port, if it ever mattered.
- **K16, K19, K25 and the fusions** have no SM or arch gate. On the 4090 and the PRO 6000 they read **1 (engages), possibly 3**:
  one default plan, gain unmeasured.
- **The ≥ 160-SM gates do not check the vendor.** From the code, not tested: a ROCm part reporting ≥ 160 would take the BW
  route under auto with `tree` decode.

## 5. Findings

1. **Three gates make the Qwen speed Qwen-only, and none is a kernel limit.**
   - K33's exact-shape allow-list leaves Granite's and OLMoE's recorded K33 plans unused.
   - `FUSION_DEFAULT_FAMILIES` keeps Mixtral's quality-passed fused stack off.
   - The `Qwen3MoeAttention` class name keeps fused q/k/v off for structurally similar attention.
   Each could become a structural or shape gate with the same safe fallback the code already has.
2. **The family allow-list is doing two jobs.** It records a quality licence: FAM failed Granite and gpt-oss at the
   family's own floor, and Gemma-4 can't be licensed with the current instrument. Code engagement is already structural.
   A per-(family, lever) licence table driven by claims would separate the two, keeping the blocks where quality failed
   and opening Mixtral once a speed read exists.
3. **e4b's `[fast]` floor is still `grouped-nf4-gemm>=0.30.0`** (`pyproject.toml:84`). The K33 GEMV defaults on only from
   grouped-nf4-gemm 0.43.0, so a fresh install at the floor doesn't get P116's speed. P116 registered raising the floor
   as its consequence; it hasn't happened. A packaging decision, not changed here.
4. **The docs are out of step with the code in two places.**
   - The serving solution page lists the glue and epilogue knobs as `=1` opt-ins, but `serve_paged` resolves them to `auto`
     for the Qwen families.
   - It says `serve_paged` reads `E4B_SERVE_ATTN_INT4_ARTIFACT`, `_FINGERPRINT` and `_DUMP`. Only `enable_from_env` reads
     them, and nothing in the package calls it.

## 6. Proposed paired readings (pre-registration drafts, NOT run)

| id | question | arms | why |
|---|---|---|---|
| R-a | does K33 at its recorded Granite and OLMoE plans pay on the served W1 step? | `GNF4_GEMV_BW=1` vs `0`, ABBA, P110 quality bar, 5090 | cells marked 2 with plans already selected |
| R-b | Mixtral fused stack speed | knobs auto vs off, ABBA, W1, 5090 | quality already PASS; the allow-list waits on this |
| R-c | K33 on sm_89 / 128 SMs | Qwen3-30B W1, `GNF4_GEMV_BW=1` vs `0`, RTX 4090 | the SM gate's only untested side |
| R-d | fused q/k/v by structure | a structural gate in place of the class name; A2000 correctness first ($0), then a paid read | OLMoE, Gemma-4, ERNIE |

$0 A2000 probes stay limited to cells the records can't settle: correctness and dispatch only, no speed. None is needed
yet: the engagement cells above come from records (P115 Phase C, FAM) or from the code.
