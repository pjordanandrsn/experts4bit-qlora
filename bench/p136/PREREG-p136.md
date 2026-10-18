# P136 — R-e served fastpath census (registered, no run)

Status: **registration only**. This page registers a future P123-method census for the missing cells in [fastpath portability](../fastpath-portability/README.md). No P136 GPU result, cost, or receipt exists. The matrix remains `inf` until a complete, independently reviewed reading can justify `rec`. P136 does not change library code, default switches, quality gates, or a public speed claim. It does not read the PRO 6000 cell.

## Question and fixed subjects

Where does the shipped all-VRAM graph server spend one steady decode step, including the non-lean W16 expert path where applicable? The subjects are the committed e4b default for **all three families**, with every serving and grouped-nf4 knob unset except `E4B_PAGED_MAX_SEQS=16`; Mixtral also fixes `E4B_PAGED_MAX_TOKENS_PER_SEQ=768` to fit its 512-token prompts and 160-token row budget. The gpt-oss W16 cell is the default NF4-arena subject (the matrix's non-lean K25), not an MXFP4 expert-store read. Each speed arm must record `int4_store_kinds == []` and `int4_expert_layers == 0`; any lever store VOIDs the family. The local e4b source pin is the **exact clean merged commit** supplied in a future envelope. It cannot be known at PR time. `p136_drive.sh` derives it from a clean checkout HEAD and refuses an override. The box verifies the installed `direct_url.json` commit before fetching a checkpoint. Grouped-nf4-gemm is v0.46.1 **commit** `70bc03cb7a61b03c4e801e7423c58c103a1b6d7d` (never a movable tag at launch), transformers 5.17.0, bitsandbytes 0.50.2, and the box image must carry torch 2.8.0+cu128. Nsight Systems CLI is pinned to P123's recorded full build, `2025.6.1.190-256136895201v0` (`p123-5090-1`, `bench/p123/receipts/p123-5090-1/versions.txt`, SHA-256 `5a2f899b5d377be205eca92a5aecf25fa80171eceda8af7a79ac24051779d457`). The box accepts that exact preinstalled binary, or installs `nsight-systems-cli-2025.6.1` from NVIDIA's devtools apt source. As checked on 2026-10-17, [its Ubuntu 20.04 `InRelease`](https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/InRelease) is signed by `AE09FE4BBD223A84B2CCFCE3F60F4B3D7FA2AF80` (the older `7fa2af80` key), while [NVIDIA's CUDA key rotation](https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/) names the newer `3bf863cc` key. The installer fingerprint-checks both keys and scopes their keyring to the devtools source; an install or version mismatch refuses before any model fetch. `nsys status -e` is retained as a diagnostic: P123's successful 5090 capture reported unavailable CPU profiling features there, while this lane disables CPU sampling. Staged scripts, map, calibration, and premise tests are SHA-256 pinned by `staged-p136.sha256`.

| Registered future run ID | Checkpoint and revision | MoE layers | Captures |
|---|---|---:|---|
| `p136-gemma4-5090-1` | `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 30 | W1 and W16 |
| `p136-mixtral-5090-1` | `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` | 32 | W1 and W16 |
| `p136-gptoss-5090-1` | `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee` | 24 | W16 only |
| `p136-a2000-dry-1` | no checkpoint | — | inventory and staged-byte check only |

The checkpoint pins are the FAM registrations. P136's files derive from P123 by named substitutions: P136 family IDs/pins, a W16-only gpt-oss loop, the Mixtral token pool, the v0.46.1 kernel pin, and a new family-neutral reducer. P109 supplies the fixed wikitext prompt rows and decode slope; P39 supplies the NF4 bake; SC1b supplies the capture window and class reducer. The `kernel_classes_nf4.json` map derives from P123 v1.1 and adds three grouped-mxfp4 expert kernels present at the pinned v0.46.1 commit; those three names cannot fire under this registered default subject. They are harmless map entries retained for a future lever read, which would need separate registration. The map remains a **starting map**, not proof that every new family is classified. Unmapped kernels or broken MoE segments VOID the cell; a map amendment needs a reviewed preregistration before a repeat read.

## Method and read rule

On one RTX 5090 (sm_120), after host refusals, install/pin tripwires, script self-tests and exactly **19 passing** card premise tests with no skipped, failed or errored cases, fetch and bake exactly one family. Two fresh, **unprofiled** default-server processes, S1a and S1b, run P109's 32/160-token decode slope (three repetitions) for the registered workloads. They provide an unprofiled reference and noise check, not a comparative speed claim. A W1 or W16 self-pair outside [0.97, 1.03] VOIDs the family reading.

For each registered workload, warm and skip 32 steps, then bracket exactly 64 full-batch decode-only steps. Nsight captures separate graph-mode and node-mode reports (`cuda-sw,nvtx`, CUDA profiler capture range). Export both to SQLite, run SC1b's `arm` reducer with this lane's frozen class map, 64-step driver receipt, exact MoE layer count, and the matching unprofiled reference. The profiled wall is never quoted as speed. Record the expert store kind and device grouping used for K25, plus per-step `P`, every kernel class, the residual/unmapped class, in-graph and out-of-graph idle, overlap, kernel count, default fusion/dispatch census, and peak memory. Raw reports and SQLite stay on the host; reduced records, logs, manifest and forensics travel as receipts.

The P136 reducer first VOIDs missing/invalid files, source-pin or model mismatch, a set knob, a missing graph server, invalid reference slope, wrong 32-skip/64-step window, wrong batch, an incomplete class/idle table or kernel count, and SC1b's incomplete-map, segment, diagnostic, clock, profiler-inflation, or node-inflation labels. It emits `READ` only for a complete family receipt with all its registered workloads. A `READ` is a **record of those cells** for review, not a matrix edit or a conclusion that a family is faster. Any failure is a VOID; an instrument or map change is a new registration, never a silent retry.

## $0 dry run, host, budget, and stop

`p136-a2000-dry-1` is a planned $0 A2000 (sm_86) inventory/staging probe only. Before even this probe, claim `gpu:a2000` on the coordination bus and obtain the future envelope ACK. Its box path verifies staged hashes, CUDA availability and machine inventory, writes `DRY_ONLY`, and exits **before install, model fetch, bake, served graph capture, or speed arms**. sm_86 cannot exercise the sm_89+ fp8 KV append and bucketed graph route, so it cannot close R-e or validate the 5090 capture. The controller's `P136_DRIVE_DRYRUN=1` prints a transport plan without connecting to a host; it is not evidence of the A2000 box path.

Paid runs require a separate post-merge envelope request and maintainer ACK naming the merged e4b SHA, current host quote, download cap, deadline, and exact run ID. Host class: verified-secure, direct SSH, one nominal 32 GB RTX 5090 with sm_120 and at least 30000 MiB VRAM measured by the box, at least 60 GiB host RAM, and free disk at launch **and immediately before fetch** of at least 200 GB for Gemma-4/gpt-oss or 220 GB for Mixtral. FAM's prior size basis is 49.9 GB for Gemma-4 and 93.4 GB for Mixtral; the latter needed at least 165 GB immediately before fetch. P136 deliberately uses the larger launch floor. The controller refuses an occupied work directory and does not create or destroy a rental.

For envelope sizing only, request an RTX 5090 at no more than **$0.85/hour**, download charge no more than **$0.011/GB**, and guards of 4 h Gemma-4, 6 h Mixtral, 4 h gpt-oss. A conservative 100 GB download cap per run gives worst planned costs of **$4.50, $6.20, and $4.50**, respectively: `rate × guard hours + download GB × $0.011`. Combined ceiling **$15.20**; this is a registration budget, not spending authority. The future quoted offer and egress estimate must fit both the per-run and combined caps. If a host or quote does not fit, stop and amend before any launch. No automatic rerun is registered.

STOP-1: bad stage hash, dirty/overridden source SHA, undeclared ID, bad CUDA/card/torch/RAM/disk, missing premise, or any tripwire ends before model fetch. STOP-2: the box checks remaining deadline before fetch, bake, each reference arm and each capture; it leaves a 600-second receipt margin. STOP-3: failed reference arms prevent captures; missing or failed captures produce VOID, not a partial `rec`. STOP-4: the controller fetches reduced receipts and excludes raw Nsight, checkpoint and arena; it verifies the nonce and completion marker. A deadline, liveness loss, or failed marker remains an incomplete attempt. A fresh run or map change needs maintainer direction and a new registration.

## Receipt and authority boundary

After an ACKed run, hand off run ID, exact source pins, `verdict_p136.json`, `summary.txt`, driver and census records, staged manifest, and SHA-256 per retained file. The handoff may use an immutable evidence URI and its digest; the public PR names only run ID, repository file and digest, never private machine paths or coordination IDs. The independent reviewer is the planner-incubation reviewer; the maintainer alone merges and grants a later run envelope. Until that envelope, **no A2000 claim/probe, GPU rental, or model fetch is authorized by this PR**.

## Amendment 1 (2026-10-18): the premise requires the norm + rotary route the kernel offers

`p136-gemma4-5090-1`, attempt 1, stopped at STOP-1 before the model fetch. The card passed the host refusals, the tripwires and every self-test (RTX 5090, 32607 MiB, torch 2.8.0+cu128). The premise ran 18 passed and 1 failed, and the box exited 25. The failing case, `test_the_folds_replay_exactly_as_their_padded_eager_step`, asserted that `rope_norm_heads` ran during capture and saw `{'rmsnorm_rows': 48, 'rmsnorm_resid_rows': 28, 'rope_norm_heads': 0}`. Since `1ddcb0ae` (#1472), round 2's attention fold calls grouped-nf4-gemm's `rope_norm_qk` (grouped-nf4-gemm#528: q's and k's norm + rotary in one launch) wherever the kernel exports it, and never calls `rope_norm_heads`. The pinned `70bc03cb` exports it. The fold ran, and the census assertion `(9, (2, 2))` passed. The test predates the switch (it was last changed in `98f919de`) and is gated to sm_89+, so the sm_86 dry run could not reach it. Receipt files, SHA-256:

- `summary.txt` `3443f22f29c66c145a8196c51a77baff14e88f6e415cea0f1892f9c27d02b073`
- `logs/premise.log` `709827ae15914c96161a60a250ceed5f0f8f42884662e7f24a3765cd0ad05250`
- `REFUSAL` `033cadc0ce6948e9788cd50fcbb27b13d027c1b7c915df7b71b44172a78a550c`
- `receipt.json` `ce2c17fbc81b6248eddb3411fae055b60d30fe75b2cf8464873cec980c77e1a2`
- `teardown-proof.json` `3d0db8dcd6c68249cd8a110668afb031e7eab59d8548bfc81fccce3b8ba09401`

The amendment is test-only. P136 stages its own copy of the test, `bench/p136/premise/test_fused_glue_decode_graphs_gpu.py`, under the same box name. `tests/test_fused_glue_decode_graphs_gpu.py` on main asserts the pre-#1472 route, so it fails on sm_89+ with a grouped-nf4-gemm that exports `rope_norm_qk`. It stays byte-pinned by P115, P115c, P115d, P123 and P125, whose kernel pins predate `rope_norm_qk`, and P136 runs its own route-exact copy. A later lane that stages the main file at such a kernel pin needs the same copy. `p136_drive.sh` and `test_registration.py` map that one staged name to the copy. The copy reads the route from the kernel module before wrapping anything, and wraps `rope_norm_qk` before the fold binds it. Where the kernel exports `rope_norm_qk`, both the capture assertion and the padded-eager assertion require `rmsnorm_rows`, `rmsnorm_resid_rows` and `rope_norm_qk` to have run. Otherwise they require `rope_norm_heads`, as before. A fold that silently takes `rope_norm_heads` on a kernel offering `rope_norm_qk` therefore fails. The graph == padded-eager comparisons are unchanged; they have not yet run on an sm_89+ card, and they must pass. The premise is still exactly 19 passing tests, with none skipped, failed or errored. `staged-p136.sha256` moves by one row: `test_fused_glue_decode_graphs_gpu.py` goes from `5a8eb6b4b0e4d515d8789f1fb729bb847fd74d53288f2faa3efac07e04e1a740` to `8e5b2f775cf5c0ddde349822624a7d44b40f3898738cc64ceb31f5f3f4ea5aae`. No other staged file, threshold, read rule, host term or budget changes.

`p136-gemma4-5090-1` has used one of its two attempts, so its next attempt is its last. The Mixtral and gpt-oss run IDs keep both of theirs. Every further run still needs its own post-merge execution approval, naming this amendment's merge commit as the source.

## Amendment 2 (2026-10-18): each family's second and last attempt has its own run ID

A run ID is used once: the launcher refuses an ID that already has a ledger row or a receipt directory. After Amendment 1,
`p136-gemma4-5090-1` therefore could not take its second attempt, and the runner and driver accepted no other gemma4 ID.

This amendment registers one more ID per family, each that family's second and last attempt:
- `p136-gemma4-5090-2`: gemma4's last attempt, after `p136-gemma4-5090-1` stopped at the premise (Amendment 1).
- `p136-mixtral-5090-2` and `p136-gptoss-5090-2`: used only if that family's `-1` fails after its box has started.

Each takes its family's registered model, revision, MoE layer count, disk floor and workloads, unchanged. Both case lists
gain the three IDs: `p136_run.sh` (staged) and `p136_drive.sh`. An attempt counts once an instance was created and its box
started; a create refused because the offer was taken or vanished is not an attempt. Two failed attempts lapse a family. Every
attempt counts against the lane's combined ceiling, which stays at **$15.20**.

**Launch rule under the ceiling.** A run launches only if the lane's billed-to-date plus that run's worst at its quoted
offer is at or under **$15.20**. The run's worst is its guard hours × the quoted hourly rate plus the 100 GB download cap
× the quoted per-GB rate. Billed-to-date is the sum of the billed cost of every `p136-*` row in the launcher's ledger, read
at the launch, not carried forward from memory. A run that does not fit waits for an amendment.

`staged-p136.sha256` moves by one row: `p136_run.sh` goes from `09feafcafafb42e66ecc7a018ddeef3b31541c6c6ff52e92bdd4c9d1dbc86290` to `79ad608032f0044d1fd4ef699d3979e4a5e7a84fadf667558dc155454f1eb79a`. No threshold, read rule, host term,
workload or other staged file changes.

## Amendment 3 (2026-10-18): a refusal before create consumes an ID but is not an attempt

`p136-gemma4-5090-2` was refused at $0 before any instance existed. The pinned offer was briefly absent from the provider's
search at the create: the same search, repeated 10 s apart, returned it, then not, then again. The launcher still records a
refused launch, with a $0 receipt (`receipt.json` `dc80f64b0865376402d4924e532c94c669a5b2352ca3f905e283d7b93dbb5a05`) and a ledger row, so the ID is spent although nothing
was created. A family therefore needs more registered IDs than attempts.

This amendment registers three more IDs per family, used in order, each taking that family's registered model, revision,
MoE layer count, disk floor and workloads unchanged:
- gemma4: `p136-gemma4-5090-3`, `p136-gemma4-5090-4`, `p136-gemma4-5090-5`;
- mixtral: `p136-mixtral-5090-3`, `p136-mixtral-5090-4`, `p136-mixtral-5090-5`;
- gpt-oss: `p136-gptoss-5090-3`, `p136-gptoss-5090-4`, `p136-gptoss-5090-5`.

The attempt rule, replacing Amendment 2's:
- a launch refused **before** create consumes its ID but is not an attempt;
- an attempt is a create whose box started;
- a family lapses after **two** box-started failures;
- gemma4 has used one (`p136-gemma4-5090-1`, Amendment 1), so it has one left.

The launch rule under the $15.20 ceiling is unchanged, with billed-to-date read from the ledger at the launch. Both case lists
gain the nine IDs: `p136_run.sh` (staged) and `p136_drive.sh`. `staged-p136.sha256` moves by one row: `p136_run.sh` goes from
`79ad608032f0044d1fd4ef699d3979e4a5e7a84fadf667558dc155454f1eb79a` to `119a26c583ae1b426dcf2bd3ee4f14b4c1f02239c91df2936f803e561d5ee9f7`. No threshold, read rule, host term, workload or other staged file changes.
