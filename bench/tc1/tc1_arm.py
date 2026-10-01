#!/usr/bin/env python3
"""tc1_arm.py -- lane TC1 (TC1-PREREG.md) per-arm driver: e4b, Unsloth, plain HF+PEFT+bnb at MATCHED adapter precision and
MATCHED LoRA init, with the frozen-base probe.

PROVENANCE: copied from bench/tp4/tp4_arm.py @ e4b main 10ce711d (the tp4 tree is a committed receipt and is never edited).
Every behaviour not named below is kept as tp4 had it: status vocabulary, receipt JSON fields, step timing (median 11..N),
held-out evals, C1, stub codes, phase clock and watchdog, the T10 detector dry-runs, the #542 structural expert selection,
the selftest scaffolding. What is new, named so the files can be diffed:

  T19 --adapter-dtype fp32|native (default fp32): after the adapters exist, EVERY framework's trainable adapter parameters
      are cast to fp32 -- e4b included. tp4 cast only the non-e4b arms (`if a.framework != "e4b"`), so its 4.490 row holds
      e4b's 192 expert-adapter tensors in bf16 against Unsloth's 576 in fp32 (TC1-PREREG "Why this lane" 2). `native`
      leaves every arm as its loader built it (e4b: bf16 expert adapters + fp32 attention adapters from
      load_moe_4bit_streaming / add_attention_lora -- the `fused_attn4_shipped` arm). adapter_dtypes_before/after and
      lora_cast_to_fp32 are recorded as before; `adapter_dtype` names the flag. fp32 adapters on e4b's paths: the reference
      path computes the delta in A's dtype and casts back (lora.py ExpertsLoRA._lora, LoRALinear.forward); the fused path
      hands A/B to grouped-nf4-gemm's fused_grouped_lora, whose lora_delta_grouped casts the activations to A.dtype on all
      three of its paths and adds `delta.to(out.dtype)` (kernel/nf4_qlora.py at the v0.34.0 tag, read 2026-10-01; the
      reference-path claim is exercised on CPU by _selftest_frozen_probe_real).
  T20 --lora-init native|matched:<seed> (default native): after the adapters exist and after the T19 cast, every LoRA A is
      overwritten with a deterministic per-slot tensor and every LoRA B is asserted zero (a non-zero B is a harness_error,
      exit 18, before any step). Slot key = (layer, kind, expert): kinds q/k/v/o (expert -1) and gate_up/down (one slot per
      expert). Generator: seed_slot = int.from_bytes(sha256(f"{seed}|{layer}|{kind}|{expert}")[:8], "big") into
      torch.Generator().manual_seed; values U(-b, b), b = 1/sqrt(fan_in), fan_in = the in-width of the [r, in] slot (PEFT's
      kaiming_uniform(a=sqrt(5)) bound), materialised in fp32 on CPU and copied into the parameter's own orientation, dtype
      and device. The mapping from parameters to slots is STRUCTURAL (lora_slots): e4b ExpertsLoRA `<prefix>_lora_A`
      [E, r, in] -> one slot per expert, kind gate_up-like when the stack reads the hidden state (in == hidden) and
      down-like when it writes it (out == hidden), the parameter's own name only as a tie-break; e4b LoRALinear lora_A
      [r, in] -> q/k/v/o by the projection's name; PEFT (HF, and Unsloth through PEFT) lora.Linear
      `…q_proj.lora_A.default.weight` [r, in] -> q/k/v/o; PEFT 0.21.2 ParamWrapper (target_parameters): ONE A [r*E, in]
      per wrapped 3-D stack, expert e = rows e*r:(e+1)*r (peft/tuners/lora/layer.py get_delta_factors: A.reshape(E, r, in)),
      the kind read from the wrapper's `parameter_name` (the module path carries none: the outer wrapper is `…experts`, the
      inner `…experts.base_layer`), classified by the same structural rule on the wrapped stack's logical shape (a
      Params4bit's quant_state.shape; `is_transposed` bases reported swapped). Receipt: `matched_init` {seed, n_slots_set,
      n_slots_expected = n_attention_projections (structural, on the adapted model) + 2 * n_layers * n_experts, unmapped,
      complete, kinds, b_nonzero, hidden, mapping_rules, distribution}; init_sha is computed after the overwrite. A
      framework whose slots cannot all be mapped records complete=false -- the reducer VOIDs it; nothing is silently native.
  T21 frozen_base_probe on every row after load: layer 0, expert 0 of the gate_up-like and down-like stacks, and layer 0's
      q_proj, each dequantised to bf16 the way the framework's own forward would (e4b: Experts4bit._dequantize_expert, the
      function its forward recomputes with; bnb Params4bit / Linear4bit: bitsandbytes.functional.dequantize_4bit with the
      parameter's own quant_state -- the first expert's blocks when the statistics are not nested, the whole stack sliced
      when they are; a bf16 stack: its own bytes), sha256 of the bytes, regime label (`nf4/64`, `nf4/64+dq` for double-quant,
      `bf16`, …), and the POSITIVE control: one byte of a COPY of the packed storage flipped, dequantised again, the sha
      asserted different. Receipt: `frozen_base_probe` {slots: {gate_up, down, q_proj: {sha, regime, n_bytes, method,
      control_detects_flip}}, control_detects_flip, experts_module, attention_module, errors}. The reducer reads SAME-BYTES /
      DIFFERENT / N-A across arms; this file compares nothing. Measured here on CPU (2026-10-01): e4b's per-expert NF4/64
      bytes equal bnb's whole-stack quantisation sliced at expert 0, so SAME-BYTES is attainable; and e4b's
      quantize_attention_projections_4bit builds its Params4bit with bnb's default compress_statistics=True, so the e4b
      attention slot reads `nf4/64+dq`.
  T22 Environment names TC1_* (TC1_INSTANCE_ID, TC1_ANCHOR_JSON, TC1_BOX_CLASS, TC1_PHASE_BUDGET_S, TC1_ARM_ALARM_S); the
      default work tree /root/tc1; receipts cite `--prereg tc1/TC1-PREREG.md`; the selftest prefix tc1_selftest_.
      The tiny selftest models now carry the two REAL adapter layouts (e4b's and PEFT 0.21.2's) over one seeded frozen
      base, so the T20 selftest can assert identical step-0 losses and measure the matched trajectories' agreement.

  P2-1 Unsloth knobs and engagement (phase 2): --unsloth-moe-backend default|grouped_mm|unsloth_triton|native_torch sets
      UNSLOTH_MOE_BACKEND in THIS process before `import unsloth` (select_moe_backend() reads it once, lru_cached:
      unsloth_zoo moe_utils.py:1061-1085, UPSTREAM-NOTES "Unsloth"); --unsloth-speed-tilt 1 sets UNSLOTH_MOE_RECOMPUTE=0
      UNSLOTH_MOE_GC_REPLAY_PIN=1 (moe_utils.py:751/689/717); both recorded under `unsloth_knobs` and `env.unsloth_env`.
      Engagement: the three backends behind select_moe_backend() -- forward_native_grouped_mm (moe_utils.py:3790),
      forward_triton_grouped_gemm (:4153), forward_native_moe_loop (:4357) -- and the bnb-4bit entry point
      forward_moe_backend_bnb4bit (moe_utils_bnb4bit.py:354-414) are wrapped with call counters (an absent name is
      recorded, never a crash); per step min/max in `unsloth_backend_calls_per_step_min/_max`, plus `moe_backend_selected`
      from select_moe_backend() after load. --unsloth-double-quant off|default: when FastLanguageModel.from_pretrained
      NAMES `bnb_4bit_use_double_quant` (or `quantization_config`) in its signature, read at runtime, the matched arms
      pass double-quant OFF; what the loaded expert Params4bit's quant_state says (`nested`) is recorded either way
      (`unsloth_double_quant`), else `how: unknown-default`.
  P2-2 --framework axolotl (phase 2, axolotl-arm-spec.md): load_axolotl builds the axolotl 0.20.0 config dict for the arm
      (axolotl_config_dict, pure), finds the attention projections and the expert stacks BY STRUCTURE on a meta-device
      skeleton of the checkpoint (hf_targets, #542), runs `load_cfg(DictDefault(...))` (cli/config.py:227-318: validation,
      prepare_plugins -> the KernelsPlugin registers when listed), `load_tokenizer(cfg)` and `ModelLoader(cfg, tok).load()`
      -- which IS the sequence axolotl.train.train uses up to the trainer (loaders/model.py:305-378: PLUGIN_MANAGER
      pre_model_load, apply_post_plugin_pre_model_load_patches -> _apply_moe_expert_quantization_patch, _build_model,
      apply_post_model_build_patches -> _finalize_moe_expert_quantization (patch_manager.py:1075), the PEFT wrap through
      load_adapter, post_model_load) -- and hands the model to run_arm's own loop. Receipt `axolotl`: version + its
      torch/transformers/peft/bitsandbytes, the config dict, quantized_moe_experts_n (get_moe_quantized_count, moe_quant.py)
      and the parametrization census (axolotl_expert_census: Bnb4bitParametrization entries by class name,
      n_bnb4bit_unwrapped = experts modules whose innermost base carries a bnb parametrization on every stack),
      experts_implementation (cfg and config._experts_implementation), expert_backend / plugins. --axolotl-best 1 adds
      plugins [KernelsPlugin], expert_backend scattermoe, moe_bnb_fast true (args.py:38/55; plugin.py:134-163 runs inside
      ModelLoader.load() through PLUGIN_MANAGER.pre_model_load). The matched init and the frozen-base probe read a
      parametrized stack's logical shape and quant_state from its Bnb4bitParametrization (bitsandbytes/nn/parametrize.py:
      the packed original is a plain uint8 [N, 1] Parameter; the quant_state lives on the parametrization module).

Exit codes added: 18 matched init impossible (a LoRA B is not zero after construction).

----- the tp4 docstring, unmodified -----

tp4_arm.py -- lane tp4 (TP4-PREREG.md) per-arm driver, THREE frameworks: e4b, Unsloth, plain HF+PEFT+bnb.

PROVENANCE: copied from bench/tp3/tp3_arm.py @ e4b main e0cfb488 (itself a byte copy of tp2_arm.py @ 769edb1d + T10);
the tp2/tp3 trees are committed receipts and are never edited. Every behaviour not named below is kept as tp3 had
it: status vocabulary, receipt JSON fields, step timing (median 11..N), held-out evals, C1, stub codes, selftest
scaffolding, the T10 structural attention census. What is new, named so the files can be diffed:

  T11 --framework hf / --arm hf: the field's plain stack -- transformers AutoModelForCausalLM with
      BitsAndBytesConfig(load_in_4bit, nf4, bf16 compute, no double quant) + PEFT LoraConfig with target_modules =
      every attention projection found BY STRUCTURE (q/k/o present, v optional -- the same predicate as e4b's
      detector) and target_parameters = every 3-D floating expert stack, ALSO found by structure since #542 -- the
      rule was a `"experts" in name` substring, which selects NOTHING on a family that names the module something
      else and then degrades silently to attention-only, so an empty or implausible selection now REFUSES (PEFT
      >= 0.17 creates one A/B pair per expert, the shape e4b's ExpertsLoRA and Unsloth's MoE LoRA also have). What
      is and is not 4-bit is what the
      census records: transformers' bnb quantizer converts nn.Linear only, so the expert stacks stay bf16 and the
      row says so (census Params4bit_expert_stacks == 0). Its engagement counter is experts_forward (module
      forward calls); a loader exception is classified exactly as Unsloth's (oom / refused / load_fault).
  T12 --template alpaca|clinical: the Unsloth notebooks' Alpaca prompt (three slots, an empty Input slot when the
      row has no input, EOS appended -- the notebook's formatting_prompts_func) beside tp2's clinical FMT. The
      tokens file records template + eos so the tokenizer_agree check re-derives rows the same way.
  T13 --micro-batch M (default 1 = tp2 byte-for-byte): M rows per micro-batch, right-padded to the longest row
      with the tokenizer's pad id (eos when absent), attention_mask passed to EVERY framework, labels -100 on
      pads; rows in fixed order ((i*accum+j)*M+k). tokens_per_step counts real tokens; tokens_padded_per_step is
      recorded beside it. With M == 1 nothing changes (no mask for e4b, all-ones for the others, as tp2).
  T14 --optim adamw_torch|adamw_8bit, --weight-decay, --lr-schedule constant|linear, --warmup-steps: the same
      optimizer call in every arm (adamw_8bit = bitsandbytes.optim.AdamW8bit, the notebooks' optim); the linear
      schedule is transformers' get_linear_schedule_with_warmup formula (lr factor step/warmup then
      (N-step)/(N-warmup), so step 0 trains at lr 0 exactly as the notebooks do). lr_per_step is recorded.
      Defaults reproduce tp2 (adamw_torch, wd 0.01, constant).
  T15 Unsloth: --unsloth-targets (comma list; the notebooks' seven by default); when FastLanguageModel refuses a
      family with a message naming FastModel / vision / multimodal, the arm retries ONCE with FastModel and the
      receipt records loader_used + loader_fallback_reason (pre-registered, TP4-PREREG "Arms").
  T16 Receipt fields added: framework-specific hf_targets (PEFT version, n target modules, the expert parameter
      names), micro_batch, optimizer string, lr_per_step, tokens_padded_per_step, template, loader_used.
      Environment names TP4_* (TP4_INSTANCE_ID, TP4_ANCHOR_JSON, TP4_BOX_CLASS).

----- the tp3 docstring, unmodified -----

tp3_arm.py -- lane tp3 per-arm driver, BOTH frameworks.

PROVENANCE: copied byte-for-byte from bench/h2h-20260906/tp2/tp2_arm.py @ 769edb1d58a77580dec758b8db010e049037970c
(then committed under tp2 at 9738404); the tp2 tree is a committed receipt and is never edited. Every behaviour not
named below is kept byte-for-byte: status vocabulary, receipt JSON fields, step timing (median 11..N), held-out
evals, C1, stub codes, selftest scaffolding. The original tp2 docstring follows unmodified, then T10.

  T10 e4b#434 (against e4b#426, merged as #435 at main 5dad2a7): the attn-4bit expected count is the library's
      STRUCTURAL census -- detect_attention_projections(model, exact_linear=True).expected_count, snapshotted
      BEFORE the conversion (exact_linear=True matches `type is nn.Linear`; after conversion Linear4bit would not
      match -- the detector docstring's rule) -- never 4 * n_layers. Post-#426 the conversion consumes the same
      census, so n_attn4 == expected holds by construction: the assert is now a consistency tripwire (harness/library
      skew, a mutated model between census and convert, detector drift between lanes) and its real product is
      provenance. A missing k_proj is REFUSED (the detector's SystemExit, caught here) with the refusing modules
      named -- never a void_attn4 count guess. The receipt gains `structural_expected_n_attn4` (census.expected_count;
      null on non-attn4 arms and non-e4b frameworks) and `detector_version` (the library's own DETECTOR_VERSION if it
      ships one, else "<experts4bit_qlora version>+det:<sha256 of the detector source>[:12]" -- deterministic, no
      library change). Families: granite 128, olmoe 64, qwen3 192, mixtral 128 (exactly 4 x n_layers -- verdicts
      unchanged); Gemma-4 115 (25 x 4 + 5 x 3, the k_eq_v layers -- compares correctly for the first time); gpt-oss
      still REFUSED on the bias rule, which fires after admission (#435 CHANGELOG). The pre-registration written into
      every receipt is the `--prereg` argument -- REQUIRED for a real run (main() refuses without it, before any cell
      or stub), so no receipt can ever cite a pre-registration the run did not pass (P41: `--prereg p41/P41-PREREG.md`);
      the selftest sets PREREG explicitly, keeping the byte-for-byte tp2 behaviour it verifies (#434 follow-up, CEO
      review on PR #447).

----- the original tp2_arm.py docstring, unmodified -----

tp2_arm.py -- lane tp2 (P40: e4b vs Unsloth per MoE family, one box, one fixture) per-arm driver, BOTH frameworks.

Base: p38_arm.py as amended (amendment 3: snapshot-dir resolution in the Unsloth branch; amendment 4: U8 on the innermost
experts module), whose measurement code is tp1_train_smoke.py's (= n17_cell.py's): eval_loss(), the C1 hashes from the
bytes that persist, the byte-flip positive control, PowerSampler / idle_power, host_fingerprint(), init_sha, per-step
timing and kernel-call counters, refusal/OOM/fault receipts with exit codes. What is new, named so the files can be diffed:

  T1  --fam / --model / --revision per family (tp1's six, tp1's staged revisions); the receipt is <fam>_<fw>_<tag>.json
      (P40's naming) and carries `fam`, `model_type`, `n_layers` (from the config; text_config for gemma4), `revision`,
      and the snapshot directory it loaded from.
  T2  e4b branch = tp1's per-family loading (`load_moe_4bit_streaming(..., offload=<--offload>, pin=True, prefetch=False,
      quant_type="nf4")`, `.to("cuda")` only when resident, `verify_moe_4bit(strict=True)` -> `verify_failed` exit 7) plus
      P38's U4 attention-4bit (`quantize_attention_projections_4bit` BEFORE `add_attention_lora`; the count is asserted
      == 4 * n_layers, else `void_attn4`; a `SystemExit` from that function -- TRAIN_ATTN_4BIT refusing a bias-carrying
      projection -- is a `refused` row, never caught into a bf16 fallback).
  T3  --arm attn_only (gpt-oss's secondary row, tp1's D1/D8): the reference path with no expert adapter present, bf16
      attention (tp1's arm), which PROBES enable_fast_train / enable_batched_train (count, then disable) and a
      side-effect-free attn-4bit probe (does any q/k/v/o carry a bias?) and writes/refreshes the family's `fused_attn4`
      and `reference_attn4` refusal stubs with what it measured (a stub the run script wrote first, citing tp1, keeps
      its citation and gains the probe).
  T4  Unsloth branch = P38's U1 generalised: --unsloth-loader FastLanguageModel (P38's) | FastModel (an amendment if used);
      the snapshot dir resolved directly (amendment 3); the registered target list; the engagement banner captured;
      U8 evaluated on the innermost experts module (amendment 4) for any module whose last name component is `experts`
      (mlp.experts / block_sparse_moe.experts / layers.N.experts). A loader exception is classified: OOM -> `oom` (5);
      NotImplementedError / ValueError / TypeError / KeyError / AssertionError / ImportError or a message that says
      "not supported" / "unsupported" -> `refused` (3); anything else (CUDA errors included) -> `load_fault` (6).
  T5  Fixture per P40: --lr 2e-4, --accum 4 (micro-batches per optimizer step; rows in fixed order i*accum+j), --autocast 1
      (torch.autocast bf16 around forward+loss, training and eval), N=60, eval every 20 on the held-out 48, r 8 / alpha 16.
      A "step" is one optimizer step; s/step, tokens/step, J/step and the kernel-call counters are per optimizer step.
      Time-to-target is NOT computed (P40: out of scope).
  T6  --tokens-sha asserted (P38 U2); --expect-trainable N (the family's first e4b receipt, passed by the run script):
      a different count is RECORDED (`trainable_mismatch`) and the arm still trains -- the reducer applies P40's
      "same trainable count" validity rule; a trainable parameter outside the adapters stays `void_trainable` (15).
      `trainable_by_group` (attention / experts / other) is recorded so a mismatch is diagnosable.
  T7  The census regexes accept every family's expert-parameter names (gate_up_proj / down_proj / w1..w3 / input_linear /
      output_linear / gate_proj / up_proj) and every experts-module path.
  T8  --selftest: CPU, a tiny synthetic model, both branches' bookkeeping paths driven through the same run_arm() with
      mocked kernels (nf4_qlora.fused_grouped_lora, unsloth_zoo's forward_moe_backend_bnb4bit and the U8 predicate),
      the same receipt writer and the same stub paths. Verifies the file here, on a box with no GPU.
  T9  Device abstraction (cuda_sync / peak_gb / reset_peak / autocast) so T8 runs; every torch.cuda call is guarded.

Exit codes: 0 ok; 3 refused; 4 C1 failed (receipt written, arm void); 5 OOM; 6 load fault; 7 verify failed;
13 dataset/tokens mismatch; 15 void_trainable (non-adapter trainable); 18 matched init impossible (TC1, see above).
"""
import argparse
import contextlib
import faulthandler
import gc
import glob
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import types

import torch
import torch.nn as nn

PREREG = "tc1/TC1-PREREG.md"   # selftest-only: real runs must pass --prereg (main() refuses otherwise; no default)
HARNESS = ("tc1_arm.py (copy of tp4_arm.py @ 10ce711d + T19: --adapter-dtype fp32|native on EVERY framework; + T20: "
           "--lora-init matched:<seed>, per-slot deterministic LoRA A by structural slot mapping, B asserted zero, "
           "matched_init receipt; + T21: frozen_base_probe, layer-0 slots dequantised the framework's own way with a "
           "byte-flip control; + T22: TC1_ environment names, /root/tc1; + P2-1: Unsloth backend/tilt/double-quant knobs "
           "and backend engagement counters; + P2-2: --framework axolotl through axolotl's own ModelLoader)")   # a receipt must say WHICH harness produced it
EXPERT_ATTRS = ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")
EXPERT_PARAM_RE = re.compile(r"experts\.(?:.*\.)?(gate_up_proj|down_proj|gate_proj|up_proj|w[123]|input_linear|output_linear)$")
FMT = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
# T12: the Unsloth notebooks' alpaca_prompt, verbatim (unslothai/notebooks, e.g. nb/Qwen3_(14B)-Alpaca.ipynb); the notebook's
# formatting_prompts_func fills all three slots for every row (an empty Input slot when the row has none) and appends EOS_TOKEN.
ALPACA_PROMPT = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                 "Write a response that appropriately completes the request.\n\n"
                 "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")
UNSLOTH_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BANNER = "Enabling LoRA on MoE parameters"
UNSLOTH_BACKENDS = ("default", "grouped_mm", "unsloth_triton", "native_torch")   # select_moe_backend()'s three + default = the loader's own choice
# unsloth_zoo 2026.9.9, read from source (UPSTREAM-NOTES "Unsloth"): (module, function, counter key)
UNSLOTH_BACKEND_FUNCS = (("unsloth_zoo.temporary_patches.moe_utils", "forward_native_grouped_mm", "unsloth_grouped_mm"),      # moe_utils.py:3790
                         ("unsloth_zoo.temporary_patches.moe_utils", "forward_triton_grouped_gemm", "unsloth_triton"),       # moe_utils.py:4153
                         ("unsloth_zoo.temporary_patches.moe_utils", "forward_native_moe_loop", "unsloth_loop"),             # moe_utils.py:4357
                         ("unsloth_zoo.temporary_patches.moe_utils_bnb4bit", "forward_moe_backend_bnb4bit", "moe_bnb4bit_backend"))   # moe_utils_bnb4bit.py:354
UNSLOTH_BACKEND_KEYS = {"grouped_mm": "unsloth_grouped_mm", "unsloth_triton": "unsloth_triton", "native_torch": "unsloth_loop"}


def apply_unsloth_knobs(a):
    """P2-1: the Unsloth knobs go into THIS process's environment before `import unsloth`. Returns the receipt block
    {moe_backend_requested, speed_tilt, double_quant_requested, env_set}; `default` leaves the environment alone so what
    the loader chooses on its own is what runs -- and is recorded by select_moe_backend() after load."""
    out = {"moe_backend_requested": getattr(a, "unsloth_moe_backend", "default") or "default",
           "speed_tilt": bool(int(getattr(a, "unsloth_speed_tilt", 0) or 0)),
           "double_quant_requested": getattr(a, "unsloth_double_quant", "off") or "off", "env_set": {}}
    if getattr(a, "framework", None) != "unsloth":
        return out
    if out["moe_backend_requested"] != "default":
        os.environ["UNSLOTH_MOE_BACKEND"] = out["moe_backend_requested"]           # moe_utils.py:1061-1085
        out["env_set"]["UNSLOTH_MOE_BACKEND"] = out["moe_backend_requested"]
    if out["speed_tilt"]:
        os.environ["UNSLOTH_MOE_RECOMPUTE"] = "0"                                  # moe_utils.py:751 (_moe_recompute_enabled)
        os.environ["UNSLOTH_MOE_GC_REPLAY_PIN"] = "1"                              # moe_utils.py:717 (_momentary_pin_fits)
        out["env_set"].update({"UNSLOTH_MOE_RECOMPUTE": "0", "UNSLOTH_MOE_GC_REPLAY_PIN": "1"})
    return out


def unsloth_double_quant_kwargs(from_pretrained_fn, want_off):
    """P2-1: the from_pretrained kwargs that turn double-quant OFF, decided from the signature READ AT RUNTIME: only a
    parameter the signature NAMES is passed (`bnb_4bit_use_double_quant`, else `quantization_config` with a
    BitsAndBytesConfig); otherwise nothing is passed and the receipt says `unknown-default`. Pure: testable without unsloth."""
    dq = {"requested": None, "how": "unknown-default", "loaded_nested": None}
    kw = {}
    if not want_off:
        dq["how"] = "loader default (not requested)"
        return kw, dq
    try:
        params = inspect.signature(from_pretrained_fn).parameters
    except (TypeError, ValueError) as e:
        dq["how"] = f"unknown-default: signature unreadable ({type(e).__name__})"
        return kw, dq
    if "bnb_4bit_use_double_quant" in params:
        kw["bnb_4bit_use_double_quant"] = False
        dq.update({"requested": False, "how": "from_pretrained(bnb_4bit_use_double_quant=False)"})
    elif "quantization_config" in params:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
        dq.update({"requested": False, "how": "from_pretrained(quantization_config=BitsAndBytesConfig(nf4, bf16 compute, double_quant=False))"})
    else:
        dq["how"] = "unknown-default: from_pretrained names neither bnb_4bit_use_double_quant nor quantization_config"
    return kw, dq


def unsloth_backend_selected():
    """select_moe_backend() after load (lru_cached, so it is the value the model's forward will use); never raises."""
    try:
        from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
        return str(select_moe_backend())
    except Exception as e:
        return f"unavailable: {type(e).__name__}: {str(e)[:120]}"


def loaded_expert_double_quant(model):
    """What the loaded expert Params4bit's quant_state says about double quantisation (`nested`): True / False, or None
    when no expert Params4bit with a quant_state is found."""
    for name, p in model.named_parameters():
        if type(p).__name__ == "Params4bit" and EXPERT_PARAM_RE.search(name) and getattr(p, "quant_state", None) is not None:
            return bool(getattr(p.quant_state, "nested", False))
    return None


# ----------------------------------------------------------------------------- P2-2: axolotl helpers (pure; no axolotl import)
def axolotl_config_dict(a, base_model, mods, params, best=False, double_quant=False, dataset_path=None, output_dir=None):
    """The axolotl 0.20.0 config for ONE arm as the dict `load_cfg` accepts (axolotl-arm-spec.md "Config keys"; keys read in
    utils/schemas/config.py, utils/schemas/model.py, loaders/model.py, monkeypatch/moe_quant.py). The trainer-side keys
    (micro_batch_size, gradient_accumulation_steps, max_steps, learning_rate, ...) are set to the fixture even though THIS
    harness runs the loop, so the dict describes the run a user would launch. `datasets` must be present
    (utils/schemas/validation.py:276 "either datasets or pretraining_dataset is required"): it names the registered Alpaca
    subset file; nothing in load_cfg reads it. Double-quant: moe_quant.py:134-136 reads cfg.bnb_4bit_use_double_quant
    (None -> True) for the expert stacks; the Linear path hard-codes True and takes bnb_config_kwargs on top
    (loaders/model.py:1057, 1066). There is no router_aux_loss_coef config key in 0.20.0 (grep of the schemas); the
    model's value is recorded in the receipt instead."""
    cfg = {
        "base_model": base_model,
        "load_in_4bit": True, "adapter": "qlora", "quantize_moe_experts": True,                           # schema 955-964; moe_quant.py:150-181
        "bnb_4bit_use_double_quant": bool(double_quant),
        "bnb_config_kwargs": {"bnb_4bit_use_double_quant": bool(double_quant)},
        "lora_r": int(a.r), "lora_alpha": int(a.alpha), "lora_dropout": 0.0,                              # dropout 0 is forced with target_parameters
        "lora_target_modules": list(mods), "lora_target_parameters": list(params),                      # loaders/adapter.py:176-195 -> LoraConfig
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},       # loaders/model.py:628-631
        "bf16": True, "sequence_len": int(a.seq), "seed": int(a.seed),
        "micro_batch_size": int(getattr(a, "micro_batch", 1) or 1), "gradient_accumulation_steps": int(a.accum), "max_steps": int(a.steps), "num_epochs": 1,
        "learning_rate": float(a.lr), "weight_decay": float(getattr(a, "weight_decay", 0.0)), "warmup_steps": int(getattr(a, "warmup_steps", 0) or 0),
        "lr_scheduler": "linear", "optimizer": "adamw_bnb_8bit",
        "datasets": [{"path": dataset_path or "/root/tc1/data/ds_alpaca.json", "type": "alpaca"}],
        "output_dir": output_dir or "/root/tc1/axolotl-out",
    }
    if best:                                                                                                # integrations/kernels/args.py:38, 55; plugin.py:134-163
        cfg.update({"plugins": ["axolotl.integrations.kernels.KernelsPlugin"], "expert_backend": "scattermoe", "moe_bnb_fast": True})
    return cfg


def axolotl_expert_census(model):
    """The parametrization census: how many parameters carry a bitsandbytes Bnb4bitParametrization (by class name --
    bitsandbytes/nn/parametrize.py), how many experts modules have EVERY stack so parametrized (n_bnb4bit_unwrapped, read on
    the innermost base under the PEFT wrappers), and the experts-module count. Pure structure; no axolotl import."""
    n_quant, n_entries, n_mods, n_mods_4bit, samples = 0, 0, 0, 0, []
    for name, m in model.named_modules():
        pz = getattr(m, "parametrizations", None)
        if pz is None or not hasattr(pz, "items"):
            continue
        for pname, plist in pz.items():
            n_entries += 1
            if any(type(x).__name__ == "Bnb4bitParametrization" for x in plist):
                n_quant += 1
                if len(samples) < 4:
                    samples.append(f"{name}.{pname}")
    for name, m in model.named_modules():
        if is_experts_module(name):
            n_mods += 1
            inner = innermost_experts(m)
            pz = getattr(inner, "parametrizations", None)
            if pz is not None and hasattr(pz, "items") and len(pz) >= 2 and all(any(type(x).__name__ == "Bnb4bitParametrization" for x in plist) for _, plist in pz.items()):
                n_mods_4bit += 1
    return {"quantized_moe_experts_n": n_quant, "parametrized_params": n_entries, "n_experts_modules": n_mods, "n_bnb4bit_unwrapped": n_mods_4bit, "samples": samples}
DEV = "cuda"
# #548: the share of an arm's own alarm the prologue may consume before the arm refuses itself (see phase_budget_for)
PROLOGUE_BUDGET_SHARE = 0.35


# ----------------------------------------------------------------------------- T9: device abstraction
def cuda_sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 3) if DEV == "cuda" else 0.0


def reset_peak():
    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def autocast_ctx(enabled):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=True)


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# ----------------------------------------------------------------------------- #548: where the time before step 1 goes
PROC_T0 = time.perf_counter()          # after the imports: the interpreter's own import cost precedes this mark
_FIRST_ARM = [True]                    # a real lane runs one arm per process; the selftest runs many in one


class Phases:
    """The named-phase clock for everything before step 1 (and the epilogue), plus a watchdog that refuses LOUDLY.

    #548: a 30B-class arm spent ~33 min between `LOAD OK` and step 1 while the receipt carried exactly one number
    about that window (`load_s`), so an arm that ALARMED and an arm that was merely slow read identically. Three
    properties are what turn this into a receipt that answers the question rather than one that carries more numbers:

      EXHAUSTIVE   `prologue_s` is measured independently of its parts and `prologue_unattributed_s` is the
                   residual, so time spent in a phase nobody thought to name still appears as a NUMBER, never as
                   a hole. A future prologue surprise is therefore visible before anyone knows what to call it.
      SYNCHRONISED every boundary cuda-syncs, so device work is billed to the phase that launched it instead of to
                   whichever later phase happens to touch the result. The prologue publishes no measurement, so a
                   sync here changes no quantity this lane quotes (the timed window still opens on its own sync).
      LOUD         the watchdog fires WHILE THE ARM IS STILL INSIDE the over-budget phase and writes a receipt
                   naming it, instead of leaving `perl alarm` to SIGKILL a process that cannot write its own stub.

    Phases do not nest: each `with PH(name)` block is a flat interval, and the loaders cover their own bodies.
    """

    POLL_S = 0.25

    def __init__(self):
        self.reset()

    def reset(self):
        self.seconds, self.order = {}, []
        self.current, self._cur_t0 = None, None
        self._lock = threading.Lock()
        self._t0, self.prologue_s, self._prologue_named_s = None, None, None
        self._exact = {}
        self._prologue_s_exact = None
        self.budget_s, self._on_over, self._stop = 0.0, None, None

    # -- recording ------------------------------------------------------------
    def begin(self, t0):
        self._t0 = t0

    def mark(self, name, seconds):
        with self._lock:
            if name not in self.seconds:
                self.order.append(name)
            self.seconds[name] = round(self.seconds.get(name, 0.0) + float(seconds), 3)
            # Unrounded, for the residual ONLY. Every phase lies inside the prologue window, so
            # exact(total) - exact(sum of parts) >= 0 always; but each part is rounded to 1 ms for
            # the table, and ~10 of those roundings can accumulate past the total, which made the
            # residual read -0.001 (e4b#629 CI). Round the residual ONCE, from unrounded inputs.
            self._exact[name] = self._exact.get(name, 0.0) + float(seconds)

    @contextlib.contextmanager
    def __call__(self, name, sync=True):
        if sync:
            _sync_quiet()
        t = time.perf_counter()
        with self._lock:
            self.current, self._cur_t0 = name, t
        try:
            yield
        finally:
            if sync:
                _sync_quiet()
            dt = time.perf_counter() - t
            with self._lock:
                self.current, self._cur_t0 = None, None
            self.mark(name, dt)

    def started(self):
        """True once an arm has opened its window: an empty table then MEANS empty, rather than absent."""
        return self._t0 is not None

    def end_prologue(self):
        """Close the window at the instant the timed training loop opens; the residual is computed here, once."""
        self.stop_watchdog()
        if self._t0 is None:
            return
        self._prologue_s_exact = time.perf_counter() - self._t0
        self.prologue_s = round(self._prologue_s_exact, 3)
        with self._lock:
            self._prologue_named_s = round(sum(self.seconds.values()), 3)
            self._prologue_named_exact = sum(self._exact.values())

    def report(self):
        """The receipt block. `prologue_unattributed_s` is the residual and is the field that keeps this honest."""
        with self._lock:
            ph = {k: self.seconds[k] for k in self.order}
        out = {"phase_seconds": ph, "phase_budget_s": (round(self.budget_s, 1) if self.budget_s else None)}
        if self.prologue_s is not None:
            out["prologue_s"] = self.prologue_s
            out["prologue_unattributed_s"] = round((self._prologue_s_exact or 0.0) - getattr(self, "_prologue_named_exact", 0.0), 3)
        return out

    def snapshot(self):
        """Safe to call from the watchdog thread: what has been recorded so far, plus the phase in flight."""
        with self._lock:
            ph = {k: self.seconds[k] for k in self.order}
            cur, cur_t0 = self.current, self._cur_t0
        if cur is not None and cur_t0 is not None:
            ph[cur + " (in flight)"] = round(time.perf_counter() - cur_t0, 3)
        return ph, cur, cur_t0

    # -- the loud half --------------------------------------------------------
    def start_watchdog(self, budget_s, on_over_budget):
        """Refuse while still inside the over-budget phase. Budget <= 0 (or no action) records only, never fires."""
        self.budget_s = float(budget_s or 0.0)
        self._on_over = on_over_budget
        if self.budget_s <= 0 or on_over_budget is None or self._t0 is None:
            return None
        self._stop = threading.Event()
        stop, t0, budget = self._stop, self._t0, self.budget_s
        poll = max(0.01, min(self.POLL_S, budget / 4.0))     # a budget the poll cannot resolve is a budget that does not fire

        def _loop():
            while not stop.wait(poll):
                since = time.perf_counter() - t0
                if since <= budget:
                    continue
                stop.set()
                ph, cur, cur_t0 = self.snapshot()
                on_over_budget(cur or "(between phases)",
                               round(time.perf_counter() - cur_t0, 1) if cur_t0 else None,
                               round(since, 1), ph)
                return
        th = threading.Thread(target=_loop, name="tc1-phase-watchdog", daemon=True)
        th.start()
        return th

    def stop_watchdog(self):
        if self._stop is not None:
            self._stop.set()


def _sync_quiet():
    """A phase boundary must never be the thing that raises; a failed sync is the next phase's problem."""
    try:
        cuda_sync()
    except Exception:
        pass


PH = Phases()


def phase_budget_for(a):
    """The prologue's share of the arm's own alarm. `--phase-budget-s` wins, then TC1_PHASE_BUDGET_S, then a share
    of TC1_ARM_ALARM_S (what `tc1_run.sh` passed to `perl -e alarm`), then nothing.

    Why a SHARE and not a literal: the budget has to mean "this arm can no longer finish", which is a fact about the
    rental window the run script chose, not about seconds. At tp4's 3600 s arm alarm the default is 1260 s -- #548's
    observed ~1970 s prologue would have been refused with a receipt naming its phase, ~11 min before SIGALRM killed
    the process silently. It is recorded in every receipt (`phase_budget_s`) so a row says what it was judged against.
    """
    if getattr(a, "phase_budget_s", None) is not None:
        return float(a.phase_budget_s)          # explicit wins, INCLUDING an explicit 0 = off
    for var, scale in (("TC1_PHASE_BUDGET_S", 1.0), ("TC1_ARM_ALARM_S", PROLOGUE_BUDGET_SHARE)):
        v = os.environ.get(var)
        if v:
            try:
                return max(60.0, float(v) * scale) if scale != 1.0 else float(v)
            except ValueError:
                pass
    return 0.0


# ----------------------------------------------------------------------------- n17 / tp1 / p38 code (unchanged)
class PowerSampler:
    def __init__(self, enabled=True):
        self.samples, self._run, self.enabled = [], False, enabled

    def _loop(self):
        while self._run:
            try:
                self.samples.append(float(subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self):
        if self.enabled:
            self._run = True
            self.t = threading.Thread(target=self._loop, daemon=True)
            self.t.start()
        return self

    def __exit__(self, *a):
        self._run = False
        if self.enabled:
            self.t.join(timeout=2)


def idle_power(n=10):
    v = []
    for _ in range(n):
        try:
            v.append(float(subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.15)
    return statistics.median(v) if v else 0.0


def host_fingerprint(instance_env="TC1_INSTANCE_ID"):
    fields = ("uuid,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,"
              "power.limit,clocks.max.sm,clocks.max.mem,vbios_version,driver_version")
    want = fields.split(",")
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        raw = r.stdout.strip().split("\n")[0] if r.stdout.strip() else ""
        cols = [v.strip() for v in raw.split(",")] if raw else []
        if r.returncode != 0:
            out["nvidia_smi_error"] = f"exit {r.returncode}: {r.stderr.strip()[:200]}"
        elif len(cols) != len(want):
            out["nvidia_smi_error"] = f"expected {len(want)} columns, got {len(cols)}: {raw[:200]}"
        else:
            out = dict(zip(want, cols))
    except Exception as e:
        out["nvidia_smi_error"] = f"{type(e).__name__}: {e}"
    try:
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.startswith("model name"):
                    out["cpu"] = ln.split(":", 1)[1].strip()
                    break
        out["cpu_threads"] = os.cpu_count()
        with open("/proc/meminfo") as fh:
            out["host_mem_gib"] = round(int(fh.readline().split()[1]) / 1048576, 1)
    except Exception as e:
        out["host_error"] = f"{type(e).__name__}: {e}"
    out["vast_instance_id"] = os.environ.get(instance_env)
    out["container_host"] = socket.gethostname()
    return out


@torch.no_grad()
def eval_loss(model, rows, fwd_kwargs, autocast):
    model.eval()
    tot = n = 0
    for ids in rows:
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEV)
        with autocast_ctx(autocast):
            tot += float(model(input_ids=x, labels=x, **fwd_kwargs(x)).loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def control_flip_fires(h):
    """Positive control: the comparison must DETECT a single flipped byte."""
    if not h:
        return False
    k = next(iter(h))
    tampered = dict(h)
    tampered[k] = ("0" if h[k][0] != "0" else "1") + h[k][1:]
    return [x for x in h if h[x] != tampered.get(x)] == [k]


def write_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def receipt_path(a, fw=None, tag=None):
    return os.path.join(a.out, f"{a.fam}_{fw or a.framework}_{tag or a.tag}.json")


# ----------------------------------------------------------------------------- frozen-byte hashes (C1), per framework
def hashes_e4b(model):
    """tp1's expert_hashes (state_dict bytes of every Experts4bit stack) + the NF4 attention projections when present."""
    try:
        import bitsandbytes as bnb
        L4 = bnb.nn.Linear4bit
    except Exception:
        L4 = None
    h, nbytes, empties = {}, 0, 0
    for name, m in model.named_modules():
        if any(hasattr(m, a) for a in EXPERT_ATTRS):
            sd = m.state_dict()
            for attr in EXPERT_ATTRS:
                t = sd.get(attr)
                if t is None:
                    continue
                b = t.detach().to("cpu").contiguous().numpy().tobytes()
                if not b:
                    empties += 1
                    continue
                nbytes += len(b)
                h[f"{name}.{attr}"] = sha_bytes(b)
        elif L4 is not None and isinstance(m, L4):
            b = m.weight.data.detach().to("cpu").contiguous().numpy().tobytes()
            if not b:
                empties += 1
                continue
            nbytes += len(b)
            h[f"{name}.weight(attn4)"] = sha_bytes(b)
    return h, nbytes, empties


def hashes_unsloth(model):
    """The frozen 4-bit bytes Unsloth holds: every Params4bit (expert stacks incl. PEFT's parametrizations.*.original,
    and the bnb Linear4bit weights), read from named_parameters(); a LoRA parameter is never hashed."""
    h, nbytes, empties = {}, 0, 0
    for name, p in model.named_parameters():
        if p.requires_grad or "lora" in name.lower():
            continue
        is4 = type(p).__name__ == "Params4bit" or (p.dtype == torch.uint8 and ("experts" in name or "proj" in name))
        if not is4:
            continue
        b = p.data.detach().to("cpu").contiguous().numpy().tobytes()
        if not b:
            empties += 1
            continue
        nbytes += len(b)
        h[name] = sha_bytes(b)
    return h, nbytes, empties


def is_experts_module(name):
    return name.split(".")[-1] == "experts"


def innermost(m):
    while hasattr(m, "base_layer"):
        m = m.base_layer
    return m


def quant_census(model):
    """U5 / T7: what is 4-bit and what is not, by class; the expert stacks, the attention projections, the router, lm_head."""
    c = {"Params4bit_expert_stacks": 0, "Params4bit_other": 0, "Linear4bit": 0, "Linear_bf16": 0, "Linear_fp32": 0,
         "Experts4bit": 0, "ExpertsLoRA": 0, "LoRALinear": 0, "experts_modules": 0, "experts_module_classes": {},
         "router_gate": [], "lm_head": None, "samples": []}
    for name, m in model.named_modules():
        cls = type(m).__name__
        if cls in ("Experts4bit", "ExpertsNbit", "GptOssExperts4bit"):
            c["Experts4bit"] += 1
        elif cls == "ExpertsLoRA":
            c["ExpertsLoRA"] += 1
        elif cls == "LoRALinear":
            c["LoRALinear"] += 1
        elif cls == "Linear4bit":
            c["Linear4bit"] += 1
        elif cls == "Linear" and "lora_" not in name:
            dt = getattr(getattr(m, "weight", None), "dtype", None)
            if dt == torch.bfloat16:
                c["Linear_bf16"] += 1
            elif dt == torch.float32:
                c["Linear_fp32"] += 1
        if is_experts_module(name):
            c["experts_modules"] += 1
            k = type(innermost(m)).__name__
            c["experts_module_classes"][k] = c["experts_module_classes"].get(k, 0) + 1
        if name.endswith("mlp.gate") or name.endswith("router") or name.endswith("block_sparse_moe.gate") or name.endswith(".gate"):
            w = getattr(m, "weight", None)
            if w is not None and len(c["router_gate"]) < 2:
                c["router_gate"].append({"name": name, "cls": cls, "dtype": str(getattr(w, "dtype", None))})
        if name.endswith("lm_head"):
            c["lm_head"] = {"cls": cls, "dtype": str(getattr(getattr(m, "weight", None), "dtype", None))}
    for name, p in model.named_parameters():
        if type(p).__name__ == "Params4bit":
            key = "Params4bit_expert_stacks" if EXPERT_PARAM_RE.search(name) else "Params4bit_other"
            c[key] += 1
            if len(c["samples"]) < 4 and key == "Params4bit_expert_stacks":
                c["samples"].append({"name": name, "dtype": str(p.dtype), "shape": list(p.shape)})
    return c


def trainable_census(model):
    tr = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_params = sum(p.numel() for _, p in tr)
    dtypes = {}
    for _, p in tr:
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    non_adapter = [n for n, _ in tr if "lora" not in n.lower()]
    groups = {"attention": 0, "experts": 0, "other": 0}
    for n, p in tr:
        if "experts" in n:
            groups["experts"] += p.numel()
        elif "self_attn" in n or re.search(r"\.(q_proj|k_proj|v_proj|o_proj)\.", n):
            groups["attention"] += p.numel()
        else:
            groups["other"] += p.numel()
    return tr, n_params, dtypes, non_adapter, groups


def trainable_sha(tr):
    h = hashlib.sha256()
    for name, p in tr:
        h.update(name.encode())
        h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _phase_alarm_action(a, ctx):
    """#548 (3): refuse LOUDLY, from inside the phase, rather than recording the overrun afterwards.

    The failure this replaces is a real row from lane tp4: `status: alarm`, reason *"the process could not write its
    own stub"* -- SIGALRM from `perl -e alarm` killed an arm 3570 s into a prologue and the receipt could not name a
    single phase. Here the arm refuses itself while still inside the offending phase, so the row carries the phase,
    its elapsed seconds, every phase already closed, and a traceback of every thread. It exits 16 (a status of its
    own) rather than 142, so a run script can tell "the prologue blew its budget" from "the whole arm ran out".
    """
    def _fire(phase, in_phase_s, since_start_s, ph):
        budget = round(PH.budget_s, 1)
        msg = (f"prologue phase '{phase}' has run {in_phase_s}s and the prologue is {since_start_s}s in, past its "
               f"{budget}s budget; refusing now so this row can name the phase (#548)")
        print("PHASE ALARM " + msg, flush=True)
        sys.stdout.flush()
        try:
            faulthandler.dump_traceback()           # every thread's stack: a stuck phase leaves evidence, not just a number
        except Exception:
            pass
        sys.stderr.flush()
        try:
            stub(a, "phase_alarm", msg, dict(ctx, phase=phase, phase_seconds=ph, phase_in_flight=phase,
                                             phase_elapsed_s=in_phase_s, prologue_s=since_start_s, phase_budget_s=budget))
        except Exception as e:                      # a receipt we could not write must still be visible on stdout
            print(f"PHASE ALARM could not write its stub: {type(e).__name__}: {e}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(16)
    return _fire


def stub(a, status, reason, extra=None, code=None, fw=None, tag=None, arm=None):
    rec = {"framework": fw or a.framework, "fam": a.fam, "model": a.model, "revision": a.revision, "arm": arm or a.arm,
           "tag": tag or a.tag, "status": status, "reason": str(reason)[:800], "steps": a.steps, "seq": a.seq,
           "accum": a.accum, "micro_batch": int(getattr(a, "micro_batch", 1) or 1), "offload": bool(a.offload), "prereg": a.prereg, "harness": HARNESS}
    ph, cur, _ = PH.snapshot()                      # #548: a REFUSED / OOM / alarmed row says where its time went too --
    if PH.started():                                # that is the row the issue was raised about, and it had no numbers at all.
        # Emitted even when EMPTY, so "this arm died before anything was timed" is distinguishable from
        # "this receipt predates the field" -- the same reason prologue_unattributed_s is always present.
        rec["phase_seconds"], rec["phase_in_flight"], rec["phase_budget_s"] = ph, cur, (round(PH.budget_s, 1) if PH.budget_s else None)
    if extra:
        rec.update(extra)
    write_json(receipt_path(a, fw, tag), rec)
    print(f"CELL {status.upper()} " + json.dumps({k: v for k, v in rec.items() if k not in ("losses", "step_ms", "microbatch_ms")}), flush=True)
    if code is not None:
        sys.exit(code)


def refresh_stub(a, tag, arm, status, reason, extra):
    """T3: write the family's refusal stub for `tag`, or merge the probe into a stub the run script already wrote
    (its citation stays; the measured probe is added)."""
    p = receipt_path(a, "e4b", tag)
    if os.path.exists(p):
        try:
            rec = json.load(open(p))
        except Exception:
            rec = {}
        if rec.get("status") == "ok":
            print(f"STUB SKIPPED: {os.path.basename(p)} is a receipt that trained; the probe is in this arm's own receipt", flush=True)
            return
        rec.setdefault("status", status)
        rec.setdefault("reason", reason)
        rec.update(extra)
        rec["probe_reason"] = str(reason)[:600]
        write_json(p, rec)
        print(f"CELL {rec['status'].upper()} (stub refreshed) " + json.dumps({k: v for k, v in rec.items() if k not in ("losses", "step_ms", "microbatch_ms")}), flush=True)
    else:
        stub(a, status, reason, extra, fw="e4b", tag=tag, arm=arm)


def capture_verbose(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = fn(*args, **kw)
    return n, buf.getvalue().strip()[-600:]


def classify_load_exception(e):
    if is_oom(e):
        return "oom", 5
    msg = str(e).lower()
    if isinstance(e, (NotImplementedError, ValueError, TypeError, KeyError, AssertionError, ImportError, SystemExit)):
        return "refused", 3
    if isinstance(e, RuntimeError) and "cuda" not in msg:
        return "refused", 3
    if re.search(r"not support|unsupported|refus|does not support|cannot", msg):
        return "refused", 3
    return "load_fault", 6


def n_layers_of(cfg):
    lm = getattr(cfg, "text_config", None) or cfg
    return getattr(lm, "num_hidden_layers", None), getattr(cfg, "model_type", None)


# ----------------------------------------------------------------------------- U2: the fixed text, tokenised once per family
def render_row(r, template="clinical", eos=""):
    """T12: the text one dataset row becomes. clinical = tp2's FMT (byte-identical to tp2/tp3 tokens files); alpaca = the
    Unsloth notebooks' three-slot prompt with EOS appended (the notebook's formatting_prompts_func, every row, empty Input
    slot when the row has none)."""
    if template == "alpaca":
        return ALPACA_PROMPT.format(r["instruction"], r.get("input", "") or "", r["output"]) + (eos or "")
    return FMT.format(instruction=r["instruction"], output=r["output"])


def encode_rows(tok, rows, seq, template="clinical", eos=""):
    out = []
    for r in rows:
        ids = tok(render_row(r, template, eos), truncation=True, max_length=seq).input_ids
        if len(ids) >= 8:
            out.append([int(i) for i in ids])
    return out


def prepare(a, tok=None):
    got = sha_bytes(open(a.data, "rb").read())
    if got != a.data_sha:
        print(f"DATASET MISMATCH {got} != {a.data_sha}")
        sys.exit(13)
    ds = json.load(open(a.data))
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    template = getattr(a, "template", "clinical")
    eos = (getattr(tok, "eos_token", None) or "") if template == "alpaca" else ""
    train, ev = encode_rows(tok, ds["train"], a.seq, template, eos), encode_rows(tok, ds["eval"], a.seq, template, eos)[:a.eval_n]
    body = json.dumps({"train": train, "eval": ev}, separators=(",", ":")).encode()
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    rec = {"fam": a.fam, "dataset": os.path.basename(a.data), "dataset_sha256": got, "tokenizer": a.model, "revision": a.revision,
           "tokenizer_class": type(tok).__name__, "format": ALPACA_PROMPT if template == "alpaca" else FMT, "template": template, "eos": eos,
           "pad_id": pad_id, "seq": a.seq, "n_train": len(train), "n_eval": len(ev),
           "train_tokens": sum(map(len, train)), "sha256": sha_bytes(body), "train": train, "eval": ev}
    write_json(a.tokens, rec)
    print(f"TOKENS fam={a.fam} template={template} n_train={len(train)} n_eval={len(ev)} train_tokens={rec['train_tokens']} mean={rec['train_tokens']/max(1,len(train)):.1f} pad_id={pad_id} sha={rec['sha256']}")
    return rec


def collate(rows, pad_id):
    """T13: right-pad M token rows to the longest; labels -100 on the pads; a 0/1 attention mask. M == 1 -> no padding."""
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), int(pad_id or 0), dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    labels = torch.full((len(rows), L), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        t = torch.tensor(r, dtype=torch.long)
        ids[i, : len(r)] = t
        mask[i, : len(r)] = 1
        labels[i, : len(r)] = t
    return ids.to(DEV), mask.to(DEV), labels.to(DEV)


# ----------------------------------------------------------------------------- U8: engagement counters
class Counters:
    def __init__(self):
        self.counts = {"fused_grouped_lora": 0, "experts_forward": 0, "moe_bnb4bit_backend": 0}
        self._restore, self._hooks, self.absent = [], [], []

    def install_e4b(self):
        try:
            import nf4_qlora
            orig = nf4_qlora.fused_grouped_lora

            def w(*a, _orig=orig, **k):
                self.counts["fused_grouped_lora"] += 1
                return _orig(*a, **k)
            nf4_qlora.fused_grouped_lora = w
            self._restore.append((nf4_qlora, "fused_grouped_lora", orig))
        except ImportError:
            pass

    def install_experts_hooks(self, model):
        """T11: module-level engagement -- one forward call per experts module per micro-batch (the HF arm's counter, and
        the second counter Unsloth's validity rule reads)."""
        for name, m in model.named_modules():
            if is_experts_module(name):
                self._hooks.append(m.register_forward_pre_hook(lambda mod, inp: self.counts.__setitem__("experts_forward", self.counts["experts_forward"] + 1)))

    def install_hf(self, model):
        self.install_experts_hooks(model)

    def install_unsloth(self, model):
        """P2-1: the bnb-4bit entry point AND the three backends behind select_moe_backend() (UNSLOTH_BACKEND_FUNCS) are
        wrapped with counters. A name the installed zoo lacks is recorded in `absent`, never a crash. Every module already
        imported under `unsloth*` / `transformers*` that holds the original function object (a `from ... import` alias,
        transformers' ALL_EXPERTS_FUNCTIONS table) gets the wrapper too, so a backend called through an alias still counts."""
        self.install_experts_hooks(model)
        for modname, fname, key in UNSLOTH_BACKEND_FUNCS:
            try:
                M = importlib.import_module(modname)
                orig = getattr(M, fname)
            except Exception as e:
                self.absent.append(f"{modname}.{fname}: {type(e).__name__}")
                continue
            self.counts.setdefault(key, 0)

            def w(*a, _orig=orig, _key=key, **k):
                self.counts[_key] += 1
                return _orig(*a, **k)
            setattr(M, fname, w)
            self._restore.append((M, fname, orig))
            for mn, mod in list(sys.modules.items()):
                if mod is None or mod is M or not (mn.startswith("unsloth") or mn.startswith("transformers")):
                    continue
                try:
                    for attr, val in list(vars(mod).items()):
                        if val is orig:
                            setattr(mod, attr, w)
                            self._restore.append((mod, attr, orig))
                        elif isinstance(val, dict) and attr == "ALL_EXPERTS_FUNCTIONS":
                            for k2, v2 in list(val.items()):
                                if v2 is orig:
                                    val[k2] = w
                                    self._restore.append((val, k2, orig))
                except Exception:
                    pass

    def snapshot(self):
        d = dict(self.counts)
        try:   # P46: which path the grouped-LoRA delta took (grouped-nf4-gemm >= 0.32.1 keeps per-path counters)
            import nf4_qlora
            for k, v in getattr(nf4_qlora, "LORA_PATH_STATS", {}).items():
                d["lora_path_" + k] = int(v)
        except ImportError:
            pass
        return d

    def uninstall(self):
        for h in self._hooks:
            h.remove()
        for mod, name, orig in self._restore:
            if isinstance(mod, dict):
                mod[name] = orig
            else:
                setattr(mod, name, orig)


# ----------------------------------------------------------------------------- the two loaders (each returns model + a dict of extras)
def attn4_bias_probe(model):
    """T3: the side-effect-free form of TRAIN_ATTN_4BIT's refusal condition (a bias on any q/k/v/o)."""
    projs = ("q_proj", "k_proj", "v_proj", "o_proj")
    biased, n = [], 0
    for name, mod in model.named_modules():
        if all(isinstance(getattr(mod, p, None), nn.Linear) for p in projs):
            for p in projs:
                n += 1
                if getattr(mod, p).bias is not None:
                    biased.append(f"{name}.{p}")
    return {"n_projections": n, "n_biased": len(biased), "would_refuse": bool(biased), "sample": biased[:2]}


def detector_version(lora_mod, lib_version):
    """T10 (#434): the detector version recorded in every e4b receipt -- the library's own DETECTOR_VERSION if it
    ships one (#435 ships none; checked at main 5dad2a7), else the library version + a short hash of the detector's
    source. Deterministic, needs no library change, and both forms name the library version."""
    import inspect
    return (getattr(lora_mod, "DETECTOR_VERSION", None)
            or f"{lib_version}+det:{hashlib.sha256(inspect.getsource(lora_mod.detect_attention_projections).encode()).hexdigest()[:12]}")


def attn4_census_check(a, model, x, detect_fn, quantize_fn):
    """T10 (#434): the attn-4bit count check against the library's structural census, never 4 * n_layers.
    The census is snapshotted BEFORE any mutation: exact_linear=True matches `type is nn.Linear`, so after
    conversion Linear4bit would not match (the detector docstring's rule). A detector SystemExit (a q/o layer
    without k_proj -- true cross-layer KV reuse) is a REFUSED row with the refusing modules named, never a
    void_attn4 guess; a quantize SystemExit (a bias-carrying projection) stays the REFUSED row tp2 wrote."""
    try:
        census = detect_fn(model, exact_linear=True)
    except SystemExit as e:
        refusing = [n for n, m in model.named_modules()
                    if type(getattr(m, "q_proj", None)) is nn.Linear and type(getattr(m, "o_proj", None)) is nn.Linear
                    and type(getattr(m, "k_proj", None)) is not nn.Linear]
        stub(a, "refused", f"detect_attention_projections refused: {e}",
             {"phase": "attn4", "attn4_probe": x.get("attn4_probe"), "attn4_refusing_modules": refusing[:8],
              "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)
    x["structural_expected_n_attn4"] = census.expected_count
    try:
        x["n_attn4"] = quantize_fn(model)                             # U4: BEFORE the attention LoRA (its docstring's rule)
    except SystemExit as e:                                           # TRAIN_ATTN_4BIT refuses a bias-carrying projection
        stub(a, "refused", f"quantize_attention_projections_4bit refused: {e}", {"phase": "attn4", "attn4_probe": x["attn4_probe"],
             "structural_expected_n_attn4": x["structural_expected_n_attn4"],
             "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)
    if x["n_attn4"] != x["structural_expected_n_attn4"]:
        stub(a, "void_attn4", f"quantize_attention_projections_4bit converted {x['n_attn4']} projections, "
             f"structural census expected {x['structural_expected_n_attn4']}",
             {"phase": "attn4", "structural_expected_n_attn4": x["structural_expected_n_attn4"],
              "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)


def load_e4b(a):
    """#548: the loader's own phases. `load_s` (the whole call) is unchanged and stays in the receipt; these split
    it, because on a 30B MoE the four steps after the weights land are not a rounding error on the weights."""
    from experts4bit_qlora import (disable_batched_train, disable_fast_train, enable_batched_train, enable_fast_train,
                                   load_moe_4bit_streaming, verify_moe_4bit)
    import experts4bit_qlora
    import experts4bit_qlora.lora as _lora_mod
    from experts4bit_qlora.lora import add_attention_lora, detect_attention_projections, quantize_attention_projections_4bit
    from transformers import AutoTokenizer
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "attn4_probe": None,
         "structural_expected_n_attn4": None, "detector_version": detector_version(_lora_mod, experts4bit_qlora.__version__)}
    with PH("load_weights"):
        model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha,
                                             offload=bool(a.offload), pin=True, prefetch=False, quant_type="nf4")
        if not a.offload:
            model.to("cuda")
    x["n_layers"], x["model_type"] = n_layers_of(cfg)
    with PH("verify"):
        try:
            rep = verify_moe_4bit(model, strict=True)
        except RuntimeError as e:
            stub(a, "verify_failed", str(e), {"phase": "verify"}, code=7)
    x["verify"] = {"n_quantized": rep.get("n_quantized"), "n_unquantized": rep.get("n_unquantized")}
    with PH("attn4"):
        x["attn4_probe"] = attn4_bias_probe(model)
        if a.attn_4bit:
            attn4_census_check(a, model, x, detect_attention_projections, quantize_attention_projections_4bit)   # T10
    with PH("lora"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        add_attention_lora(model, a.r, a.alpha, torch.float32)
    with PH("enable"):
        if a.arm == "attn_only":
            for n, p in model.named_parameters():
                if "lora" in n and "experts" in n:
                    p.requires_grad_(False)
            nf, why_f = capture_verbose(enable_fast_train, model, verbose=True, dgrad=True)
            disable_fast_train(model)
            nb, why_b = capture_verbose(enable_batched_train, model, verbose=True)
            disable_batched_train(model)
            x["probes"] = {"fused": {"n_patched": nf, "reason": why_f}, "batched": {"n_patched": nb, "reason": why_b}}
            common = {"model_type": x["model_type"], "n_layers": x["n_layers"], "probed_by": "attn_only", "n_patched": 0}
            if nf == 0:
                refresh_stub(a, "fused_attn4", "fused", "refused", f"enable_fast_train(dgrad=True) patched 0 modules on this box: {why_f}", dict(common, probe_n_patched=nf))
            if x["attn4_probe"]["would_refuse"]:
                refresh_stub(a, "reference_attn4", "reference", "refused",
                             f"TRAIN_ATTN_4BIT would refuse: {x['attn4_probe']['n_biased']} of {x['attn4_probe']['n_projections']} attention projections carry a bias "
                             f"(quantize_attention_projections_4bit raises SystemExit on a bias; e.g. {x['attn4_probe']['sample']})", dict(common, attn4_probe=x["attn4_probe"]))
        elif a.arm == "fused":
            dg = bool(getattr(a, "dgrad", 1))
            x["n_patched"], x["reason"] = capture_verbose(enable_fast_train, model, verbose=True, dgrad=dg)
            if x["n_patched"] == 0:
                stub(a, "refused", f"enable_fast_train(dgrad={dg}) patched 0 modules: {x['reason']}", {"phase": "enable", "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)
        elif a.arm == "batched":
            # P56. Counted like the fused arm, but its proof-of-work is DIFFERENT and is
            # recorded below: this path silently falls back to the reference forward per
            # call (pad waste, evicted storage, empty batch), and a fallback is invisible
            # in the output. An arm that fell back on every call IS a second reference arm,
            # and its parity number would read ~0 while measuring nothing.
            x["n_patched"], x["reason"] = capture_verbose(enable_batched_train, model, verbose=True)
            if x["n_patched"] == 0:
                stub(a, "refused", f"enable_batched_train patched 0 modules: {x['reason']}", {"phase": "enable", "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)
        else:
            disable_fast_train(model)
            disable_batched_train(model)
    with PH("tokenizer"):
        x["tokenizer_obj"] = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    x["ckpt_mode"] = "hf:use_reentrant=False"
    x["hashes"] = hashes_e4b
    x["fwd_kwargs"] = lambda t: {}
    x["snapshot_dir"] = None
    return model, x


def snapshot_dir_for(model_id, revision):
    return os.path.join(os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")),
                        "models--" + model_id.replace("/", "--"), "snapshots", revision)


FASTMODEL_HINT_RE = re.compile(r"FastModel|vision|multimodal|ForConditionalGeneration|image", re.I)


def unsloth_targets_of(a):
    raw = getattr(a, "unsloth_targets", None) or ",".join(UNSLOTH_TARGETS)
    return [t.strip() for t in raw.split(",") if t.strip()]


def load_unsloth(a):
    import unsloth
    loader = getattr(unsloth, a.unsloth_loader)
    from huggingface_hub import snapshot_download
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "attn4_probe": None,
         "structural_expected_n_attn4": None, "detector_version": None,   # T10: e4b-only fields, null here
         "loader_used": a.unsloth_loader, "loader_fallback_reason": None, "unsloth_targets": unsloth_targets_of(a)}
    with PH("snapshot"):                                                                 # #548
        _cand = snapshot_dir_for(a.model, a.revision)                                    # amendment 3: the pinned snapshot dir, same bytes
        local = _cand if os.path.isdir(_cand) else snapshot_download(a.model, revision=a.revision)
    x["snapshot_dir"] = local

    dq = {"requested": None, "how": "unknown-default", "loaded_nested": None}
    want_dq_off = (getattr(a, "unsloth_double_quant", "off") or "off") == "off"

    def _load(ldr):
        kw, dq_ = unsloth_double_quant_kwargs(ldr.from_pretrained, want_dq_off)   # P2-1: only a NAMED parameter is passed
        dq.update(dq_)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model, tokenizer_obj = ldr.from_pretrained(model_name=local, max_seq_length=a.seq, dtype=torch.bfloat16, load_in_4bit=True, **kw)
            model = ldr.get_peft_model(
                model, r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", target_modules=list(x["unsloth_targets"]),
                use_gradient_checkpointing=("unsloth" if a.grad_ckpt == "unsloth" else True), random_state=a.seed)
        return model, tokenizer_obj, buf.getvalue()
    with PH("load_weights"):                                                             # #548: from_pretrained + get_peft_model
        try:
            model, tokenizer_obj, out = _load(loader)
        except Exception as e:                                      # T15: one recorded retry with FastModel when the message asks for it
            if a.unsloth_loader == "FastLanguageModel" and FASTMODEL_HINT_RE.search(str(e)) and hasattr(unsloth, "FastModel"):
                x["loader_used"], x["loader_fallback_reason"] = "FastModel (fallback)", f"{type(e).__name__}: {str(e)[:300]}"
                print(f"LOADER FALLBACK FastLanguageModel -> FastModel: {x['loader_fallback_reason']}", flush=True)
                model, tokenizer_obj, out = _load(unsloth.FastModel)
            else:
                raise
    x["banner_lines"] = [l for l in out.splitlines() if re.search(r"MoE|Params4bit|4.?bit|LoRA on|backend", l, re.I)][:16]
    print("\n".join(out.splitlines()[-40:]), flush=True)
    if not any(BANNER in l for l in x["banner_lines"]):
        x["banner_lines"].append(f"NO '{BANNER}' banner on stdout (the census below decides)")
    dq["loaded_nested"] = loaded_expert_double_quant(model)            # P2-1: what the loaded expert Params4bit says (None = none found)
    x["unsloth_double_quant"] = dq
    x["moe_backend_selected"] = unsloth_backend_selected()             # P2-1: select_moe_backend() after load (lru_cached)
    x["verify"] = {"n_quantized": None, "n_unquantized": None}
    x["n_layers"], x["model_type"] = n_layers_of(model.config)
    model.config.use_cache = False
    x["tokenizer_obj"] = tokenizer_obj
    x["ckpt_mode"] = "unsloth" if a.grad_ckpt == "unsloth" else "hf:True (via get_peft_model)"
    x["hashes"] = hashes_unsloth
    x["fwd_kwargs"] = lambda t: {"attention_mask": torch.ones_like(t)}     # all-ones = no masking; identical semantics
    return model, x


def is_linear_like(m):
    """nn.Linear or a bitsandbytes Linear4bit (after load_in_4bit): the attention-projection predicate for the HF arm."""
    return m is not None and isinstance(m, nn.Module) and hasattr(m, "in_features") and hasattr(m, "out_features") and hasattr(m, "weight")


def per_expert_2d_groups(model):
    """#542, diagnostic only: how many modules hold their experts PER EXPERT -- numerically named sibling submodules,
    each carrying linear-like children. That layout has no fused stack for PEFT's target_parameters to adapt, so it is
    the commonest thing an empty structural selection means; counting it turns a bare refusal into a legible one."""
    n = 0
    for _, m in model.named_modules():
        kids = list(m.named_children())
        if len(kids) >= 2 and all(k.isdigit() for k, _ in kids) \
                and all(any(is_linear_like(c) for _, c in kid.named_children()) for _, kid in kids):
            n += 1
    return n


def declared_num_experts(cfg):
    """The expert count the CONFIG declares, if it declares one. Used only to disambiguate a structural selection that
    found more than one leading dimension -- never to make the selection."""
    for owner in (getattr(cfg, "text_config", None), cfg):
        if owner is None:
            continue
        for k in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", "num_experts_per_layer"):
            v = getattr(owner, k, None)
            if isinstance(v, int) and not isinstance(v, bool) and v > 1:
                return v
    return None


def hf_expert_parameters(model, n_layers=None, model_type=None):
    """#542: the fused expert stacks PEFT must adapt, selected BY STRUCTURE and never by a family's word for the module.

    The predicate is the SHAPE: a 3-D floating-point parameter (E, *, *). A decoder carries no other one -- attention
    and MLP weights are 2-D, norms and biases 1-D, embeddings 2-D, and bitsandbytes' Params4bit is uint8 -- so nothing
    here reads a name. The pre-#542 rule was ``p.ndim == 3 and "experts" in name``, a NAME substring, which is the
    shape #426/#435 already removed from the attention path: it selects nothing on a family that calls the module
    something else (GraniteMoe's own ``block_sparse_moe.input_linear`` / ``output_linear``) and then degrades SILENTLY
    to an attention-only run. ``EXPERT_PARAM_RE``, the other candidate the issue names, does not fix that either --
    it requires a literal ``experts.`` component in the path, so it misses the very layout whose leaf names it lists.

    ``requires_grad`` is deliberately NOT part of the predicate, against the issue's sketch: this runs BEFORE
    ``get_peft_model``, where transformers has not frozen the base weights, so a frozen-only filter would select
    nothing on a real load.

    An empty or implausible selection RAISES ``NotImplementedError`` -- ``run_arm`` classifies that as ``refused``
    (code 3), the same way ``load_hf`` already refuses when no attention projection is found. An arm that adapts
    attention only is not the comparator this lane registered, and a comparator that quietly becomes something else
    is worse than a missing row.

    Returns (names, diag); the diag goes into the receipt so a future run can be audited against this rule.
    """
    by_lead, quantized_3d = {}, []
    for name, p in model.named_parameters():
        if p.ndim != 3:
            continue
        if not p.is_floating_point():                  # a quantized stack is not a PEFT target_parameters candidate
            quantized_3d.append(name)
            continue
        by_lead.setdefault(int(p.shape[0]), []).append(name)
    declared = declared_num_experts(getattr(model, "config", None))
    flat = [n for names in by_lead.values() for n in names]
    diag = {"rule": "structural: 3-D floating parameter (E, *, *), no name substring (#542)",
            "n_3d_floating": len(flat), "leading_dims": {str(k): len(v) for k, v in sorted(by_lead.items())},
            "declared_num_experts": declared, "n_3d_non_floating": len(quantized_3d),
            "n_by_name_substring": sum("experts" in n for n in flat),        # what the pre-#542 rule would have taken
            "n_by_expert_param_re": sum(bool(EXPERT_PARAM_RE.search(n)) for n in flat),
            "per_expert_2d_groups": per_expert_2d_groups(model),
            "n_layers": n_layers, "model_type": model_type}

    def refusal(why):     # the reason is truncated to 700 chars in the receipt, so everything load-bearing comes first
        return NotImplementedError(
            f"HF arm REFUSES its expert selection: {why} (model_type={model_type!r}, n_layers={n_layers}). Adapting "
            f"attention only is not this lane's HF comparator (#542). Looked for a 3-D floating expert stack (E, *, *) "
            f"BY STRUCTURE; selection census {json.dumps(diag, sort_keys=True)}")

    if not flat:
        raise refusal("no expert stack found")
    if declared is not None and declared in by_lead:
        lead = declared
    elif len(by_lead) == 1:
        # One leading dimension is unambiguous STRUCTURE, so it wins even when the config declares a different number:
        # the declared field means different things across families (routed vs shared experts), and structure is the
        # rule here. The disagreement is recorded in the diag, so a receipt shows it rather than hiding it.
        lead = next(iter(by_lead))
    elif declared is not None:
        raise refusal(f"the config declares {declared} experts but no 3-D floating parameter has that leading dimension")
    else:
        raise refusal(f"3-D floating parameters disagree on their leading dimension {sorted(by_lead)} "
                      f"and the config declares no expert count")
    if lead < 2:
        raise refusal(f"the only 3-D floating parameters have leading dimension {lead}, which is not an expert stack")
    names = by_lead[lead]
    if n_layers and len(names) % n_layers:
        raise refusal(f"{len(names)} expert stacks over {n_layers} layers is not a whole number per layer")
    diag["n_selected"], diag["experts_per_stack"] = len(names), lead
    diag["stacks_per_layer"] = (len(names) // n_layers) if n_layers else None
    return names, diag


def hf_targets(model, n_layers=None, model_type=None):
    """T11 / #542: what PEFT adapts, found BY STRUCTURE -- the attention projections by the same predicate as e4b's
    detector (q/k/o present, v optional), the expert stacks by shape via hf_expert_parameters, which REFUSES rather
    than handing back an empty list. Full names, so PEFT's suffix match is exact."""
    mods = []
    for name, m in model.named_modules():
        if all(is_linear_like(getattr(m, p, None)) for p in ("q_proj", "k_proj", "o_proj")):
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if is_linear_like(getattr(m, p, None)):
                    mods.append(f"{name}.{p}")
    params, expert_diag = hf_expert_parameters(model, n_layers, model_type)
    return mods, params, expert_diag


def load_hf(a):
    """T11: the field's plain stack -- transformers + bitsandbytes 4-bit + PEFT LoRA (target_modules by structure,
    target_parameters on the 3-D expert stacks, BOTH by structure since #542 -- a family whose expert stacks cannot be
    found refuses here rather than training an attention-only arm). The census, not this function, says what ended up
    4-bit."""
    import peft
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "attn4_probe": None,
         "structural_expected_n_attn4": None, "detector_version": None, "loader_used": "AutoModelForCausalLM"}
    with PH("snapshot"):                                                                 # #548
        _cand = snapshot_dir_for(a.model, a.revision)
        local = _cand if os.path.isdir(_cand) else snapshot_download(a.model, revision=a.revision)
    x["snapshot_dir"] = local
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=False)
    with PH("load_weights"):                                                             # #548
        model = AutoModelForCausalLM.from_pretrained(local, quantization_config=bnb_cfg, dtype=torch.bfloat16, device_map={"": 0})
    x["n_layers"], x["model_type"] = n_layers_of(model.config)
    with PH("attn4"):                                                                    # #548: the probe only (the HF arm converts nothing)
        x["attn4_probe"] = attn4_bias_probe(model)
    with PH("lora"):                                                                     # #548
        mods, params, expert_diag = hf_targets(model, x["n_layers"], x["model_type"])   # #542: params is never empty -- it refuses
        if not mods:
            raise NotImplementedError("HF arm: no attention projection found by structure (q_proj/k_proj/o_proj); refusing rather than guessing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", target_modules=mods,
                         target_parameters=params, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.config.use_cache = False
    x["hf_targets"] = {"peft": peft.__version__, "n_target_modules": len(mods), "target_modules_sample": mods[:4],
                       "n_target_parameters": len(params), "target_parameters": params[:8] + (["..."] if len(params) > 8 else []),
                       "expert_selection": expert_diag,      # #542: the structural rule + what the old substring would have taken
                       "bnb": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": False}}
    x["banner_lines"] = [f"PEFT {peft.__version__}: target_modules={len(mods)} target_parameters={len(params)}"]
    x["verify"] = {"n_quantized": None, "n_unquantized": None}
    with PH("tokenizer"):                                                                # #548
        x["tokenizer_obj"] = AutoTokenizer.from_pretrained(local)
    x["ckpt_mode"] = "hf:use_reentrant=False"
    x["hashes"] = hashes_unsloth                                           # the same frozen-bytes hasher: Params4bit + uint8 stacks
    x["fwd_kwargs"] = lambda t: {"attention_mask": torch.ones_like(t)}
    return model, x


def load_axolotl(a):
    """P2-2: axolotl 0.20.0's own model stack under this harness's loop (axolotl-arm-spec.md). Everything axolotl.train.train
    does up to the trainer is `load_cfg` + `load_tokenizer` + `ModelLoader(cfg, tok).load()` (loaders/model.py:305-378 runs
    the plugin hooks, the MoE quantization patch, the build, the finalize, the PEFT wrap and post_model_load in order), so
    that is what runs here. The targets are found BY STRUCTURE on a meta-device skeleton of the checkpoint (no weights),
    with the HF arm's rule (#542). A loader exception is classified by run_arm exactly as the HF arm's."""
    import importlib.metadata as md
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "attn4_probe": None,
         "structural_expected_n_attn4": None, "detector_version": None, "loader_used": "axolotl ModelLoader.load()"}
    with PH("snapshot"):
        _cand = snapshot_dir_for(a.model, a.revision)
        local = _cand if os.path.isdir(_cand) else snapshot_download(a.model, revision=a.revision)
    x["snapshot_dir"] = local
    with PH("targets"):                                                                  # structure first, on a weightless skeleton
        cfg0 = AutoConfig.from_pretrained(local)
        x["n_layers"], x["model_type"] = n_layers_of(cfg0)
        with torch.device("meta"):
            skel = AutoModelForCausalLM.from_config(cfg0)
        mods, params, expert_diag = hf_targets(skel, x["n_layers"], x["model_type"])    # #542: refuses rather than an empty list
        if not mods:
            raise NotImplementedError("axolotl arm: no attention projection found by structure (q_proj/k_proj/o_proj); refusing rather than guessing")
        del skel
    best = bool(int(getattr(a, "axolotl_best", 0) or 0))
    cfg_dict = axolotl_config_dict(a, local, mods, params, best=best, double_quant=False,
                                   dataset_path=getattr(a, "axolotl_dataset", None), output_dir=os.path.join(a.out, "axolotl-out"))
    import axolotl
    from axolotl.cli.config import load_cfg
    from axolotl.loaders import ModelLoader, load_tokenizer
    from axolotl.utils.dict import DictDefault
    with PH("load_weights"):
        cfg = load_cfg(DictDefault(cfg_dict))                   # cli/config.py:227-318 (validate, prepare_plugins, normalize)
        tokenizer_obj = load_tokenizer(cfg)                     # train.py:73
        model, lora_config = ModelLoader(cfg, tokenizer_obj).load()   # train.py:83-84 -> loaders/model.py:305-378
    try:
        from axolotl.monkeypatch.moe_quant import get_moe_quantized_count
        n_quant_log = int(get_moe_quantized_count())            # what _finalize_moe_expert_quantization logged (patch_manager.py:1075-1093)
    except Exception as e:
        n_quant_log = f"unavailable: {type(e).__name__}"
    census = axolotl_expert_census(model)
    mcfg = getattr(getattr(model, "config", None), "to_dict", lambda: {})()
    x["axolotl"] = {"version": getattr(axolotl, "__version__", None), "torch": torch.__version__,
                    "transformers": md.version("transformers"), "peft": md.version("peft"), "bitsandbytes": md.version("bitsandbytes"),
                    "config": cfg_dict, "best": best, "plugins": cfg_dict.get("plugins"), "expert_backend": cfg_dict.get("expert_backend"),
                    "cfg_experts_implementation": cfg.get("experts_implementation"), "cfg_use_scattermoe": cfg.get("use_scattermoe"),
                    "config_experts_implementation": getattr(getattr(model, "config", None), "_experts_implementation", None),
                    "router_aux_loss_coef": mcfg.get("router_aux_loss_coef"),
                    "quantized_moe_experts_n": n_quant_log, "census": census,
                    "moe_experts_quantized_flag": getattr(getattr(model, "base_model", None), "model", None) is not None and getattr(getattr(model.base_model, "model", None), "_moe_experts_quantized", None),
                    "lora_config": {"r": getattr(lora_config, "r", None), "lora_alpha": getattr(lora_config, "lora_alpha", None),
                                    "n_target_modules": len(mods), "n_target_parameters": len(params), "init_lora_weights": getattr(lora_config, "init_lora_weights", None)}}
    x["hf_targets"] = {"peft": md.version("peft"), "n_target_modules": len(mods), "target_modules_sample": mods[:4],
                       "n_target_parameters": len(params), "target_parameters": params[:8] + (["..."] if len(params) > 8 else []),
                       "expert_selection": expert_diag, "bnb": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": False, "quantize_moe_experts": True}}
    x["axolotl_targets"] = x["hf_targets"]
    x["axolotl_bnb4bit_modules"] = census
    x["banner_lines"] = [f"axolotl {x['axolotl']['version']}: quantized_moe_experts_n={n_quant_log} parametrized={census['parametrized_params']} "
                         f"n_bnb4bit_unwrapped={census['n_bnb4bit_unwrapped']} experts_implementation cfg={cfg.get('experts_implementation')} "
                         f"config={x['axolotl']['config_experts_implementation']} plugins={cfg_dict.get('plugins')}"]
    x["verify"] = {"n_quantized": None, "n_unquantized": None}
    x["ckpt_mode"] = f"axolotl cfg gradient_checkpointing={cfg.get('gradient_checkpointing')} kwargs={cfg.get('gradient_checkpointing_kwargs')}"
    model.config.use_cache = False
    x["tokenizer_obj"] = tokenizer_obj
    x["hashes"] = hashes_unsloth                                  # Params4bit + the uint8 packed originals under `experts.parametrizations.*.original`
    x["fwd_kwargs"] = lambda t: {"attention_mask": torch.ones_like(t)}
    return model, x


def u8_bnb4bit(model):
    """U8 as amended (amendment 4): the predicate on the module named experts AND on the innermost module."""
    try:
        from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import _moe_uses_bnb4bit_expert_weights as pred
    except Exception as e:
        return f"unchecked: {e}"
    mods = [(n, m) for n, m in model.named_modules() if is_experts_module(n)]
    out = {"n_experts_modules": len(mods), "n_bnb4bit": sum(bool(pred(m)) for _, m in mods)}
    try:
        out["n_bnb4bit_unwrapped"] = sum(bool(pred(innermost(m))) for _, m in mods)
        out["inner_param_types"] = sorted({type(getattr(innermost(m), k, None)).__name__ for _, m in mods for k in ("gate_up_proj", "down_proj")})
        out["wrap_depth"] = max([sum(1 for _ in iter_base_layers(m)) for _, m in mods] or [0])
    except Exception as e:
        out["unwrapped"] = f"unchecked: {e}"
    return out


def iter_base_layers(m):
    while hasattr(m, "base_layer"):
        m = m.base_layer
        yield m


# ----------------------------------------------------------------------------- T20 (TC1): matched LoRA init -- one deterministic A per slot
ATTN_KINDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
MATCHED_DISTRIBUTION = ("A[slot] ~ U(-b, b), b = 1/sqrt(fan_in), fan_in = in-width of the [r, in] slot = PEFT's "
                        "kaiming_uniform(a=sqrt(5)) on a [r, in] weight; drawn in fp32 on CPU from "
                        "torch.Generator().manual_seed(int.from_bytes(sha256('<seed>|<layer>|<kind>|<expert>')[:8], 'big')); "
                        "B left as built and asserted zero")


def parse_lora_init(spec):
    """`native` -> None (today's behaviour); `matched:<seed>` -> the integer seed; anything else is refused."""
    if spec in (None, "", "native"):
        return None
    m = re.fullmatch(r"matched:(\d+)", str(spec))
    if not m:
        raise ValueError(f"--lora-init must be `native` or `matched:<seed>`, got {spec!r}")
    return int(m.group(1))


def layer_index_of(name):
    """The decoder-layer index of a module/parameter path: its FIRST all-digit dotted component (`model.layers.7.…`,
    `base_model.model.model.layers.7.…`, `language_model.model.layers.7.…`). No family's word for the list is read.
    -1 when the path carries no index; such a slot is reported as unmapped, never guessed."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def slot_seed(seed, layer, kind, expert):
    """The registered per-slot generator seed: sha256(f"{seed}|{layer}|{kind}|{expert}"), first 8 bytes, big-endian."""
    return int.from_bytes(hashlib.sha256(f"{seed}|{layer}|{kind}|{expert}".encode()).digest()[:8], "big")


def matched_lora_A(seed, layer, kind, expert, r, fan_in):
    """PEFT's default LoRA A distribution materialised deterministically for ONE slot, in fp32 on CPU, orientation
    [r, in]. kaiming_uniform(a=sqrt(5)) on a [r, fan_in] weight: gain = sqrt(2/(1+5)) = sqrt(1/3), std = gain/sqrt(fan_in),
    bound = sqrt(3)*std = 1/sqrt(fan_in) (torch.nn.init; peft/tuners/lora/layer.py reset_lora_parameters)."""
    g = torch.Generator().manual_seed(slot_seed(seed, layer, kind, expert))
    b = 1.0 / math.sqrt(fan_in)
    return torch.empty(r, fan_in, dtype=torch.float32).uniform_(-b, b, generator=g)


def _in_features(m):
    for owner in (m, getattr(m, "base", None), getattr(m, "base_layer", None)):
        v = getattr(owner, "in_features", None) if owner is not None else None
        if isinstance(v, int) and v > 0:
            return v
    return None


def hidden_size_of(model):
    """The width of the hidden state, which is what decides gate_up-like (reads it) vs down-like (writes it): the config's
    hidden_size (text_config first), else the structural fallback -- the input width of a q_proj that sits beside a k_proj
    and an o_proj. (value, rule)."""
    cfg = getattr(model, "config", None)
    for owner in (getattr(cfg, "text_config", None), cfg):
        v = getattr(owner, "hidden_size", None) if owner is not None else None
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v, "config.hidden_size"
    for _, m in model.named_modules():
        if all(getattr(m, p, None) is not None for p in ("q_proj", "k_proj", "o_proj")):
            w = _in_features(m.q_proj)
            if w:
                return w, "q_proj.in_features (structural)"
    return None, "unknown"


def count_attention_projections(model):
    """n_attention_projections for n_slots_expected, by the lane's structural rule (q/k/o present, v optional), on the
    ADAPTED model: a child is counted by its name on the block, whatever class wraps it now (LoRALinear, peft Linear,
    Linear4bit) -- the same census the detectors take before wrapping, re-taken after."""
    n = 0
    for _, m in model.named_modules():
        if all(getattr(m, p, None) is not None for p in ("q_proj", "k_proj", "o_proj")):
            n += sum(getattr(m, p, None) is not None for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
    return n


def _raw_param(owner, name):
    """The stored parameter, bypassing a torch parametrization when one is registered on it (reading the property would
    run the parametrization -- on a PEFT-parametrized expert stack that materialises W + delta for the whole stack)."""
    pz = getattr(owner, "parametrizations", None)
    if pz is not None and name in pz:
        return pz[name].original
    return getattr(owner, name, None)


def quant_state_of(owner, name, p=None):
    """The bitsandbytes quant_state that describes `owner.<name>`: on the Params4bit itself (Unsloth, Linear4bit), else on
    the Bnb4bitParametrization registered for it (axolotl's quantize_moe_experts: the packed original is a plain uint8
    [N, 1] Parameter and the state lives on the parametrization module -- bitsandbytes/nn/parametrize.py). None otherwise."""
    p = _raw_param(owner, name) if p is None else p
    qs = getattr(p, "quant_state", None)
    if qs is not None:
        return qs
    pz = getattr(owner, "parametrizations", None)
    if pz is not None and name in pz:
        for mod in pz[name]:
            qs = getattr(mod, "quant_state", None)
            if qs is not None:
                return qs
    return None


def logical_stack_shape(owner, name):
    """The logical (E, d1, d2) of an expert stack whatever its storage: Unsloth's packed Params4bit carries `_original_shape`
    (moe_utils_bnb4bit.py:245/734 -- UNVERIFIED locally, read from the inspection note) and a quant_state; a parametrized
    stack's quant_state.shape; a plain tensor's own shape. None when no 3-D shape is found."""
    p = _raw_param(owner, name)
    if p is None:
        return None
    cands = [getattr(p, "_original_shape", None)]
    qs = quant_state_of(owner, name, p)
    cands.append(getattr(qs, "shape", None) if qs is not None else None)
    cands.append(tuple(p.shape))
    for c in cands:
        if c is not None and len(tuple(c)) == 3:
            return tuple(int(v) for v in c)
    return None


def expert_stack_shape(wrapper, parameter_name):
    """(E, out, in) of the 3-D expert stack a PEFT ParamWrapper adapts, read from the innermost base (logical_stack_shape);
    a base flagged `is_transposed` (transformers' grouped-GEMM convention) stores [E, in, out] and is reported swapped.
    None when the attribute is not a 3-D stack."""
    inner = wrapper
    while hasattr(inner, "base_layer"):
        inner = inner.base_layer
    shape = logical_stack_shape(inner, parameter_name)
    if shape is None:
        return None
    E, d1, d2 = shape
    return (E, d2, d1) if getattr(inner, "is_transposed", False) else (E, d1, d2)


def expert_kind(shape3, hidden, hint):
    """gate_up-like reads the hidden state (in == hidden), down-like writes it (out == hidden) -- STRUCTURE first; the
    parameter's own name decides only when structure cannot (hidden unknown, or a square stack where both hold).
    (kind | None, rule)."""
    E, out, inn = shape3
    if hidden:
        reads, writes = inn == hidden, out == hidden
        if reads and not writes:
            return "gate_up", "structure: in == hidden"
        if writes and not reads:
            return "down", "structure: out == hidden"
    h = (hint or "").lower()
    if any(k in h for k in ("down", "output", "out_", "w2")):
        return "down", f"name: {hint}"
    if any(k in h for k in ("gate_up", "gate", "up", "input", "in_", "w1", "w3")):
        return "gate_up", f"name: {hint}"
    return None, f"unmapped: shape {shape3}, hidden {hidden}, name {hint!r}"


def lora_slots(model, hidden=None):
    """Every LoRA slot on `model` -> (layer, kind, expert) plus where its A lives and which B belongs to it. By
    STRUCTURE and tensor shape, three layouts (the slot key is the same whichever framework produced it):

      PEFT ParamWrapper (HF, Unsloth through PEFT): a module with `parameter_name`, `num_experts` and lora_A / lora_B
        ModuleDicts; ONE A [r*E, in] per wrapped 3-D parameter, expert e = rows e*r:(e+1)*r (peft 0.21.2
        get_delta_factors: A.reshape(E, r, in)); B [out, r*E] viewed [out, r, E]. The module path carries no parameter
        name (the outer wrapper is `…experts`, the inner `…experts.base_layer`), so `parameter_name` is what is read.
      PEFT Linear (attention): lora_A ModuleDict without `parameter_name`; A [r, in]; kind from the projection's name.
      e4b ExpertsLoRA: 3-D `<prefix>_lora_A` [E, r, in] / `<prefix>_lora_B` [E, out, r] parameter pairs on one module.
      e4b LoRALinear (attention): 2-D `lora_A` [r, in] / `lora_B` [out, r] parameters on a module with a `base`.

    Returns (slots, unmapped, rules). A slot: {key, r, fan_in, A (the parameter), sel (how to address this slot's rows
    in A: ("rows", lo, hi) | ("index", e) | ("all",)), B, module}. Nothing here draws a number."""
    slots, unmapped, rules = [], [], {}
    for name, m in model.named_modules():
        L = layer_index_of(name)
        last = name.split(".")[-1]
        lora_A = getattr(m, "lora_A", None)
        if isinstance(lora_A, nn.ModuleDict):
            keys = list(lora_A.keys())
            if len(keys) != 1:
                unmapped.append(f"{name}: {len(keys)} adapters in lora_A ({keys}); one is mappable")
                continue
            A, B = lora_A[keys[0]].weight, m.lora_B[keys[0]].weight
            if hasattr(m, "parameter_name"):                                   # ParamWrapper
                E = int(getattr(m, "num_experts", 0) or 0)
                shp = expert_stack_shape(m, m.parameter_name)
                if E < 1 or shp is None or shp[0] != E or A.shape[0] % E:
                    unmapped.append(f"{name}: parameter_name={m.parameter_name!r} num_experts={E} stack={shp} A={list(A.shape)}")
                    continue
                kind, rule = expert_kind(shp, hidden, m.parameter_name)
                if kind is None:
                    unmapped.append(f"{name}: {rule}")
                    continue
                rules[f"{name} ({m.parameter_name})"] = f"{kind} by {rule}"
                r = A.shape[0] // E
                for e in range(E):
                    slots.append({"key": (L, kind, e), "r": r, "fan_in": int(A.shape[1]), "A": A, "sel": ("rows", e * r, (e + 1) * r), "B": B, "module": name, "B_name": f"{name}.lora_B.{keys[0]}.weight"})
            else:                                                              # peft Linear on a projection
                kind = ATTN_KINDS.get(last, last)
                rules[name] = f"{kind} by name ({'attention' if last in ATTN_KINDS else 'EXTRA: not an attention projection'})"
                slots.append({"key": (L, kind, -1), "r": int(A.shape[0]), "fan_in": int(A.shape[1]), "A": A, "sel": ("all",), "B": B, "module": name, "B_name": f"{name}.lora_B.{keys[0]}.weight"})
            continue
        direct = dict(m.named_parameters(recurse=False))
        if isinstance(lora_A, nn.Parameter) and lora_A.ndim == 2 and isinstance(getattr(m, "lora_B", None), nn.Parameter):
            kind = ATTN_KINDS.get(last, last)                                  # e4b LoRALinear
            rules[name] = f"{kind} by name ({'attention' if last in ATTN_KINDS else 'EXTRA: not an attention projection'})"
            slots.append({"key": (L, kind, -1), "r": int(lora_A.shape[0]), "fan_in": int(lora_A.shape[1]), "A": lora_A, "sel": ("all",), "B": m.lora_B, "module": name, "B_name": f"{name}.lora_B"})
            continue
        for pn, A in direct.items():                                           # e4b ExpertsLoRA: <prefix>_lora_A [E, r, in]
            if not pn.endswith("lora_A") or A.ndim != 3:
                continue
            B = direct.get(pn[:-1] + "B")
            if B is None or B.ndim != 3 or B.shape[0] != A.shape[0]:
                unmapped.append(f"{name}.{pn}: no matching 3-D {pn[:-1]}B")
                continue
            E, r, inn = (int(v) for v in A.shape)
            kind, rule = expert_kind((E, int(B.shape[1]), inn), hidden, pn[: -len("_lora_A")] if pn.endswith("_lora_A") else pn)
            if kind is None:
                unmapped.append(f"{name}.{pn}: {rule}")
                continue
            rules[f"{name}.{pn}"] = f"{kind} by {rule}"
            for e in range(E):
                slots.append({"key": (L, kind, e), "r": r, "fan_in": inn, "A": A, "sel": ("index", e), "B": B, "module": f"{name}.{pn}", "B_name": f"{name}.{pn[:-1]}B"})
    return slots, unmapped, rules


def apply_matched_init(model, spec, n_layers=None, cfg=None):
    """T20: overwrite every LoRA A with its slot's deterministic tensor (fp32 on CPU, copied into the parameter's own
    orientation, dtype and device) and check every LoRA B is zero. Returns the `matched_init` receipt block:
    {seed, n_slots_set, n_slots_expected, expected_parts, unmapped, complete, kinds, b_nonzero, hidden, mapping_rules,
    distribution}. `complete` is False whenever a slot could not be mapped, a slot was set twice, the count does not
    equal the structural expectation (n_attention_projections + 2 * n_layers * n_experts) or a B is not zero -- the
    reducer VOIDs such an arm for the matched set; nothing is silently left native."""
    seed = parse_lora_init(spec)
    if seed is None:
        return None
    hidden, hidden_rule = hidden_size_of(model)
    slots, unmapped, rules = lora_slots(model, hidden)
    n_set, kinds, seen, b_checked, b_nonzero, max_E = 0, {}, set(), set(), [], 0
    with torch.no_grad():
        for s in slots:
            L, kind, e = s["key"]
            if L < 0:
                unmapped.append(f"{s['module']}: no layer index in the path")
                continue
            if s["key"] in seen:
                unmapped.append(f"{s['module']}: slot {s['key']} set twice")
                continue
            seen.add(s["key"])
            max_E = max(max_E, e + 1)
            A = matched_lora_A(seed, L, kind, e, s["r"], s["fan_in"])
            sel = s["sel"]
            if sel[0] == "rows":
                dst = s["A"].data[sel[1]:sel[2]]
            elif sel[0] == "index":
                dst = s["A"].data[sel[1]]
            else:
                dst = s["A"].data
            if tuple(dst.shape) != tuple(A.shape):                          # a [in, r] slot would transpose here; none of
                unmapped.append(f"{s['module']}: slot {s['key']} is {list(dst.shape)}, expected {list(A.shape)} ([r, in])")  # the lane's layouts stores one
                continue
            dst.copy_(A.to(dst.dtype))
            n_set += 1
            kinds[kind] = kinds.get(kind, 0) + 1
            if id(s["B"]) not in b_checked:
                b_checked.add(id(s["B"]))
                if bool(s["B"].detach().any()):
                    b_nonzero.append(s["B_name"])
    n_attn = count_attention_projections(model)
    n_experts = declared_num_experts(cfg) if cfg is not None else None
    if n_experts is None:
        n_experts = max_E or None
    expected = (n_attn + 2 * int(n_layers) * int(n_experts)) if (n_layers and n_experts) else None
    complete = bool(expected is not None and n_set == expected and not unmapped and not b_nonzero)
    return {"seed": seed, "n_slots_set": n_set, "n_slots_expected": expected,
            "expected_parts": {"n_attention_projections": n_attn, "n_layers": n_layers, "n_experts": n_experts},
            "unmapped": unmapped[:24], "n_unmapped": len(unmapped), "complete": complete, "kinds": kinds, "b_nonzero": b_nonzero[:8],
            "hidden": {"value": hidden, "rule": hidden_rule},
            "mapping_rules": dict(list(rules.items())[:12]), "n_mapping_rules": len(rules), "distribution": MATCHED_DISTRIBUTION}


# ----------------------------------------------------------------------------- T21 (TC1): the frozen-base probe (registered slots, dequantised the framework's own way)
def _dtype_label(dt):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32", torch.uint8: "u8"}.get(dt, str(dt))


def tensor_bytes(t):
    """The raw bytes of a tensor whatever its dtype (bf16 has no numpy dtype): viewed as uint8 after a host copy."""
    t = t.detach().to("cpu").contiguous()
    return t.view(torch.uint8).numpy().tobytes()


def flip_first_byte(t):
    """A COPY of `t` whose first storage byte is XOR 0xFF -- the positive control's tamper, on any dtype."""
    c = t.detach().clone().contiguous()
    b = c.view(torch.uint8).view(-1)
    b[0] = b[0] ^ 0xFF
    return c


def innermost_experts(m):
    """Peel the adapters off an experts module: PEFT ParamWrappers (`base_layer`), then e4b's ExpertsLoRA (`base`)."""
    while hasattr(m, "base_layer"):
        m = m.base_layer
    if not hasattr(m, "gate_up_proj") and hasattr(m, "base"):
        m = m.base
    return m


def _slot_record(deq, regime, method, source, control_deq):
    sha, sha2 = sha_bytes(tensor_bytes(deq)), sha_bytes(tensor_bytes(control_deq))
    return {"sha": sha, "regime": regime, "n_bytes": int(deq.numel() * deq.element_size()), "shape": [int(v) for v in deq.shape],
            "dtype": _dtype_label(deq.dtype), "method": method, "source": source, "control_sha": sha2, "control_detects_flip": sha != sha2}


def _probe_e4b_stack(inner, slot, pk, am, shp):
    """e4b: the library's own per-expert dequant (Experts4bit._dequantize_expert, the function its forward recomputes
    with) on expert 0, to bf16; the control re-dequantises a 1-row COPY of the packed storage with one byte flipped."""
    packed, absmax, shape = getattr(inner, pk), getattr(inner, am), tuple(int(v) for v in getattr(inner, shp))
    deq = inner._dequantize_expert(packed, absmax, shape, 0, torch.bfloat16)
    deq2 = inner._dequantize_expert(flip_first_byte(packed[0:1]), absmax, shape, 0, torch.bfloat16)
    bits = int(getattr(inner, "bits", 4))
    regime = f"{inner.quant_type}/{inner.blocksize}" if bits != 16 else f"{_dtype_label(packed.dtype)} passthrough"
    return _slot_record(deq, regime, f"e4b {type(inner).__name__}._dequantize_expert(expert 0) -> bf16", f"{pk}[0]", deq2)


def _probe_stack_param(p, inner, attr):
    """HF / Unsloth: expert 0 of a 3-D expert stack. A bnb Params4bit is dequantised with ITS OWN quant_state through
    bitsandbytes.functional.dequantize_4bit (what Linear4bit / the bnb MoE backend call) -- the first expert's blocks only
    when the statistics are not nested (double-quant off; blocks of `blocksize` tile each expert exactly because
    out*in is a multiple of it), the whole stack sliced otherwise -- then cast to bf16 (bnb's forward casts the dequant to
    the activation dtype). A floating stack (HF's bf16 experts) is its own bytes, regime = its dtype. A Params4bit with
    no quant_state (the selftest's stand-in) is raw bytes, labelled so."""
    import bitsandbytes.functional as BF
    from bitsandbytes.functional import QuantState
    qs = quant_state_of(inner, attr, p)
    transposed = bool(getattr(inner, "is_transposed", False))
    if qs is not None:
        E, d1, d2 = (int(v) for v in qs.shape)
        numel_e = d1 * d2
        nested = bool(getattr(qs, "nested", False))
        regime = f"{qs.quant_type}/{qs.blocksize}" + ("+dq" if nested else "")
        if not nested:
            packed_e = p.data.reshape(-1)[: numel_e // 2].reshape(-1, 1)
            absmax_e = qs.absmax[: numel_e // int(qs.blocksize)]

            def deq_of(packed):
                st = QuantState(absmax=absmax_e, shape=torch.Size((d1, d2)), code=qs.code, blocksize=qs.blocksize, quant_type=qs.quant_type, dtype=qs.dtype)
                w = BF.dequantize_4bit(packed, quant_state=st).to(torch.bfloat16)
                return w.t().contiguous() if transposed else w
            return _slot_record(deq_of(packed_e), regime, "bnb dequantize_4bit(expert-0 blocks, own quant_state) -> bf16", f"{attr}[0]", deq_of(flip_first_byte(packed_e)))

        def deq_whole(packed):
            w = BF.dequantize_4bit(packed, quant_state=qs)[0].to(torch.bfloat16)
            return w.t().contiguous() if transposed else w
        return _slot_record(deq_whole(p.data), regime, "bnb dequantize_4bit(whole stack, nested statistics)[0] -> bf16", f"{attr}[0]", deq_whole(flip_first_byte(p.data)))
    if p.is_floating_point():
        w = p.data[0]
        w = w.t().contiguous() if transposed else w
        return _slot_record(w, _dtype_label(w.dtype), "bytes as stored (not quantised; no dequant in the forward)", f"{attr}[0]", flip_first_byte(w))
    w = p.data[0]
    return _slot_record(w, f"{_dtype_label(w.dtype)}-raw (no quant_state)", "raw bytes (a packed parameter without quant_state; selftest stand-in)", f"{attr}[0]", flip_first_byte(w))


def _probe_attention(q):
    """Layer 0 q_proj: a bnb Linear4bit weight is dequantised with its own quant_state (what Linear4bit.forward does
    through matmul_4bit) to bf16; a plain Linear is its bytes, regime = its dtype."""
    import bitsandbytes.functional as BF
    for owner in (q, getattr(q, "base", None), getattr(q, "base_layer", None)):
        w = getattr(owner, "weight", None) if owner is not None else None
        if w is not None:
            break
    if w is None:
        return None
    qs = getattr(w, "quant_state", None)
    if qs is not None:
        regime = f"{qs.quant_type}/{qs.blocksize}" + ("+dq" if bool(getattr(qs, "nested", False)) else "")

        def deq_of(packed):
            return BF.dequantize_4bit(packed, quant_state=qs).to(torch.bfloat16)
        return _slot_record(deq_of(w.data), regime, "bnb dequantize_4bit(weight, own quant_state) -> bf16", "q_proj.weight", deq_of(flip_first_byte(w.data)))
    return _slot_record(w.data, _dtype_label(w.dtype), "bytes as stored (not quantised)", "q_proj.weight", flip_first_byte(w.data))


def frozen_base_probe(model, framework):
    """T21: {slots: {gate_up, down, q_proj -> {sha, regime, n_bytes, method, control_detects_flip, ...}}, control_detects_flip,
    experts_module, attention_module, errors}. The registered slot is layer 0, expert 0 (the first experts module in
    module order, peeled to its frozen base) and layer 0's q_proj. Each slot's bytes are the bf16 the framework's own
    forward would see; each carries its own byte-flip control. The reducer reads SAME-BYTES / DIFFERENT / N-A across
    arms from the sha and the regime; this function compares nothing."""
    out = {"slots": {}, "control_detects_flip": None, "experts_module": None, "attention_module": None, "errors": []}
    with torch.no_grad():
        try:
            name, mod = next(((n, m) for n, m in model.named_modules() if is_experts_module(n)), (None, None))
            if mod is None:
                out["errors"].append("no module named experts")
            else:
                out["experts_module"] = name
                inner = innermost_experts(mod)
                if hasattr(inner, "_dequantize_expert") and hasattr(inner, "gate_up_absmax"):
                    out["slots"]["gate_up"] = _probe_e4b_stack(inner, "gate_up", "gate_up_proj", "gate_up_absmax", "_gate_up_shape")
                    out["slots"]["down"] = _probe_e4b_stack(inner, "down", "down_proj", "down_absmax", "_down_shape")
                else:
                    hidden, _ = hidden_size_of(model)
                    cands = [n for n, _ in inner.named_parameters(recurse=False)]
                    pz = getattr(inner, "parametrizations", None)          # a parametrized stack is not in named_parameters
                    if pz is not None:
                        cands += [k for k in pz.keys() if k not in cands]
                    for attr in cands:
                        p = _raw_param(inner, attr)
                        shp = expert_stack_shape(inner, attr)
                        if p is None or shp is None or (p.ndim != 3 and quant_state_of(inner, attr, p) is None and getattr(p, "_original_shape", None) is None):
                            continue
                        kind, rule = expert_kind(shp, hidden, attr)
                        if kind is None or kind in out["slots"]:
                            out["errors"].append(f"{attr}: {rule if kind is None else 'second ' + kind + ' stack'}")
                            continue
                        rec = _probe_stack_param(p, inner, attr)
                        rec["mapped_by"] = rule
                        out["slots"][kind] = rec
        except Exception as e:
            out["errors"].append(f"experts: {type(e).__name__}: {str(e)[:200]}")
        try:
            name, blk = next(((n, m) for n, m in model.named_modules() if all(getattr(m, p, None) is not None for p in ("q_proj", "k_proj", "o_proj"))), (None, None))
            if blk is None:
                out["errors"].append("no attention block (q_proj/k_proj/o_proj) found")
            else:
                out["attention_module"] = name
                rec = _probe_attention(blk.q_proj)
                if rec is None:
                    out["errors"].append(f"{name}.q_proj: no weight found")
                else:
                    out["slots"]["q_proj"] = rec
        except Exception as e:
            out["errors"].append(f"attention: {type(e).__name__}: {str(e)[:200]}")
    if out["slots"]:
        out["control_detects_flip"] = all(s["control_detects_flip"] for s in out["slots"].values())
    return out


# ----------------------------------------------------------------------------- the arm (both frameworks, one code path)
# ------------------------------------------------------------------ P45: where a training step's time goes (host vs device)
PROFILE_FAMILIES = [                     # first match wins; the order is the registration (P45-PREREG.md)
    ("memcpy", r"Memcpy|Memset|copy_|_to_copy|\.to\b|contiguous|clone"),
    ("fused_kernel", r"nf4_qlora|grouped|dgrad|fused_|int4_b32|gemm_4bit|nf4_grouped|mxfp4|triton|_gemm_int4|smallm|dequant"),
    ("routing", r"index_select|index_add|index_put|index_copy|gather|scatter|where|nonzero|topk|argsort|\bsort\b|bincount|cumsum|unique|masked_|one_hot|repeat_interleave|argmax|softmax|sigmoid|embedding"),
    ("optimizer", r"adam|Adam|optimizer|_foreach|lerp|addcdiv|addcmul|zero_grad|bnb|bitsandbytes|dequantize_blockwise|quantize_blockwise"),
    ("autograd", r"autograd|AccumulateGrad|Backward|backward|CheckpointFunction|checkpoint|torch::autograd"),
    ("matmul", r"\bmm\b|matmul|bmm|addmm|baddbmm|linear|cublas|cutlass|gemm|gemv"),
    ("norm_act", r"norm|silu|gelu|rsqrt|mul\b|add\b|sub\b|div\b|pow\b|mean|sum\b|exp\b|log\b|cat\b|split|chunk|view|reshape|transpose|permute|expand|slice|select|unsqueeze|squeeze|fill_|zeros|ones|empty|arange"),
]


def _family(name: str) -> str:
    import re
    for fam, rx in PROFILE_FAMILIES:
        if re.search(rx, name):
            return fam
    return "other"


def summarize_profile(prof, wall_s: float, n_steps: int, out_path: str) -> dict:
    """The census P45 registers: device-busy fraction (sum of device-side self time over the profiled steps' wall --
    overlapping streams can push it past 1.0 and that is reported, not clipped), device events (launches + memcpys)
    per step, and CPU self time by op family from key_averages(). Full top tables go to `out_path`; the returned dict
    is the summary the CELL line carries. Nothing here is a timed number: the timed s/step excludes nothing, the
    profiled steps simply carry the profiler's overhead and are flagged in the receipt."""
    from torch.autograd import DeviceType
    ka = prof.key_averages()
    dev_ms = 0.0; memcpy_ms = 0.0; n_dev = 0; n_cpu = 0; cpu_ms = 0.0
    by_fam_cpu, by_fam_dev = {}, {}
    rows = []
    for e in ka:
        name = e.key
        cpu_self = float(getattr(e, "self_cpu_time_total", 0.0)) / 1e3
        dev_self = float(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0))) / 1e3
        cnt = int(e.count)
        fam = _family(name)
        is_dev = getattr(e, "device_type", None) == DeviceType.CUDA
        if is_dev:
            n_dev += cnt; dev_ms += dev_self
            if "Memcpy" in name or "Memset" in name:
                memcpy_ms += dev_self
            by_fam_dev[fam] = by_fam_dev.get(fam, 0.0) + dev_self
        else:
            n_cpu += cnt; cpu_ms += cpu_self
            by_fam_cpu[fam] = by_fam_cpu.get(fam, 0.0) + cpu_self
        rows.append({"name": name[:120], "family": fam, "device": bool(is_dev), "count": cnt,
                     "self_cpu_ms": round(cpu_self, 3), "self_device_ms": round(dev_self, 3)})
    wall_ms = wall_s * 1e3
    summ = {
        "profiled_steps": n_steps, "wall_ms": round(wall_ms, 1), "wall_ms_per_step": round(wall_ms / max(n_steps, 1), 1),
        "device_ms": round(dev_ms, 1), "device_busy_fraction": round(dev_ms / wall_ms, 4) if wall_ms else None,
        "memcpy_ms": round(memcpy_ms, 1), "memcpy_fraction_of_device": round(memcpy_ms / dev_ms, 4) if dev_ms else None,
        "device_events_per_step": round(n_dev / max(n_steps, 1)), "cpu_ops_per_step": round(n_cpu / max(n_steps, 1)),
        "cpu_self_ms": round(cpu_ms, 1), "cpu_self_fraction_of_wall": round(cpu_ms / wall_ms, 4) if wall_ms else None,
        "cpu_self_by_family_fraction": {k: round(v / cpu_ms, 4) for k, v in sorted(by_fam_cpu.items(), key=lambda kv: -kv[1])} if cpu_ms else {},
        "device_by_family_fraction": {k: round(v / dev_ms, 4) for k, v in sorted(by_fam_dev.items(), key=lambda kv: -kv[1])} if dev_ms else {},
        "note": "device_busy_fraction sums device self time across streams and can exceed 1.0; cpu_self is the profiler's op-level self time and excludes interpreter time between ops",
    }
    top_cpu = sorted([r for r in rows if not r["device"]], key=lambda r: -r["self_cpu_ms"])[:80]
    top_dev = sorted([r for r in rows if r["device"]], key=lambda r: -r["self_device_ms"])[:80]
    write_json(out_path, {"summary": summ, "top_cpu": top_cpu, "top_device": top_dev, "families": [f for f, _ in PROFILE_FAMILIES]})
    summ["path"] = os.path.basename(out_path)
    return summ


def run_arm(a, load_fn, sampler=True):
    import importlib.metadata as md
    # #548: the window opens HERE, not at LOAD OK -- so the receipt accounts for the whole process, not a chosen slice.
    PH.reset()
    PH.begin(PROC_T0 if _FIRST_ARM[0] else time.perf_counter())
    unsloth_knobs = apply_unsloth_knobs(a)         # P2-1: into the environment BEFORE `import unsloth` (and before `env` is snapshotted)
    if a.framework == "e4b":                        # P67: the switch's counter is per ARM (the selftest runs many in one process)
        try:
            from experts4bit_qlora.lora import reset_reference_order_stats
            reset_reference_order_stats()
        except Exception:                           # an e4b without the switch; the receipt then records calls=None
            pass
    _FIRST_ARM[0] = False
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.adapter_dir, exist_ok=True)
    tk = json.load(open(a.tokens))
    body = json.dumps({"train": tk["train"], "eval": tk["eval"]}, separators=(",", ":")).encode()
    if sha_bytes(body) != tk["sha256"] or (a.tokens_sha and tk["sha256"] != a.tokens_sha):
        stub(a, "tokens_mismatch", f"{a.tokens}: sha {sha_bytes(body)[:12]} != {tk['sha256'][:12]} / registered {str(a.tokens_sha)[:12]}", code=13)
    if tk.get("fam") not in (None, a.fam):
        stub(a, "tokens_mismatch", f"{a.tokens} was tokenised for fam {tk.get('fam')}, arm is {a.fam}", code=13)
    train, ev = tk["train"], tk["eval"][:a.eval_n]
    template, eos, pad_id = tk.get("template", "clinical"), tk.get("eos", "") or "", tk.get("pad_id")
    M = int(getattr(a, "micro_batch", 1) or 1)

    env = {"framework": a.framework, "torch": torch.__version__, "device": DEV,
           "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "cap": list(torch.cuda.get_device_capability()) if DEV == "cuda" else None,
           "host": host_fingerprint(), "anchor_json": os.environ.get("TC1_ANCHOR_JSON"), "box_class": os.environ.get("TC1_BOX_CLASS"),
           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "unsloth_loader": a.unsloth_loader if a.framework == "unsloth" else None,
           "unsloth_env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_")}}
    for pkg in ("transformers", "experts4bit-qlora", "grouped-nf4-gemm", "bitsandbytes", "peft", "unsloth", "unsloth_zoo", "triton", "huggingface_hub"):
        try:
            env[pkg] = md.version(pkg)
        except Exception:
            env[pkg] = None

    idle_w = idle_power() if sampler else 0.0
    torch.manual_seed(a.seed)
    t_load = time.perf_counter()
    PH.mark("preamble", t_load - PH._t0)            # #548: argv, the tokens file, the version census, the idle-power probe
    alarm_ctx = {}                                  # what the watchdog can name about the model, filled in as it is learned
    PH.start_watchdog(phase_budget_for(a), _phase_alarm_action(a, alarm_ctx))
    try:
        model, x = load_fn(a)
    except SystemExit as e:                      # a stub already written (int code) propagates; a framework's SystemExit(message) is a refusal row
        if isinstance(e.code, int) or e.code is None:
            raise
        stub(a, "refused", f"SystemExit: {str(e.code)[:700]}", {"phase": "load"}, code=3)
    except Exception as e:
        st, code = classify_load_exception(e)
        stub(a, st, f"{type(e).__name__}: {str(e)[:700]}", {"phase": "load"}, code=code)
    load_s = time.perf_counter() - t_load
    if a.framework == "unsloth" and not x.get("moe_backend_selected"):   # P2-1: select_moe_backend() after load, on every Unsloth arm
        x["moe_backend_selected"] = unsloth_backend_selected()
    n_attn4, n_patched, reason, banner_lines = x["n_attn4"], x["n_patched"], x["reason"], x["banner_lines"]
    hashes, fwd_kwargs, tokenizer_obj = x["hashes"], x["fwd_kwargs"], x.get("tokenizer_obj")

    alarm_ctx.update({"n_layers": x.get("n_layers"), "model_type": x.get("model_type"), "load_s": round(load_s, 1)})
    # U5: adapters fp32 (Unsloth: cast if needed, recorded), the censuses, U3/T6: the trainable count
    _census_phase = PH("census")                    # #548: the two censuses + the tokenizer agreement re-derivation
    _census_phase.__enter__()
    tr, n_trainable, dtypes_before, non_adapter, groups = trainable_census(model)
    cast = 0
    adapter_dtype = getattr(a, "adapter_dtype", "fp32") or "fp32"
    if adapter_dtype == "fp32":                   # T19 (TC1): adapters fp32 in EVERY arm, e4b included (tp4: `if a.framework != "e4b"`)
        for n, p in tr:
            if p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
                cast += 1
    lora_init = getattr(a, "lora_init", "native") or "native"
    matched = None
    if parse_lora_init(lora_init) is not None:    # T20 (TC1): the per-slot deterministic A, after the adapters exist and after the cast
        matched = apply_matched_init(model, lora_init, x.get("n_layers"), getattr(model, "config", None))
        if matched["b_nonzero"]:
            stub(a, "harness_error", f"matched init: {len(matched['b_nonzero'])} LoRA B tensor(s) are not zero after construction "
                 f"(e.g. {matched['b_nonzero'][:2]}); a matched start is impossible on this arm",
                 {"phase": "census", "matched_init": matched, "adapter_dtype": adapter_dtype, "lora_init": lora_init,
                  "n_layers": x.get("n_layers"), "model_type": x.get("model_type")}, code=18)
    tr, n_trainable, dtypes_after, non_adapter, groups = trainable_census(model)
    census = quant_census(model)
    if pad_id is None and tokenizer_obj is not None:
        pad_id = getattr(tokenizer_obj, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer_obj, "eos_token_id", None)
    tokenizer_agree = None
    if tokenizer_obj is not None:
        try:   # re-derive 8 train rows from the raw dataset if it sits beside the tokens file; informational
            raw = glob.glob(os.path.join(os.path.dirname(os.path.abspath(a.tokens)), "data", f"ds_{tk.get('dataset', 'x')[3:-5]}.json")) \
                or glob.glob(os.path.join(os.path.dirname(os.path.abspath(a.tokens)), "data", "ds_*.json"))
            if raw:
                rows = json.load(open(raw[0]))["train"]
                tokenizer_agree = encode_rows(tokenizer_obj, rows[:8], a.seq, template, eos) == train[:8]
        except Exception as e:
            tokenizer_agree = f"unchecked: {e}"
    _census_phase.__exit__(None, None, None)
    with PH("frozen_probe"):                      # T21 (TC1): the registered layer-0 slots, dequantised the framework's own way + the byte-flip control
        probe = frozen_base_probe(model, a.framework)
    trainable_mismatch = None
    if a.expect_trainable and n_trainable != a.expect_trainable:
        trainable_mismatch = {"expected": a.expect_trainable, "got": n_trainable, "by_group": groups}
    print("LOAD OK " + json.dumps({"fam": a.fam, "framework": a.framework, "arm": a.arm, "model_type": x.get("model_type"), "n_layers": x.get("n_layers"),
                                   "load_s": round(load_s, 1), "verify": x.get("verify"), "n_attn4": n_attn4, "n_patched": n_patched,
                                   "structural_expected_n_attn4": x.get("structural_expected_n_attn4"), "detector_version": x.get("detector_version"),
                                   "trainable_params": n_trainable, "trainable_tensors": len(tr), "trainable_by_group": groups, "trainable_mismatch": trainable_mismatch,
                                   "dtypes_before": dtypes_before, "dtypes_after": dtypes_after, "lora_cast_to_fp32": cast,
                                   "adapter_dtype": adapter_dtype, "lora_init": lora_init, "matched_init": matched,
                                   "frozen_base_probe": {"slots": {k: {"sha": v["sha"][:16], "regime": v["regime"], "control_detects_flip": v["control_detects_flip"]} for k, v in probe["slots"].items()},
                                                         "control_detects_flip": probe["control_detects_flip"], "errors": probe["errors"]},
                                   "non_adapter_trainable": non_adapter[:4], "census": {k: v for k, v in census.items() if k != "samples"},
                                   "tokenizer_agree": tokenizer_agree, "ckpt": x["ckpt_mode"], "probes": x.get("probes"), "attn4_probe": x.get("attn4_probe")}), flush=True)
    for l in banner_lines:
        print("ENGAGE " + l[:300], flush=True)
    if non_adapter:
        stub(a, "void_trainable", f"non-adapter trainables {non_adapter[:4]}",
             {"trainable_params": n_trainable, "non_adapter_trainable": non_adapter[:8], "census": census, "n_layers": x.get("n_layers"), "model_type": x.get("model_type")}, code=15)
    with PH("trainable_sha"):                       # #548: every trainable parameter cast to CPU fp32 purely to be hashed
        init_sha = trainable_sha(tr)

    with PH("counters"):
        counter = Counters()
        if a.framework == "e4b":
            counter.install_e4b()
        elif a.framework in ("hf", "axolotl"):
            counter.install_hf(model)
        else:
            counter.install_unsloth(model)

    with PH("c1_before"):                           # #548: the frozen expert bytes, copied to CPU and sha256'd (C1_bytes_hashed says how many)
        h_before, bytes_before, empties_before = hashes(model)
        assert bytes_before > 0, "C1 hashed ZERO bytes -- gate is vacuous"
        assert empties_before == 0, f"C1 saw {empties_before} empty frozen tensors"
        assert control_flip_fires(h_before), "C1 positive control did not fire -- the check cannot fail"

    with PH("eval0"):                               # #548: also the first forward -- any JIT / autotune on the forward path lands here
        ev0 = eval_loss(model, ev, fwd_kwargs, a.autocast)
    curve = [{"step": 0, "heldout_loss": round(ev0, 5), "train_wall_s": 0.0}]
    _opt_phase = PH("optimizer")                    # #548
    _opt_phase.__enter__()
    params = [p for _, p in tr]
    optim_name, wd = getattr(a, "optim", "adamw_torch"), float(getattr(a, "weight_decay", 0.01))
    if optim_name == "adamw_8bit":                     # T14: the notebooks' optimizer, the same call in every arm
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            stub(a, "harness_error", f"--optim adamw_8bit needs bitsandbytes in this environment: {e}", {"phase": "optimizer"}, code=10)
        opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=wd)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=wd)   # tp2's call (torch defaults, wd 0.01) when the defaults are kept
    schedule, warmup = getattr(a, "lr_schedule", "constant"), int(getattr(a, "warmup_steps", 0) or 0)
    sched = None
    if schedule == "linear":                           # transformers.get_linear_schedule_with_warmup, exactly

        def _lam(step, N=a.steps, W=warmup):
            if step < W:
                return step / max(1, W)
            return max(0.0, (N - step) / max(1, N - W))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lam)
    optimizer_str = f"{optim_name}(lr={a.lr}, weight_decay={wd}) schedule={schedule} warmup_steps={warmup}"
    model.train()
    gc.collect()
    reset_peak()
    _opt_phase.__exit__(None, None, None)

    losses, step_ms, tokens_per_step, tokens_padded_per_step, lr_per_step, kcalls = [], [], [], [], [], []
    profile_summary = None                                              # P45
    microbatch_ms = []                                                # T17: per step, one entry per micro-batch (only with --microbatch-timing)
    train_wall, steps_done = 0.0, 0
    common_stub = lambda: {"n_patched": n_patched, "n_attn4": n_attn4, "init_sha": init_sha, "load_s": round(load_s, 1),
                           "trainable_params": n_trainable, "census": census, "n_layers": x.get("n_layers"), "model_type": x.get("model_type"),
                           "micro_batch": M, "optimizer": optimizer_str,
                           "adapter_dtype": adapter_dtype, "lora_init": lora_init, "matched_init": matched, "frozen_base_probe": probe,
                           "adapter_dtypes_before": dtypes_before, "adapter_dtypes_after": dtypes_after, "lora_cast_to_fp32": cast}
    try:
        with PowerSampler(enabled=sampler) as ps:
            cuda_sync()
            PH.end_prologue()                       # #548: the window closes exactly where the timed window opens
            print("PROLOGUE " + json.dumps(PH.report()), flush=True)
            t0 = time.perf_counter()
            prof, prof_wall = None, 0.0                                       # P45: profiled steps are inside the window and marked in the receipt
            for i in range(a.steps):
                if a.profile_steps and i == a.profile_warm:
                    from torch.profiler import ProfilerActivity, profile as _tprofile
                    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
                    prof = _tprofile(activities=acts, record_shapes=False, profile_memory=False, with_stack=False)
                    prof.start()
                before = counter.snapshot()
                ts = time.perf_counter()
                loss_sum, ntok, npad = 0.0, 0, 0
                mb_ms = []
                lr_per_step.append(float(opt.param_groups[0]["lr"]))
                for j in range(a.accum):                                    # T5/T13: accum micro-batches of M rows, rows in fixed order
                    tm = time.perf_counter() if a.microbatch_timing else None
                    if M == 1:
                        ids = torch.tensor(train[(i * a.accum + j) % len(train)], dtype=torch.long).unsqueeze(0).to(DEV)
                        kw, labels, nreal = fwd_kwargs(ids), ids, int(ids.numel())
                    else:
                        rows = [train[((i * a.accum + j) * M + k) % len(train)] for k in range(M)]
                        ids, mask, labels = collate(rows, pad_id)
                        kw, nreal = {"attention_mask": mask}, sum(len(r) for r in rows)
                    with autocast_ctx(a.autocast):
                        out = model(input_ids=ids, labels=labels, **kw)
                        loss = out.loss / a.accum
                    loss.backward()
                    if tm is not None:                                      # T17: a sync per micro-batch, so the number is the micro-batch's
                        cuda_sync()
                        mb_ms.append(round((time.perf_counter() - tm) * 1e3, 1))
                    loss_sum += float(out.loss.detach())
                    ntok += nreal
                    npad += int(ids.numel()) - nreal
                opt.step()
                if sched is not None:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                cuda_sync()
                dt = time.perf_counter() - ts
                if prof is not None and a.profile_warm <= i < a.profile_warm + a.profile_steps:
                    prof_wall += dt
                    if i == a.profile_warm + a.profile_steps - 1:
                        prof.stop()
                        profile_summary = summarize_profile(prof, prof_wall, a.profile_steps, os.path.splitext(receipt_path(a))[0] + "_profile.json")
                        prof = None
                        print("PROFILE " + json.dumps(profile_summary), flush=True)
                train_wall += dt
                step_ms.append(round(dt * 1e3, 1))
                losses.append(round(loss_sum / a.accum, 5))
                tokens_per_step.append(ntok)
                tokens_padded_per_step.append(npad)
                if a.microbatch_timing:
                    microbatch_ms.append(mb_ms)
                after = counter.snapshot()
                kcalls.append({k: after[k] - before[k] for k in after})
                steps_done = i + 1
                if steps_done % max(1, int(a.log_every)) == 0:
                    mb = f" mb_ms {mb_ms}" if a.microbatch_timing else ""
                    print(f"    step {steps_done}/{a.steps} loss {losses[-1]} {step_ms[-1]} ms{mb} kcalls {kcalls[-1]}", flush=True)
                if steps_done % a.eval_every == 0:
                    e = eval_loss(model, ev, fwd_kwargs, a.autocast)
                    curve.append({"step": steps_done, "heldout_loss": round(e, 5), "train_wall_s": round(train_wall, 2)})
                    print(f"    eval@{steps_done} heldout {e:.5f} train_wall {train_wall:.1f}s", flush=True)
            cuda_sync()
            wall = time.perf_counter() - t0
    except Exception as e:
        counter.uninstall()
        if is_oom(e):
            stub(a, "oom", f"OOM at step {steps_done + 1}: {str(e)[:200]}",
                 dict(common_stub(), phase="train", steps_done=steps_done, losses=losses, step_ms=step_ms, microbatch_ms=microbatch_ms, peak_vram_gb=peak_gb()), code=5)
        raise
    counter.uninstall()
    peak = peak_gb()

    if curve[-1]["step"] != steps_done:
        with PH("eval_final"):                      # #548
            e = eval_loss(model, ev, fwd_kwargs, a.autocast)
        curve.append({"step": steps_done, "heldout_loss": round(e, 5), "train_wall_s": round(train_wall, 2)})
    ev1 = curve[-1]["heldout_loss"]
    with PH("c1_after"):                            # #548: the SECOND full pass over the frozen expert bytes
        h_after, bytes_after, empties_after = hashes(model)
    changed = [k for k in h_before if h_before[k] != h_after.get(k)]
    c1_ok = (not changed) and bytes_after == bytes_before and empties_after == 0

    bnb4 = u8_bnb4bit(model) if a.framework == "unsloth" else None

    # U7: the adapter
    adapter = {}
    atag = f"{a.fam}_{a.tag}"
    _adapter_phase = PH("adapter_save")              # #548
    _adapter_phase.__enter__()
    try:
        if a.framework == "e4b":
            if a.selftest:
                sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k}
                pth = os.path.join(a.adapter_dir, f"adapter_{atag}.pt")
                torch.save(sd, pth)
                n_t = len(sd)
            else:
                from experts4bit_qlora.train import save_adapter
                n_t = save_adapter(model, a.adapter_dir, atag)
                pth = os.path.join(a.adapter_dir, f"adapter_{atag}.pt")
            sd = torch.load(pth, map_location="cpu")
            adapter = {"path": pth, "bytes": os.path.getsize(pth), "tensors": n_t, "params": sum(v.numel() for v in sd.values()),
                       "dtypes": sorted({str(v.dtype) for v in sd.values()})}
        else:
            d = os.path.join(a.adapter_dir, atag)
            model.save_pretrained(d)
            files = [os.path.join(d, f) for f in os.listdir(d)]
            adapter = {"path": d, "bytes": sum(os.path.getsize(f) for f in files), "files": sorted(os.path.basename(f) for f in files)}
            try:
                from safetensors import safe_open
                st = [f for f in files if f.endswith(".safetensors")]
                if st:
                    with safe_open(st[0], "pt") as fh:
                        keys = list(fh.keys())
                        adapter["tensors"] = len(keys)
                        adapter["params"] = sum(math.prod(fh.get_slice(k).get_shape()) for k in keys)
                        adapter["dtypes"] = sorted({str(fh.get_slice(k).get_dtype()) for k in keys})
            except Exception as e:
                adapter["read_error"] = str(e)[:200]
    except Exception as e:
        adapter = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    _adapter_phase.__exit__(None, None, None)

    mean_w = statistics.mean(ps.samples) if ps.samples else None
    net_w = (mean_w - idle_w) if mean_w else None
    # P56: what RAN on the batched arm, not what was patched. `enable_batched_train`
    # returns a count of modules PATCHED; every one of them can still fall back to the
    # reference forward on any given call, indistinguishably in the output. The reducer
    # VOIDs a batched arm with fallback_calls > 0, so the number has to reach the receipt.
    batched_stats = None
    if a.arm == "batched":
        from experts4bit_qlora import batched_fallback_stats
        batched_stats = batched_fallback_stats(model)
        batched_stats.pop("per_module", None)     # totals + by_reason; per-module is 30 dicts
    # P67: what the perturbed-reference switch DID on this arm, read from the loop itself, never from the
    # environment alone -- an order that was set but never reached ExpertsLoRA.forward would make a floor
    # arm a plain repeat, and p67_reduce.py refuses such an arm as a floor draw. None on every other arm.
    reference_order = None
    if a.framework == "e4b" and a.arm == "reference":
        reference_order = {"requested": os.environ.get("E4B_REFERENCE_EXPERT_ORDER") or None}
        try:
            from experts4bit_qlora.lora import reference_order_stats
            reference_order.update(reference_order_stats())
        except Exception:                         # an e4b without the switch: say so rather than guess (the reducer refuses it)
            reference_order["calls"] = None
            reference_order["order"] = None
    key = {"e4b": "fused_grouped_lora", "unsloth": "moe_bnb4bit_backend", "hf": "experts_forward", "axolotl": "experts_forward"}[a.framework]
    kps = [k[key] for k in kcalls]
    efw = [k["experts_forward"] for k in kcalls]
    ub_keys = [k for _, _, k in UNSLOTH_BACKEND_FUNCS]            # P2-1: per-step min/max of every Unsloth backend counter
    ub_min = {k: min(c.get(k, 0) for c in kcalls) for k in ub_keys} if (a.framework == "unsloth" and kcalls) else None
    ub_max = {k: max(c.get(k, 0) for c in kcalls) for k in ub_keys} if (a.framework == "unsloth" and kcalls) else None
    steady = step_ms[10:] if len(step_ms) > 10 else step_ms
    cell = {
        "framework": a.framework, "fam": a.fam, "model": a.model, "revision": a.revision, "model_type": x.get("model_type"), "n_layers": x.get("n_layers"),
        "snapshot_dir": x.get("snapshot_dir"), "arm": a.arm, "tag": a.tag,
        "status": "ok" if c1_ok else "c1_failed", "steps": a.steps, "seq": a.seq, "accum": a.accum, "micro_batch": M, "autocast": bool(a.autocast),
        "r": a.r, "alpha": a.alpha, "lr": a.lr, "seed": a.seed, "offload": bool(a.offload),
        "optimizer": optimizer_str, "lr_per_step": [round(v, 8) for v in lr_per_step], "template": template,
        "grad_ckpt": x["ckpt_mode"], "attn_4bit": bool(a.attn_4bit), "n_attn4": n_attn4, "attn4_probe": x.get("attn4_probe"),
        "structural_expected_n_attn4": x.get("structural_expected_n_attn4"), "detector_version": x.get("detector_version"),
        "loader_used": x.get("loader_used"), "loader_fallback_reason": x.get("loader_fallback_reason"), "unsloth_targets": x.get("unsloth_targets"),
        "hf_targets": x.get("hf_targets"), "axolotl_targets": x.get("axolotl_targets"), "axolotl": x.get("axolotl"),      # P2-2
        "axolotl_bnb4bit_modules": x.get("axolotl_bnb4bit_modules"),
        "unsloth_knobs": unsloth_knobs, "moe_backend_selected": x.get("moe_backend_selected"),                            # P2-1
        "unsloth_double_quant": x.get("unsloth_double_quant"),
        "unsloth_backend_calls_per_step_min": ub_min, "unsloth_backend_calls_per_step_max": ub_max, "unsloth_backend_absent": counter.absent,
        "tokens": {"path": os.path.basename(a.tokens), "sha256": tk["sha256"], "n_train": len(train), "eval_rows_used": len(ev), "tokenizer_agree": tokenizer_agree,
                   "pad_id": pad_id},
        "prereg": a.prereg, "harness": HARNESS, "env": env, "load_s": round(load_s, 1),
        **PH.report(),                              # #548: phase_seconds, prologue_s, prologue_unattributed_s, phase_budget_s
        "verify": x.get("verify"), "census": census, "engagement_banners": banner_lines, "unsloth_bnb4bit_modules": bnb4,
        "trainable_params": n_trainable, "trainable_tensors": len(tr), "trainable_by_group": groups,
        "expect_trainable": a.expect_trainable, "trainable_mismatch": trainable_mismatch,
        "adapter_dtypes_before": dtypes_before, "adapter_dtypes_after": dtypes_after,
        "lora_cast_to_fp32": cast, "init_sha": init_sha, "n_patched": n_patched, "enable_reason": reason, "probes": x.get("probes"),
        "adapter_dtype": adapter_dtype, "lora_init": lora_init, "matched_init": matched,      # T19 / T20 (TC1)
        "frozen_base_probe": probe,                                                           # T21 (TC1)
        "batched_stats": batched_stats,          # P56: None unless --arm batched; see below
        "reference_order": reference_order,      # P67: None unless an e4b reference arm; see above
        "dgrad": (bool(getattr(a, "dgrad", 1)) if a.arm == "fused" else None),   # P56: which backward the fused arm ran
        "kernel_counter_key": key, "kernel_calls_per_step": kps, "kernel_calls_per_step_min": (min(kps) if kps else 0),
        "experts_forward_calls_per_step_min": (min(efw) if efw else 0), "kernel_calls_all": kcalls,
        "C1_tensors_hashed": len(h_before), "C1_bytes_hashed": bytes_before, "C1_empties_skipped": empties_before,
        "C1_control_detects_flipped_byte": True, "C1_experts_changed": len(changed), "C1_bit_exact": c1_ok, "C1_changed_sample": changed[:3],
        "loss_first": losses[0], "loss_last": losses[-1], "loss_mean_last20": round(statistics.mean(losses[-20:]), 5),
        "eval_loss_step0": round(ev0, 5), "eval_loss_final": ev1, "eval_curve": curve,
        "s_per_step": round(wall / a.steps, 4), "s_per_step_median_11plus": round(statistics.median(steady) / 1e3, 4), "step_ms": step_ms, "microbatch_ms": microbatch_ms, "log_every": int(a.log_every), "microbatch_timing": bool(a.microbatch_timing),
        "train_wall_s": round(train_wall, 2), "window_wall_s": round(wall, 2),
        "tokens_per_step": tokens_per_step, "tokens_total": sum(tokens_per_step), "tokens_per_s": round(sum(tokens_per_step) / train_wall, 1) if train_wall else None,
        "tokens_padded_per_step": tokens_padded_per_step, "tokens_padded_total": sum(tokens_padded_per_step),
        "peak_vram_gb": peak, "idle_w": round(idle_w, 1), "mean_w": round(mean_w, 1) if mean_w else None, "power_samples": len(ps.samples),
        "sampler": bool(sampler), "joules_per_step": round(net_w * (train_wall / a.steps), 2) if net_w else None,
        "adapter": adapter, "losses": losses,
        "profile": profile_summary, "profile_steps": int(a.profile_steps), "profile_warm": int(a.profile_warm),
    }
    write_json(receipt_path(a), cell)
    print(("CELL OK " if c1_ok else "CELL C1_FAILED ") + json.dumps(
        {k: v for k, v in cell.items() if k not in ("losses", "step_ms", "microbatch_ms", "tokens_per_step", "tokens_padded_per_step", "lr_per_step", "kernel_calls_all",
                                                   "env", "census", "eval_curve", "kernel_calls_per_step")}), flush=True)
    if not c1_ok:
        sys.exit(4)
    return cell


# ----------------------------------------------------------------------------- T8: the CPU self-test (tiny synthetic model, mocked kernels)
# TC1: the tiny models carry the REAL adapter layouts of the two sides -- e4b's (ExpertsLoRA / LoRALinear shapes, A ~ N(0, 1/r),
# bf16 expert adapters as the streaming loader builds them) and PEFT 0.21.2's (ParamWrapper / lora.Linear shapes, A ~
# kaiming_uniform, one [r*E, in] A per wrapped stack) -- over the SAME frozen base (every base tensor is drawn from its own
# seeded generator, so construction order cannot change it). So the step-0 loss is identical across the tiny arms by
# construction, the matched init gives the two layouts the same function, and the native inits give different ones: what
# the T20 selftest asserts. The frozen stacks stay the uint8 stand-ins of tp4 (the bytes C1 hashes), with the Experts4bit
# attributes the T21 probe reads bolted on in e4b mode; the REAL bnb / e4b dequant paths are exercised by
# _selftest_frozen_probe_real on genuine CPU-quantised tensors.
class _FakeTok:
    """Byte-level tokenizer stand-in with AutoTokenizer's call shape."""
    def __call__(self, text, truncation=True, max_length=512):
        ids = [1 + (b % 62) for b in text.encode()][:max_length]
        return types.SimpleNamespace(input_ids=ids)


class Params4bit(nn.Parameter):
    """A uint8 parameter whose class NAME is what the census and hashers key on (bitsandbytes' Params4bit)."""
    def __new__(cls, data):
        return super().__new__(cls, data, requires_grad=False)


_BASE_SEED = 1000       # every frozen base tensor of the tiny models: the same bytes in both layouts (SAME-BASE by construction)


def _seeded_randn(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def _seeded_uniform(shape, seed, bound):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape).uniform_(-bound, bound, generator=g)


class _LoRALinear(nn.Module):
    """e4b's LoRALinear shape: `base`, lora_A [r, in], lora_B [out, r]; A ~ N(0, 1/r) (experts4bit_qlora/lora.py), B = 0,
    scaling alpha/r; the delta computed in the adapter dtype and cast back (the library's dtype rule)."""
    def __init__(self, base, r, alpha, seed, dtype):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(_seeded_randn((r, base.in_features), seed, 1.0 / r).to(dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dtype))

    def forward(self, x):
        d = self.scaling * ((x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t())
        return self.base(x) + d.to(x.dtype)


class _PeftLinear(nn.Module):
    """PEFT lora.Linear's shape on an attention projection (Unsloth and the HF arm): `base_layer`, lora_A / lora_B
    ModuleDicts keyed by the adapter name holding nn.Linear(in, r) / nn.Linear(r, out); A ~ kaiming_uniform(a=sqrt(5)) =
    U(-1/sqrt(in), 1/sqrt(in)), B = 0 (peft/tuners/lora/layer.py reset_lora_parameters); adapters in the base's dtype."""
    def __init__(self, base, r, alpha, seed, dtype):
        super().__init__()
        self.base_layer = base
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.in_features, self.out_features, self.scaling = base.in_features, base.out_features, alpha / r
        A, B = nn.Linear(base.in_features, r, bias=False), nn.Linear(r, base.out_features, bias=False)
        with torch.no_grad():
            A.weight.copy_(_seeded_uniform((r, base.in_features), seed, 1.0 / math.sqrt(base.in_features)))
            B.weight.zero_()
        self.lora_A, self.lora_B = nn.ModuleDict({"default": A.to(dtype)}), nn.ModuleDict({"default": B.to(dtype)})

    def forward(self, x):
        A, B = self.lora_A["default"].weight, self.lora_B["default"].weight
        d = self.scaling * ((x.to(A.dtype) @ A.t()) @ B.t())
        return self.base_layer(x) + d.to(x.dtype)


class _TinyStack(nn.Module):
    """The frozen expert stacks both layouts share: gate_up_proj [E, 2I, H] and down_proj [E, H, inter] as uint8 'packed'
    storage (the bytes C1 hashes; the selftest's dequant is (u8/255 - 0.5) * 0.05), plus -- in e4b mode -- the absmax
    buffers and the Experts4bit attributes the T21 probe reads (_dequantize_expert, quant_type, blocksize, bits,
    _gate_up_shape, _down_shape). In peft mode the stacks are the fake `Params4bit` (the class NAME the census keys on)
    with no quant_state, so the probe records them as raw bytes and says so."""
    def __init__(self, E, H, inter, mode, param_cls):
        super().__init__()
        self.E, self.H, self.inter, self.mode, self.num_experts = E, H, inter, mode, E
        g = torch.Generator().manual_seed(7)
        if mode == "axolotl":
            # P2-2: what quantize_moe_experts leaves behind (moe_quant.py:150-181 -> bitsandbytes replace_parameter_4bit): bf16
            # stacks become a plain uint8 [N, 1] `parametrizations.<name>.original` with the quant_state on the
            # Bnb4bitParametrization; reading `self.gate_up_proj` dequantises. Real bnb, on CPU.
            from bitsandbytes.nn.parametrize import replace_parameter_4bit
            self.gate_up_proj = nn.Parameter(_seeded_randn((E, 2 * inter, H), _BASE_SEED + 700, 0.05).to(torch.bfloat16))
            self.down_proj = nn.Parameter(_seeded_randn((E, H, inter), _BASE_SEED + 701, 0.05).to(torch.bfloat16))
            for n in ("gate_up_proj", "down_proj"):
                replace_parameter_4bit(self, n, compress_statistics=False, quant_type="nf4", blocksize=64)
            return
        self.gate_up_proj = param_cls(torch.randint(0, 255, (E, 2 * inter, H), generator=g, dtype=torch.uint8))
        self.down_proj = param_cls(torch.randint(0, 255, (E, H, inter), generator=g, dtype=torch.uint8))
        if mode == "e4b":
            self.gate_up_absmax = nn.Parameter(torch.rand(E, 4, generator=g), requires_grad=False)
            self.down_absmax = nn.Parameter(torch.rand(E, 4, generator=g), requires_grad=False)
            self.quant_type, self.blocksize, self.bits = "selftest_u8", 1, 8
            self._gate_up_shape, self._down_shape = (2 * inter, H), (H, inter)

    @staticmethod
    def _deq(packed):
        return (packed.float() / 255.0 - 0.5) * 0.05

    def _dequantize_expert(self, packed, absmax, shape, expert_idx, dtype):
        return self._deq(packed[expert_idx]).reshape(shape).to(dtype)

    def forward(self, x, A1, B1, A2, B2, scaling):
        """act(W1 x + s*B1 A1 x), then W2 h + s*B2 A2 h, every expert on every token, mean over experts (a dense stand-in
        for routing). A1 [E, r, H], B1 [E, 2I, r], A2 [E, r, inter], B2 [E, H, r] -- the e4b orientation, which the peft
        layout is VIEWED into, so both tiny models compute the same function from the same slot tensors."""
        if self.mode == "axolotl":
            w1, w2 = self.gate_up_proj.float(), self.down_proj.float()     # the parametrization dequantises on access
        else:
            w1, w2 = self._deq(self.gate_up_proj), self._deq(self.down_proj)
        h = torch.einsum("bth,eih->beti", x, w1)
        h = h + scaling * torch.einsum("betr,eir->beti", torch.einsum("bth,erh->betr", x.to(A1.dtype), A1), B1).to(h.dtype)
        gate, up = h.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        y = torch.einsum("beti,ehi->beth", act, w2)
        y = y + scaling * torch.einsum("betr,ehr->beth", torch.einsum("beti,eri->betr", act.to(A2.dtype), A2), B2).to(y.dtype)
        return y.mean(1)


class _TinyExpertsE4b(nn.Module):
    """e4b's ExpertsLoRA shape: `base` (the frozen stack), gate_up_lora_A [E, r, H] / gate_up_lora_B [E, 2I, r] /
    down_lora_A [E, r, inter] / down_lora_B [E, H, r]; A ~ N(0, 1/r) (lora.py ExpertsLoRA.__init__), B = 0, in the loader's
    adapter dtype (bf16 on the streaming loader: the 'shipped' configuration). When `patched` (the fused arm) the forward
    goes through the mocked kernel entry point, which the counter wraps."""
    def __init__(self, base, r, alpha, seed, dtype):
        super().__init__()
        self.base = base
        E, H, inter = base.E, base.H, base.inter
        self.scaling = alpha / r
        self.gate_up_lora_A = nn.Parameter(_seeded_randn((E, r, H), seed, 1.0 / r).to(dtype))
        self.gate_up_lora_B = nn.Parameter(torch.zeros(E, 2 * inter, r, dtype=dtype))
        self.down_lora_A = nn.Parameter(_seeded_randn((E, r, inter), seed + 1, 1.0 / r).to(dtype))
        self.down_lora_B = nn.Parameter(torch.zeros(E, H, r, dtype=dtype))
        self.patched = False

    def _reference(self, x):
        return self.base(x, self.gate_up_lora_A, self.gate_up_lora_B, self.down_lora_A, self.down_lora_B, self.scaling)

    def forward(self, x):
        if self.patched:
            import nf4_qlora
            return nf4_qlora.fused_grouped_lora(self, x)
        return self._reference(x)


class _ParamWrapper(nn.Module):
    """PEFT 0.21.2's lora.ParamWrapper shape (peft/tuners/lora/layer.py): `base_layer`, `parameter_name`, `num_experts`,
    lora_A / lora_B ModuleDicts holding nn.Linear(in, r*E) / nn.Linear(r*E, out) -- ONE A/B per wrapped 3-D parameter;
    expert e = rows e*r:(e+1)*r of A viewed [E, r, in], and B viewed [out, r, E].permute(2, 0, 1) (get_delta_factors);
    A ~ kaiming_uniform(a=sqrt(5)) on the [r*E, in] weight = U(-1/sqrt(in), 1/sqrt(in)); B = 0. Two wrappers nest on
    one experts module, the OUTER wrapping the parameter targeted LAST (down_proj when target_parameters lists gate_up
    then down, as get_peft_model does), and the module path carries no parameter name: the mapping reads
    `parameter_name`. `flavour` unsloth routes the forward through the mocked bnb4bit backend (the counter)."""
    def __init__(self, base_layer, parameter_name, E, inn, out, r, alpha, seed, dtype, flavour):
        super().__init__()
        self.base_layer, self.parameter_name, self.num_experts, self.flavour = base_layer, parameter_name, E, flavour
        self.in_features, self.out_features, self.scaling = inn, out, alpha / r
        A, B = nn.Linear(inn, r * E, bias=False), nn.Linear(r * E, out, bias=False)
        with torch.no_grad():
            A.weight.copy_(_seeded_uniform((r * E, inn), seed, 1.0 / math.sqrt(inn)))
            B.weight.zero_()
        self.lora_A, self.lora_B = nn.ModuleDict({"default": A.to(dtype)}), nn.ModuleDict({"default": B.to(dtype)})

    def factors(self):
        A, B = self.lora_A["default"].weight, self.lora_B["default"].weight
        E = self.num_experts
        return A.reshape(E, -1, A.shape[-1]), B.reshape(B.shape[0], -1, E).permute(2, 0, 1)

    def _reference(self, x):
        wr, m = {}, self
        while isinstance(m, _ParamWrapper):
            wr[m.parameter_name] = m
            m = m.base_layer
        A1, B1 = wr["gate_up_proj"].factors()
        A2, B2 = wr["down_proj"].factors()
        return m(x, A1, B1, A2, B2, self.scaling)

    def forward(self, x):
        if self.flavour == "unsloth":
            from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as M
            return M.forward_moe_backend_bnb4bit(self, x)
        return self._reference(x)


class _Layer(nn.Module):
    def __init__(self, i, H, E, inter, r, alpha, mode, extra_lora=False):
        super().__init__()
        self.self_attn = nn.Module()
        for j, p in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
            base = nn.Linear(H, H, bias=False)
            with torch.no_grad():
                base.weight.copy_(_seeded_randn((H, H), _BASE_SEED + 10 * i + j, 1.0 / math.sqrt(H)))
            seed = 5000 + 10 * i + j
            # e4b: add_attention_lora(..., torch.float32) -> fp32 attention adapters; PEFT casts the adapter to the base's dtype (bf16)
            setattr(self.self_attn, p, _LoRALinear(base, r, alpha, seed, torch.float32) if mode == "e4b" else _PeftLinear(base, r, alpha, seed, torch.bfloat16))
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(H, E, bias=False)
        with torch.no_grad():
            self.mlp.gate.weight.copy_(_seeded_randn((E, H), _BASE_SEED + 100 + i))
        self.mlp.gate.weight.requires_grad_(False)
        if mode == "e4b":
            stack = _TinyStack(E, H, inter, "e4b", lambda t: nn.Parameter(t, requires_grad=False))
            self.mlp.experts = _TinyExpertsE4b(stack, r, alpha, 6000 + i, torch.bfloat16)      # the streaming loader's bf16 expert adapters
        else:
            stack = _TinyStack(E, H, inter, mode, Params4bit)
            inner = _ParamWrapper(stack, "gate_up_proj", E, H, 2 * inter, r, alpha, 7000 + i, torch.bfloat16, "unsloth" if mode == "unsloth" else "hf")
            self.mlp.experts = _ParamWrapper(inner, "down_proj", E, inter, H, r, alpha, 7100 + i, torch.bfloat16, "unsloth" if mode == "unsloth" else "hf")
            if extra_lora:                    # a LoRA on a module no slot census expects: the mapping must report the arm incomplete
                dense = nn.Linear(H, H, bias=False)
                dense.weight.requires_grad_(False)
                self.mlp.up_proj = _PeftLinear(dense, r, alpha, 7200 + i, torch.bfloat16)

    def forward(self, x):
        a = self.self_attn
        x = x + a.o_proj(a.v_proj(x) * torch.sigmoid(a.q_proj(x) * a.k_proj(x)).mean(-1, keepdim=True))
        if hasattr(self.mlp, "up_proj"):
            x = x + 0.01 * self.mlp.up_proj(x)
        return x + self.mlp.experts(x)


class _TinyLM(nn.Module):
    def __init__(self, mode, V=64, H=16, E=4, inter=6, L=2, r=2, alpha=4, extra_lora=False):   # 2*inter != H: the stacks are not square, so STRUCTURE decides the kind
        super().__init__()
        self.config = types.SimpleNamespace(model_type=f"tiny_{mode}", num_hidden_layers=L, num_experts=E, hidden_size=H, use_cache=False)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(V, H)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(_seeded_randn((V, H), _BASE_SEED + 500))
        self.model.embed_tokens.weight.requires_grad_(False)
        self.model.layers = nn.ModuleList([_Layer(i, H, E, inter, r, alpha, mode, extra_lora) for i in range(L)])
        self.lm_head = nn.Linear(H, V, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(_seeded_randn((V, H), _BASE_SEED + 600, 1.0 / math.sqrt(H)))
        self.lm_head.weight.requires_grad_(False)

    def gradient_checkpointing_enable(self, **k):
        pass

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = self.model.embed_tokens(input_ids)
        for l in self.model.layers:
            x = l(x)
        logits = self.lm_head(x)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1))
        return types.SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if "lora" in k}, os.path.join(d, "adapter_model.safetensors"))
        open(os.path.join(d, "adapter_config.json"), "w").write("{}")


def _install_fake_modules():
    nf4 = types.ModuleType("nf4_qlora")
    nf4.fused_grouped_lora = lambda mod, x: mod._reference(x)
    sys.modules["nf4_qlora"] = nf4
    zoo = types.ModuleType("unsloth_zoo")
    tp = types.ModuleType("unsloth_zoo.temporary_patches")
    mu = types.ModuleType("unsloth_zoo.temporary_patches.moe_utils")       # P2-1: the three backends behind select_moe_backend()
    mu.forward_native_grouped_mm = lambda mod, x: mod._reference(x)
    mu.forward_triton_grouped_gemm = lambda mod, x: mod._reference(x)
    mu.forward_native_moe_loop = lambda mod, x: mod._reference(x)
    mu.select_moe_backend = lambda: os.environ.get("UNSLOTH_MOE_BACKEND") or "grouped_mm"
    mb = types.ModuleType("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")

    def _bnb4bit(mod, x):                 # the real entry point dispatches on select_moe_backend() (moe_utils_bnb4bit.py:354-414); looked up at call time
        fn = {"grouped_mm": "forward_native_grouped_mm", "unsloth_triton": "forward_triton_grouped_gemm", "native_torch": "forward_native_moe_loop"}[mu.select_moe_backend()]
        return getattr(mu, fn)(mod, x)
    mb.forward_moe_backend_bnb4bit = _bnb4bit
    mb._moe_uses_bnb4bit_expert_weights = lambda m: type(getattr(m, "gate_up_proj", None)).__name__ == "Params4bit"
    sys.modules["unsloth_zoo"], sys.modules["unsloth_zoo.temporary_patches"] = zoo, tp
    sys.modules["unsloth_zoo.temporary_patches.moe_utils"], sys.modules["unsloth_zoo.temporary_patches.moe_utils_bnb4bit"] = mu, mb
    zoo.temporary_patches = tp
    tp.moe_utils, tp.moe_utils_bnb4bit = mu, mb


def _selftest_load_e4b(a):
    with PH("load_weights"):                        # #548: the loaders' own phase names, exercised on CPU
        m = _TinyLM("e4b")
        if a.model == "selftest/bnonzero":          # T20: a B that is not zero after construction -> the arm must refuse loudly
            with torch.no_grad():
                m.model.layers[0].mlp.experts.down_lora_B.fill_(0.5)
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "n_layers": 2, "model_type": "tiny_e4b",
         "verify": {"n_quantized": 2, "n_unquantized": 0}, "ckpt_mode": "hf:use_reentrant=False", "hashes": hashes_e4b,
         "fwd_kwargs": lambda t: {}, "tokenizer_obj": _FakeTok(), "snapshot_dir": None, "attn4_probe": attn4_bias_probe(m),
         "structural_expected_n_attn4": None, "detector_version": None}
    with PH("attn4"):
        if a.attn_4bit:
            x["n_attn4"] = 4 * 2
            x["structural_expected_n_attn4"] = 4 * 2     # T10: simulated census here; the REAL detector is dry-run tested below
            x["detector_version"] = "selftest"
    if a.arm == "fused":
        for l in m.model.layers:
            l.mlp.experts.patched = True
        x["n_patched"], x["reason"] = 2, "[selftest] fused on 2 module(s)"
    elif a.arm == "attn_only":
        for n, p in m.named_parameters():
            if "lora" in n and "experts" in n:
                p.requires_grad_(False)
        x["probes"] = {"fused": {"n_patched": 0, "reason": "[selftest] 0 ExpertsLoRA"}, "batched": {"n_patched": 0, "reason": "[selftest] 0"}}
        refresh_stub(a, "fused_attn4", "fused", "refused", "enable_fast_train(dgrad=True) patched 0 modules on this box: [selftest]", {"probed_by": "attn_only", "n_patched": 0, "n_layers": 2})
    return m, x


def _selftest_load_hf(a):
    """T11: the HF arm's bookkeeping on the tiny PEFT-shaped model (Params4bit stacks under two nested ParamWrappers)."""
    with PH("load_weights"):
        m = _TinyLM("hf", extra_lora=(a.model == "selftest/extra"))
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": ["PEFT selftest: target_modules=8 target_parameters=4"],
         "probes": {}, "n_layers": 2, "model_type": "tiny_hf", "verify": {"n_quantized": None, "n_unquantized": None},
         "ckpt_mode": "hf:use_reentrant=False", "hashes": hashes_unsloth, "fwd_kwargs": lambda t: {"attention_mask": torch.ones_like(t)},
         "tokenizer_obj": _FakeTok(), "snapshot_dir": "/selftest/snapshots/deadbeef", "structural_expected_n_attn4": None, "detector_version": None,
         "loader_used": "AutoModelForCausalLM", "hf_targets": {"peft": "selftest", "n_target_modules": 8, "n_target_parameters": 4}}
    return m, x


def _selftest_load_unsloth(a):
    if a.model == "selftest/refuse":
        raise NotImplementedError("selftest: loader refuses this family")
    with PH("load_weights"):
        m = _TinyLM("unsloth")
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [f"Unsloth: Detected MoE model. {BANNER}: ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']"],
         "probes": {}, "n_layers": 2, "model_type": "tiny_unsloth", "verify": {"n_quantized": None, "n_unquantized": None},
         "ckpt_mode": "unsloth" if a.grad_ckpt == "unsloth" else "hf:True (via get_peft_model)", "hashes": hashes_unsloth,
         "fwd_kwargs": lambda t: {"attention_mask": torch.ones_like(t)},
         "tokenizer_obj": _FakeTok(), "snapshot_dir": "/selftest/snapshots/deadbeef",
         "structural_expected_n_attn4": None, "detector_version": None}
    return m, x


def _selftest_load_axolotl(a):
    """P2-2: the axolotl arm's bookkeeping on the tiny PEFT-shaped model over REAL bnb-parametrized stacks (what
    quantize_moe_experts produces), with the config dict and the census the real loader records."""
    with PH("load_weights"):
        m = _TinyLM("axolotl")
    mods = [f"model.layers.{i}.self_attn.{p}" for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    params = [f"model.layers.{i}.mlp.experts.{n}" for i in range(2) for n in ("gate_up_proj", "down_proj")]
    best = bool(int(getattr(a, "axolotl_best", 0) or 0))
    cfg_dict = axolotl_config_dict(a, "selftest/tiny", mods, params, best=best, double_quant=False, dataset_path="/selftest/ds_alpaca.json", output_dir="/selftest/out")
    census = axolotl_expert_census(m)
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [f"axolotl selftest: quantized_moe_experts_n={census['quantized_moe_experts_n']}"],
         "probes": {}, "n_layers": 2, "model_type": "tiny_axolotl", "verify": {"n_quantized": None, "n_unquantized": None},
         "ckpt_mode": "axolotl cfg gradient_checkpointing=True kwargs={'use_reentrant': False}", "hashes": hashes_unsloth,
         "fwd_kwargs": lambda t: {"attention_mask": torch.ones_like(t)}, "tokenizer_obj": _FakeTok(), "snapshot_dir": "/selftest/snapshots/deadbeef",
         "structural_expected_n_attn4": None, "detector_version": None, "loader_used": "axolotl ModelLoader.load() (selftest stand-in)",
         "hf_targets": {"peft": "selftest", "n_target_modules": 8, "n_target_parameters": 4},
         "axolotl": {"version": "selftest", "torch": torch.__version__, "config": cfg_dict, "best": best, "plugins": cfg_dict.get("plugins"),
                     "expert_backend": cfg_dict.get("expert_backend"), "quantized_moe_experts_n": census["quantized_moe_experts_n"], "census": census,
                     "cfg_experts_implementation": "scattermoe" if best else None, "config_experts_implementation": None},
         "axolotl_bnb4bit_modules": census}
    x["axolotl_targets"] = x["hf_targets"]
    return m, x


def _selftest_unsloth_backend_and_axolotl(a, d, rec, e_ref):
    """P2-1 / P2-2 end to end on the tiny models: the backend knobs reach the environment and the counters, the absent-name
    path records instead of crashing, the double-quant kwarg decision follows the signature, and the axolotl arm runs through
    run_arm with its census, its matched init over parametrized stacks and its probe."""
    out = {}
    a.fam, a.model, a.tokens, a.tokens_sha = "tiny", "selftest/tiny", os.path.join(d, "tokens_tiny.json"), rec["sha256"]
    a.framework, a.arm, a.attn_4bit, a.expect_trainable, a.lora_init, a.adapter_dtype = "unsloth", "unsloth", 0, None, "native", "fp32"
    env_keys = ("UNSLOTH_MOE_BACKEND", "UNSLOTH_MOE_RECOMPUTE", "UNSLOTH_MOE_GC_REPLAY_PIN")
    saved = {k: os.environ.get(k) for k in env_keys}
    try:
        for backend, tag, want_key in (("grouped_mm", "ckpt_unsloth_gmm", "unsloth_grouped_mm"), ("native_torch", "ckpt_unsloth_loop", "unsloth_loop"), ("unsloth_triton", "ckpt_unsloth_tri", "unsloth_triton")):
            for k in env_keys:
                os.environ.pop(k, None)                   # one arm per process in a real run; here they share one
            a.tag, a.unsloth_moe_backend, a.unsloth_speed_tilt = tag, backend, int(backend == "grouped_mm")
            r = run_arm(a, _selftest_load_unsloth, sampler=False)
            assert r["status"] == "ok" and r["unsloth_knobs"]["moe_backend_requested"] == backend and r["moe_backend_selected"] == backend, (tag, r["unsloth_knobs"], r["moe_backend_selected"])
            assert r["unsloth_knobs"]["env_set"]["UNSLOTH_MOE_BACKEND"] == backend and r["env"]["unsloth_env"].get("UNSLOTH_MOE_BACKEND") == backend, (tag, r["env"]["unsloth_env"])
            mins = r["unsloth_backend_calls_per_step_min"]
            assert mins[want_key] == 2 * a.accum and mins["moe_bnb4bit_backend"] == 2 * a.accum and sum(v for k, v in mins.items() if k not in (want_key, "moe_bnb4bit_backend")) == 0, (tag, mins)
            assert r["unsloth_backend_calls_per_step_max"][want_key] == 2 * a.accum and r["unsloth_backend_absent"] == [], (tag, r["unsloth_backend_absent"])
            if backend == "grouped_mm":
                assert r["unsloth_knobs"]["speed_tilt"] is True and r["unsloth_knobs"]["env_set"]["UNSLOTH_MOE_RECOMPUTE"] == "0" and r["env"]["unsloth_env"]["UNSLOTH_MOE_GC_REPLAY_PIN"] == "1", r["unsloth_knobs"]
            else:
                assert r["unsloth_knobs"]["speed_tilt"] is False and "UNSLOTH_MOE_RECOMPUTE" not in r["env"]["unsloth_env"], r["unsloth_knobs"]
            out[tag] = {k: v for k, v in mins.items()}
        # `default` leaves the environment alone: the loader's own choice is what runs and is recorded
        for k in env_keys:
            os.environ.pop(k, None)
        a.tag, a.unsloth_moe_backend, a.unsloth_speed_tilt = "ckpt_unsloth_default", "default", 0
        r = run_arm(a, _selftest_load_unsloth, sampler=False)
        assert r["unsloth_knobs"]["env_set"] == {} and r["moe_backend_selected"] == "grouped_mm" and r["unsloth_backend_calls_per_step_min"]["unsloth_grouped_mm"] == 2 * a.accum, r["unsloth_knobs"]
        # an absent backend name is recorded, never a crash
        mu = sys.modules["unsloth_zoo.temporary_patches.moe_utils"]
        tri = mu.forward_triton_grouped_gemm
        del mu.forward_triton_grouped_gemm
        try:
            a.tag = "ckpt_unsloth_absent"
            r = run_arm(a, _selftest_load_unsloth, sampler=False)
            assert r["status"] == "ok" and r["unsloth_backend_absent"] == ["unsloth_zoo.temporary_patches.moe_utils.forward_triton_grouped_gemm: AttributeError"], r["unsloth_backend_absent"]
            assert r["unsloth_backend_calls_per_step_min"]["unsloth_triton"] == 0
        finally:
            mu.forward_triton_grouped_gemm = tri
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    # the double-quant kwarg decision follows the signature READ AT RUNTIME
    def fp_named(model_name, max_seq_length, dtype, load_in_4bit, bnb_4bit_use_double_quant=True):
        pass

    def fp_qc(model_name, max_seq_length, dtype, load_in_4bit, quantization_config=None):
        pass

    def fp_neither(model_name, max_seq_length, dtype, load_in_4bit, **kwargs):
        pass
    kw, dq = unsloth_double_quant_kwargs(fp_named, True)
    assert kw == {"bnb_4bit_use_double_quant": False} and dq["requested"] is False and dq["how"].startswith("from_pretrained(bnb_4bit_use_double_quant=False)"), dq
    kw, dq = unsloth_double_quant_kwargs(fp_neither, True)
    assert kw == {} and dq["requested"] is None and dq["how"].startswith("unknown-default"), dq
    kw, dq = unsloth_double_quant_kwargs(fp_named, False)
    assert kw == {} and "not requested" in dq["how"], dq
    try:
        kw, dq = unsloth_double_quant_kwargs(fp_qc, True)
        assert "quantization_config" in kw and kw["quantization_config"].bnb_4bit_use_double_quant is False, dq
        out["dq_via_quantization_config"] = True
    except ImportError as e:                       # transformers absent here: the branch cannot be exercised, said so
        out["dq_via_quantization_config"] = f"skipped: {e}"
    # loaded_expert_double_quant reads the expert Params4bit's quant_state.nested (real bnb, CPU) or None
    import bitsandbytes as bnb
    probe = nn.Module()
    probe.layers = nn.ModuleList([nn.Module()])
    probe.layers[0].mlp = nn.Module()
    probe.layers[0].mlp.experts = nn.Module()
    for nested in (False, True):
        w = bnb.nn.Params4bit(torch.randn(4, 24, 64, dtype=torch.bfloat16), requires_grad=False, quant_type="nf4", compress_statistics=nested, blocksize=64).to("cpu")
        if getattr(w, "quant_state", None) is None:
            out["loaded_dq"] = "skipped: Params4bit.to('cpu') did not quantise here"
            break
        probe.layers[0].mlp.experts.gate_up_proj = w
        assert loaded_expert_double_quant(probe) is nested, (nested, w.quant_state.nested)
        out["loaded_dq"] = "ok"
    assert loaded_expert_double_quant(_TinyLM("hf")) is None            # no quant_state on the fake Params4bit
    # ---- the axolotl arm end to end (real bnb parametrized stacks on CPU)
    try:
        _TinyLM("axolotl")
    except Exception as e:
        out["axolotl"] = {"skipped": f"bnb replace_parameter_4bit unavailable on CPU here: {type(e).__name__}: {str(e)[:120]}"}
        return out
    a.framework, a.arm, a.attn_4bit, a.expect_trainable = "axolotl", "axolotl", 0, None
    a.lora_init, a.adapter_dtype, a.axolotl_best = "matched:3407", "fp32", 0
    a.tag = "ckpt_axolotl_m"
    ax = run_arm(a, _selftest_load_axolotl, sampler=False)
    assert ax["status"] == "ok" and ax["kernel_counter_key"] == "experts_forward" and ax["kernel_calls_per_step_min"] == 2 * a.accum, (ax["status"], ax.get("reason"))
    c = ax["axolotl"]["census"]
    assert c["quantized_moe_experts_n"] == 4 and c["n_bnb4bit_unwrapped"] == 2 and c["n_experts_modules"] == 2 and c["parametrized_params"] == 4, c
    assert ax["axolotl_bnb4bit_modules"]["n_bnb4bit_unwrapped"] == 2 and ax["axolotl_targets"]["n_target_parameters"] == 4
    cfgd = ax["axolotl"]["config"]
    assert cfgd["quantize_moe_experts"] is True and cfgd["load_in_4bit"] is True and cfgd["adapter"] == "qlora" and cfgd["bnb_4bit_use_double_quant"] is False
    assert len(cfgd["lora_target_parameters"]) == 4 and len(cfgd["lora_target_modules"]) == 8 and cfgd["lora_dropout"] == 0.0 and cfgd["datasets"] and "plugins" not in cfgd
    assert cfgd["gradient_checkpointing"] is True and cfgd["gradient_checkpointing_kwargs"] == {"use_reentrant": False} and cfgd["optimizer"] == "adamw_bnb_8bit" and cfgd["lr_scheduler"] == "linear"
    assert cfgd["max_steps"] == a.steps and cfgd["gradient_accumulation_steps"] == a.accum and cfgd["lora_r"] == a.r and cfgd["seed"] == a.seed
    mi = ax["matched_init"]
    assert mi["complete"] is True and mi["n_slots_set"] == mi["n_slots_expected"] == 24 and all("structure" in v for k, v in mi["mapping_rules"].items() if "experts" in k), mi
    assert list(ax["adapter_dtypes_after"]) == ["torch.float32"]
    fp = ax["frozen_base_probe"]
    assert fp["slots"]["gate_up"]["regime"] == "nf4/64" and fp["slots"]["down"]["regime"] == "nf4/64" and fp["control_detects_flip"] is True and not fp["errors"], fp
    assert "expert-0 blocks" in fp["slots"]["gate_up"]["method"], fp["slots"]["gate_up"]["method"]
    assert ax["C1_bit_exact"] and ax["C1_bytes_hashed"] > 0 and ax["C1_tensors_hashed"] == 4, (ax["C1_bytes_hashed"], ax["C1_tensors_hashed"])   # the 4 packed originals
    assert ax["census"]["Params4bit_expert_stacks"] == 0, "a parametrized stack is a plain uint8 Parameter, not a Params4bit"
    a.tag, a.axolotl_best, a.lora_init, a.adapter_dtype = "ckpt_axolotl_best", 1, "native", "fp32"
    axb = run_arm(a, _selftest_load_axolotl, sampler=False)
    assert axb["status"] == "ok" and axb["axolotl"]["config"]["plugins"] == ["axolotl.integrations.kernels.KernelsPlugin"] and axb["axolotl"]["config"]["expert_backend"] == "scattermoe" and axb["axolotl"]["config"]["moe_bnb_fast"] is True
    assert axb["matched_init"] is None and axb["axolotl"]["best"] is True
    a.axolotl_best = 0
    out["axolotl"] = {"quantized_moe_experts_n": c["quantized_moe_experts_n"], "n_bnb4bit_unwrapped": c["n_bnb4bit_unwrapped"], "slots": mi["n_slots_set"], "probe": {k: v["regime"] for k, v in fp["slots"].items()},
                      "eval_loss_step0": ax["eval_loss_step0"]}
    return out


def _selftest_frozen_probe_real():
    """T21 on GENUINE storage, CPU: a real Experts4bit (e4b's per-expert NF4/64 stacks, bitsandbytes CPU quantisation)
    under a real ExpertsLoRA, the same bf16 stack quantised WHOLE as one bnb Params4bit (what Unsloth's loader holds,
    the HF arm's census says bf16 instead), and a real Linear4bit q_proj via quantize_attention_projections_4bit
    (double-quant ON by bnb's Params4bit default -- recorded as `+dq`). Asserts: every control detects its flip; e4b's
    expert-0 dequant bytes == the whole-stack slice (SAME-BYTES is attainable, the P10 reading is not vacuous); the
    attention slot's regime carries +dq. Returns the regimes and the comparison, or {"skipped": why} when the CPU 4-bit
    path is unavailable here (reported in the SELFTEST OK line, never silent)."""
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as BF
        try:
            from experts4bit_qlora._vendor.experts import Experts4bit
            from experts4bit_qlora.lora import ExpertsLoRA, add_attention_lora, quantize_attention_projections_4bit
        except ImportError:                      # the package is not installed in this interpreter: the repo checkout two levels up is it
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            from experts4bit_qlora._vendor.experts import Experts4bit
            from experts4bit_qlora.lora import ExpertsLoRA, add_attention_lora, quantize_attention_projections_4bit
    except Exception as e:
        return {"skipped": f"{type(e).__name__}: {str(e)[:120]}"}
    E, H, inter = 4, 64, 128
    g = torch.Generator().manual_seed(11)
    gu = torch.randn(E, 2 * inter, H, generator=g, dtype=torch.bfloat16)
    dn = torch.randn(E, H, inter, generator=g, dtype=torch.bfloat16)
    try:
        ex = Experts4bit.from_float(gu, dn, blocksize=64, quant_type="nf4")
    except Exception as e:
        return {"skipped": f"bnb CPU 4-bit quantisation unavailable: {type(e).__name__}: {str(e)[:120]}"}
    # the e4b side: ExpertsLoRA over Experts4bit, cast to fp32 adapters, one forward+backward on CPU (the reference path)
    lo = ExpertsLoRA(ex, r=2, alpha=4, dtype=torch.bfloat16)
    for p in lo.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    lo.train()
    xx = torch.randn(5, H, dtype=torch.bfloat16)
    y = lo(xx, torch.tensor([[0, 1]] * 5), torch.full((5, 2), 0.5, dtype=torch.bfloat16))
    y.float().sum().backward()
    assert lo.gate_up_lora_A.grad is not None and lo.gate_up_lora_A.dtype == torch.float32
    e4b_m = nn.Module()
    e4b_m.layers = nn.ModuleList([nn.Module()])
    e4b_m.layers[0].self_attn = nn.Module()
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(e4b_m.layers[0].self_attn, p, nn.Linear(H, H, bias=False).to(torch.bfloat16))
    e4b_m.layers[0].mlp = nn.Module()
    e4b_m.layers[0].mlp.experts = lo
    e4b_m.config = types.SimpleNamespace(hidden_size=H, num_hidden_layers=1, num_experts=E)
    assert quantize_attention_projections_4bit(e4b_m) == 4 and add_attention_lora(e4b_m, 2, 4, torch.float32) == 4
    pe = frozen_base_probe(e4b_m, "e4b")
    # the bnb side: the SAME bf16 stacks as whole-tensor Params4bit under a PEFT-shaped wrapper chain, HF-style bf16 q_proj
    class _Stack(nn.Module):
        pass
    st = _Stack()
    st.gate_up_proj = bnb.nn.Params4bit(gu.contiguous(), requires_grad=False, quant_type="nf4", compress_statistics=False, blocksize=64)
    st.down_proj = bnb.nn.Params4bit(dn.contiguous(), requires_grad=False, quant_type="nf4", compress_statistics=False, blocksize=64)
    st.gate_up_proj = st.gate_up_proj.to("cpu")    # Params4bit quantises on transfer (bnb 0.50: CPU supported)
    st.down_proj = st.down_proj.to("cpu")
    if getattr(st.gate_up_proj, "quant_state", None) is None:
        q1, s1 = BF.quantize_4bit(gu.contiguous(), blocksize=64, compress_statistics=False, quant_type="nf4")
        q2, s2 = BF.quantize_4bit(dn.contiguous(), blocksize=64, compress_statistics=False, quant_type="nf4")
        st.gate_up_proj = bnb.nn.Params4bit(q1, requires_grad=False, quant_state=s1, quant_type="nf4", blocksize=64, bnb_quantized=True)
        st.down_proj = bnb.nn.Params4bit(q2, requires_grad=False, quant_state=s2, quant_type="nf4", blocksize=64, bnb_quantized=True)
    hf_m = nn.Module()
    hf_m.layers = nn.ModuleList([nn.Module()])
    hf_m.layers[0].self_attn = nn.Module()
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(hf_m.layers[0].self_attn, p, nn.Linear(H, H, bias=False).to(torch.bfloat16))
    hf_m.layers[0].mlp = nn.Module()
    w1 = _ParamWrapper(st, "gate_up_proj", E, H, 2 * inter, 2, 4, 1, torch.bfloat16, "hf")
    hf_m.layers[0].mlp.experts = _ParamWrapper(w1, "down_proj", E, inter, H, 2, 4, 2, torch.bfloat16, "hf")
    hf_m.config = types.SimpleNamespace(hidden_size=H, num_hidden_layers=1, num_experts=E)
    ph = frozen_base_probe(hf_m, "hf")
    for tag, pr in (("e4b", pe), ("hf", ph)):
        assert not pr["errors"], (tag, pr["errors"])
        assert set(pr["slots"]) == {"gate_up", "down", "q_proj"}, (tag, sorted(pr["slots"]))
        assert pr["control_detects_flip"] is True, (tag, {k: v["control_detects_flip"] for k, v in pr["slots"].items()})
    assert pe["slots"]["gate_up"]["regime"] == "nf4/64" and ph["slots"]["gate_up"]["regime"] == "nf4/64", (pe["slots"]["gate_up"]["regime"], ph["slots"]["gate_up"]["regime"])
    same = {k: pe["slots"][k]["sha"] == ph["slots"][k]["sha"] for k in ("gate_up", "down")}
    assert all(same.values()), ("e4b per-expert dequant != bnb whole-stack slice on the same bf16 bytes", same)
    assert pe["slots"]["gate_up"]["shape"] == [2 * inter, H] and pe["slots"]["down"]["shape"] == [H, inter]
    assert pe["slots"]["q_proj"]["regime"].startswith("nf4/64"), pe["slots"]["q_proj"]["regime"]
    assert ph["slots"]["q_proj"]["regime"] == "bf16", ph["slots"]["q_proj"]["regime"]
    # the matched init on the REAL ExpertsLoRA + LoRALinear shapes maps every slot
    mi = apply_matched_init(e4b_m, "matched:3407", 1, e4b_m.config)
    assert mi["complete"] and mi["n_slots_set"] == mi["n_slots_expected"] == 4 + 2 * E, mi
    A0 = matched_lora_A(3407, 0, "gate_up", 0, 2, H)
    assert torch.equal(lo.gate_up_lora_A.data[0], A0), "the e4b expert slot does not hold its registered tensor"
    return {"e4b_regimes": {k: v["regime"] for k, v in pe["slots"].items()}, "hf_regimes": {k: v["regime"] for k, v in ph["slots"].items()},
            "expert_slots_same_bytes": same, "attention_dq_on_e4b": pe["slots"]["q_proj"]["regime"].endswith("+dq"), "matched_init_real_e4b": mi["n_slots_set"]}


def _selftest_detector(d, a):
    """T10 (#434) dry-runs: the REAL structural detector against synthetic stacks, driven through the same
    attn4_census_check the e4b arm uses. CPU-only; the library's package __init__ needs bitsandbytes but lora.py
    itself does not, so on a box without the installed package lora.py is loaded by path from the repo checkout."""
    try:
        from experts4bit_qlora.lora import detect_attention_projections as det
    except Exception:
        import importlib.util
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experts4bit_qlora", "lora.py"))
        spec = importlib.util.spec_from_file_location("e4b_lora_selftest", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod   # dataclass() resolves cls.__module__ through sys.modules
        spec.loader.exec_module(mod)
        det = mod.detect_attention_projections

    def stack(L, keqv=(), drop_k=()):
        m = nn.Module()
        blocks = []
        for i in range(L):
            b = nn.Module()
            b.q_proj = nn.Linear(8, 8, bias=False)
            if i not in drop_k:
                b.k_proj = nn.Linear(8, 8, bias=False)
            b.v_proj = None if i in keqv else nn.Linear(8, 8, bias=False)   # Gemma-4 k_eq_v: transformers sets v_proj = None
            b.o_proj = nn.Linear(8, 8, bias=False)
            blocks.append(b)
        m.layers = nn.ModuleList(blocks)
        return m

    def sim_quantize(model, skip=0):
        c = det(model, exact_linear=True)
        for j, (mod, name) in enumerate(c.candidates):
            if j >= skip:
                setattr(mod, name, nn.Module())   # stands in for Linear4bit: afterwards not exactly nn.Linear
        return len(c.candidates) - skip

    def arm_x(fam, tag):
        a.fam, a.framework, a.arm, a.tag = fam, "e4b", "reference", tag
        return {"n_attn4": 0, "n_layers": None, "model_type": "tiny_det", "attn4_probe": None,
                "structural_expected_n_attn4": None, "detector_version": "selftest"}

    # 1. tiny_keqv_30: 30 layers, 5 K-as-V -> census 115; converting exactly the candidates passes;
    #    a simulated 114-conversion fires void_attn4 naming 115.
    x = arm_x("tcdet", "keqv30")
    attn4_census_check(a, stack(30, keqv={0, 6, 12, 18, 24}), x, det, sim_quantize)
    assert x["structural_expected_n_attn4"] == 115 and x["n_attn4"] == 115, x
    x = arm_x("tcdet", "keqv30_drift")
    try:
        attn4_census_check(a, stack(30, keqv={0, 6, 12, 18, 24}), x, det, lambda mm: sim_quantize(mm, skip=1))
        raise AssertionError("a 114-conversion did not fire void_attn4")
    except SystemExit as e:
        assert e.code == 3, e.code
    r = json.load(open(os.path.join(d, "tcdet_e4b_keqv30_drift.json")))
    assert r["status"] == "void_attn4" and r["structural_expected_n_attn4"] == 115 and "115" in r["reason"], r

    # 2. tiny_plain_4: an ordinary 4-projection stack -> expected 4 x L; passes (the four-projection families untouched).
    x = arm_x("tcdet", "plain4")
    attn4_census_check(a, stack(4), x, det, sim_quantize)
    assert x["structural_expected_n_attn4"] == 16 == 4 * 4 and x["n_attn4"] == 16, x

    # 3. tiny_missing_k: q/o present but k_proj missing -> a REFUSED row (code 3, the layer named), never void_attn4, never a guess.
    x = arm_x("tcdet", "missing_k")
    try:
        attn4_census_check(a, stack(3, drop_k={1}), x, det, sim_quantize)
        raise AssertionError("a missing k_proj did not refuse")
    except SystemExit as e:
        assert e.code == 3, e.code
    r = json.load(open(os.path.join(d, "tcdet_e4b_missing_k.json")))
    assert r["status"] == "refused" and r["phase"] == "attn4" and r["attn4_refusing_modules"] == ["layers.1"], r
    assert "structural_expected_n_attn4" not in r, r   # the census never completed -- nothing guessed
    return {"tiny_keqv30": 115, "tiny_plain4": 16, "tiny_missing_k": r["attn4_refusing_modules"]}


def _moe_stub(L=3, E=4, layout="fused_experts", H=8, declare_experts=True, drop_last_stack=False):
    """#542 selftest fixture: a decoder of L layers whose MoE block stores its experts in ONE of the layouts the
    families in this lane actually use. Only the NAMES and the storage shape differ between layouts."""
    blocks = []
    for i in range(L):
        b = nn.Module()
        b.self_attn = nn.Module()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(b.self_attn, p, nn.Linear(H, H, bias=False))
        blk = nn.Module()
        blk.gate = nn.Linear(H, E, bias=False)
        last = drop_last_stack and i == L - 1
        if layout == "fused_experts":            # transformers v5 fuses these under a module literally named `experts`
            blk.experts = nn.Module()
            blk.experts.gate_up_proj = nn.Parameter(torch.zeros(E, H, 2 * H))
            if not last:
                blk.experts.down_proj = nn.Parameter(torch.zeros(E, 2 * H, H))
        elif layout == "granite_on_disk":        # GraniteMoe's OWN names: not one `experts` component in the path
            blk.input_linear = nn.Module()
            blk.input_linear.weight = nn.Parameter(torch.zeros(E, 2 * H, H))
            blk.output_linear = nn.Module()
            blk.output_linear.weight = nn.Parameter(torch.zeros(E, H, H))
        elif layout == "per_expert_2d":          # OLMoE / Mixtral as stored: one 2-D Linear triple per expert
            blk.experts = nn.ModuleList()
            for _ in range(E):
                e = nn.Module()
                e.gate_proj = nn.Linear(H, H, bias=False)
                e.up_proj = nn.Linear(H, H, bias=False)
                e.down_proj = nn.Linear(H, H, bias=False)
                blk.experts.append(e)
        elif layout == "dense":                  # no MoE at all
            blk.up_proj = nn.Linear(H, 2 * H, bias=False)
            blk.down_proj = nn.Linear(2 * H, H, bias=False)
        elif layout == "ragged":                 # two 3-D floating stacks that disagree on their leading dimension
            blk.experts = nn.Module()
            blk.experts.gate_up_proj = nn.Parameter(torch.zeros(E, H, 2 * H))
            blk.experts.down_proj = nn.Parameter(torch.zeros(E + 1, 2 * H, H))
        b.mlp = blk
        blocks.append(b)
    inner = nn.Module()
    inner.layers = nn.ModuleList(blocks)
    m = nn.Module()
    m.model = inner
    m.config = types.SimpleNamespace(num_hidden_layers=L, model_type=f"tiny_{layout}")
    if declare_experts == "mismatch":             # a config whose declared count matches no stack on the loaded model
        m.config.num_experts = 99
    elif declare_experts:
        m.config.num_experts = E
    return m


def _selftest_hf_expert_selection():
    """#542 dry-runs, the shape _selftest_detector has for attention: the HF arm's expert selection is STRUCTURAL, and
    an empty or implausible selection REFUSES instead of degrading to an attention-only run.

    The load-bearing case is `granite_on_disk`: GraniteMoe's own names carry no `experts` component, so the pre-#542
    substring (and `EXPERT_PARAM_RE`, which requires a literal `experts.`) would both have taken ZERO -- recorded here
    as n_by_name_substring / n_by_expert_param_re beside a full structural selection."""
    L, E = 3, 4
    out = {}
    fused = _moe_stub(L, E, "fused_experts")
    mods, params, diag = hf_targets(fused, L, "tiny_fused_experts")
    assert len(mods) == 4 * L and len(params) == 2 * L and diag["stacks_per_layer"] == 2, (len(mods), diag)
    assert diag["n_by_name_substring"] == 2 * L, diag       # the old rule and the new one agree on this layout
    out["fused"] = {"n": len(params), "substring": diag["n_by_name_substring"]}

    gran = _moe_stub(L, E, "granite_on_disk")
    _, params, diag = hf_targets(gran, L, "tiny_granite_on_disk")
    assert len(params) == 2 * L and diag["experts_per_stack"] == E, (params, diag)
    assert diag["n_by_name_substring"] == 0 and diag["n_by_expert_param_re"] == 0, diag   # both name rules take nothing
    assert all(n.endswith(("input_linear.weight", "output_linear.weight")) for n in params), params
    out["granite_on_disk"] = {"n": len(params), "substring": diag["n_by_name_substring"], "param_re": diag["n_by_expert_param_re"]}

    refusals = {}
    for tag, model, needle in (
            ("per_expert_2d", _moe_stub(L, E, "per_expert_2d"), "no expert stack found"),
            ("dense", _moe_stub(L, E, "dense"), "no expert stack found"),
            ("ragged_no_config", _moe_stub(L, E, "ragged", declare_experts=False), "disagree on their leading dimension"),
            ("ragged_declared_matches_nothing", _moe_stub(L, E, "ragged", declare_experts="mismatch"), "declares 99 experts"),
            ("not_per_layer", _moe_stub(L, E, "fused_experts", drop_last_stack=True), "is not a whole number per layer")):
        try:
            hf_targets(model, L, f"tiny_{tag}")
            raise AssertionError(f"#542: {tag} did not refuse -- an empty/implausible selection ran attention-only")
        except NotImplementedError as e:
            msg = str(e)
            assert "(#542)" in msg and f"n_layers={L}" in msg, msg
            if needle:
                assert needle in msg, msg
            refusals[tag] = type(e).__name__
    # a config that DECLARES its expert count disambiguates a model carrying more than one 3-D leading dimension
    _, params, diag = hf_targets(_moe_stub(L, E, "ragged"), L, "tiny_ragged")
    assert len(params) == L and diag["experts_per_stack"] == E and diag["leading_dims"] == {"4": L, "5": L}, diag
    # the per-expert refusal NAMES the layout it found instead of a bare "empty"
    try:
        hf_targets(_moe_stub(L, E, "per_expert_2d"), L, "tiny_per_expert_2d")
    except NotImplementedError as e:
        assert f'"per_expert_2d_groups": {L}' in str(e), str(e)
    # the classifier run_arm uses turns every one of these into a `refused` row (code 3), never a silent arm
    assert classify_load_exception(NotImplementedError("x")) == ("refused", 3)
    out["refusals"] = sorted(refusals)
    return out


def selftest(a):
    global DEV
    DEV = "cpu"
    a.prereg = PREREG   # explicit: selftest receipts cite this lane's document; a real run must pass --prereg (no default)
    _install_fake_modules()
    import tempfile
    d = tempfile.mkdtemp(prefix="tc1_selftest_")
    os.makedirs(os.path.join(d, "data"))
    rows = [{"instruction": f"Q{i}: describe patient {i}", "output": f"Patient {i} is stable; plan {i % 5} continues. " * 2} for i in range(40)]
    ds = {"train": rows[:32], "eval": rows[32:]}
    dp = os.path.join(d, "data", "ds_selftest.json")
    json.dump(ds, open(dp, "w"))
    a.fam, a.model, a.revision, a.data, a.data_sha = "tiny", "selftest/tiny", "0" * 40, dp, sha_bytes(open(dp, "rb").read())
    a.seq, a.steps, a.eval_every, a.eval_n, a.accum, a.autocast, a.lr, a.r, a.alpha = 64, 12, 4, 6, a.accum, a.autocast, 1e-3, 2, 4
    a.micro_batch, a.optim, a.weight_decay, a.lr_schedule, a.warmup_steps, a.template = 1, "adamw_torch", 0.01, "constant", 0, "clinical"
    a.log_every, a.microbatch_timing = 1, 1                          # T17: every step printed, every micro-batch timed
    a.adapter_dtype, a.lora_init = "fp32", "native"                  # T19/T20: the defaults; the TC1 block below varies them
    a.out, a.adapter_dir, a.tokens = d, os.path.join(d, "adapters"), os.path.join(d, "tokens_tiny.json")
    rec = prepare(a, tok=_FakeTok())
    a.tokens_sha = rec["sha256"]
    got = {}
    for fw, arm, tag, kw in (("e4b", "reference", "reference_attn4", {"attn_4bit": 1}), ("e4b", "fused", "fused_attn4", {"attn_4bit": 1}),
                             ("unsloth", "unsloth", "ckpt_unsloth", {"attn_4bit": 0}), ("unsloth", "unsloth", "ckpt_hf", {"attn_4bit": 0, "grad_ckpt": "hf"})):
        a.framework, a.arm, a.tag, a.attn_4bit, a.grad_ckpt = fw, arm, tag, kw["attn_4bit"], kw.get("grad_ckpt", "unsloth")
        a.expect_trainable = got.get("e4b_reference_attn4", {}).get("trainable_params") if fw == "unsloth" else None
        got[f"{fw}_{tag}"] = run_arm(a, _selftest_load_e4b if fw == "e4b" else _selftest_load_unsloth, sampler=False)
    # gpt-oss shape (its own family, its own tokens file): attn_only over bare experts (expert LoRA frozen), with the
    # fused_attn4 refusal stub written first by "the run script" (the tp1 citation), refreshed by the arm's own probe
    a.fam, a.tokens = "tinygo", os.path.join(d, "tokens_tinygo.json")
    rec_go = prepare(a, tok=_FakeTok())
    a.tokens_sha = rec_go["sha256"]
    a.framework, a.arm, a.tag, a.attn_4bit, a.expect_trainable = "e4b", "attn_only", "attn_only", 0, None
    stub(a, "refused", "SKIPPED as REFUSED: tp1 row cited (selftest)", {"cited": "tp1"}, fw="e4b", tag="fused_attn4", arm="fused")
    got["e4b_attn_only"] = run_arm(a, _selftest_load_e4b, sampler=False)
    a.fam, a.tokens, a.tokens_sha = "tiny", os.path.join(d, "tokens_tiny.json"), rec["sha256"]
    # a loader refusal -> refused stub, exit 3 (caught here)
    a.framework, a.arm, a.tag, a.model = "unsloth", "unsloth", "ckpt_unsloth_refuse", "selftest/refuse"
    try:
        run_arm(a, _selftest_load_unsloth, sampler=False)
        raise AssertionError("refusal did not exit")
    except SystemExit as e:
        assert e.code == 3, e.code
    a.model = "selftest/tiny"
    # a tokens-sha mismatch -> exit 13
    a.tokens_sha = "f" * 64
    a.tag = "ckpt_unsloth_badsha"
    try:
        run_arm(a, _selftest_load_unsloth, sampler=False)
        raise AssertionError("tokens mismatch did not exit")
    except SystemExit as e:
        assert e.code == 13, e.code
    a.tokens_sha = rec["sha256"]

    # ---- assertions on the receipts
    R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(d, "tiny*_*.json"))}
    need = ["s_per_step_median_11plus", "peak_vram_gb", "tokens_per_s", "joules_per_step", "losses", "eval_curve", "eval_loss_final",
            "kernel_calls_per_step_min", "n_patched", "n_attn4", "census", "unsloth_bnb4bit_modules", "C1_bit_exact", "C1_bytes_hashed",
            "C1_empties_skipped", "C1_control_detects_flipped_byte", "adapter", "env", "prereg", "init_sha", "trainable_params", "n_layers", "accum", "autocast",
            "structural_expected_n_attn4", "detector_version",
            "adapter_dtype", "lora_init", "matched_init", "frozen_base_probe"]                       # TC1: T19 / T20 / T21 on every row
    for k in ("tiny_e4b_reference_attn4", "tiny_e4b_fused_attn4", "tiny_unsloth_ckpt_unsloth", "tiny_unsloth_ckpt_hf", "tinygo_e4b_attn_only"):
        r = R[k]
        assert r["status"] == "ok", (k, r.get("status"), r.get("reason"))
        for f in need:
            assert f in r, (k, f)
        assert r["C1_bit_exact"] and r["C1_bytes_hashed"] > 0 and r["C1_empties_skipped"] == 0
        assert len(r["losses"]) == a.steps and r["eval_curve"][-1]["step"] == a.steps and len(r["eval_curve"]) == 1 + a.steps // a.eval_every
        assert r["adapter"].get("bytes", 0) > 0 and r["adapter"].get("dtypes"), r["adapter"]
        assert r["tokens"]["sha256"] == (rec_go if k.startswith("tinygo") else rec)["sha256"] and r["tokens_total"] == sum(r["tokens_per_step"])
        assert all(t == sum(len(rec["train"][(i * a.accum + j) % len(rec["train"])]) for j in range(a.accum)) for i, t in enumerate(r["tokens_per_step"]))
        # T19: the default casts EVERY arm's adapters to fp32, e4b included (tp4 cast only the non-e4b arms)
        assert r["adapter_dtype"] == "fp32" and list(r["adapter_dtypes_after"]) == ["torch.float32"], (k, r["adapter_dtypes_after"])
        if k == "tinygo_e4b_attn_only":           # its bf16 expert adapters are FROZEN (not trainable): only the fp32 attention LoRA is counted
            assert r["adapter_dtypes_before"] == {"torch.float32": 16} and r["lora_cast_to_fp32"] == 0, (k, r["adapter_dtypes_before"], r["lora_cast_to_fp32"])
        else:
            assert "torch.bfloat16" in r["adapter_dtypes_before"] and r["lora_cast_to_fp32"] == r["adapter_dtypes_before"]["torch.bfloat16"], (k, r["adapter_dtypes_before"], r["lora_cast_to_fp32"])
        assert r["lora_init"] == "native" and r["matched_init"] is None, (k, r["lora_init"], r["matched_init"])
        # T21: the probe on every row, three slots, each control detecting its flip
        fp = r["frozen_base_probe"]
        assert set(fp["slots"]) == {"gate_up", "down", "q_proj"} and fp["control_detects_flip"] is True and not fp["errors"], (k, fp)
        assert all(s["sha"] != s["control_sha"] and s["n_bytes"] > 0 for s in fp["slots"].values()), (k, fp)
    e_ref, e_fu, u1, u2, ao = R["tiny_e4b_reference_attn4"], R["tiny_e4b_fused_attn4"], R["tiny_unsloth_ckpt_unsloth"], R["tiny_unsloth_ckpt_hf"], R["tinygo_e4b_attn_only"]
    assert e_ref["init_sha"] == e_fu["init_sha"], "e4b arms did not start bit-identical"
    assert e_ref["n_attn4"] == 8 and e_fu["n_attn4"] == 8 and e_fu["n_patched"] == 2 and e_ref["n_patched"] == 0
    assert e_ref["structural_expected_n_attn4"] == 8 and e_fu["structural_expected_n_attn4"] == 8 and e_ref["detector_version"] == "selftest"
    assert u1["structural_expected_n_attn4"] is None and u1["detector_version"] is None and ao["detector_version"] is None
    assert e_fu["kernel_calls_per_step_min"] == 2 * a.accum and e_ref["kernel_calls_per_step_min"] == 0, (e_fu["kernel_calls_per_step_min"], e_ref["kernel_calls_per_step_min"])
    assert e_fu["kernel_counter_key"] == "fused_grouped_lora"
    assert u1["kernel_counter_key"] == "moe_bnb4bit_backend" and u1["kernel_calls_per_step_min"] == 2 * a.accum and u1["experts_forward_calls_per_step_min"] == 2 * a.accum
    assert u1["census"]["Params4bit_expert_stacks"] == 4 and u1["unsloth_bnb4bit_modules"]["n_bnb4bit"] == 0 and u1["unsloth_bnb4bit_modules"]["n_bnb4bit_unwrapped"] == 2
    assert u1["unsloth_bnb4bit_modules"]["wrap_depth"] == 2 and u1["unsloth_bnb4bit_modules"]["inner_param_types"] == ["Params4bit"]
    assert any(BANNER in s for s in u1["engagement_banners"])
    assert u1["init_sha"] == u2["init_sha"] and u1["grad_ckpt"] == "unsloth" and u2["grad_ckpt"].startswith("hf")
    assert u1["trainable_params"] == e_ref["trainable_params"] and u1["expect_trainable"] == e_ref["trainable_params"] and u1["trainable_mismatch"] is None
    assert e_ref["trainable_by_group"]["experts"] > 0 and ao["trainable_by_group"]["experts"] == 0 and ao["trainable_params"] < e_ref["trainable_params"]
    assert ao["probes"]["fused"]["n_patched"] == 0
    # the SAME frozen base in both layouts: identical step-0 held-out loss (B = 0, the base alone), and the e4b stack
    # dequant regime the probe records differs from the peft side's raw-bytes stand-in (the reducer reads N-A for that pair)
    assert e_ref["eval_loss_step0"] == u1["eval_loss_step0"] == e_fu["eval_loss_step0"], (e_ref["eval_loss_step0"], u1["eval_loss_step0"], e_fu["eval_loss_step0"])
    assert e_ref["frozen_base_probe"]["slots"]["gate_up"]["regime"] == "selftest_u8/1" and u1["frozen_base_probe"]["slots"]["gate_up"]["regime"].startswith("u8-raw")
    assert e_ref["frozen_base_probe"]["slots"]["q_proj"]["regime"] == "fp32" and u1["frozen_base_probe"]["slots"]["q_proj"]["regime"] == "fp32"
    st = R["tinygo_e4b_fused_attn4"]   # the run-script stub (tp1 citation) refreshed by the attn_only arm's own probe
    assert st["status"] == "refused" and st["cited"] == "tp1" and st["probed_by"] == "attn_only" and "patched 0 modules on this box" in st["probe_reason"], st
    assert R["tiny_e4b_fused_attn4"]["status"] == "ok" and "probe_reason" not in R["tiny_e4b_fused_attn4"], "a receipt that trained must never be touched by refresh_stub"
    assert "tinygo_e4b_reference_attn4" not in R, "no bias on the tiny attention -> no reference_attn4 refusal stub"
    rf = R["tiny_unsloth_ckpt_unsloth_refuse"]
    assert rf["status"] == "refused" and rf["phase"] == "load" and "NotImplementedError" in rf["reason"]
    bs = R["tiny_unsloth_ckpt_unsloth_badsha"]
    assert bs["status"] == "tokens_mismatch"
    assert R["tiny_e4b_fused_attn4"]["accum"] == a.accum and R["tiny_e4b_fused_attn4"]["autocast"] == bool(a.autocast)
    # T6: a trainable mismatch is recorded, not a stub
    a.framework, a.arm, a.tag, a.expect_trainable = "unsloth", "unsloth", "ckpt_unsloth_mismatch", e_ref["trainable_params"] + 1
    r = run_arm(a, _selftest_load_unsloth, sampler=False)
    assert r["status"] == "ok" and r["trainable_mismatch"]["expected"] == e_ref["trainable_params"] + 1 and r["trainable_mismatch"]["got"] == e_ref["trainable_params"]
    # --prereg wiring (#434 follow-up): a run's receipts name its own pre-registration, never another lane's by accident
    assert e_ref["prereg"] == PREREG and r["prereg"] == PREREG   # selftest sets a.prereg explicitly above; there is no argparse default
    a.framework, a.arm, a.tag, a.prereg, a.expect_trainable = "unsloth", "unsloth", "ckpt_unsloth_prereg", "p41/P41-PREREG.md", None
    r = run_arm(a, _selftest_load_unsloth, sampler=False)
    assert r["prereg"] == "p41/P41-PREREG.md", r["prereg"]
    a.prereg = PREREG
    det = _selftest_detector(d, a)   # T10: the three dry-run tests against the REAL structural detector (#434)
    sel = _selftest_hf_expert_selection()   # #542: the HF arm's expert selection is structural and refuses when empty

    # ---- T11: the hf arm through the same run_arm (module-level counter, fp32 cast, C1 on Params4bit bytes)
    a.fam, a.model, a.tokens, a.tokens_sha = "tiny", "selftest/tiny", os.path.join(d, "tokens_tiny.json"), rec["sha256"]   # _selftest_detector left a.fam at tcdet
    a.framework, a.arm, a.tag, a.expect_trainable, a.attn_4bit = "hf", "hf", "hf_peft", e_ref["trainable_params"], 0
    hfr = run_arm(a, _selftest_load_hf, sampler=False)
    assert hfr["status"] == "ok" and hfr["kernel_counter_key"] == "experts_forward" and hfr["kernel_calls_per_step_min"] == 2 * a.accum, hfr["kernel_calls_per_step_min"]
    assert hfr["trainable_params"] == e_ref["trainable_params"] and hfr["trainable_mismatch"] is None and hfr["lora_cast_to_fp32"] == hfr["trainable_tensors"]
    assert hfr["census"]["Params4bit_expert_stacks"] == 4 and hfr["hf_targets"]["n_target_parameters"] == 4 and hfr["loader_used"] == "AutoModelForCausalLM"
    assert hfr["C1_bit_exact"] and hfr["micro_batch"] == 1 and hfr["tokens_padded_total"] == 0 and hfr["template"] == "clinical"
    assert hfr["optimizer"].startswith("adamw_torch(lr=0.001, weight_decay=0.01) schedule=constant") and all(v == a.lr for v in hfr["lr_per_step"])
    assert hfr["eval_loss_step0"] == e_ref["eval_loss_step0"], "the hf tiny model does not share the e4b tiny model's frozen base"

    # ---- T12/T13/T14: the alpaca template, micro-batches of 2 with padding, the linear schedule -- through every framework
    rows_a = [{"instruction": f"Task {i}", "input": ("context " * (i % 3)).strip(), "output": ("answer " * (3 + i % 7)).strip()} for i in range(40)]
    dpa = os.path.join(d, "data", "ds_alpaca_selftest.json")
    json.dump({"train": rows_a[:32], "eval": rows_a[32:]}, open(dpa, "w"))
    a.fam, a.data, a.data_sha, a.tokens, a.template = "tinya", dpa, sha_bytes(open(dpa, "rb").read()), os.path.join(d, "tokens_tinya.json"), "alpaca"
    a.seq = 320    # the byte tokenizer makes a token per character: the ~190-char template must fit or every row truncates to one length and nothing pads
    rec_a = prepare(a, tok=_FakeTok())
    assert len({len(r) for r in rec_a["train"]}) > 1, "selftest rows must differ in length so padding is exercised"
    assert rec_a["template"] == "alpaca" and rec_a["format"] == ALPACA_PROMPT and rec_a["eos"] == "" and rec_a["pad_id"] is None
    assert rec_a["train"][0] == encode_rows(_FakeTok(), rows_a[:1], a.seq, "alpaca", "")[0] and rec_a["train"][0] != encode_rows(_FakeTok(), rows_a[:1], a.seq)[0]
    a.tokens_sha, a.micro_batch, a.lr_schedule, a.warmup_steps, a.expect_trainable = rec_a["sha256"], 2, "linear", 3, None
    mb = {}
    for fw, arm, tag, fn in (("e4b", "fused", "fused_attn4", _selftest_load_e4b), ("unsloth", "unsloth", "ckpt_unsloth", _selftest_load_unsloth), ("hf", "hf", "hf_peft", _selftest_load_hf)):
        a.framework, a.arm, a.tag, a.attn_4bit = fw, arm, tag, int(fw == "e4b")
        mb[fw] = run_arm(a, fn, sampler=False)
    for fw, r in mb.items():
        assert r["status"] == "ok" and r["micro_batch"] == 2 and r["template"] == "alpaca", (fw, r["status"])
        # rows (i*accum+j)*2+k, real tokens counted, pads counted beside them, every padded position has a -100 label (loss finite)
        for i, t in enumerate(r["tokens_per_step"]):
            want = sum(len(rec_a["train"][((i * a.accum + j) * 2 + k) % len(rec_a["train"])]) for j in range(a.accum) for k in range(2))
            assert t == want, (fw, i, t, want)
        assert r["tokens_padded_total"] > 0 and all(math.isfinite(v) for v in r["losses"]), fw
        assert r["kernel_calls_per_step_min"] == 2 * a.accum, (fw, r["kernel_calls_per_step_min"])   # one kernel call per micro-batch per layer, whatever M is
        lr = r["lr_per_step"]   # recorded to 8 decimals: step 0 at lr 0, warmup to lr at step W, linear decay after
        assert lr[0] == 0.0 and abs(lr[1] - a.lr / 3) < 1e-7 and abs(lr[3] - a.lr) < 1e-7 and lr[-1] < lr[3] and len(lr) == a.steps, (fw, lr)
        assert r["optimizer"].endswith("schedule=linear warmup_steps=3"), r["optimizer"]
    a.micro_batch, a.lr_schedule, a.warmup_steps, a.template, a.seq = 1, "constant", 0, "clinical", 64
    a.fam, a.tokens, a.tokens_sha = "tiny", os.path.join(d, "tokens_tiny.json"), rec["sha256"]

    # ---- TC1 T19/T20: the MATCHED set -- fp32 adapters + matched:3407 on e4b fused, e4b reference, hf and unsloth.
    # Same base, same A, B = 0 -> identical step 0 by construction; the trajectories agree to the tolerance MEASURED
    # below (printed, asserted against a bound that is a ceiling, never tuned to pass); the native inits (the control)
    # must NOT agree, or the override did nothing.
    a.lora_init, a.adapter_dtype = "matched:3407", "fp32"
    MATCHED = (("e4b", "fused", "fused_attn4_m", _selftest_load_e4b), ("e4b", "reference", "reference_attn4_m", _selftest_load_e4b),
               ("hf", "hf", "hf_peft_m", _selftest_load_hf), ("unsloth", "unsloth", "ckpt_unsloth_m", _selftest_load_unsloth))
    M, N = {}, {}
    for fw, arm, tag, fn in MATCHED:
        a.framework, a.arm, a.tag, a.attn_4bit, a.expect_trainable = fw, arm, tag, int(fw == "e4b"), None
        M[tag] = run_arm(a, fn, sampler=False)
    a.lora_init = "native"
    for fw, arm, tag, fn in MATCHED:
        a.framework, a.arm, a.tag, a.attn_4bit = fw, arm, tag + "_native", int(fw == "e4b")
        N[tag] = run_arm(a, fn, sampler=False)
    a.lora_init = "matched:3407"
    n_slots = 8 + 2 * 2 * 4                       # 8 attention projections (4 x 2 layers) + 2 kinds x 2 layers x 4 experts = 24
    for tag, r in M.items():
        mi = r["matched_init"]
        assert r["status"] == "ok" and r["lora_init"] == "matched:3407" and mi["seed"] == 3407, (tag, r["status"], r["lora_init"])
        assert mi["complete"] is True and mi["n_slots_set"] == mi["n_slots_expected"] == n_slots and not mi["unmapped"] and not mi["b_nonzero"], (tag, mi)
        assert mi["kinds"] == {"q": 2, "k": 2, "v": 2, "o": 2, "gate_up": 8, "down": 8}, (tag, mi["kinds"])
        assert mi["expected_parts"] == {"n_attention_projections": 8, "n_layers": 2, "n_experts": 4}, (tag, mi["expected_parts"])
        assert list(r["adapter_dtypes_after"]) == ["torch.float32"], (tag, r["adapter_dtypes_after"])
        assert r["init_sha"] != N[tag]["init_sha"], (tag, "the matched override did not change the trainable bytes")
    assert all("structure" in v for k, v in M["hf_peft_m"]["matched_init"]["mapping_rules"].items() if "experts" in k), M["hf_peft_m"]["matched_init"]["mapping_rules"]
    assert M["fused_attn4_m"]["init_sha"] == M["reference_attn4_m"]["init_sha"] and M["hf_peft_m"]["init_sha"] == M["ckpt_unsloth_m"]["init_sha"]
    s0 = {t: r["eval_loss_step0"] for t, r in M.items()}
    assert len(set(s0.values())) == 1, ("matched arms must report IDENTICAL eval_loss_step0", s0)
    assert torch.equal(torch.tensor(M["fused_attn4_m"]["losses"]), torch.tensor(M["reference_attn4_m"]["losses"])), "fused vs reference: same code path in the selftest, must be bit-identical"

    def _tol(x, y):
        return {"train_max": max(abs(p - q) for p, q in zip(x["losses"], y["losses"])),
                "train_median": statistics.median(abs(p - q) for p, q in zip(x["losses"], y["losses"])),
                "heldout_final": abs(x["eval_loss_final"] - y["eval_loss_final"])}
    tol = {"hf_vs_e4b": _tol(M["hf_peft_m"], M["fused_attn4_m"]), "unsloth_vs_e4b": _tol(M["ckpt_unsloth_m"], M["fused_attn4_m"]),
           "native_hf_vs_e4b": _tol(N["hf_peft_m"], N["fused_attn4_m"])}
    # the e4b and PEFT layouts hold the same slot tensors viewed differently; the einsum orders differ, so agreement is
    # float rounding, not bit identity. The ceiling here is 1e-3 on 4-decimal-rounded losses; the measured value is printed.
    for k in ("hf_vs_e4b", "unsloth_vs_e4b"):
        assert tol[k]["train_max"] <= 1e-3 and tol[k]["heldout_final"] <= 1e-3, (k, tol[k])
    assert tol["native_hf_vs_e4b"]["train_max"] > 10 * max(tol["hf_vs_e4b"]["train_max"], 1e-6), ("the native inits agree as well as the matched ones: the override is inert", tol)
    # the slot tensors themselves: expert 0 of layer 0's gate_up on the e4b side == the registered draw
    assert all(r["losses"][0] == M["fused_attn4_m"]["losses"][0] for r in N.values()), "step-0 train loss depends on A while B is 0"
    # ---- T19: --adapter-dtype native = the SHIPPED e4b configuration (bf16 expert adapters, N(0, 1/r) init), native init
    a.framework, a.arm, a.tag, a.attn_4bit, a.adapter_dtype, a.lora_init = "e4b", "fused", "fused_attn4_shipped", 1, "native", "native"
    sh = run_arm(a, _selftest_load_e4b, sampler=False)
    assert sh["status"] == "ok" and sh["adapter_dtype"] == "native" and sh["lora_cast_to_fp32"] == 0 and sh["matched_init"] is None
    assert sh["adapter_dtypes_after"] == sh["adapter_dtypes_before"] and "torch.bfloat16" in sh["adapter_dtypes_after"], sh["adapter_dtypes_after"]
    a.adapter_dtype, a.lora_init = "fp32", "matched:3407"
    # ---- T20: an arm whose slots cannot all be mapped is recorded INCOMPLETE (the reducer VOIDs it), never silently native
    a.framework, a.arm, a.tag, a.attn_4bit, a.model = "hf", "hf", "hf_peft_m_extra", 0, "selftest/extra"
    ex = run_arm(a, _selftest_load_hf, sampler=False)
    mi = ex["matched_init"]
    assert ex["status"] == "ok" and mi["complete"] is False and mi["n_slots_set"] == n_slots + 2 and mi["n_slots_expected"] == n_slots, mi
    assert any("EXTRA" in v for v in mi["mapping_rules"].values()), mi["mapping_rules"]
    # ---- T20: a LoRA B that is not zero makes a matched start impossible: harness_error, exit 18, before any step
    a.framework, a.arm, a.tag, a.attn_4bit, a.model = "e4b", "fused", "fused_attn4_m_bnz", 1, "selftest/bnonzero"
    try:
        run_arm(a, _selftest_load_e4b, sampler=False)
        raise AssertionError("a non-zero LoRA B did not refuse")
    except SystemExit as e:
        assert e.code == 18, e.code
    bnz = json.load(open(os.path.join(d, "tiny_e4b_fused_attn4_m_bnz.json")))
    assert bnz["status"] == "harness_error" and "not zero" in bnz["reason"] and bnz["matched_init"]["b_nonzero"] == ["model.layers.0.mlp.experts.down_lora_B"], bnz["matched_init"]["b_nonzero"]
    a.model, a.lora_init, a.adapter_dtype = "selftest/tiny", "native", "fp32"
    # ---- T20: the registered generator, pinned: the slot tensor is a pure function of (seed, layer, kind, expert, r, fan_in)
    A0 = matched_lora_A(3407, 0, "gate_up", 0, 2, 16)
    assert torch.equal(A0, matched_lora_A(3407, 0, "gate_up", 0, 2, 16)) and not torch.equal(A0, matched_lora_A(3407, 0, "gate_up", 1, 2, 16))
    assert not torch.equal(A0, matched_lora_A(3407, 1, "gate_up", 0, 2, 16)) and not torch.equal(A0, matched_lora_A(3408, 0, "gate_up", 0, 2, 16))
    assert float(A0.abs().max()) <= 1.0 / math.sqrt(16) and A0.dtype == torch.float32 and tuple(A0.shape) == (2, 16)
    assert slot_seed(3407, 0, "gate_up", 0) == int.from_bytes(hashlib.sha256(b"3407|0|gate_up|0").digest()[:8], "big")
    for bad in ("matched", "matched:", "matched:x", "seeded:1"):
        try:
            parse_lora_init(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    # ---- T21 on genuine storage (real e4b Experts4bit / ExpertsLoRA, real bnb Params4bit + Linear4bit, CPU)
    probe_real = _selftest_frozen_probe_real()
    # ---- P2-1 / P2-2: the Unsloth backend knobs and counters, the double-quant decision, the axolotl arm
    p2 = _selftest_unsloth_backend_and_axolotl(a, d, rec, e_ref)
    a.framework, a.arm, a.tag, a.lora_init, a.adapter_dtype = "e4b", "fused", "fused_attn4", "native", "fp32"

    print(f"SELFTEST OK dir={d} receipts={sorted(R)} e4b ref/fused loss_last {e_ref['loss_last']}/{e_fu['loss_last']} unsloth {u1['loss_last']} "
          f"hf {hfr['loss_last']} accum={a.accum} autocast={a.autocast} kcalls fused={e_fu['kernel_calls_per_step_min']} unsloth={u1['kernel_calls_per_step_min']} "
          f"hf={hfr['kernel_calls_per_step_min']} mb2_pads={ {k: v['tokens_padded_total'] for k, v in mb.items()} } detector_dryruns={det} "
          f"expert_selection_dryruns={sel} matched_step0={s0} matched_tolerance={tol} matched_slots={n_slots} "
          f"frozen_probe_real={probe_real} phase2={p2}")
    return d


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="T8: CPU, tiny synthetic model, both branches, mocked kernels; + T10 detector dry-runs (#434)")
    ap.add_argument("--framework", choices=["e4b", "unsloth", "hf", "axolotl"], default="e4b")
    ap.add_argument("--arm", choices=["reference", "fused", "batched", "attn_only", "unsloth", "hf", "axolotl"], default="fused",
                    help="P56: `batched` is enable_batched_train -- e4b's KERNEL-FREE group-sorted path. It computes the "
                         "same function as `reference` with the same arithmetic REORDERING as `fused`, but through torch "
                         "+ bitsandbytes rather than grouped-nf4-gemm, so a parity pair against `reference` separates a "
                         "kernel defect from a trajectory floor.")
    ap.add_argument("--template", choices=["clinical", "alpaca"], default="clinical", help="T12: the prompt template the tokens file is built with (--prepare)")
    ap.add_argument("--micro-batch", type=int, default=1, help="T13: rows per micro-batch (right-padded, masked, -100 labels on pads); 1 = tp2 byte-for-byte")
    ap.add_argument("--optim", choices=["adamw_torch", "adamw_8bit"], default="adamw_torch", help="T14: the same optimizer call in every arm")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="T14: tp2 kept torch's default 0.01; the notebooks use 0.001")
    ap.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant", help="T14: linear = transformers' linear-with-warmup formula")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--unsloth-targets", default=",".join(UNSLOTH_TARGETS), help="T15: Unsloth get_peft_model target_modules (comma list; the notebooks' seven)")
    ap.add_argument("--unsloth-moe-backend", choices=list(UNSLOTH_BACKENDS), default="default",
                    help="P2-1 (TC1): sets UNSLOTH_MOE_BACKEND in this process before `import unsloth` (select_moe_backend(), moe_utils.py:1061-1085); "
                         "default = leave the loader's own choice, recorded after load")
    ap.add_argument("--unsloth-speed-tilt", type=int, default=0, help="P2-1 (TC1): 1 = UNSLOTH_MOE_RECOMPUTE=0 UNSLOTH_MOE_GC_REPLAY_PIN=1 (recorded)")
    ap.add_argument("--unsloth-double-quant", choices=["off", "default"], default="off",
                    help="P2-1 (TC1): off = pass double-quant OFF through a from_pretrained parameter the signature names (read at runtime), "
                         "else the loader's default; the loaded expert Params4bit's quant_state.nested is recorded either way")
    ap.add_argument("--axolotl-best", type=int, default=0, help="P2-2 (TC1): 1 = plugins [KernelsPlugin], expert_backend scattermoe, moe_bnb_fast true")
    ap.add_argument("--axolotl-dataset", default=None, help="P2-2 (TC1): the registered Alpaca subset file named in the axolotl config's `datasets` (validation needs one; nothing reads it here)")
    ap.add_argument("--adapter-dtype", choices=["fp32", "native"], default="fp32",
                    help="T19 (TC1): fp32 = cast EVERY framework's trainable adapters to fp32 after construction (e4b included; tp4 cast only "
                         "the non-e4b arms); native = leave them as the loader built them (e4b: bf16 expert adapters -- the `shipped` arm)")
    ap.add_argument("--lora-init", default="native",
                    help="T20 (TC1): native = each framework's own init; matched:<seed> = overwrite every LoRA A with the registered per-slot "
                         "deterministic tensor (U(-1/sqrt(fan_in), 1/sqrt(fan_in)) from sha256(seed|layer|kind|expert)) and assert every B zero; "
                         "the receipt's matched_init.complete says whether every slot was mapped")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fam", default="qwen3")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--log-every", type=int, default=10,
                    help="T17 (P43): print a step line every N optimizer steps (default 10 = tp4 as run; 1 = every step)")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="P45: wrap this many optimizer steps (after --profile-warm) in torch.profiler (CPU+CUDA) and write <receipt>_profile.json: "
                         "device-busy fraction, launches/step, CPU self time by op family; 0 = off (the timed number is never taken from profiled steps)")
    ap.add_argument("--profile-warm", type=int, default=3, help="P45: steps to run before the profiler starts")
    ap.add_argument("--microbatch-timing", type=int, default=0,
                    help="T17 (P43): time every micro-batch (a cuda sync per micro-batch) and record microbatch_ms per step; 0 = tp4 as run")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--accum", type=int, default=4, help="T5: micro-batches (batch 1 each) per optimizer step (P40: 4)")
    ap.add_argument("--autocast", type=int, default=1, help="T5: torch.autocast bf16 around forward+loss (P40: bf16 autocast)")
    ap.add_argument("--offload", type=int, default=0, help="e4b: expert offload to pinned host RAM (tp1: Mixtral only)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--data-sha", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--tokens-sha", default=None)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dgrad", type=int, default=1,
                    help="P56: enable_fast_train(dgrad=...) on the `fused` arm. 1 (default, every tp1-tp4 row to date) "
                         "routes the BACKWARD through grouped-nf4-gemm's single-launch dgrad kernel; 0 keeps its "
                         "per-expert decode loop, which decodes with the same oracle the reference uses and is EXACT. "
                         "The pair separates the forward fusion's error from the backward kernel's.")
    ap.add_argument("--attn-4bit", type=int, default=0, help="e4b: quantize_attention_projections_4bit before the attention LoRA (U4)")
    ap.add_argument("--grad-ckpt", choices=["unsloth", "hf"], default="unsloth", help="Unsloth: use_gradient_checkpointing mode (U1)")
    ap.add_argument("--unsloth-loader", choices=["FastLanguageModel", "FastModel"], default="FastLanguageModel", help="T4: P38's loader; FastModel is an amendment")
    ap.add_argument("--expect-trainable", type=int, default=None, help="T6: the family's e4b trainable count; a mismatch is recorded")
    ap.add_argument("--prereg", default=None, help="REQUIRED for a real run: the governing pre-registration path written into every receipt and stub (P41: p41/P41-PREREG.md). No default — a receipt must never cite a pre-registration the run did not pass")
    ap.add_argument("--no-sampler", type=int, default=0)
    ap.add_argument("--phase-budget-s", type=float, default=None,
                    help="#548: refuse the arm (status phase_alarm, exit 16) while still inside a prologue phase once the whole "
                         f"prologue passes this many seconds. Default: TC1_PHASE_BUDGET_S, else {PROLOGUE_BUDGET_SHARE:g} x TC1_ARM_ALARM_S "
                         "(what tc1_run.sh gave perl's alarm), else off. phase_seconds is recorded either way")
    ap.add_argument("--out", default="/root/tc1")
    ap.add_argument("--adapter-dir", default="/root/tc1/adapters")
    a = ap.parse_args()
    a.tag = a.tag or a.arm
    try:
        parse_lora_init(a.lora_init)              # T20: refused at the command line, before any receipt exists
    except ValueError as e:
        ap.error(str(e))
    if a.selftest:
        return selftest(a)
    if a.prepare:
        return prepare(a)
    if not a.prereg:
        ap.error("--prereg is required for a real run: a receipt must name the pre-registration it ran under (TC1: tc1/TC1-PREREG.md)")
    # P67: the perturbed-reference order is a switch for the e4b REFERENCE arm only. Anywhere else it reaches the
    # per-expert loop only through a fallback (batched) or not at all (fused), so the receipt would describe an arm
    # that did not run. Refused before anything loads -- and before the CUDA check, so CI can drive the refusal.
    if os.environ.get("E4B_REFERENCE_EXPERT_ORDER") and not (a.framework == "e4b" and a.arm == "reference"):
        stub(a, "harness_error", f"E4B_REFERENCE_EXPERT_ORDER={os.environ['E4B_REFERENCE_EXPERT_ORDER']!r} is set on "
             f"{a.framework}/{a.arm}; it is a reference-arm switch only (bench/p67/P67-PREREG.md)", {"phase": "preamble"}, code=19)
    if not torch.cuda.is_available():
        stub(a, "harness_error", "torch.cuda.is_available() is False on a GPU lane", code=10)
    loader = {"e4b": load_e4b, "unsloth": load_unsloth, "hf": load_hf, "axolotl": load_axolotl}[a.framework]
    return run_arm(a, loader, sampler=not a.no_sampler)


if __name__ == "__main__":
    main()
