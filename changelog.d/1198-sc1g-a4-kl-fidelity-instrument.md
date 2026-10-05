### SC1g amendment A4 registered (#846): engines graded by KL to a bf16-dequant reference of gpt-oss-20b, cross-engine NLL descriptive (bench and tests only)

- **Why.** A3's read showed teacher-forced NLL on off-policy chat text can rank the less faithful path first (NF4 under the
  native MXFP4 store). So A1's cross-engine NLL is now descriptive, and engines are graded by KL from P44's reference.
- **Estimator (`bench/sc2/sc1g_kl.py`).** KL65 on a partition of the reference's top-64 named tokens plus rest.
  - Each engine returns its exact log-probs on those named tokens:
    - vLLM `logprob_token_ids`;
    - SGLang `token_ids_logprob`;
    - llama.cpp's harness `--named`;
    - e4b through a `torch` proxy in `sc1g_k8.py`. The shared harness bytes stay untouched; alignment is proved to 1e-9
      against the arm's own NLL.
  - Registered rules: rest clamp ε 1e-9 with a 1 % clamp ceiling; a 0.99 coverage floor; a top-K ≥ 256 fallback.
- **Box R (`bench/sc2/sc1g-r/`, one H100 NVL, declared $3.50/h, maintainer-approved).**
  - Scores the reference decode-shaped on the five committed windows and writes hashed artifacts.
  - Calibrates KL65 against the full-vocabulary KL on two real perturbations (decode vs prefill, and an NF4 fake-quant). The
    ratio must be ≥ 0.90 on both, or the estimator is UNREAD.
  - Reads its own OK / UNREAD / VOID verdict.
- **Box I under A4.** Named KL rows for every engine on four conversations, then descriptive prefill rows. Its proof reads
  one named row per engine path.
- **Predictions.**
  - K-A: NF4 ≥ 3× MXFP4. A prediction, not a gate.
  - L1: e4b ≤ 2× the best comparator.
  - L2: every native-MXFP4 engine sits below R's NF4-requant scale.
