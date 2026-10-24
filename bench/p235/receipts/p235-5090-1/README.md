# `p235-5090-1`: P58's same-box vLLM head-to-head at the release pins

One RTX 5090 (cc 12.0, driver 595.91.07, a **470 W** power limit recorded on the box and never ruled) on an Intel Core
Ultra 9 285K host, 2026-10-24T19:42:46Z to 20:31:01Z, through the guarded launcher. Cost $0.86.
- **Code:** the harness at e4b `7be9e610` (e4b#2119's merge), recorded and never installed. Installed from PyPI and
  proven by their bytes: experts4bit-qlora 0.58.0, grouped-nf4-gemm 0.51.0, vLLM 0.32.0 (primary) and 0.30.0
  (secondary, each in its own venv).
- **Stack:** e4b on torch 2.8.0+cu129, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.1; each vLLM venv on
  torch 2.13.0+cu130, triton 3.7.1.
- **Model:** Qwen/Qwen3-30B-A3B @ad44e777 (e4b's NF4 arena baked on the box) and Qwen/Qwen3-30B-A3B-GPTQ-Int4
  @9b534e43 (vLLM's checkpoint).

**Before the arms:** the pre-install guard fit passed (`GUARD_FIT fit=6060s need=7575s left=10542s`); provenance and the
fast-check release proofs passed for every distribution; the tripwire passed; both vLLM venvs installed (rc 0).

**The run:** P58's 18 arms in the registered order, every process rc 0, every arm VALID, no arm skipped, no DRIFT. The
binding release proofs ran in the arms' own import environment right before each build's first arm.

**Read:** vLLM 0.32.0 / e4b 0.58.0 int4 = **1.038** at B=1 and **1.203** at B=16 (vLLM ahead); vLLM 0.30.0 / e4b int4 =
1.031 and 1.187. The full read is in [../../RESULTS-p235.md](../../RESULTS-p235.md); the reducer's own output is
`RESULTS.txt` and `RESULTS-p235-generated.md` here.

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store** (`STORE-ONLY.sha256`, by path within the run's receipt): the nine
  vLLM arm receipts (`vllm_*.json`), which record the provider's instance id; the launcher's teardown proof and instance
  record. The launcher's receipt, its ledger row, the host facts, the guard files, the heartbeat file and the launcher's
  known_hosts stay there too.
- **Not copied:** the staged scripts, calibration file and hook, each byte-identical to the file `staged.sha256` pins at
  `7be9e610`; the nonce markers, the run nonce and the instance id file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name, an endpoint or a GPU UUID (the
  runner does not collect the UUID).
- **Byte-identical, carriage returns kept:** the engine, fetch, bake and prompt logs and `outer.log` carry the bare
  carriage returns of their progress bars (no CRLF line ending anywhere); they are kept as the box wrote them, so
  `SHA256SUMS` matches the store.
