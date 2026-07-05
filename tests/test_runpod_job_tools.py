"""CPU-safe tests for the distributed-validation job tools (claimer, manifest, summarizer).

No GPU, no models: claiming atomicity via lock dirs, {job_dir} substitution and failure capture
in run_job (payloads are tiny `python -c` commands), manifest generation incl. the phase-3
adapters-must-exist gate, and the summarizer's aggregation + claim rules over fixture results.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_olmoe_repeat_manifest as manifest_mod  # noqa: E402
import runpod_claim_and_run as runner  # noqa: E402
import summarize_runpod_jobs as summarizer  # noqa: E402


def test_claim_is_atomic_and_exclusive(tmp_path):
    locks = str(tmp_path / "locks")
    os.makedirs(locks)
    ident_a = {"pod_id": "pod-a", "hostname": "a", "pid": 1, "commit": "x", "claimed_at": "t"}
    ident_b = {"pod_id": "pod-b", "hostname": "b", "pid": 2, "commit": "x", "claimed_at": "t"}
    assert runner.claim(locks, "job1", ident_a) is True
    assert runner.claim(locks, "job1", ident_b) is False  # second claimer loses, atomically
    owner = json.load(open(tmp_path / "locks" / "job1.lock" / "owner.json"))
    assert owner["pod_id"] == "pod-a"  # the lock records its owner


def test_run_job_substitutes_job_dir_and_lifts_rows(tmp_path):
    jobs_root = str(tmp_path / "jobs")
    job = {
        "job_id": "toy",
        "job_type": "train",
        "command": [sys.executable, "-c",
                    "import json,sys; open(sys.argv[1] + '/result_rows.jsonl','w')"
                    ".write(json.dumps({'status':'pass','train_eval_best':1.01}) + '\\n')",
                    "{job_dir}"],
        "params": {"seed": 7, "storage_mode": "nf4", "offload": False},
    }
    ident = {"pod_id": "pod-a", "hostname": "a", "pid": 1, "commit": "x", "claimed_at": "t"}
    assert runner.run_job(job, jobs_root, ident) == "pass"
    result = json.load(open(os.path.join(jobs_root, "toy", "result.json")))
    assert result["train_eval_best"] == 1.01  # lifted from the payload's row
    assert result["seed"] == 7 and result["pod_id"] == "pod-a"  # params + identity merged
    assert os.path.exists(os.path.join(jobs_root, "toy", "command.sh"))
    assert json.load(open(os.path.join(jobs_root, "toy", "status.json")))["status"] == "pass"


def test_run_job_records_failure_loudly(tmp_path):
    job = {"job_id": "boom", "job_type": "decode", "command": [sys.executable, "-c", "raise SystemExit(3)"],
           "params": {}}
    ident = {"pod_id": "p", "hostname": "h", "pid": 1, "commit": None, "claimed_at": "t"}
    assert runner.run_job(job, str(tmp_path), ident) == "fail"
    result = json.load(open(tmp_path / "boom" / "result.json"))
    assert result["status"] == "fail" and result["exit_code"] == 3
    assert "run.log" in result["fail_or_skip_reason"]


def test_manifest_phase1_and_2_ids():
    jobs = [manifest_mod._train_job(label, seed)
            for label in manifest_mod.TRAIN_MODES for seed in manifest_mod.SEEDS]
    ids = {j["job_id"] for j in jobs}
    assert len(ids) == 12
    assert "train_olmoe_nf4_resident_seed1337" in ids
    assert "train_olmoe_int8_offload_seed3407" in ids
    assert all("qwen3" not in i for i in ids)  # the model-label correction, pinned
    dec = manifest_mod._decode_job("fp4")
    assert dec["job_id"] == "decode_olmoe_fp4_resident_repeat5"
    assert "QUANT_TYPE=fp4" in dec["command"]


def test_manifest_phase3_gates_on_existing_adapters(tmp_path):
    jobs_root = str(tmp_path)
    # Only one train job's adapter exists on disk.
    d = tmp_path / "train_olmoe_nf4_resident_seed1337" / "nf4"
    os.makedirs(d)
    (d / "adapter_best.pt").write_bytes(b"x")
    jobs, missing = manifest_mod._query_jobs(jobs_root)
    assert len(jobs) == 2  # one adapter x two query modes
    assert {j["job_id"] for j in jobs} == {
        "query_olmoe_train-nf4-resident-seed1337_query-nf4-resident",
        "query_olmoe_train-nf4-resident-seed1337_query-int8-resident",
    }
    assert len(set(missing)) == 11  # the other 11 train jobs are gated out, loudly


def _train_result(mode_label, seed, best, before, peak, spstep):
    storage, offload = mode_label.split("-")[0], mode_label.endswith("offload")
    return {"job_id": f"train_olmoe_{storage}_{'offload' if offload else 'resident'}_seed{seed}",
            "job_type": "train", "status": "pass", "storage_mode": storage, "offload": offload,
            "seed": seed, "train_eval_best": best, "train_eval_after": best + 0.002,
            "train_eval_before": before, "peak_train_gpu_gb": peak, "train_s_per_step": spstep}


def test_summarizer_tables_and_claims(tmp_path):
    jobs_root = tmp_path / "jobs"
    fixtures = []
    for seed in (1337, 2027, 3407):
        fixtures += [
            _train_result("nf4-resident", seed, 1.030, 1.4905, 5.28, 12.4),
            _train_result("nf4-offload", seed, 1.031, 1.4905, 2.52, 16.7),
            _train_result("int8-resident", seed, 1.025, 1.4811, 8.50, 12.1),
            _train_result("int8-offload", seed, 1.014, 1.4811, 2.72, 18.5),
        ]
    fixtures.append({"job_id": "decode_olmoe_nf4_resident_repeat5", "job_type": "decode",
                     "status": "pass", "storage_mode": "nf4", "offload": False, "samples": 5,
                     "tok_s_mean": 10.1, "tok_s_std": 0.2, "tok_s_min": 9.9, "tok_s_max": 10.4,
                     "peak_gpu_gb": 4.72})
    fixtures.append({"job_id": "decode_olmoe_fp4_resident_repeat5", "job_type": "decode",
                     "status": "pass", "storage_mode": "fp4", "offload": False, "samples": 5,
                     "tok_s_mean": 12.5, "tok_s_std": 0.3, "tok_s_min": 12.1, "tok_s_max": 12.9,
                     "peak_gpu_gb": 4.72})
    fixtures.append({"job_id": "train_olmoe_broken", "job_type": "train", "status": "fail",
                     "fail_or_skip_reason": "synthetic failure"})
    for r in fixtures:
        d = jobs_root / r["job_id"]
        os.makedirs(d)
        (d / "result.json").write_text(json.dumps(r))

    rows = summarizer.load_results(str(jobs_root))
    train = [r for r in rows if r.get("job_type") == "train"]
    decode = [r for r in rows if r.get("job_type") == "decode"]

    agg = "\n".join(summarizer.training_tables(train))
    assert "int8-offload" in agg and "1.0140" in agg
    claims = "\n".join(summarizer.claims_table(train, decode))
    # 3/3 seed wins -> stable; separated decode means -> stable; both host-specific.
    assert "best-eval wins in 3/3 seeds" in claims and claims.count("Stable (host-specific)") >= 4
    dec_md = "\n".join(summarizer.decode_table(decode))
    assert "12.50 ± 0.30" in dec_md
    # The failed job stays visible.
    assert any(r["status"] == "fail" for r in rows)
