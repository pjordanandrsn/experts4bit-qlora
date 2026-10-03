"""The shape of the SC1 box script and its controller, pinned in CI (lane SC1, experts4bit-qlora#846; the registration's
`bench/sc1/SC1-PREREG.md` is the spec):

- every shell file parses;
- the refusals (class, disk, driver, box A's host vendor) come before the e4b install, and the lane's failure codes use the
  launcher's machine-exclusion codes only where the registration says so (disk 13, driver / vendor 18);
- the PROVE block fetches no Qwen3 (Granite only), after the installs, before any Qwen3 fetch;
- the three boxes run their phases in v3's order (marker strings, first occurrence, strictly increasing);
- every `SC1_*` knob the box script reads is forwarded by the driver (tp4's lesson);
- every arm name v3's phases list appears in the script (the one renamed pair, `_detok`, is stated);
- P88's environment strings and command lines are byte-identical; every e4b arm names the four route knobs;
- the quiescence gate precedes the pack build and the first timed arm; the sched arms target the fixed `_apply_fusions`;
- the two Python helpers' selftests pass on CPU.
"""
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sc1"
RUN = (LANE / "sc1_run.sh").read_text()
DRIVE = (LANE / "sc1_drive.sh").read_text()
SCHED = (LANE / "sc1_e4b_sched.py").read_text()
SAMPLER = (LANE / "sc1_sampler.sh").read_text()
P88 = (REPO / "bench" / "p88" / "p88_run.sh").read_text()
P94 = (REPO / "bench" / "p94" / "p94_run.sh").read_text()
ENV = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": "/tmp"}


def _body(fn: str, nxt: str) -> str:
    return RUN[RUN.index(f"{fn}(){{"):RUN.index(f"{nxt}(){{")] if nxt else RUN[RUN.index(f"{fn}(){{"):]


def _phases(body: str) -> int:
    return len(re.findall(r'^\s*phase [A-Za-z0-9-]+ "', body, re.M))


def _in_order(body: str, markers):
    idx = []
    for m in markers:
        assert m in body, f"marker missing: {m!r}"
        idx.append(body.index(m))
    assert idx == sorted(idx), [m for m, i in zip(markers, idx) if idx.index(i) != markers.index(m)] or list(zip(markers, idx))


def test_shell_files_parse():
    for f in ("sc1_run.sh", "sc1_drive.sh", "sc1_sampler.sh", "make_pin.sh"):
        r = subprocess.run(["bash", "-n", str(LANE / f)], capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, (f, r.stderr)


def test_refusals_precede_install_and_use_the_registered_codes():
    install = RUN.index('say "install e4b @')
    pin = RUN.index("sha256sum -c staged.sha256")
    for refusal in ("finish 15;", "finish 13;", "finish 18;"):              # class, disk, driver + box A's vendor
        assert RUN.index(refusal) < install, refusal
        assert pin < RUN.index(refusal)
    assert RUN.count("finish 18;") == 2 and 'echo "refused: driver' in RUN and 'echo "refused: cpu vendor' in RUN
    assert "MIN_DRIVER=${SC1_MIN_DRIVER:-580}" in RUN and "MIN_DISK_GB=${SC1_MIN_DISK_GB:-320}" in RUN
    assert 'CPU_VENDOR=${SC1_CPU_VENDOR:-AuthenticAMD}' in RUN and 'if [ "$BOX" = A ]; then [ "$HOST_VENDOR" = "$CPU_VENDOR" ]' in RUN
    for field in ("pcie.link.gen.current", "pcie.link.width.current", "nproc", "/sys/fs/cgroup/cpu.max", "/proc/loadavg"):
        assert field in RUN[:install], field
    codes = {int(c) for c in re.findall(r"finish (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes & {13, 14, 17, 18}


def test_the_proof_fetches_no_qwen3():
    install = RUN.index('say "install e4b @')
    prove = RUN.index('if [ "$PROVE" = 1 ]; then')
    end = RUN.index(": > PROVED; finish 0")
    first_qwen3_fetch = RUN.index('fetch qwen3 "$MID" "$REV"')
    assert install < prove < end < first_qwen3_fetch
    block = RUN[prove:end]
    assert "Qwen3-30B" not in block and 'fetch qwen3' not in block and '"$MID"' not in block and "$REV" not in block
    assert 'fetch granite "$GR" "$GR_REV" 900' in block and 'bake granite "$GR" 1500' in block
    assert 'sched_smoke granite_b$B $B "$GR_ENV" 0 "$GR" "$GR_REV" "$GA"' in block and "for B in 1 16" in block
    assert "finish 23" in block and "SC1_PROVE_SGLANG_MODEL" in block
    assert "GR=ibm-granite/granite-3.1-3b-a800m-instruct; GR_REV=a02780686e08a03fe0d2679a293b5c74a90efa89" in RUN and "GR_REV=a02780686e08a03fe0d2679a293b5c74a90efa89" in P94
    assert 'GR_ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"' in RUN and 'GR_ENV="E4B_SERVE_EXP_INT4=0' in P94
    # the smoke's own gate: --smoke exits 0 only when every SET lever counts nonzero and every bucket captured
    assert "def lever_census" in SCHED and 'return 0 if ok else 1' in SCHED


BOX_A = ['phase 0 ', 'fetch_common || finish 11', 'bake_qwen3 || finish 12', 'prompts || finish 19',
         'sched_smoke qwen3_fused_b$B $B "$SPEEDENV" 1', 'sched_smoke qwen3_folds_b$B $B "$SPEEDENV" 0', 'quiesce build',
         'can_run 5400 k8_build && e4b_build', '\n  premise\n',
         'phase A ', 'quiesce arms', 'arm_lic_window r1 16', 'arm_vllm_gptq r1 16', 'arm_lic_window r1 1;', 'arm_vllm_gptq r1 1;',
         'phase A2 ', 'arm_lic_sched r1 16', 'arm_lic_sched r1 1;',
         'phase B ', 'e4b_k8 lic_auto_$SRC', 'E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto" $SRC', 'k8_census; rec', 'e4b_k8 lic_1_$SRC',
         'E4B_INT4_GROUPED_SMALLM=1 E4B_INT4_LEAN_GLUE=auto" $SRC', 'e4b_k8 nf4_$SRC "$NF4ENV" $SRC',
         'phase D-vLLM ', 'vllm_nll prefill $SRC', 'vllm_nll served $SRC;', 'e4b_k8 nll_e4b_prefill_$SRC "$LICENV $PACKENV" $SRC --ppl-oracle eager',
         'phase G1 ', 'arm_lic_window r2 16', 'arm_vllm_gptq r2 16', 'arm_lic_window r2 1;', 'arm_vllm_gptq r2 1;', 'arm_lic_sched r2 16', 'arm_lic_sched r2 1;',
         'phase C ', 'arm_vllm_fp8kv r1 $B', 'arm_rtn r1 $B', 'arm_nf4 r1 $B',
         'phase F ', 'arm_vllm_sameprompt r1', 'arm_sched_sameprompt r1', 'arm_degraded r1',
         'phase E ', 'ttft_sched 512; ttft_sched 4096; ttft_vllm 512; ttft_vllm 4096',
         'phase EN ', 'energy_sched 1; energy_sched 16; energy_vllm 1; energy_vllm 16',
         'phase G2 ', 'arm_vllm_fp8kv r2 $B', 'arm_rtn r2 $B', 'arm_vllm_sameprompt r2', 'arm_sched_sameprompt r2', 'arm_degraded r2',
         'third e4b lic_b$B arm_lic_window $B', 'third e4bsched lic_sched_b$B arm_lic_sched $B', 'third vllm gptq_graph_b$B arm_vllm_gptq $B',
         'third vllm gptq_fp8kv_b$B', 'third e4b rtn_b$B', 'arm_vllm_nodetok $B', 'arm_nf4 r2 $B',
         'vllm_nll served $SRC fp8', 'reduce; }']
BOX_B = ['phase 0 ', 'fetch_common || finish 11', 'fetch_ggufs', 'fetch_exl3', 'bake_qwen3 || finish 12', 'prompts || finish 19', 'quiesce arms',
         'phase AN ', 'arm_int4_window r1 $B', 'arm_int4_sched r1 $B', 'arm_vllm_gptq r1 $B',
         'phase OR ', 'oracle $SRC',
         'phase CMP ', 'arm_llamacpp_q4km r1 $B', 'arm_exl3 r1 $B', 'arm_lmdeploy r1 $B', 'arm_llamacpp_iq4xs',
         'phase Q ', 'llamacpp_nll $W/gguf/$GGUF_Q4KM $M $SRC', 'exl3_nll $M $SRC', 'lmdeploy_nll $M $SRC',
         'phase AN2 ', 'arm_int4_window r2 $B', 'arm_int4_sched r2 $B', 'arm_vllm_gptq r2 $B',
         'phase TT ', 'ttft_sched 512', 'ttft_vllm 4096', 'llamacpp_up $W/gguf/$GGUF_Q4KM 1 8192', 'ttft_llamacpp 4096', 'ttft_exl3 512', 'ttft_lmdeploy 4096',
         'phase EN ', 'energy_llamacpp; energy_exl3; energy_lmdeploy',
         'phase CMP2 ', 'arm_llamacpp_q4km r2 $B', 'arm_exl3 r2 $B', 'arm_lmdeploy r2 $B', 'third e4b int4_b$B', 'third e4bsched int4_sched_b$B',
         'third llamacpp q4km_b$B', 'third lmdeploy w4a16_b$B', 'reduce; }']
BOX_C = ['phase 0 ', 'fetch_common || finish 11', 'fetch_exl3', 'bake_qwen3 || finish 12', 'prompts || finish 19', 'quiesce arms',
         'phase AN ', 'arm_int4_window r1 $B', 'arm_int4_sched r1 $B', 'arm_vllm_gptq r1 $B',
         'phase SG0 ', 'can_run 2700 sglang_install && install_sglang', 'sglang_up matched; quiesce sglang',
         'phase SG ', 'arm_sglang_matched r1 $B', 'sglang_down; sglang_up native', 'arm_sglang_native r1 $B',
         'phase SGQ ', 'sglang_up quality', 'sglang_nll prefill $SRC', 'sglang_nll served $SRC',
         'phase EX ', 'arm_exl3_cu13 $B',
         'phase TT ', 'ttft_sched 512', 'ttft_vllm 4096', 'sglang_up ttft_matched', 'ttft_sglang 4096', 'ttft_exl3 512; ttft_exl3 4096',
         'phase SG2 ', 'arm_sglang_matched r2 $B', 'third sglang gptq_matched_b$B', 'arm_sglang_native r2 $B', 'third sglang gptq_native_b$B',
         'arm_int4_window r2 $B', 'arm_int4_sched r2 $B', 'arm_vllm_gptq r2 $B', 'reduce; }']


def test_box_a_phases_run_in_v3_order():
    body = _body("box_a", "box_b")
    _in_order(body, BOX_A)
    assert _phases(body) == 11                              # 0 A A2 B D-vLLM G1 C F E EN G2
    assert 'SCHED_NAME=lic; SCHED_STACK="$LICENV $PACKENV"' in body


def test_box_b_phases_run_in_v3_order():
    body = _body("box_b", "box_c")
    _in_order(body, BOX_B)
    assert _phases(body) == 9 and 'SCHED_NAME=int4; SCHED_STACK="$SPEEDENV"' in body   # 0 AN OR CMP Q AN2 TT EN CMP2


def test_box_c_phases_run_in_v3_order():
    body = _body("box_c", "")
    body = body[:body.index('case "$BOX" in A) box_a;; B) box_b;; C) box_c;; esac')]
    _in_order(body, BOX_C)
    assert _phases(body) == 8 and 'SCHED_NAME=int4; SCHED_STACK="$SPEEDENV"' in body   # 0 AN SG0 SG SGQ EX TT SG2
    assert 'case "$BOX" in A) box_a;; B) box_b;; C) box_c;; esac' in RUN


def test_every_env_knob_the_box_reads_is_forwarded_by_the_driver():
    """tp4's lesson (TP4-PREREG amendment 2): a knob the box script reads that the controller does not forward silently never
    reaches the box. Every SC1_* read by sc1_run.sh (plus the comparator install flavour install.sh reads) is in the driver's list."""
    read = set(re.findall(r"\$\{(SC1_[A-Z0-9_]+)", RUN))
    for loop in re.findall(r"for v in ((?:SC1_[A-Z0-9_]+ )+)", RUN):
        read |= set(loop.split())
    set_by_box = set(re.findall(r"(?:^|[\s;{(])(SC1_[A-Z0-9_]+)=", RUN, re.M)) | {"SC1_SAMPLES_DIR"}
    forwarded = set(re.findall(r"SC1_[A-Z0-9_]+", DRIVE[DRIVE.index('PASS="SC1_BOX='):DRIVE.index('if [ "${SC1_DRIVE_DRYRUN:-0}"')]))
    missing = sorted(read - set_by_box - forwarded)
    assert not missing, f"read by sc1_run.sh but never forwarded by sc1_drive.sh: {missing}"
    for must in ("SC1_BOX", "SC1_PROVE", "SC1_PROVE_SGLANG_MODEL", "SC1_GPU_CLASS", "SC1_MIN_DISK_GB", "SC1_MIN_DRIVER", "SC1_CPU_VENDOR",
                 "SC1_CALIB_NSEQ", "SC1_REHEARSAL", "SC1_QUIESCE_S", "SC1_VLLM_WHEEL"):
        assert must in forwarded, must
    assert "SC1_VLLM_WHEEL" in (LANE / "vllm" / "install.sh").read_text()     # read by the vLLM install, hence forwarded
    assert 'PASS="$PASS $v=$(printf %q "${!v}")"' in DRIVE                     # %q: a value with spaces must survive the remote re-parse
    # the sched driver's own contract is set by the box, never forwarded
    for k in ("SC1_ARM", "SC1_BATCH", "SC1_PROMPTS", "SC1_OUT", "SC1_INSTANCE_ID"):
        assert f'"{k}"' in SCHED, k
        assert f"{k}=" in RUN, k


V3_ARMS = (  # the receipt stems v3's phases name, in the reducer's vocabulary (B loops over 16 1; D over r1 r2 r3)
    "lic_b${B}_$D", "lic_sched_b${B}_$D", "gptq_graph_b${B}_$D", "gptq_fp8kv_b${B}_$D", "rtn_b${B}_$D", "nf4_ctrl_b${B}_$D",
    "gptq_graph_sameprompt_b16_$D", "lic_sched_sameprompt_b16_$D", "lic_degraded_b16_$D",
    "gptq_graph_nodetok_b${B}_r1",                                  # the F7 pair (detokenize=False beside the matched detokenize=True arms, both B)
    "int4_b${B}_$D", "int4_sched_b${B}_$D", "q4km_b${B}_$D", "iq4xs_b1_r1", "4bpw_b${B}_$D", "w4a16_b${B}_$D",
    "gptq_matched_b${B}_$D", "gptq_native_b${B}_$D", "4bpw_cu13_b${B}_r1",
    "k8_build_wikitext", "lic_auto_$SRC", "lic_1_$SRC", "nf4_$SRC", "nll_e4b_prefill_$SRC", "oracle_$SRC", "k19census_b1", "census_k8_lic_1_$SRC",
    "nll_vllm${3:+_fp8kv}_${MODE}_$SRC", "nll_sglang_$1_$2", "nll_llamacpp_$2_$3", "nll_exl3_$1_$2", "nll_lmdeploy_$1_$2",
    "ttft_e4b_$L", "ttft_vllm_$1", "ttft_sglang_$L", "ttft_llamacpp_$L", "ttft_exl3_$L", "ttft_lmdeploy_$L",
    "energy e4b_b$B e4b", "energy vllm_b$B vllm", "energy llamacpp_b1 llamacpp", "energy exl3_b1 exl3", "energy lmdeploy_b1 lmdeploy",
    "smoke_qwen3_fused_b$B", "smoke_qwen3_folds_b$B", "smoke_granite_b$B")


def test_every_v3_arm_name_appears_in_the_script():
    missing = [a for a in V3_ARMS if a not in RUN]
    assert not missing, missing
    assert RUN.count("vllm_arm gptq_graph_nodetok_b${B}_r1 nodetok") == 1 and not re.search(r"(?<!no)detok(?!eni)", RUN)
    assert '"nodetok"' in (LANE / "vllm" / "sc1_vllm_common.py").read_text()       # the driver's own arm name (8cb77d1)
    for mode in ("prefill", "decode", "decode_prefix", "served"):
        assert f" {mode} " in RUN or f"{mode}$" in RUN or f" {mode};" in RUN or f"{mode} $SRC" in RUN or f"M in prefill {mode}" in RUN, mode


def test_p88s_environment_and_command_lines_are_byte_identical():
    for line in ('FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"',
                 'SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"',
                 'LICENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 E4B_CALIB_NSEQ=$NSEQ $FOLDS"',
                 'K8ARGS="--placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext"',
                 "--batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed", "E4B_CALIB_LAYERS_PER_PASS=10 E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact",
                 'grep -qa "INT4EXP calibrated experts"', "-ge 1500 ]", "MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
                 'K8_MODEL="$M" K8_WORK="$W/work_$TAG"'.replace('"$M" K8_WORK="$W/work_$TAG"', '"$MID" K8_WORK="$W/work_qwen3"')[:len('K8_MODEL=')]):
        assert line in RUN, line
        assert line in P88, ("not P88's line", line)
    assert "GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4; GPTQ_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441" in RUN
    assert "GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213" in RUN            # v0.34.1's COMMIT (the tag object is e7ae8e2e)
    assert 'ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"' in RUN
    for fn, nxt in (("e4b_window", "e4b_k8"), ("e4b_k8", "k8_census"), ("e4b_build", "lic_ready")):
        assert "env $ROUTEENV " in _body(fn, nxt), fn                            # every e4b arm names the four route knobs
    for fn, nxt in (("sched_arm", "sched_smoke"), ("sched_smoke", "vllm_arm")):
        assert "env PYTHONPATH= $ROUTEENV " in _body(fn, nxt), fn               # ... the scheduler arms with the hook off (A4)
    assert 'e4b_window lic_degraded_b16_$D 16 "$LICENV $PACKENV" --fuse-qkv 1 E4B_INT4_GROUPED_SMALLM=0' in RUN
    assert "E4B_INT4_ARTIFACT_DIR=$W/artifact E4B_INT4_EXPECTED_FINGERPRINT=$FP" in RUN and 'echo "PACK $FP"' in RUN
    assert "test_k19_row_exact_gpu.py -q -p no:cacheprovider" in RUN and "finish 25" in RUN


def test_quiescence_gate_and_the_sched_arms_target_the_fixed_fusions():
    a, b, c = _body("box_a", "box_b"), _body("box_b", "box_c"), _body("box_c", "")
    assert a.index("quiesce build") < a.index("e4b_build") and a.index("quiesce arms") < a.index("arm_lic_window r1 16")
    assert b.index("quiesce arms") < b.index("arm_int4_window r1 $B") and c.index("quiesce arms") < c.index("arm_int4_window r1 $B")
    assert c.index("sglang_up matched; quiesce sglang") < c.index("arm_sglang_matched r1 $B")
    q = _body("quiesce", "sampler_start")
    assert "pip install|cmake|ninja|nvcc|git clone" in q and "l < h" in q and "nproc" in q and "cpu.max" in q
    # the sched engine env: MAX_SEQS=B, graphs, buckets, fuse_qkv=1 WITH the folds (the fixed _apply_fusions), the pack by fingerprint
    se = _body("sched_env", "sched_arm")
    for k in ("E4B_PAGED_MODEL", "E4B_PAGED_ARENA", "E4B_PAGED_CALIB", "E4B_PAGED_PLACEMENT=all-vram", "E4B_PAGED_MAX_SEQS=$B",
              "E4B_PAGED_MAX_TOKENS_PER_SEQ=$MTS", "E4B_PAGED_CHUNK_TOKENS", "E4B_PAGED_GRAPHS=1", "E4B_PAGED_BUCKETS=$BK", "E4B_PAGED_FUSE_QKV=$FUSE"):
        assert k in se, k
    assert 'sched_arm lic_sched_b${B}_$D $B "$LICENV $PACKENV" 2048 1 $W/prompts_b$B.json' in RUN       # fuse_qkv=1 + LICENV's $FOLDS
    assert 'sched_smoke qwen3_fused_b$B $B "$SPEEDENV" 1' in RUN and 'sched_smoke qwen3_folds_b$B $B "$SPEEDENV" 0' in RUN
    assert "2719171" in RUN and "2719171" in SCHED
    assert "--system-site-packages" in RUN and "site.ENABLE_USER_SITE" in RUN


def test_the_sampler_and_the_receipt_conventions():
    assert "timestamp,memory.used,utilization.gpu,clocks.sm,power.draw.instant,pcie.link.gen.current" in SAMPLER and "-lms 50" in SAMPLER
    assert "ps -eo pid=,rss=,pcpu=,comm= --sort=-rss" in SAMPLER
    for fn in ("e4b_window", "e4b_k8", "sched_arm", "vllm_arm", "sglang_arm", "llamacpp_arm", "exl3_arm", "lmdeploy_arm", "oracle"):
        body = RUN[RUN.index(f"{fn}(){{"):]
        body = body[:body.index("return $rc; }") + 13]
        assert "sampler_start" in body and "sampler_stop" in body and "stub $W/" in body and 'line "' in body, fn
    assert '"status": status, "reason": reason' in RUN and '"log_tail": tail' in RUN and "unsupported_$ENGINE.json" in RUN
    assert 'SC1_SHORT=32 SC1_LONG=1024 SC1_REPS=1' in RUN and "--energy" in RUN and "--sameprompt" in RUN and "--ttft" in RUN
    assert 'engine: "e4b-sched"' in SCHED.replace("ENGINE = \"e4b-sched\"", 'engine: "e4b-sched"')


def test_the_helpers_selftests_pass_on_cpu():
    for f in ("sc1_prompts.py", "sc1_e4b_sched.py"):
        r = subprocess.run([sys.executable, str(LANE / f), "--selftest"], capture_output=True, text=True)
        assert r.returncode == 0 and "selftest OK" in r.stdout, (f, r.stdout, r.stderr)


def test_receipt_names_match_the_reducers_vocabulary():
    """The reducer (bench/sc1/sc1_reduce.py, another builder's file) is written to a receipt vocabulary; the runner's names must parse
    under ITS regexes and name ITS anchors / arms, or every row lands UNREAD. Skipped until the reducer is on the branch."""
    import importlib.util
    path = LANE / "sc1_reduce.py"
    if not path.is_file():
        import pytest
        pytest.skip("sc1_reduce.py not integrated yet")
    spec = importlib.util.spec_from_file_location("sc1_reduce_under_test", path)
    red = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = red                                      # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(red)
    timed = ["e4b_lic_b16_r1.json", "e4b_lic_degraded_b16_r1.json", "e4b_nf4_ctrl_b1_r2.json", "e4b_rtn_b16_r3.json", "e4b_int4_b1_r1.json",
             "e4bsched_lic_sched_b1_r1.json", "e4bsched_lic_sched_sameprompt_b16_r1.json", "e4bsched_int4_sched_b16_r2.json",
             "vllm_gptq_graph_b16_r1.json", "vllm_gptq_fp8kv_b1_r2.json", "vllm_gptq_graph_nodetok_b1_r1.json",
             "vllm_gptq_graph_sameprompt_b16_r1.json", "sglang_gptq_matched_b16_r1.json", "sglang_gptq_native_b1_r1.json",
             "llamacpp_q4km_b16_r1.json", "llamacpp_iq4xs_b1_r1.json", "exl3_4bpw_b1_r1.json", "exl3_4bpw_cu13_b16_r1.json", "lmdeploy_w4a16_b16_r2.json"]
    for name in timed:
        m = red.ARM_RE.match(name)
        assert m, name
        engine, arm = m.group(1), m.group(2)
        assert engine in ("e4b", "e4bsched") or arm in set(red.MATCHED_ARMS.values()) | {"gptq_fp8kv", "gptq_graph_nodetok", "gptq_graph_sameprompt", "iq4xs"} \
            | set().union(*red.NATIVE_ARMS.values()), (engine, arm)
    for name in ("smoke_qwen3_fused_b1.json", "e4b_k19census_b1.json", "e4bsched_energy_b1.json", "vllm_energy_b16.json", "llamacpp_energy_q4km_b1.json"):
        assert not red.ARM_RE.match(name), name                     # not timed rows
    for name in ("nll_vllm_prefill_wikitext.json", "nll_vllm_served_c4val1.json", "nll_vllm_fp8kv_served_wikitext.json", "nll_e4b_prefill_c4val1.json",
                 "nll_sglang_served_wikitext.json", "nll_llamacpp_decode_c4val1.json", "nll_exl3_prefill_wikitext.json", "nll_lmdeploy_decode_prefix_wikitext.json"):
        assert red.NLL_RE.match(name), name
    for name in ("ttft_e4b_512.json", "ttft_vllm_4096.json", "ttft_sglang_512.json", "ttft_llamacpp_4096.json", "ttft_exl3_512.json", "ttft_lmdeploy_4096.json"):
        assert re.match(r"^ttft_([a-z0-9]+)_(512|4096)\.json$", name), name
    assert red.ANCHOR_SCHED == {"A": "lic_sched", "B": "int4_sched", "C": "int4_sched"} and red.ANCHOR_WINDOW == {"A": "lic", "B": "int4", "C": "int4"}
    for st in ("unsupported", "harness_error", "alarm", "not_run", "load_fault", "void", "ok"):
        assert st in red.STATUS_MAP, st                              # every status the runner's stubs write
    assert 'e4b_k8 lic_auto_$SRC' in RUN and "k8_{name}" in (LANE / "sc1_reduce.py").read_text()   # k8_<name>: the runner writes k8_lic_auto_<src> etc.
    assert 'sc1_reduce.py $W --box $BOX --out-dir $W' in RUN           # its CLI
    assert '"SC1_BOX": "%s"' in RUN                                    # box.json


def test_g2_droppables_in_the_registered_order_and_no_awq_arm():
    """v4 Phase C cuts the AWQ arm; G2's droppables run nodetok (both B) first, nf4_ctrl r2 second, the fp8-KV served rows last."""
    g2 = _body("box_a", "box_b")
    g2 = g2[g2.index('phase G2 '):]
    nodetok = g2.index('for B in 1 16; do can_run 900 vllm_gptq_graph_nodetok_b${B}_r1 && { arm_vllm_nodetok $B; rec $?; }; done')
    nf4 = g2.index('for B in 16 1; do can_run 900 e4b_nf4_ctrl_b${B}_r2 && { arm_nf4 r2 $B; rec $?; }; done')
    fp8q = g2.index('vllm_nll served $SRC fp8')
    assert g2.index('third e4b rtn_b$B') < nodetok < nf4 < fp8q < g2.index('reduce; }')
    assert not re.search(r"(?i)awq", RUN.replace("the AWQ arm is CUT from SC1 -- v4 Phase C", "")), "the AWQ arm is cut (v4 Phase C)"
    assert "awq" not in DRIVE.lower()



def _bash(script: str) -> str:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=ENV, check=False).stdout


def test_quiesce_busy_count_reads_0_when_nothing_matches():
    """sc1a-prove-3 (Amendment A2): `pgrep -fc ... || echo 0` read "0\\n0" on procps -- `pgrep -c` prints 0 AND exits 1 when
    nothing matches -- so quiesce could never succeed and every call burned SC1_QUIESCE_S. Run the script's OWN expression on
    a pattern that matches nothing (the [p] keeps the probe from matching its own `bash -c` command line) and require "0"."""
    m = re.search(r"busy=\$\((.*?)\); load1=", RUN)
    assert m, "quiesce's busy= expression moved"
    expr = m.group(1)
    assert "'pip install|cmake|ninja|nvcc|git clone'" in expr
    probe = expr.replace("'pip install|cmake|ninja|nvcc|git clone'", "'sc1-no-such-[p]rocess-7f3a9c'")
    assert _bash(f"set -uo pipefail; x=$({probe}); printf '%s' \"$x\"") == "0"
    # positive control for the trap itself: a counter that prints 0 AND exits 1 (procps pgrep -c on no match) under `|| echo 0`
    assert _bash("x=$( (echo 0; exit 1) || echo 0); printf '%s' \"$x\"") == "0\n0"
    for text in (RUN, DRIVE, SAMPLER):
        assert "|| echo 0" not in text


def _prove_block() -> str:
    return RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index(": > PROVED; finish 0")]


def _rec() -> str:
    m = re.search(r"rc_any=0; rec\(\)\{.*?\}", RUN)
    assert m, "rec() moved"
    return m.group(0)


def _run_prove_line(marker: str, stubs: str) -> str:
    line = next(ln for ln in _prove_block().splitlines() if marker in ln)
    return _bash(f"{_rec()}\nsay(){{ :; }}\n{stubs}\n{line}\nprintf '%s' \"$rc_any\"")


def test_the_proof_is_not_proved_when_a_smoke_is_skipped_or_fails():
    """sc1a-prove-3 (Amendment A2): both Granite smokes were skipped by can_run, rec was never called, rc_any stayed 0 and
    PROVED was written. The smoke loop, run with stubs: admitted + passing -> 0; a failing smoke keeps its rc; skipped -> 23."""
    loop = "for B in 1 16; do"
    base = 'GR_ENV=x GR=x GR_REV=x GA=x'
    assert _run_prove_line(loop, f"{base}; can_run(){{ return 0; }}; sched_smoke(){{ return 0; }}") == "0"
    assert _run_prove_line(loop, f"{base}; can_run(){{ return 0; }}; sched_smoke(){{ return 7; }}") == "7"
    assert _run_prove_line(loop, f"{base}; can_run(){{ return 1; }}; sched_smoke(){{ return 0; }}") == "23"


def test_the_proof_needs_every_comparator_the_box_installs_and_box_c_needs_the_jit():
    needs = "for E in $PROVE_NEEDS; do"
    have = 'have(){ case $1 in vllm) return 0;; *) return 1;; esac; }'
    assert _run_prove_line(needs, f"{have}; PROVE_NEEDS='vllm'") == "0"
    assert _run_prove_line(needs, f"{have}; PROVE_NEEDS='vllm exl3'") == "23"
    block = _prove_block()
    assert 'A) PROVE_NEEDS="vllm";;' in block and 'B) PROVE_NEEDS="vllm llamacpp exl3";;' in block   # A10: no LMDeploy on box B
    assert 'C) PROVE_NEEDS="vllm exl3 sglang";;' in block
    # the install dispatch installs exactly those sets (box C's SGLang only in the proof)
    assert "A) install_vllm ;;" in RUN and 'B) install_vllm; install_llamacpp; install_exl3 cu128; unsupported lmdeploy "$LMD_UNSUPPORTED" ;;' in RUN
    assert 'C) install_vllm; install_exl3 cu132; [ "$PROVE" = 1 ] && install_sglang ;;' in RUN
    assert "can_run 900 prove_sglang_jit" in block and 'if [ -z "${SC1_PROVE_SGLANG_MODEL:-}" ]' in block


def test_the_sched_arms_run_without_the_harness_hook():
    """sc1a-5090-1 (Amendment A4): the P42 hook (PYTHONPATH=$W/hook) applies the int4 levers right after the tier, and
    serve_paged.build_engine applies the SAME levers at the same point (its _apply_levers IS the hook's _apply_lanes); with
    both, the second enable refused ('E4B_SERVE_ATTN_INT4=1 matched no attention projections'). Every sc1_e4b_sched.py
    invocation clears PYTHONPATH; every step_decomp.py invocation (the harness path) keeps the hook."""
    lines = RUN.splitlines()
    sched = [i for i, ln in enumerate(lines) if "$W/sc1_e4b_sched.py" in ln and "perl -e" in ln]
    assert len(sched) == 2, sched
    for i in sched:
        env_line = lines[i - 1]
        assert env_line.lstrip().startswith("env PYTHONPATH= $ROUTEENV "), env_line
    for i, ln in enumerate(lines):
        if "$W/step_decomp.py" in ln and "exec @ARGV" in ln:
            ctx = lines[i - 1] + ln
            assert "PYTHONPATH=" not in ctx, ln
    assert "harness_hook_loaded()" in SCHED and "REFUSED: the P42 harness hook is loaded" in SCHED


def test_the_tripwire_requires_each_route_knob_read_not_its_default():
    """sc1b-prove-8 (Amendment A6): release 0.40.0 (#878) moved E4B_NF4_GROUPED_SMALLM's default 0 -> auto and the box
    tripwire, which asserted each knob's DEFAULT, refused. Every e4b arm pins all four knobs explicitly (ROUTEENV), so what
    the arms need is that the library READS each knob from the environment; the default is logged, not asserted. Run the
    tripwire's own knob loop on stand-in library sources."""
    lines = RUN.splitlines()
    a = next(i for i, ln in enumerate(lines) if ln.startswith("for knob in (") and "E4B_NF4_GROUPED_SMALLM" in ln)
    b = a + 1
    while b < len(lines) and lines[b].startswith("    "):
        b += 1
    loop = "\n".join(lines[a:b])
    def run(src):
        g = {"re": __import__("re"), "src": src}
        exec(loop, g)
    knobs = ("E4B_INT4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "E4B_MXFP4_GROUPED_SMALLM", "E4B_NF4_GROUPED_SMALLM")
    today = "\n".join(f'v = os.environ.get("{k}", "{"auto"}")' for k in knobs)              # main after #878
    before = today.replace('"E4B_NF4_GROUPED_SMALLM", "auto"', '"E4B_NF4_GROUPED_SMALLM", "0"')  # 0a2a0c8
    run(today)
    run(before)
    import pytest
    with pytest.raises(AssertionError):
        run(today.replace("E4B_NF4_GROUPED_SMALLM", "E4B_NF4_SOMETHING_ELSE"))             # a knob no longer read: refuse
    for k in knobs:
        assert f"{k}=" in RUN.split("ROUTEENV=", 1)[1].split("\n", 1)[0], k                 # ... and the arms pin all four


def test_box_c_installs_the_python_headers_triton_compiles_against():
    """sc1c-prove-12 (Amendment A7): both Granite smokes on box C's CUDA-13 image died in Triton's driver build -- gcc
    "fatal error: Python.h: No such file or directory" -- because apt installed python3 without python3-dev; the PyTorch
    images of boxes A and B ship the headers."""
    apt = next(ln for ln in RUN.splitlines() if "apt-get install" in ln)
    assert " python3-dev " in apt and " build-essential" in apt, apt
