#!/usr/bin/env python3
"""tp1_reduce.py (v2, the bundle's reducer) -- the parity table for lane tp1 in the row-status vocabulary of the
2026-09-05 14:45Z phase directive. The copy that ran on the box (v1, its table superseded by this one) is
logs/tp1_reduce.py; the verdict rule below is v1's, verbatim -- only the row classification and the matrix are new.

ROW STATUS, exactly one of: OK, REFUSED, HARNESS_ERROR, ALARM, OOM, NOT_RUN, EXPERIMENTAL.
PARITY VERDICT, a separate column, only for OK rows of an accelerated arm (fused / batched): PASS / FAIL / VOID;
the reference arm prints REF, attn_only prints "no pair".

Classification is MECHANICAL, from the lane's own artefacts, never from the absence of an error:
  summary.txt            `<fam>/<arm> rc=N [CELL ...]` -- one line per ATTEMPT (a tp1b re-run appends a second
                         line for the same arm); `<fam>: FETCH FAILED rc=N` -- a family's fetch failure.
  <fam>_train_<arm>.json the receipt of the LAST attempt: status ok / refused / alarm / oom / load_fault /
                         verify_failed; refusal stubs are written by the probing arm (D8), alarm stubs by tp1_run.sh.
  logs/outer.log         `arm <fam>/<arm>` start lines, `fetch <fam>` lines, the `tp1b:` marker (post-TP_DONE
                         re-runs: Granite's corrected-counter fused arm, amendment 3; the Gemma-4 / Mixtral redo,
                         amendment 4).
  logs/run_<fam>_<arm>[.attempt<k>].log  the per-attempt console; a failed attempt's traceback lives here.
  TP_DONE / TP2_DONE / BOX_REFUSED       marker files, reported when shipped.

Rules, in order, per attempt:
  rc == 0 and receipt.status == "ok" and arm == "mxfp4"   -> EXPERIMENTAL (never licensed; no verdict)
  rc == 0 and receipt.status == "ok"                      -> OK (verdict computed below)
  rc == 3 or receipt.status == "refused"                  -> REFUSED (reason: the enabler's own line)
  rc == 4 (C1 failed, receipt written)                    -> OK, verdict VOID (C1)
  rc == 5 or receipt.status == "oom"                      -> OOM
  rc == 142 or receipt.status == "alarm"                  -> ALARM
  rc in (6, 7) (load_fault / verify_failed)               -> NOT_RUN (the arm never trained; the receipt's reason)
  no receipt, traceback's innermost frame in tp1_train_smoke.py -> HARNESS_ERROR (the traceback's last line)
  no receipt, any other traceback                         -> NOT_RUN (uncaught exception rc=N: the last line)
  rc == 0 and no receipt                                  -> NOT_RUN ("rc=0 without a receipt": never inferred OK)
  no summary line, receipt.status == "refused"            -> REFUSED (a probe stub, D8)
  no summary line, the family's fetch failed              -> ALARM if that fetch rc was 142, else NOT_RUN
  no summary line, an `arm` start line in outer.log       -> NOT_RUN (started <ts>, no result at the snapshot)
  no summary line, nothing in outer.log                   -> NOT_RUN (not reached at the snapshot)
A row's rc comes from summary.txt; a receipt whose summary line is missing is flagged "rc unverified".

Verdict (v1, PREREG-flagship-matrix B2 / -model2 C2, P36 "Registered criteria"): VOID if init_sha != the
reference's, C1 not bit-exact / 0 bytes / empties > 0 / control silent (either arm), n_patched == 0, kernel
calls/step min < 2 * n_patched (batched.py's silent _PAD_WASTE_LIMIT fallback), or a different step count; else
PASS iff |loss_last_arm - loss_last_ref| <= 0.05 AND median_i |loss_i_arm - loss_i_ref| <= 0.05; else FAIL.
Cost (s/step, tok/s, peak GB, J/step) is reported, never gated. Eval deltas are printed labelled "not the band".

Amendment references are attached from evidence, so no attempt disappears behind a clean table:
  amendment 3: a HARNESS_ERROR whose traceback carries the late-binding counter's TypeError; a granite/fused
               attempt started after the `tp1b:` marker is "the corrected-counter re-run (the row that counts)".
  amendment 4: a gemma4 fetch failure by alarm; any gemma4 / mixtral attempt after the `tp1b:` marker is the redo.
  amendments 1 and 2 are lane-level (pre-flight floor and doubled fetch alarms: logs/tp1_run.sh vs
               logs/tp1_run.sh.pre-amend; the helper-fetch false start: logs/outer.attempt1.log) -- header lines.
stdlib only. Usage: tp1_reduce.py <dir> [--md out.md]
"""
import argparse
import glob
import json
import os
import re
import statistics

BAND = 0.05
FAMS = ["granite", "olmoe", "gptoss", "qwen3", "gemma4", "mixtral"]
NAMES = {"granite": "Granite-3.1-3B-A800M", "olmoe": "OLMoE-1B-7B-0924-Instruct",
         "gptoss": "gpt-oss-20b", "qwen3": "Qwen3-30B-A3B", "gemma4": "Gemma-4-26B-A4B-it",
         "mixtral": "Mixtral-8x7B-Instruct-v0.1"}
ARMS = {"granite": ["reference", "fused", "batched"], "olmoe": ["reference", "fused", "batched"],
        "gptoss": ["fused", "batched", "attn_only", "mxfp4"], "qwen3": ["reference", "fused", "batched"],
        "gemma4": ["reference", "fused", "batched"], "mixtral": ["reference", "fused", "batched", "resident_probe"]}
ACCEL = ("fused", "batched")
STATUSES = ("OK", "REFUSED", "HARNESS_ERROR", "ALARM", "OOM", "NOT_RUN", "EXPERIMENTAL")
RC_MEANING = {0: "ok", 3: "refused", 4: "C1 failed (receipt written)", 5: "oom", 6: "load_fault", 7: "verify_failed", 142: "alarm (SIGALRM)"}
TS = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] (.*)$")
SUMMARY_ARM = re.compile(r"(?<![\w/])(\w+)/(\w+) rc=(\d+)")   # finditer: an arm that wrote no CELL line leaves no newline, so
                                                                # the next arm's result lands on the same line (Granite fused + batched)
SUMMARY_FETCH = re.compile(r"^(\w+): FETCH FAILED rc=(\d+)")
FILE_LINE = re.compile(r'^\s*File "([^"]+)", line \d+')
EXC_LINE = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Interrupt|Exit|Killed)\b.*)$")


# ------------------------------------------------------------------------------------------ artefacts
def read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def load_receipts(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_train_*.json"))):
        base = os.path.basename(p)[:-5]
        fam, _, arm = base.partition("_train_")
        try:
            out.setdefault(fam, {})[arm] = json.load(open(p))
        except Exception as e:
            out.setdefault(fam, {})[arm] = {"fam": fam, "arm": arm, "status": "unreadable", "reason": str(e)}
    return out


def parse_summary(d):
    attempts, fetch_fail, other = {}, {}, []
    for ln in read_lines(os.path.join(d, "summary.txt")):
        hits = list(SUMMARY_ARM.finditer(ln))
        if hits:
            for i, m in enumerate(hits):
                end = hits[i + 1].start() if i + 1 < len(hits) else len(ln)
                attempts.setdefault((m.group(1), m.group(2)), []).append(
                    {"rc": int(m.group(3)), "rest": ln[m.end():end].strip(), "shared_line": len(hits) > 1})
            continue
        m = SUMMARY_FETCH.match(ln)
        if m:
            fetch_fail.setdefault(m.group(1), []).append(int(m.group(2)))
            continue
        if ln.strip():
            other.append(ln.strip())
    return attempts, fetch_fail, other


def parse_outer(d):
    """arm start lines, fetch lines, staged lines and the tp1b marker, each with its line index for ordering."""
    starts, fetches, staged, tp1b_idx, done = {}, {}, {}, None, []
    cur = None  # the most recent arm start line; harness stdout is indented four spaces (LOAD/CELL/traceback lines)
    for i, ln in enumerate(read_lines(os.path.join(d, "logs", "outer.log"))):
        m = TS.match(ln)
        body = m.group(2) if m else ln
        ts = m.group(1) if m else None
        if body.startswith("tp1b:"):
            if tp1b_idx is None:
                tp1b_idx = i
            cur = None
        mm = re.match(r"^arm (\w+)/(\w+) ", body)
        if mm:
            cur = {"idx": i, "ts": ts, "after_tp1b": tp1b_idx is not None, "ran": False}
            starts.setdefault((mm.group(1), mm.group(2)), []).append(cur)
            continue
        mm = re.match(r"^fetch (\w+) ", body)
        if mm:
            fetches.setdefault(mm.group(1), []).append({"idx": i, "ts": ts, "after_tp1b": tp1b_idx is not None})
            continue
        if body.startswith("staged in ") and fetches:
            last_fam = max(fetches, key=lambda f: fetches[f][-1]["idx"])
            staged.setdefault(last_fam, []).append({"idx": i, "line": body})
            continue
        if re.match(r"^(TP_DONE|TP1B DONE|BOX_REFUSED|[A-Z0-9]+ DONE|GRANITE FUSED RERUN DONE)", body):
            done.append((ts, body))
            continue
        if cur is not None and ln.strip() and not (m or re.match(r"^\S+: line \d+: ", ln) or re.match(r"^\d+ MiB$", ln.strip())):
            cur["ran"] = True  # something other than the launcher's own lines followed this start: the harness ran
    return {"starts": starts, "fetches": fetches, "staged": staged, "tp1b_idx": tp1b_idx, "done": done}


def attempt_log(d, fam, arm, k, n):
    p = os.path.join(d, "logs", f"run_{fam}_{arm}.attempt{k}.log")
    if os.path.exists(p):
        return p
    if k == n:
        p = os.path.join(d, "logs", f"run_{fam}_{arm}.log")
        return p if os.path.exists(p) else None
    return None


def traceback_tail(path):
    """(innermost File path, exception line) of the LAST traceback in a run log; (None, last error-ish line) otherwise."""
    lines = read_lines(path) if path else []
    inner, exc, in_tb = None, None, False
    for ln in lines:
        if ln.startswith("Traceback (most recent call last)"):
            in_tb, inner, exc = True, None, None
            continue
        if in_tb:
            m = FILE_LINE.match(ln)
            if m:
                inner = m.group(1)
                continue
            m = EXC_LINE.match(ln)
            if m and not ln.startswith(" "):
                exc = m.group(1)
                in_tb = False
    if exc is None:
        errs = [ln for ln in lines if re.search(r"\b(Error|error:|Killed|OutOfMemory)\b", ln)]
        exc = errs[-1].strip() if errs else None
    return inner, exc


# ------------------------------------------------------------------------------------------ classification
def c1_ok(r):
    return bool(r.get("C1_bit_exact")) and r.get("C1_bytes_hashed", 0) > 0 \
        and r.get("C1_empties_skipped", 1) == 0 and bool(r.get("C1_control_detects_flipped_byte"))


def verdict(ref, arm):
    """(verdict, d_final, med, why) in the registered units for an OK accelerated row. Never touches eval loss."""
    if ref is None or ref.get("status") != "ok":
        return "VOID", None, None, "no OK reference arm to read against"
    why = []
    if arm.get("init_sha") != ref.get("init_sha"):
        why.append("init_sha differs from reference (arms did not start identical)")
    for tag, r in (("ref", ref), ("arm", arm)):
        if not c1_ok(r):
            why.append(f"C1 not clean on {tag}")
    if arm.get("n_patched", 0) == 0:
        why.append("n_patched == 0")
    need = 2 * arm.get("n_patched", 0)
    if arm.get("kernel_calls_per_step_min", 0) < need:
        why.append(f"kernel calls/step min {arm.get('kernel_calls_per_step_min')} < 2*n_patched={need} (not engaged on every layer)")
    if len(arm.get("losses", [])) != len(ref.get("losses", [])) or not arm.get("losses"):
        why.append("step counts differ")
    if why:
        return "VOID", None, None, "; ".join(why)
    d_final = abs(arm["loss_last"] - ref["loss_last"])
    med = statistics.median(abs(x - y) for x, y in zip(arm["losses"], ref["losses"]))
    return ("PASS" if (d_final <= BAND and med <= BAND) else "FAIL"), d_final, med, ""


def classify(d, fam, arm, attempts, fetch_fail, outer, receipt):
    """One dict per attempt: status, reason, rc, receipt (last attempt only), log path, amendment tag."""
    rows = []
    starts = outer["starts"].get((fam, arm), [])
    n = len(attempts)
    # A start line without harness output after it is a launcher abort before the harness ran (tp1b's first start died
    # on an unset shell variable). Result lines pair with the start lines that RAN, in order; the aborts are surfaced on
    # the attempt that followed them (v2.1 -- v2 paired results with the last n starts, which mislabelled a real first
    # attempt as the abort when the abort sat between two real attempts). Never drop a start line.
    ran = [x for x in starts if x.get("ran")]
    aborts = [x for x in starts if not x.get("ran")]
    align_flag = None
    if n and len(ran) == n:
        aligned, extra = ran, aborts
    else:  # counts disagree: fall back to positional pairing and say so on every row
        extra = starts[:-n] if n and len(starts) > n else ([] if n else starts)
        aligned = starts[-n:] if n and len(starts) >= n else starts
        if n and starts:
            align_flag = f"start lines paired positionally ({len(ran)} ran, {len(aborts)} without harness output, {n} result lines)"
    for k, att in enumerate(attempts, 1):
        rc = att["rc"]
        rec = receipt if k == n else None
        log = attempt_log(d, fam, arm, k, n)
        start = aligned[k - 1] if k - 1 < len(aligned) else None
        st = rec.get("status") if rec else None
        row = {"attempt": k, "of": n, "rc": rc, "receipt": rec, "log": log, "start": start, "flags": []}
        if att.get("shared_line") and not att["rest"]:
            row["flags"].append("no CELL line in summary.txt (the next arm's result shares its line)")
        if rc == 0 and st == "ok" and arm == "mxfp4":
            row.update(status="EXPERIMENTAL", reason="grouped-nf4-gemm run_mxfp4_20b_qlora on its own text; never licensed")
        elif rc == 0 and st == "ok":
            row.update(status="OK", reason="")
        elif rc == 3 or st == "refused":
            row.update(status="REFUSED", reason=(rec or {}).get("reason", "") or att["rest"][:160])
        elif rc == 4:
            row.update(status="OK", reason="exit 4: C1 failed, receipt written -- verdict VOID (C1)")
        elif rc == 5 or st == "oom":
            row.update(status="OOM", reason=(rec or {}).get("reason", "") or "exit 5 (OOM)")
        elif rc == 142 or st == "alarm":
            row.update(status="ALARM", reason=(rec or {}).get("reason", "") or "exit 142 (SIGALRM)")
        elif rc in (6, 7) or st in ("load_fault", "verify_failed"):
            row.update(status="NOT_RUN", reason=f"{st or RC_MEANING.get(rc)} (exit {rc}): {(rec or {}).get('reason', '')}".strip())
        elif rec is None:
            inner, exc = traceback_tail(log)
            if inner and os.path.basename(inner) == "tp1_train_smoke.py":
                row.update(status="HARNESS_ERROR", reason=f"rc={rc}; {exc or 'traceback in the harness'} (innermost frame {os.path.basename(inner)}, {os.path.basename(log)})")
            elif rc == 0:
                row.update(status="NOT_RUN", reason="rc=0 without a receipt -- success is never inferred from a missing error")
            else:
                row.update(status="NOT_RUN", reason=f"uncaught exception rc={rc}: {exc or 'see the run log'}" + (f" (innermost frame {os.path.basename(inner)})" if inner else ""))
        else:
            row.update(status="NOT_RUN", reason=f"rc={rc} ({RC_MEANING.get(rc, 'unknown')}) with receipt status {st!r}: unreadable combination")
            row["flags"].append("unreadable rc/status combination")
        mine = [x for x in extra if start is None or (x["idx"] < start["idx"] and (k == 1 or x["idx"] > aligned[k - 2]["idx"]))]
        if k == n:
            mine = mine or ([x for x in extra if start is not None and x["idx"] > start["idx"]] if not any(
                x["idx"] < start["idx"] for x in extra) else mine) if start is not None else extra
        if mine:
            row["flags"].append(f"{len(mine)} earlier start line(s) without a result line at "
                                + ", ".join(x["ts"] or "?" for x in mine) + " (a launcher abort before the harness ran; outer.log)")
        if align_flag:
            row["flags"].append(align_flag)
        rows.append(row)
    if not attempts:
        row = {"attempt": 1, "of": 1, "rc": None, "receipt": receipt, "log": attempt_log(d, fam, arm, 1, 1), "start": starts[0] if starts else None, "flags": []}
        st = receipt.get("status") if receipt else None
        if st == "refused":
            row.update(status="REFUSED", reason=receipt.get("reason", "") + " (stub written by the probing arm, D8)")
        elif st == "ok":
            row.update(status="EXPERIMENTAL" if arm == "mxfp4" else "OK", reason="")
            row["flags"].append("rc unverified: no summary line for this arm")
        elif st == "alarm":
            row.update(status="ALARM", reason=receipt.get("reason", ""))
        elif st == "oom":
            row.update(status="OOM", reason=receipt.get("reason", ""))
        elif st in ("load_fault", "verify_failed"):
            row.update(status="NOT_RUN", reason=f"{st}: {receipt.get('reason', '')}")
        elif fam in fetch_fail and not outer["staged"].get(fam):
            rcs = fetch_fail[fam]
            row["cause"] = "fetch_fail"
            if 142 in rcs:
                row.update(status="ALARM", reason="the family's fetch died by its alarm (FETCH FAILED rc=142); this arm never started")
            else:
                row.update(status="NOT_RUN", reason=f"the family's fetch failed (rc={rcs[-1]}); this arm never started")
        elif starts:
            row.update(status="NOT_RUN", reason=f"started {starts[-1]['ts']} (outer.log), no result line at the snapshot -- in progress or lost")
        elif outer["fetches"].get(fam) and not outer["staged"].get(fam):
            row.update(status="NOT_RUN", reason=f"the family's fetch started {outer['fetches'][fam][-1]['ts']} and had not staged at the snapshot")
        elif outer["staged"].get(fam):
            row.update(status="NOT_RUN", reason="not started at the snapshot (the checkpoint had staged; an earlier arm of the family was still running)")
        else:
            row.update(status="NOT_RUN", reason="not reached at the snapshot (no fetch or arm line in outer.log)")
        rows.append(row)
    for row in rows:
        row["amend"] = amendment_tag(fam, arm, row, fetch_fail)
        assert row["status"] in STATUSES, row["status"]
    return rows


def amendment_tag(fam, arm, row, fetch_fail):
    after = bool(row.get("start") and row["start"]["after_tp1b"])
    if row["status"] == "HARNESS_ERROR" and "_dequant_whole() got an unexpected keyword argument" in (row.get("reason") or ""):
        return "amendment 3 (late-binding counter closure in the harness; patched in flight; re-run by tp1b)"
    if fam == "granite" and arm == "fused" and after:
        return "amendment 3: the corrected-counter re-run (tp1b) -- the row that counts"
    if fam == "gemma4" and row.get("cause") == "fetch_fail":
        return "amendment 4 (fetch hang left to its alarm; conditional redo by tp1b)"
    if fam in ("gemma4", "mixtral") and after:
        return "amendment 4: the tp1b redo"
    return ""


# ------------------------------------------------------------------------------------------ rendering
def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ratio(num, den):
    try:
        return num / den if den else None
    except Exception:
        return None


def family_block(d, fam, recs, attempts, fetch_fail, outer, rows_out):
    lines = [f"\n### {NAMES.get(fam, fam)} (`{fam}`)"]
    ld = recs.get("load") or next((r for a, r in recs.items() if r.get("status") == "ok"), None)
    if ld:
        g = ld.get("geometry", {})
        lines.append(f"- load: status={ld.get('status')} model_type={ld.get('model_type')} load_s={fmt(ld.get('load_s'),1)} "
                     f"loaded_gb={fmt(ld.get('loaded_gb'))} verify={ld.get('verify')} wrapped={ld.get('n_lora_wrapped')} "
                     f"bare={ld.get('n_bare_experts')} classes={ld.get('expert_classes')} offload={ld.get('offload')} "
                     f"geometry(H/I/L/E/k)={g.get('hidden_size')}/{g.get('moe_intermediate_size') or g.get('intermediate_size')}/"
                     f"{g.get('num_hidden_layers')}/{g.get('num_experts') or g.get('num_local_experts')}/{g.get('num_experts_per_tok')}")
        env = ld.get("env", {})
        if env:
            lines.append(f"- env: e4b {env.get('e4b')} gnf4 {env.get('gnf4')} torch {env.get('torch')} transformers {env.get('transformers')} "
                         f"gpu {env.get('gpu')} box_class {env.get('box_class')} host_mem_gib {env.get('host', {}).get('host_mem_gib')}")
    else:
        lines.append("- load: (no receipt at this snapshot)")
    if fam in fetch_fail:
        lines.append(f"- fetch: FETCH FAILED rc={fetch_fail[fam]} (summary.txt)" + ("; a later fetch staged" if outer["staged"].get(fam) else "; no later staging at this snapshot"))
    ref = recs.get("reference") if (recs.get("reference") or {}).get("status") == "ok" else None
    hdr = ("| arm | attempt | status | verdict | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | "
           "loss first→last | eval 0→final (not the band) | Δ final train | median step \\|Δ\\| | reason / amendment |")
    lines.append(hdr)
    lines.append("|" + "---|" * 17)
    arms = ARMS.get(fam, [])
    for arm in [a for a in arms if a != "resident_probe" or a in recs or (fam, a) in attempts]:
        rows = classify(d, fam, arm, attempts.get((fam, arm), []), fetch_fail, outer, recs.get(arm))
        for row in rows:
            r = row["receipt"] or {}
            v, d_final, med, why = ("—", None, None, "")
            if row["status"] == "OK":
                if arm in ACCEL:
                    v, d_final, med, why = verdict(ref, r)
                    if row["rc"] == 4:
                        v, why = "VOID", (why + "; " if why else "") + "C1 failed (exit 4)"
                elif arm == "reference":
                    v = "REF"
                else:
                    v = "no pair"
            cost, ev_d = "", ""
            if row["status"] in ("OK",) and ref and arm != "reference":
                sp, pk, jj = r.get("s_per_step"), r.get("peak_vram_gb"), r.get("joules_per_step")
                cost = (f" (×{fmt(ratio(ref.get('s_per_step'), sp),2)} faster, peak ×{fmt(ratio(pk, ref.get('peak_vram_gb')),3)}"
                        + (f", J ×{fmt(ratio(jj, ref.get('joules_per_step')),3)}" if jj and ref.get('joules_per_step') else "") + ")")
                if ref.get("eval_loss_final") is not None and r.get("eval_loss_final") is not None:
                    ev_d = f" (Δ vs ref {r['eval_loss_final'] - ref['eval_loss_final']:+.4f})"
            need = 2 * r.get("n_patched", 0) if arm in ACCEL else 0
            c1 = "ok" if (r and c1_ok(r)) else ("**FAIL**" if r.get("status") == "ok" else "—")
            reason = row["reason"] or ""
            if why:
                reason = (reason + "; " if reason else "") + why
            if row["flags"]:
                reason = (reason + "; " if reason else "") + "; ".join(row["flags"])
            if row["amend"]:
                reason = (reason + " — " if reason else "") + row["amend"]
            if row["status"] == "EXPERIMENTAL":
                art = r.get("artifact", {}) or {}
                can, prov, ev = art.get("canary", {}) or {}, art.get("provenance", {}) or {}, art.get("eval_loss", {}) or {}
                evk = sorted(ev, key=lambda k: int(k))
                lines.append(f"| mxfp4 (gnf4 ExpertsMxfp4LoRA, own text) | {row['attempt']}/{row['of']} | **EXPERIMENTAL** | — | — | — | "
                             f"prov pre==post: {prov.get('pre_equals_post')} | {len(r.get('losses', []))} | "
                             f"{fmt(statistics.mean(r['step_s']) if r.get('step_s') else None)} | — | {fmt(r.get('peak_vram_gb'))} | — | "
                             f"{fmt(r['losses'][0] if r.get('losses') else None)}→{fmt(r['losses'][-1] if r.get('losses') else None)} | "
                             f"{fmt(ev.get(evk[0]) if evk else None)}→{fmt(ev.get(evk[-1]) if evk else None)} | — | — | "
                             f"NOT LICENSED; canary top1={fmt(can.get('top1'))} kl={fmt(can.get('kl'),5)}; {reason} |")
            else:
                lines.append(f"| {arm} | {row['attempt']}/{row['of']} | **{row['status']}** | {('**' + v + '**') if v not in ('—',) else v}{cost} | "
                             f"{r.get('n_patched', '—') if r else '—'} | {r.get('kernel_calls_per_step_min', '—') if r else '—'} ({need}) | {c1} | "
                             f"{(len(r.get('losses', [])) or r.get('steps_done', 0)) if r else '—'} | {fmt(r.get('s_per_step'), 3)} | {fmt(r.get('tokens_per_s'), 1)} | "
                             f"{fmt(r.get('peak_vram_gb'))} | {fmt(r.get('joules_per_step'), 1)} | "
                             f"{fmt(r.get('loss_first'), 4)}→{fmt(r.get('loss_last'), 4)} | {fmt(r.get('eval_loss_step0'), 4)}→{fmt(r.get('eval_loss_final'), 4)}{ev_d} | "
                             f"{fmt(d_final, 5)} | {fmt(med, 5)} | {reason or '—'} |")
            rows_out.setdefault(fam, {}).setdefault(arm, []).append({"row": row, "verdict": v, "d_final": d_final, "med": med, "receipt": r})
    if ref:
        lines.append(f"- init_sha (reference) `{ref.get('init_sha', '')[:16]}`; dataset `{ref.get('dataset', {}).get('path')}` sha "
                     f"`{ref.get('dataset', {}).get('sha256', '')[:12]}`; trainable {ref.get('trainable_params')} params in {ref.get('trainable_tensors')} tensors")
    return lines


def support(fam, arm, rows_out):
    """The capability-vocabulary reading of an arm's LAST attempt: supported / void / fail / refused / harness_error /
    alarm / oom / not_tested / experimental."""
    rs = rows_out.get(fam, {}).get(arm)
    if not rs:
        return "not_tested (no row)", None
    last = rs[-1]
    st, v = last["row"]["status"], last["verdict"]
    reason = last["row"]["reason"]
    if st == "OK":
        if arm in ACCEL:
            return ({"PASS": "supported", "VOID": "void", "FAIL": "fail"}.get(v, v.lower()) + f" ({v})"), reason
        return "supported (OK)", reason
    return {"REFUSED": "refused", "HARNESS_ERROR": "harness_error", "ALARM": "alarm", "OOM": "oom",
            "NOT_RUN": "not_tested", "EXPERIMENTAL": "experimental"}[st] + (f" — {reason}" if reason else ""), reason


def matrix(rows_out, recs_all):
    lines = ["\n## Per-family matrix (the directive's ten columns; support words in the capability vocabulary, read from the LAST attempt of each arm)",
             "| family | reference support | fused support | batched support | native-format route | loss-parity result | s/step (ref / fused / batched) | peak GB (ref / fused / batched) | tok/s (ref / fused / batched) | evidence tier | limitations / refusal reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        fr = rows_out.get(fam, {})
        recs = recs_all.get(fam, {})
        def cell(arm):
            s, _ = support(fam, arm, rows_out)
            return s
        def last_receipt(arm):
            rs = fr.get(arm)
            return (rs[-1]["receipt"] if rs and rs[-1]["row"]["status"] == "OK" else {}) if rs else {}
        if fam == "gptoss":
            ao = fr.get("attn_only", [{}])[-1]
            ref_s = ("refused — the loader builds the experts bare (no ExpertsLoRA); attention-only QLoRA over them: attn_only "
                     + (ao.get("row", {}).get("status", "NOT_RUN") if ao else "NOT_RUN") + " (no pair)")
            native = cell("mxfp4")
        else:
            ref_s = cell("reference")
            native = "n/a (NF4 checkpoint)"
        parity = []
        for arm in ACCEL:
            rs = fr.get(arm)
            if rs and rs[-1]["row"]["status"] == "OK":
                parity.append(f"{arm} {rs[-1]['verdict']}" + (f" {fmt(rs[-1]['d_final'],4)} / {fmt(rs[-1]['med'],4)}" if rs[-1]["d_final"] is not None else ""))
            elif rs:
                parity.append(f"{arm} — ({rs[-1]['row']['status']})")
        r_ref, r_f, r_b = last_receipt("reference") if fam != "gptoss" else last_receipt("attn_only"), last_receipt("fused"), last_receipt("batched")
        sps = " / ".join(fmt(x.get("s_per_step"), 3) if x else "—" for x in (r_ref, r_f, r_b))
        pks = " / ".join(fmt(x.get("peak_vram_gb"), 3) if x else "—" for x in (r_ref, r_f, r_b))
        tks = " / ".join(fmt(x.get("tokens_per_s"), 1) if x else "—" for x in (r_ref, r_f, r_b))
        if fam == "gptoss":
            sps += " (ref = attn_only)"
        have = [a for a, rs in fr.items() if rs[-1]["row"]["receipt"]]
        tier = ("measured (receipts in this directory: " + ", ".join(have) + ")") if have else "pending (no receipt at this snapshot)"
        if "mxfp4" in have:
            tier += "; mxfp4 experimental, not licensed"
        lims = []
        for arm, rs in fr.items():
            last = rs[-1]["row"]
            if last["status"] != "OK" or (rs[-1]["verdict"] in ("VOID", "FAIL")):
                lims.append(f"{arm}: {last['status']}" + (f"/{rs[-1]['verdict']}" if last["status"] == "OK" else "") + (f" — {last['reason']}" if last["reason"] else "") + (f" [{last['amend']}]" if last["amend"] else ""))
            if len(rs) > 1:
                lims.append(f"{arm}: {len(rs)} attempts (" + "; ".join(f"{x['row']['attempt']}: {x['row']['status']}" + (f" [{x['row']['amend']}]" if x['row']['amend'] else "") for x in rs) + ")")
        lines.append(f"| {NAMES[fam]} | {ref_s} | {cell('fused')} | {cell('batched')} | {native} | {'; '.join(parity) or '—'} | {sps} | {pks} | {tks} | {tier} | {'; '.join(lims) or '—'} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    d = a.dir
    recs_all = load_receipts(d)
    attempts, fetch_fail, other = parse_summary(d)
    outer = parse_outer(d)
    anchor = {}
    try:
        anchor = json.load(open(os.path.join(d, "anchor.json")))
    except Exception:
        pass
    markers = [m for m in ("TP_DONE", "TP2_DONE", "BOX_REFUSED") if os.path.exists(os.path.join(d, m))]
    out = [f"# tp1 — training parity table (`{os.path.basename(os.path.abspath(d))}`, reducer v2)",
           "Row status ∈ {OK, REFUSED, HARNESS_ERROR, ALARM, OOM, NOT_RUN, EXPERIMENTAL}, classified from summary.txt (rc per attempt), the "
           "receipt / stub, outer.log and the run logs — never from a missing error. Verdict (OK accelerated rows only): PASS iff |Δ final "
           f"train loss| ≤ {BAND} AND median step-wise |Δ| ≤ {BAND} vs the family's reference arm (PREREG-flagship-matrix B2 / -model2 C2); "
           "VOID when the arm cannot be read (init_sha, C1, n_patched, kernel engagement, steps). Cost is reported, not gated. Eval deltas "
           "are not the band. No cross-family, cross-lane or cross-card ratio appears here.",
           f"- lane header (summary.txt): {' · '.join(other) if other else '(none)'}",
           f"- anchor.json: {anchor.get('status')} class per summary; flops median {anchor.get('flops', {}).get('tflops', {}).get('median')} TFLOP/s, "
           f"launch {anchor.get('launch', {}).get('launches_per_s', {}).get('median')} /s, h2d {anchor.get('h2d', {}).get('h2d_gb_s', {}).get('median')} GB/s",
           f"- markers shipped: {', '.join(markers) or 'none'}; tp1b marker in outer.log: {'yes (line %d)' % outer['tp1b_idx'] if outer['tp1b_idx'] is not None else 'no'}; "
           f"lane end lines: {', '.join((ts + ' ' if ts else '') + b for ts, b in outer['done']) or 'none at this snapshot'}",
           "- lane-level amendments: 1 (pre-flight floor 40 MB/s, fetch alarms doubled — logs/tp1_run.sh vs logs/tp1_run.sh.pre-amend"
           + (", both shipped" if os.path.exists(os.path.join(d, "logs", "tp1_run.sh.pre-amend")) else "") + "); 2 (helper fetch by archive tarball after the "
           "fetch-by-sha false start — logs/outer.attempt1.log" + (" shipped" if os.path.exists(os.path.join(d, "logs", "outer.attempt1.log")) else " absent") + "); "
           "3 and 4 are attached to their rows below."]
    rows_out = {}
    for fam in FAMS:
        if fam in recs_all or any(k[0] == fam for k in attempts) or fam in fetch_fail or any(k[0] == fam for k in outer["starts"]) or fam in outer["fetches"]:
            out += family_block(d, fam, recs_all.get(fam, {}), attempts, fetch_fail, outer, rows_out)
        else:
            out += [f"\n### {NAMES[fam]} (`{fam}`)", "- load: (no receipt at this snapshot)",
                    "| arm | attempt | status | verdict | reason / amendment |", "|---|---|---|---|---|"]
            for arm in ARMS[fam]:
                if arm == "resident_probe":
                    continue
                rows = classify(d, fam, arm, [], fetch_fail, outer, None)
                for row in rows:
                    out.append(f"| {arm} | {row['attempt']}/{row['of']} | **{row['status']}** | — | {row['reason']}{(' — ' + row['amend']) if row['amend'] else ''} |")
                    rows_out.setdefault(fam, {}).setdefault(arm, []).append({"row": row, "verdict": "—", "d_final": None, "med": None, "receipt": {}})
    out += matrix(rows_out, recs_all)
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
