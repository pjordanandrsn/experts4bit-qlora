#!/usr/bin/env python3
"""Five synthetic instruction datasets with deliberately DISPARATE structure.

Not a claim about real industry data -- the point is that the token statistics
differ sharply (prose vs numeric vs code vs list), so a training matrix over
them exercises different shapes rather than five flavours of the same text.
Seeded and shipped with the receipts, so every cell is reproducible.
"""
import hashlib
import json
import random
import sys

N_TRAIN, N_EVAL = 1200, 200


def legal(rng):
    party = rng.choice(["Licensor", "Vendor", "Contractor", "Discloser"])
    other = rng.choice(["Licensee", "Client", "Company", "Recipient"])
    days = rng.choice([15, 30, 45, 60, 90])
    cap = rng.choice(["fees paid in the preceding 12 months", "USD 50,000",
                      "the aggregate contract value", "direct damages only"])
    clause = (f"Section {rng.randint(2,14)}.{rng.randint(1,9)}. The {party} shall "
              f"indemnify the {other} against third-party claims arising from "
              f"breach of confidentiality, provided the {other} gives written "
              f"notice within {days} days. Liability is limited to {cap}.")
    out = (f"Obligated party: {party}\nTrigger: third-party claim from "
           f"confidentiality breach\nNotice window: {days} days (written)\n"
           f"Liability cap: {cap}\nSurvives termination: yes")
    return f"Extract the obligations from this clause.\n\n{clause}", out


def clinical(rng):
    age = rng.randint(19, 88)
    sex = rng.choice(["M", "F"])
    sym = rng.choice(["pleuritic chest pain", "sudden dyspnoea",
                      "unilateral calf swelling", "syncope on exertion"])
    hr = rng.randint(55, 138); spo2 = rng.randint(86, 99)
    temp = round(rng.uniform(36.1, 39.4), 1)
    note = (f"{age}{sex} presents with {sym} for {rng.randint(1,72)}h. "
            f"HR {hr}, SpO2 {spo2}%, T {temp}C. No trauma. "
            f"{'Recent long-haul flight. ' if rng.random()<0.4 else ''}"
            f"{'On combined oral contraceptive. ' if sex=='F' and rng.random()<0.3 else ''}")
    acuity = "immediate" if (spo2 < 92 or hr > 120) else "urgent" if hr > 100 else "routine"
    out = (f"acuity: {acuity}\nvitals_flagged: "
           f"{'SpO2' if spo2<92 else ''}{',' if spo2<92 and hr>120 else ''}"
           f"{'HR' if hr>120 else ''}{'none' if spo2>=92 and hr<=120 else ''}\n"
           f"primary_concern: {sym}\nnext_step: "
           f"{'resuscitation bay' if acuity=='immediate' else 'ED assessment' if acuity=='urgent' else 'ambulatory clinic'}")
    return f"Triage this note into structured fields.\n\n{note}", out


def finance(rng):
    amt = round(rng.uniform(12.5, 48000), 2)
    ch = rng.choice(["card-present", "card-not-present", "ACH", "wire", "SEPA"])
    ctry = rng.choice(["US", "GB", "NG", "SG", "RO", "BR"])
    hour = rng.randint(0, 23)
    prior = rng.randint(0, 400)
    score = 0
    score += 2 if amt > 10000 else 0
    score += 2 if ch in ("wire", "card-not-present") else 0
    score += 2 if ctry in ("NG", "RO", "BR") else 0
    score += 1 if hour < 5 else 0
    score += 1 if prior < 3 else 0
    band = "high" if score >= 5 else "medium" if score >= 3 else "low"
    tx = (f"amount={amt} channel={ch} country={ctry} local_hour={hour:02d} "
          f"prior_txn_count={prior} mcc={rng.choice([5411,7995,4829,5967,8999])}")
    out = (f"risk_band: {band}\nscore: {score}/8\ndrivers: "
           f"{'high amount; ' if amt>10000 else ''}{'channel ' + ch + '; ' if ch in ('wire','card-not-present') else ''}"
           f"{'geography ' + ctry + '; ' if ctry in ('NG','RO','BR') else ''}"
           f"{'off-hours; ' if hour<5 else ''}{'thin history' if prior<3 else 'established history'}\n"
           f"action: {'manual review' if band=='high' else 'step-up auth' if band=='medium' else 'approve'}")
    return f"Assess transaction risk.\n\n{tx}", out


def code(rng):
    # Entropy note: an earlier version drew only from a handful of templates
    # and could not produce 1400 DISJOINT rows -- the dedup loop spun forever.
    # Identifiers and literals now carry the entropy, so uniqueness is
    # reachable by construction.
    var = rng.choice(["total", "acc", "count", "buf", "agg", "sum_", "n_hit",
                      "tally", "carry", "seen"]) + str(rng.randint(0, 99))
    n = rng.choice(["items", "rows", "vals", "recs", "batch", "chunk",
                    "entries", "keys"]) + str(rng.randint(0, 99))
    bug, fix, why = rng.choice([
        (f"for i in range(len({n})):\n    {var} += {n}[i+1]",
         f"for i in range(len({n})):\n    {var} += {n}[i]",
         "off-by-one: i+1 reads past the last index on the final iteration"),
        (f"if {var} == None:\n    return 0",
         f"if {var} is None:\n    return 0",
         "identity comparison: use `is None`, not `== None`"),
        (f"{var} = []\nfor r in {n}:\n    {var}.append(r)\nreturn {var}[0]",
         f"{var} = [r for r in {n}]\nif not {var}:\n    return None\nreturn {var}[0]",
         "unguarded index: an empty input raises IndexError"),
        (f"d = {{}}\nfor k in {n}:\n    d[k] = d[k] + 1",
         f"d = {{}}\nfor k in {n}:\n    d[k] = d.get(k, 0) + 1",
         "KeyError on first sight of a key: initialise with .get"),
    ])
    return f"Find and fix the bug, then explain it.\n\n```python\n{bug}\n```", \
           f"```python\n{fix}\n```\nbug: {why}"


def support(rng):
    prod = rng.choice(["router", "thermostat", "camera", "doorbell", "hub",
                       "sensor", "smart plug", "light bridge"])
    issue = rng.choice(["will not pair", "drops offline nightly",
                        "firmware update fails at %d%%" % rng.randint(5, 95),
                        "app shows it as offline",
                        "reboots every %d minutes" % rng.randint(5, 180),
                        "LED stays %s" % rng.choice(["amber", "red", "off"])])
    tone = rng.choice(["", "This is the third time I've written. ",
                       "Really frustrated here. ", "Hoping you can help. ",
                       "Second ticket on this. "])
    model = "%s-%d" % (rng.choice(["AX", "GX", "NX", "PRO"]), rng.randint(100, 999))
    msg = (f"{tone}My {prod} ({model}) {issue}. I've already restarted it "
           f"{rng.choice(['twice','three times','several times'])} and "
           f"{rng.choice(['reseated the power','swapped the cable','moved it closer to the hub'])}.")
    out = ("1. Confirm 2.4 GHz SSID is broadcasting (device does not join 5 GHz only)\n"
           "2. Factory reset: hold the button 10s until the LED blinks amber\n"
           "3. Re-pair within 3 m of the access point\n"
           "4. If it fails again, collect the app diagnostic log and escalate to tier 2\n"
           f"tone: {'escalated' if tone else 'neutral'}")
    return f"Draft resolution steps for this ticket.\n\n{msg}", out


GENS = {"legal": legal, "clinical": clinical, "finance": finance,
        "code": code, "support": support}


def build(name, fn, seed):
    rng = random.Random(seed)
    seen, rows, tries = set(), [], 0
    target = N_TRAIN + N_EVAL
    while len(rows) < target:
        tries += 1
        if tries > 200 * target:
            raise SystemExit(
                f"{name}: only {len(rows)}/{target} unique rows after {tries} "
                "draws -- the generator's entropy is too low. Fix the generator; "
                "do not silently ship a short dataset.")
        ins, out = fn(rng)
        key = hashlib.sha256((ins + out).encode()).hexdigest()
        if key in seen:            # disjoint by construction
            continue
        seen.add(key)
        rows.append({"instruction": ins, "output": out})
    return {"name": name, "seed": seed,
            "train": rows[:N_TRAIN], "eval": rows[N_TRAIN:]}


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    manifest = {}
    for i, (name, fn) in enumerate(sorted(GENS.items())):
        d = build(name, fn, 1000 + i)
        path = f"{outdir}/ds_{name}.json"
        blob = json.dumps(d, indent=1, sort_keys=True)
        open(path, "w").write(blob)
        tr_chars = sum(len(r["instruction"]) + len(r["output"]) for r in d["train"])
        manifest[name] = {"sha256": hashlib.sha256(blob.encode()).hexdigest(),
                          "n_train": len(d["train"]), "n_eval": len(d["eval"]),
                          "mean_chars_per_example": round(tr_chars / len(d["train"]), 1),
                          "seed": d["seed"]}
        print(f"  {name:9s} {manifest[name]['n_train']}/{manifest[name]['n_eval']} "
              f"mean {manifest[name]['mean_chars_per_example']} chars  "
              f"sha {manifest[name]['sha256'][:12]}")
    json.dump(manifest, open(f"{outdir}/ds_manifest.json", "w"), indent=1)
    print("manifest written")
