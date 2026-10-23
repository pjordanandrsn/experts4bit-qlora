"""Plumbing only: a receipt-shaped JSON whose data record is the demo's own, for testing d0_data.py end to end."""
import json
import sys

from transformers import AutoTokenizer
from loggetta.data import TrainingData, prepare_data

tok = AutoTokenizer.from_pretrained(sys.argv[1])
spec = TrainingData.from_dict({"source": "tatsu-lab/alpaca", "revision": "dce01c9b08f87459cf36a430d809084718273017",
                                 "format": "text", "text_field": "text"})
p = prepare_data(tok, int(sys.argv[3]), 512, spec, seed=42)
json.dump({"data": {**p.info, "options": spec.to_dict()}}, open(sys.argv[2], "w"), default=str)
print("examples_used", p.info["examples_used"], "tokens", p.info["tokens"])
