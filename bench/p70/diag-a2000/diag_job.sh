#!/bin/bash
export PYTHONPATH=/w/pydeps:/w/src/e4b HF_HOME=/w/hf
python /w/p70diag.py
echo "DIAG_RC=$?"
