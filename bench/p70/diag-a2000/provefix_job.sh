#!/bin/bash
export PYTHONPATH=/w/pydeps:/w/src/e4b HF_HOME=/w/hf
cd /w/provefix && python -u kl_router.py --prove-cast --out prove_cast.json
echo "PROVE_RC=$?"
