#!/bin/bash
# CI's lint-and-test job, reproduced on Linux/py3.11 with no GPU: gnf4 at the CI pin first, then e4b editable [test],
# the int4 tripwire, ruff (pinned), pytest tests/.
set -uo pipefail
cd /w/e4b
[ -x /w/venv-ci/bin/python ] || python -m venv --system-site-packages /w/venv-ci
P=/w/venv-ci/bin/python
$P -m pip install -q --cache-dir /w/pipcache "ruff==0.15.22" > /w/logs/ci_pip.log 2>&1
$P -m pip install -q --cache-dir /w/pipcache --force-reinstall --no-deps /w/gnf4 >> /w/logs/ci_pip.log 2>&1
$P -m pip install -q --cache-dir /w/pipcache -e ".[test]" >> /w/logs/ci_pip.log 2>&1 || { tail -5 /w/logs/ci_pip.log; exit 9; }
$P -c "import int4_b32, int4_pack_ref, nf4_grouped, gptq_pack; print('int4 tripwire ok')" || exit 9
$P -m ruff check experts4bit_qlora tests scripts tools; echo "ruff rc=$?"
$P -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -40
echo "pytest rc=${PIPESTATUS[0]}"
