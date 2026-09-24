#!/bin/bash
# CI's lint-and-test job on Linux/py3.11 against the REAL repository (.git shipped, as actions/checkout gives CI), with
# git installed -- the image lacks it, and two test files shell out to git. First those two files alone, then the lot.
set -uo pipefail
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq git >/dev/null 2>&1; git --version
git config --global --add safe.directory '*'
cd /w/e4b; git log --oneline -1; git status --short | head -3; echo "(status above)"
P=/w/venv-ci/bin/python
$P -m pip install -q --cache-dir /w/pipcache --force-reinstall --no-deps /w/gnf4 > /w/logs/ci_git_pip.log 2>&1
$P -m pip install -q --cache-dir /w/pipcache -e ".[test]" >> /w/logs/ci_git_pip.log 2>&1 || { tail -5 /w/logs/ci_git_pip.log; exit 9; }
$P -c "import int4_b32, int4_pack_ref, nf4_grouped, gptq_pack; print('int4 tripwire ok')" || exit 9
$P -m ruff check experts4bit_qlora tests scripts tools; echo "ruff rc=$?"
echo "== the two git-dependent files"; $P -m pytest tests/test_check_claims_register.py tests/test_readability_checks.py -q -p no:cacheprovider 2>&1 | tail -3
echo "== full suite"; $P -m pytest tests/ -q -p no:cacheprovider -rfE 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | tail -30
echo "pytest rc=${PIPESTATUS[0]}"
