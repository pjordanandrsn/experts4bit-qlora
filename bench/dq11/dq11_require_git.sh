#!/bin/bash
# The box provisions its own VCS tool before any wheel/model fetch.
set -euo pipefail
if command -v git >/dev/null 2>&1; then exit 0; fi
if ! command -v apt-get >/dev/null 2>&1 || ! command -v timeout >/dev/null 2>&1; then
  echo 'DQ11 git prerequisite unavailable: no bounded package installer'
  exit 20
fi
if ! timeout 120 apt-get update || ! timeout 120 env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git; then
  echo 'DQ11 git prerequisite installation failed'
  exit 20
fi
if ! command -v git >/dev/null 2>&1; then
  echo 'DQ11 git prerequisite still absent after installer'
  exit 20
fi
