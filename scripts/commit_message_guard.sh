#!/usr/bin/env bash
# commit_message_guard.sh: refuses private control-plane references in the text a squash merge publishes.
#
# The squash commit on main is built from the PR title and the branch's commit messages, and both are public the
# moment they land. private-marker-guard scans the tree but never reads a message; this scans the messages, the PR
# title and body, and the lines the change adds (for the private-name rule only, which the tree guard can't carry).
#
# RULES
#   public   every pattern in .github/private-markers.txt (plain lines and #added-only rules) plus the bus message-id
#            shape below. The same extended regexes, the same case-insensitive grep, as the tree guard.
#   private  one extended regex from the COMMIT_GUARD_PRIVATE_RE secret (names that must not appear, kept out of
#            the public tree). Applied to messages, title, body and added lines. Never printed.
#
# A line carrying `guard-allow` is exempt, as in the tree guard. Prefer rewording.
# Hits are reported by location and rule class only; the matched text is never echoed into the (public) log.
#
# Usage:
#   commit_message_guard.sh --self-test
#   BASE_SHA=... HEAD_SHA=... PR_TITLE=... PR_BODY=... [COMMIT_GUARD_PRIVATE_RE=...] commit_message_guard.sh
# SAME_REPO=0 (a fork PR, which receives no secrets) runs the public rules alone and says so; on a same-repo PR a
# missing secret is an error, so the private rule can't go silently toothless.
set -euo pipefail

MARKERS=.github/private-markers.txt
BUS_ID='[0-9]{8}T[0-9]{6}Z-[0-9]+'
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

build_public_patterns() {
  { grep -vE '^[[:space:]]*(#|$)' "$MARKERS"
    grep -E '^#added-only: ' "$MARKERS" | sed 's/^#added-only: //'
    printf '%s\n' "$BUS_ID"
  } > "$WORK/public.txt"
  local np
  np=$(wc -l < "$WORK/public.txt" | tr -d ' ')
  if [ "$np" -lt 20 ]; then
    echo "::error::public pattern set collapsed to ${np} patterns; refusing a toothless scan"
    exit 1
  fi
}

# scan_text WHERE FILE: report every line of FILE (minus guard-allow lines) that a rule matches; adds to HITS.
HITS=0
scan_text() {
  local where=$1 file=$2 n line p
  n=0
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    case "$line" in *guard-allow*) continue ;; esac
    if printf '%s\n' "$line" | grep -q -i -E -f "$WORK/public.txt"; then
      while IFS= read -r p; do
        if printf '%s\n' "$line" | grep -q -i -E -e "$p"; then
          echo "::error::${where}, line ${n}: matches the public marker '${p}'"
          break
        fi
      done < "$WORK/public.txt"
      HITS=$((HITS + 1))
    elif [ -n "${COMMIT_GUARD_PRIVATE_RE:-}" ] && printf '%s\n' "$line" | grep -q -i -E -e "$COMMIT_GUARD_PRIVATE_RE"; then
      echo "::error::${where}, line ${n}: matches the private-name rule"
      HITS=$((HITS + 1))
    fi
  done < "$file"
}

self_test() {
  build_public_patterns
  # Positive controls: each class must be flagged, and an ordinary message must not be. The canary source lines carry
  # a trailing guard-allow comment for the tree guard; the planted text itself does not.
  printf '%s\n' 'ruling at bus 20261017T194309Z-18922' > "$WORK/c1"  # guard-allow
  printf '%s\n' 'receipt at adertha://adertha-agents%23190/receipts/run-1' > "$WORK/c2"  # guard-allow
  printf '%s\n' 'see the run lock at ~/.adertha/rent-live.lock' > "$WORK/c3"  # guard-allow
  printf '%s\n' 'ruling at bus 20261017T194309Z-18922 guard-allow' > "$WORK/c4"  # guard-allow
  cat > "$WORK/clean" <<'CLEAN'
fix: typo in README
The maintainer reviewed this change; families of MoE models share the route.
A follow-up read: one position differs at bucket 1 (row 15, k 31), run example-5090-1, cost USD 0.04.
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Timestamp 20261017T194309Z alone is not a message id.
CLEAN
  local f
  for f in c1 c2 c3; do
    HITS=0
    scan_text "self-test ${f}" "$WORK/$f" > /dev/null
    [ "$HITS" -eq 1 ] || { echo "::error::self-test: control ${f} was not flagged"; exit 1; }
  done
  HITS=0; scan_text "self-test c4" "$WORK/c4" > /dev/null
  [ "$HITS" -eq 0 ] || { echo "::error::self-test: a guard-allow line was flagged"; exit 1; }
  HITS=0; scan_text "self-test clean" "$WORK/clean"
  [ "$HITS" -eq 0 ] || { echo "::error::self-test: ${HITS} ordinary line(s) flagged"; exit 1; }
  if [ -n "${COMMIT_GUARD_PRIVATE_RE:-}" ]; then
    local rc=0
    printf '' | grep -q -E -e "$COMMIT_GUARD_PRIVATE_RE" || rc=$?
    [ "$rc" -le 1 ] || { echo "::error::self-test: the private-name rule is not a valid extended regex"; exit 1; }
    # Every literal alternative must match itself. The canaries come from the secret and stay in memory.
    local alt lit=0 other=0
    while IFS= read -r alt; do
      [ -n "$alt" ] || continue
      if printf '%s' "$alt" | grep -q -E '^[A-Za-z0-9 _./-]+$'; then
        printf '%s\n' "x ${alt} y" > "$WORK/p"
        HITS=0; scan_text "self-test private" "$WORK/p" > /dev/null
        [ "$HITS" -eq 1 ] || { echo "::error::self-test: a literal alternative of the private-name rule matched nothing"; exit 1; }
        lit=$((lit + 1))
      else
        other=$((other + 1))
      fi
    done < <(printf '%s\n' "$COMMIT_GUARD_PRIVATE_RE" | tr '|' '\n')
    [ "$lit" -ge 1 ] || { echo "::error::self-test: the private-name rule has no literal alternative to exercise"; exit 1; }
    echo "self-test: private-name rule compiles, ${lit} literal alternative(s) exercised, ${other} pattern alternative(s) not"
  fi
  echo "self-test: $(wc -l < "$WORK/public.txt" | tr -d ' ') public patterns; every control behaves"
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit 0
fi

: "${BASE_SHA:?BASE_SHA is required}" "${HEAD_SHA:?HEAD_SHA is required}"
if [ -z "${COMMIT_GUARD_PRIVATE_RE:-}" ]; then
  if [ "${SAME_REPO:-1}" = "1" ]; then
    echo "::error::COMMIT_GUARD_PRIVATE_RE is empty on a same-repo PR; the private-name rule would be skipped"
    exit 1
  fi
  echo "::notice::fork PR: no secrets, so the private-name rule is not applied; public rules only"
fi
self_test
HITS=0
printf '%s\n' "${PR_TITLE:-}" > "$WORK/title"
printf '%s\n' "${PR_BODY:-}" > "$WORK/body"
scan_text "PR title" "$WORK/title"
scan_text "PR body" "$WORK/body"
ncommits=0
for c in $(git rev-list "${BASE_SHA}..${HEAD_SHA}"); do
  git log -1 --format=%B "$c" > "$WORK/msg"
  scan_text "commit ${c:0:12} message" "$WORK/msg"
  ncommits=$((ncommits + 1))
done
# The tree guard can't carry the private names, so the lines this change adds are checked for them here.
if [ -n "${COMMIT_GUARD_PRIVATE_RE:-}" ]; then
  mb=$(git merge-base "$BASE_SHA" "$HEAD_SHA")
  git diff -U0 --no-color "$mb" "$HEAD_SHA" -- . ':!.github/private-markers.txt' \
    | awk '/^\+\+\+ (b\/|\/dev\/null)/ { f = substr($0, 7); next } /^\+/ { print f "\t" substr($0, 2) }' \
    > "$WORK/added"
  while IFS=$'\t' read -r f line; do
    case "$line" in *guard-allow*) continue ;; esac
    if printf '%s\n' "$line" | grep -q -i -E -e "$COMMIT_GUARD_PRIVATE_RE"; then
      echo "::error::${f}: an added line matches the private-name rule"
      HITS=$((HITS + 1))
    fi
  done < "$WORK/added"
fi
if [ "$HITS" -gt 0 ]; then
  echo ""
  echo "${HITS} line(s) carry a private control-plane reference into text that becomes public on merge."
  echo "Reword them (git commit --amend / an interactive rebase for commit messages; edit the PR title or body),"
  echo "or append 'guard-allow' on the line if the mention is deliberate and safe."
  exit 1
fi
echo "clean: PR title, body, ${ncommits} commit message(s) and the added lines"
