### P126 Amendment 1 (#846): NOISY is per candidate from attempt 2 on; attempt 2 runs on another host and records it

Attempt 1 read NOISY because one runner carried a level offset of about 3 % for its whole life, which longer blocks
cannot remove. From attempt 2 on:
- **The rule.** A candidate whose blocks disagree by more than 1.5 % is ineligible, and the verdict picks among the
  eligible ones under the same 0.98 bar. NOISY is read only when no candidate is eligible.
- **Attempt 1 is untouched.** The reducer keys the rule on the record's `amendment`, so attempt 1 reduces as registered,
  byte for byte.
- **The host.** `box.json` records the card's power cap and the host's co-tenancy. Attempt 2 avoids attempt 1's machine.
- **The rest.** New predictions, a 45-case self-test, and a budget that fits the lane's remaining $1.942.
