### SC2j Amendment 3 (pre-reading): the stack under test is pinned to Amendment 1's merge

- SC2j's box installs `E4B_SHA=ea298854` (Amendment 1's merge) explicitly; the kit comes from a later `main` checkout.
  An opt-in training change landed in between, so a package-identity check against the checkout could never pass.
- `bench/sc2/sc2_kit_hold.py package-hold --e4b-sha` refuses unless `E4B_SHA` is that commit in full, then checks the
  staged kit, `make_pin.sh` and rule 8's one driver token. `verdict-stack --stack` reads the fetched run's installed
  commit and launch `E4B_SHA` against the lane's pin, a VOID condition of SC2j's rule.
