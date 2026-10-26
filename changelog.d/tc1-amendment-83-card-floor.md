### TC1 amendment 83: the card floor's clock binds; SwPowerCap is recorded, Reliability refuses only when persistent

`bench/common/card_clock.py`'s default `clock` mode now judges the burn and a training run's clock log on the clock: the busy SM median
against the lane's floor, no slowdown bit on any sample, and Reliability refused only on >= 95 % of busy samples. SwPowerCap is recorded
and never refuses. A $0 scan of the store's 737 per-second training clock logs (`bench/tc1/am83_scan.py`, counts in
`bench/tc1/am83_scan.json`) showed SwPowerCap on more than 5 % of busy samples in 117 of 422 healthy-clock RTX 5090 logs, and the first
healthy card ever burned (a stock 575 W card at 2,827 MHz) refused on it. Amendment 80's bars stay available as the named `pct_clean`
mode, and the burn's JSON now records every sample. TC1's pos80 kit re-pins `card_clock.py` here; P238 and P237 re-pin in their own
changes. No reading changes: no pos80 box has run.
