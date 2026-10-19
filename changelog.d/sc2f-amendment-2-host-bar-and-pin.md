### SC2f Amendment 2 (#846): the host bar is clean runs >= 2, and hosts are pinned under the market rule (registration only)

- **Why.** Under the registered selection, two watcher windows found no rentable host. Over 15.4 h and 5,327
  machine-filtered reads (envelope v4), four of the five bar machines were never listed. The fifth listed one offer
  that refused every create (`no_such_ask`, four attempts, $0, nothing created).
- **The change.**
  - The bar drops from clean runs >= 3 to >= 2, and keeps every measurement-quality term (575 W on record, no host
    failure, no capped block).
  - Hosts are pinned like every other lane: up to three stable machines, under the standing spending rule. This
    replaces the per-run ACK and the watcher envelope.
  - New run ids start at `sc2f-prove-11`.
- The arms, gates, predictions, outcomes and budget are unchanged.
